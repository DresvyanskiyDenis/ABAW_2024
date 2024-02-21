import gc
from typing import Optional, Tuple, List

from wandb.integration import torch


def train_step(model: torch.nn.Module, criterions: List[torch.nn.Module],
               inputs: List[torch.Tensor], ground_truths: List[torch.Tensor],
               masks: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
    """ Performs one training step for a model.

    :param model:
        model to train
    :param criterions:
        list of criterions to calculate loss
    :param inputs:
        list of input tensors
    :param ground_truths:
        list of ground truth tensors
    :param masks:
        list of masks for each output of the model. The size of each element is the size of the correcponding
        ground truth element of the list. It the values of the mask equal to True, the corresponding element of the
        ground truth should be used for the loss calculation. If the value of the mask is False, the corresponding
        element of the ground truth should be ignored in the loss calculation.
    :return: List[torch.Tensor]
        list of losses
    """
    # forward pass
    if len(inputs) == 1:
        inputs = inputs[0]
    outputs = model(inputs) # output is a tuple of tensors (batch_size, num_timesteps, num_classes), (batch_size, num_timesteps, 2)
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    # outputs are tuple of tensors. First element is class probabilities, second element is regression output (arousal + valence)
    # transform them to list of tensors, keeping 2D
    outputs = [outputs[0], outputs[1][:, :, 0].unsqueeze(-1), outputs[1][:, :, 1].unsqueeze(-1)]
    # transform ground truth labels to fit predictions and sklearn metrics (originally, it is arousal, valence, one-hot encoded labels)
    ground_truths = [ground_truths[0], ground_truths[1][:, :, 0].unsqueeze(-1), ground_truths[1][:, :, 1].unsqueeze(-1)]
    # transform masks to fit predictions
    if masks is not None:
        masks = [masks[0][:, :, 2:], masks[0][:, :, 0].unsqueeze(1), masks[0][:, :, 1].unsqueeze(1)] # TODO: check if it is correct
    # checking input parameters
    if len(criterions) != len(outputs):
        raise ValueError("Number of criterions should be equal to number of outputs of the model.")
    # calculating criterions
    # apply masks for each output of the model and the corresponding ground truth
    if masks is not None: # TODO: check the masking
        shapes = [gt.shape for gt in ground_truths]
        ground_truths = [torch.masked_select(gt, mask) for gt, mask in zip(ground_truths, masks)]
        outputs = [output[mask] for output, mask in zip(outputs, masks)]
        # reshape ground truth and outputs to the original shape (keep second dimension)
        ground_truths = [gt.view(-1, shape[1]) for gt, shape in zip(ground_truths, shapes)]
        outputs = [output.view(-1, shape[1]) for output, shape in zip(outputs, shapes)]
    losses = []
    for criterion, output, gt in zip(criterions, outputs, ground_truths):
        # calculate loss. If gt is empty (there are bo labels), we should not calculate loss and just append 0.0
        losses.append(criterion(output, gt) if not gt.shape[0] == 0 else torch.tensor(0.0).to(gt.device))
    # clear RAM from unused variables
    del outputs, ground_truths, masks
    return losses


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterions: List[Tuple[torch.nn.Module, float]],
                device: torch.device, print_step: Optional[int] = 100,
                accumulate_gradients: Optional[int] = 1,
                batch_wise_lr_scheduller: Optional[object] = None,
                loss_multiplication_factor: Optional[float] = None) -> float:
    """ Performs one epoch of training for a model.

    :param model: torch.nn.Module
            Model to train.
    :param train_generator: torch.utils.data.DataLoader
            Generator for training data. Note that it should output the ground truths as a tuple of torch.Tensor
            (thus, we have several outputs).
    :param optimizer: torch.optim.Optimizer
            Optimizer for training.
    :param criterions: List[Tuple[torch.nn.Module, float]]
            Loss functions for each output of the model with the corresponding weight.
    :param device: torch.device
            Device to use for training.
    :param print_step: int
            Number of mini-batches between two prints of the running loss.
    :param accumulate_gradients: Optional[int]
            Number of mini-batches to accumulate gradients for. If 1, no accumulation is performed.
    :param batch_wise_lr_scheduller: Optional[torch.optim.lr_scheduler]
            Learning rate scheduller in case we have lr scheduller that executes the lr changing every mini-batch.
    :return: float
            Average loss for the epoch.
    """
    # extract criterions and their weights
    criterions, criterion_weights = zip(*criterions)
    running_loss = 0.0
    total_loss = 0.0
    counter = 0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data  # labels: arousal, valence, one-hot encoded labels
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(labels, list):
            labels = [labels]
        # generate masks for each output of the model
        masks = [~torch.isnan(lb) for lb in labels] # TODO: check the masking generation
        # move data to device
        inputs = [inp.float().to(device) for inp in inputs]
        labels = [lb.to(device) for lb in labels]
        masks = [mask.to(device) for mask in masks]

        # do train step
        with torch.set_grad_enabled(True):
            # form indecex of labels which should be one-hot encoded
            step_losses = train_step(model, criterions, inputs, labels, masks)
            # normalize losses by number of accumulate gradient steps
            step_losses = [step_loss / accumulate_gradients for step_loss in step_losses]
            # apply weights to each loss
            step_losses = [step_loss * weight for step_loss, weight in zip(step_losses, criterion_weights)]
            # backward pass
            sum_losses = sum(step_losses)
            if loss_multiplication_factor is not None:
                sum_losses = sum_losses * loss_multiplication_factor
            sum_losses.backward()
            # update weights if we have accumulated enough gradients
            if (i + 1) % accumulate_gradients == 0 or (i + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                if batch_wise_lr_scheduller is not None:
                    batch_wise_lr_scheduller.step()

        # print statistics
        running_loss += sum_losses.item()
        total_loss += sum_losses.item()
        counter += 1
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
        # clear RAM from all the intermediate variables
        del inputs, labels, step_losses, sum_losses
    # clear RAM at the end of the epoch
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / counter