import gc
from typing import Optional, Tuple, List

import torch


def train_step(model: torch.nn.Module, criterion: torch.nn.Module,
               inputs: List[torch.Tensor], ground_truths: List[torch.Tensor]) -> List[torch.Tensor]:
    """ Performs one training step for a model.

    :param model:
        model to train
    :param criterion:
        criterion to calculate loss
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
    outputs = model(inputs) # output is either Exp:(batch_size, num_timesteps, num_classes) or VA:(batch_size, num_timesteps, 2)
    # calculating criterions
    #print("predicted classes: ", outputs[0, :, 0].detach().cpu().numpy())
    #print("labels: ", ground_truths[0, :, 0].detach().cpu().numpy())
    loss = criterion(outputs, ground_truths)
    #print("Calculated loss: ", loss.item())
    #print("----------------------------------")
    # clear RAM from unused variables
    del outputs, ground_truths
    return [loss]


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                device: torch.device, print_step: Optional[int] = 100,
                accumulate_gradients: Optional[int] = 1,
                batch_wise_lr_scheduller: Optional[object] = None,
                loss_multiplication_factor: Optional[float] = None,
                gradient_clipping_value: Optional[float]=None,
                downgrade_to_1_fps:Optional[bool]=None) -> float:
    # extract criterions and their weights
    running_loss = 0.0
    total_loss = 0.0
    counter = 0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data  # labels: arousal, valence, one-hot encoded labels
        if not isinstance(inputs, list):
            inputs = [inputs]
        # take every n-th timestep. However, in this case, we do not want to start from 0. Instead, we want to start from
        # every_n_timestep parameter
        if downgrade_to_1_fps is not False:
            labels = labels[:, 4::5]
        # move data to device
        inputs = [inp.float().to(device) for inp in inputs]
        labels = labels.float().to(device)

        # do train step
        with torch.set_grad_enabled(True):
            # form indecex of labels which should be one-hot encoded
            step_losses = train_step(model, criterion, inputs, labels)
            # normalize losses by number of accumulate gradient steps
            step_losses = [step_loss / accumulate_gradients for step_loss in step_losses]
            # backward pass
            sum_losses = sum(step_losses)
            if loss_multiplication_factor is not None:
                sum_losses = sum_losses * loss_multiplication_factor
            sum_losses.backward()
            if gradient_clipping_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_value, norm_type=2)
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