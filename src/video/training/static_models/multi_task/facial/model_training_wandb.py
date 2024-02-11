import sys
import os

from pytorch_utils.models.Pose_estimation.HRNet import Modified_HRNet

# infer the path to the project
path_to_the_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir,
                 os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
# dynamic appending of the path to the project to the sys.path (6 folding up)
sys.path.append(path_to_the_project)
# the same, but change "ABAW_2023_SIU" to "datatools"
sys.path.append(path_to_the_project.replace("ABAW_2023_SIU", "datatools"))

import argparse
from torchinfo import summary
import gc
from functools import partial
from typing import Tuple, List, Dict, Optional, Union

import wandb
import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, mean_absolute_error, \
    mean_squared_error

from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, \
    Modified_EfficientNet_B4, Modified_ViT_B_16
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping, GradualLayersUnfreezer, gradually_decrease_lr
from pytorch_utils.training_utils.losses import SoftFocalLoss, RMSELoss
from src.video.training.static_models.multi_task.facial.data_preparation import load_data_and_construct_dataloaders


def evaluate_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device) -> Dict[
    object, float]:
    evaluation_metrics_classification = {'val_accuracy': accuracy_score,
                                         'val_precision': partial(precision_score, average='macro'),
                                         'val_recall': partial(recall_score, average='macro'),
                                         'val_f1': partial(f1_score, average='macro'),
                                         }
    evaluation_metrics_arousal = {'val_mae_arousal': mean_absolute_error,
                                  'val_rmse_arousal': lambda y_true, y_pred: np.sqrt(
                                      mean_squared_error(y_true, y_pred)),
                                  }
    evaluation_metrics_valence = {'val_mae_valence': mean_absolute_error,
                                  'val_rmse_valence': lambda y_true, y_pred: np.sqrt(
                                      mean_squared_error(y_true, y_pred)),
                                  }

    # create arrays for predictions and ground truth labels
    predictions_classifier = []
    ground_truth_classifier = []

    predictions_arousal = []
    ground_truth_arousal = []
    predictions_valence = []
    ground_truth_valence = []

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if not isinstance(labels, list):
                labels = [labels]
            # generate masks for each output of the model
            masks = [~torch.isnan(lb) for lb in labels]
            # move data to device
            inputs = inputs.float().to(device)

            # forward pass
            outputs = model(inputs)
            # outputs are tuple of tensors. First element is class probabilities, second element is regression output (arousal + valence)
            # transform them to list of tensors, keeping 2D
            outputs = [outputs[0].cpu(), outputs[1][:, 0].cpu().unsqueeze(1), outputs[1][:, 1].cpu().unsqueeze(1)]
            # transform ground truth labels to fit predictions and sklearn metrics (originally, it is arousal, valence, one-hot encoded labels)
            ground_truths = [labels[0][:, 2:], labels[0][:, 0].unsqueeze(1),
                             labels[0][:, 1].unsqueeze(1)]
            # transform masks to fit predictions and sklearn metrics
            masks = [masks[0][:, 2:], masks[0][:, 0].unsqueeze(1), masks[0][:, 1].unsqueeze(1)]
            shapes = [gt.shape for gt in ground_truths]
            ground_truths = [torch.masked_select(gt, mask) for gt, mask in zip(ground_truths, masks)]
            outputs = [output[mask] for output, mask in zip(outputs, masks)]
            # reshape ground truth and outputs to the original shape (keep second dimension)
            ground_truths = [gt.view(-1, shape[1]) for gt, shape in zip(ground_truths, shapes)]
            outputs = [output.view(-1, shape[1]) for output, shape in zip(outputs, shapes)]

            #  put different predictions/groundtruths to different variables
            classification_output = outputs[0]
            classification_ground_truth = ground_truths[0]
            arousal_output = outputs[1]
            arousal_ground_truth = ground_truths[1]
            valence_output = outputs[2]
            valence_ground_truth = ground_truths[2]

            # transform classification output to fit labels
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = classification_output.cpu().numpy()
            classification_output = np.argmax(classification_output, axis=-1)
            # transform ground truth labels to fit predictions and sklearn metrics
            classification_ground_truth = classification_ground_truth.cpu().numpy()
            classification_ground_truth = np.argmax(classification_ground_truth, axis=-1)

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions_classifier.append(classification_output)
            ground_truth_classifier.append(classification_ground_truth)
            predictions_arousal.append(arousal_output.cpu().numpy().squeeze())
            ground_truth_arousal.append(arousal_ground_truth.cpu().numpy().squeeze())
            predictions_valence.append(valence_output.cpu().numpy().squeeze())
            ground_truth_valence.append(valence_ground_truth.cpu().numpy().squeeze())

        # before concatenation, check if all elements have at least 1 shape length and reshape them to (-1,) else keep them as they are
        predictions_classifier = [pred.reshape(-1, ) if len(pred.shape) == 0 else pred for pred in
                                  predictions_classifier]
        ground_truth_classifier = [gt.reshape(-1, ) if len(gt.shape) == 0 else gt for gt in ground_truth_classifier]
        predictions_arousal = [pred.reshape(-1, ) if len(pred.shape) == 0 else pred for pred in predictions_arousal]
        ground_truth_arousal = [gt.reshape(-1, ) if len(gt.shape) == 0 else gt for gt in ground_truth_arousal]
        predictions_valence = [pred.reshape(-1, ) if len(pred.shape) == 0 else pred for pred in predictions_valence]
        ground_truth_valence = [gt.reshape(-1, ) if len(gt.shape) == 0 else gt for gt in ground_truth_valence]

        # concatenate all predictions and ground truth labels
        predictions_classifier = np.concatenate(predictions_classifier, axis=0)
        ground_truth_classifier = np.concatenate(ground_truth_classifier, axis=0)
        predictions_arousal = np.concatenate(predictions_arousal, axis=0)
        ground_truth_arousal = np.concatenate(ground_truth_arousal, axis=0)
        predictions_valence = np.concatenate(predictions_valence, axis=0)
        ground_truth_valence = np.concatenate(ground_truth_valence, axis=0)

        # calculate evaluation metrics
        evaluation_metrics_classifier = {
            metric: evaluation_metrics_classification[metric](ground_truth_classifier, predictions_classifier) for
            metric in evaluation_metrics_classification}
        evaluation_metrics_arousal = {
            metric: evaluation_metrics_arousal[metric](ground_truth_arousal, predictions_arousal) for
            metric in evaluation_metrics_arousal}
        evaluation_metrics_valence = {
            metric: evaluation_metrics_valence[metric](ground_truth_valence, predictions_valence) for
            metric in evaluation_metrics_valence}
        # merge all evaluation metrics
        evaluation_metrics_results = {**evaluation_metrics_classifier, **evaluation_metrics_arousal,
                                      **evaluation_metrics_valence}
        # print evaluation metrics
        print('Evaluation metrics:')
        for metric_name, metric_value in evaluation_metrics_results.items():
            print("%s: %.4f" % (metric_name, metric_value))
    # clear RAM from unused variables
    del inputs, labels, outputs, classification_output, classification_ground_truth
    torch.cuda.empty_cache()
    return evaluation_metrics_results


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
    outputs = model(inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    # outputs are tuple of tensors. First element is class probabilities, second element is regression output (arousal + valence)
    # transform them to list of tensors, keeping 2D
    outputs = [outputs[0], outputs[1][:, 0].unsqueeze(1), outputs[1][:, 1].unsqueeze(1)]
    # transform ground truth labels to fit predictions and sklearn metrics (originally, it is arousal, valence, one-hot encoded labels)
    ground_truths = [ground_truths[0][:, 2:], ground_truths[0][:, 0].unsqueeze(1), ground_truths[0][:, 1].unsqueeze(1)]
    # transform masks to fit predictions
    if masks is not None:
        masks = [masks[0][:, 2:], masks[0][:, 0].unsqueeze(1), masks[0][:, 1].unsqueeze(1)]
    # checking input parameters
    if len(criterions) != len(outputs):
        raise ValueError("Number of criterions should be equal to number of outputs of the model.")
    # calculating criterions
    # apply masks for each output of the model and the corresponding ground truth
    if masks is not None:
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
        masks = [~torch.isnan(lb) for lb in labels]
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


def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader,
                device: torch.device, class_weights: torch.Tensor, training_config) -> None:
    training_config['device'] = device
    print("____________________________________________________")
    print("Training params:")
    for key, value in training_config.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb.init(project="ABAW_2023_static_multi_task", config=training_config)
    config = wandb.config
    wandb.config.update({'best_model_save_path': wandb.run.dir}, allow_val_change=True)

    # create model
    if config.model_type == "EfficientNet-B1":
        model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=config.num_classes,
                                         num_regression_neurons=2)
    elif config.model_type == "EfficientNet-B4":
        model = Modified_EfficientNet_B4(embeddings_layer_neurons=256, num_classes=config.num_classes,
                                         num_regression_neurons=2)
    elif config.model_type == "ViT_b_16":
        model = Modified_ViT_B_16(embeddings_layer_neurons=256, num_classes=config.num_classes,
                                  num_regression_neurons=2)
    elif config.model_type == "Modified_HRNet":
        model = Modified_HRNet(pretrained=True,
                               path_to_weights=config.path_hrnet_weights,
                               embeddings_layer_neurons=256, num_classes=config.NUM_CLASSES,
                               num_regression_neurons=None,
                               consider_only_upper_body=True)
    else:
        raise ValueError("Unknown model type: %s" % config.model_type)
    # load model weights
    if not config.model_type == "Modified_HRNet":
        model.load_state_dict(torch.load(config.path_to_pretrained_model))
    # send model to GPU or CPU
    model = model.to(device)
    # print model architecture
    summary(model, (2, 3, training_config['model_input_size'][config.model_type],
                    training_config['model_input_size'][config.model_type]))

    # define all model layers (params), which will be used by optimizer
    if config.model_type == "EfficientNet-B1" or config.model_type == "EfficientNet-B4":
        model_layers = [
            *list(list(model.children())[0].features.children()),
            *list(list(model.children())[0].children())[1:],
            *list(model.children())[1:]  # added layers
        ]
    elif config.model_type == "ViT_b_16":
        model_layers = [list(model.model.children())[0],  # first conv layer
                        # encoder
                        list(list(model.model.children())[1].children())[0],  # Dropout of encoder
                        # the encoder itself
                        *list(list(list(model.model.children())[1].children())[1].children()),  # blocks of encoder
                        # end of the encoder itself
                        list(list(model.model.children())[1].children())[2],  # LayerNorm of encoder
                        list(model.model.children())[2],  # last linear layer
                        *list(model.children())[1:]  # added layers
                        ]
    elif config.model_type == "Modified_HRNet":
        model_layers = [
            # we do not take HRNet part, it should be always frozen (do not have enough data for fine-tuning)
            *list(list(model.children())[1].children()),  # new conv part
            *list(model.children())[2:],  # final part (embeddings layer and outputs)
        ]
    else:
        raise ValueError("Unknown model type: %s" % config.model_type)
    # layers unfreezer
    if config.gradual_unfreezing:
        layers_unfreezer = GradualLayersUnfreezer(model=model, layers=model_layers,
                                                  layers_per_epoch=config.unfreezing_layers_per_epoch,
                                                  layers_to_unfreeze_before_start=config.layers_to_unfreeze_before_start,
                                                  input_shape=(config.BATCH_SIZE, 3,
                                                               training_config['model_input_size'][config.model_type],
                                                               training_config['model_input_size'][config.model_type]),
                                                  verbose=True)
    # if discriminative learning is applied
    if config.discriminative_learning:
        model_parameters = gradually_decrease_lr(layers=model_layers,
                                                 initial_lr=config.discriminative_learning_initial_lr,
                                                 multiplicator=config.discriminative_learning_multiplicator,
                                                 minimal_lr=config.discriminative_learning_minimal_lr,
                                                 step=config.discriminative_learning_step,
                                                 start_layer=config.discriminative_learning_start_layer)
        for param_group in model_parameters:
            print("size: {}, lr: {}".format(param_group['params'].shape, param_group['lr']))
        print(
            'The learning rate was changed for each layer according to discriminative learning approach. The new learning rates are:')
    else:
        model_parameters = model.parameters()
    # select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    optimizer = optimizers[config.optimizer](model_parameters, lr=config.lr_max_cyclic,
                                             weight_decay=config.weight_decay)
    # Loss functions
    class_weights = class_weights.to(device)
    criterions = [SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2),
                  RMSELoss(), RMSELoss()]
    criterion_weights = config.loss_weights
    criterions = list(zip(criterions, criterion_weights))
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.annealing_period,
                                                             eta_min=config.lr_min_cyclic),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
        'Warmup_cyclic': WarmUpScheduler(optimizer=optimizer,
                                         lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                                 T_max=config.annealing_period,
                                                                                                 eta_min=config.lr_min_cyclic),
                                         len_loader=len(train_generator) // config.accumulate_gradients,
                                         warmup_steps=config.warmup_steps,
                                         warmup_start_lr=config.lr_min_warmup,
                                         warmup_mode=config.warmup_mode)
    }
    # if we use discriminative learning, we don't need LR scheduler
    if config.discriminative_learning:
        lr_scheduller = None
    else:
        lr_scheduller = lr_schedullers[config.lr_scheduller]
        # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
        if config.lr_scheduller == 'Warmup_cyclic':
            optimizer.param_groups[0]['lr'] = config.lr_min_warmup

    # early stopping
    best_val_f1 = 0
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.early_stopping_patience,
                                                 save_path=config.best_model_save_path,
                                                 mode="max")

    # train model
    for epoch in range(config.num_epochs):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        train_loss = train_epoch(model, train_generator, optimizer, criterions, device, print_step=100,
                                 accumulate_gradients=config.accumulate_gradients,
                                 batch_wise_lr_scheduller=lr_scheduller if config.lr_scheduller == 'Warmup_cyclic' else None,
                                 loss_multiplication_factor=config.loss_multiplication_factor)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metrics = evaluate_model(model, dev_generator, device)

        # update best val metrics got on validation set and log them using wandb
        # also, save model if we got better recall
        if val_metrics['val_f1'] > best_val_f1:
            best_val_f1 = val_metrics['val_f1']
            wandb.config.update({'best_val_f1': best_val_f1},
                                allow_val_change=True)
            # save best model
            if not os.path.exists(config.best_model_save_path):
                os.makedirs(config.best_model_save_path)
            torch.save(model.state_dict(), os.path.join(config.best_model_save_path, 'best_model_f1.pth'))

        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log(val_metrics, commit=False)
        wandb.log({'train_loss': train_loss})
        # update LR if needed
        if config.lr_scheduller == 'ReduceLRonPlateau':
            lr_scheduller.step(best_val_f1)
        elif config.lr_scheduller == 'Cyclic':
            lr_scheduller.step()

        # check early stopping
        early_stopping_result = early_stopping_callback(best_val_f1, model)
        if early_stopping_result:
            print("Early stopping")
            break
        # unfreeze next n layers
        if config.gradual_unfreezing:
            layers_unfreezer()
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_config_file(path_to_config_file) -> Dict[str, Union[str, int, float, bool]]:
    sys.path.insert(1, str(os.path.sep).join(path_to_config_file.split(os.path.sep)[:-1]))
    name = os.path.basename(path_to_config_file).split(".")[0]
    config = __import__(name)
    # convert it to dict
    config = vars(config)
    # exclude all system varibles
    config = {key: value for key, value in config.items() if not key.startswith("__")}
    return config


def main(training_config):
    print("Start of the script....")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get data loaders
    (train_generator, dev_generator), class_weights = load_data_and_construct_dataloaders(training_config)
    # train the model
    train_model(train_generator, dev_generator,
                device, class_weights, training_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ABAW static model training',
        epilog='Parameters: path_to_config_file: str')
    # short name -p, full name --path_to_config_file
    parser.add_argument('--path_to_config_file', '-p', type=str, help='Path to the config file', required=True)
    args = parser.parse_args()
    training_config = load_config_file(args.path_to_config_file)
    # run main script with passed args
    main(training_config)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()
