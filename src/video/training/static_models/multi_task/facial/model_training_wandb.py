import sys

from src.video.training.static_models.multi_task.facial.data_preparation import load_data_and_construct_dataloaders

sys.path.append(
    "/nfs/home/ddresvya/scripts/datatools/")  # path to the datatools project (https://github.com/DresvyanskiyDenis/datatools)
sys.path.append(
    "/nfs/home/ddresvya/scripts/ABAW_2023_SIU/")  # TODO: path to the project (https://github.com/DresvyanskiyDenis/ABAW_2023_SIU)

import argparse
from torchinfo import summary
import gc
import os
from functools import partial
from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, mean_absolute_error, \
    mean_squared_error

from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, \
    Modified_EfficientNet_B4, Modified_ViT_B_16
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping, GradualLayersUnfreezer, gradually_decrease_lr
from pytorch_utils.training_utils.losses import SoftFocalLoss, RMSELoss

import wandb



def evaluate_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device) -> Dict[
    object, float]:
    evaluation_metrics_classification = {'val_accuracy': accuracy_score,
                                         'val_precision': partial(precision_score, average='macro'),
                                         'val_recall': partial(recall_score, average='macro'),
                                         'val_f1': partial(f1_score, average='macro'),
                                         'val_mae_arousal': mean_absolute_error,
                                         'val_mae_valence': mean_absolute_error,
                                         'val_rmse_arousal': lambda y_true, y_pred: np.sqrt(
                                             mean_squared_error(y_true, y_pred)),
                                         'val_rmse_valence': lambda y_true, y_pred: np.sqrt(
                                             mean_squared_error(y_true, y_pred))
                                         }

    # create arrays for predictions and ground truth labels
    predictions_classifier = []
    ground_truth_classifier = []

    predictions_arousal = []
    ground_truth_arousal = []
    predictions_valence = []
    ground_truth_valence = []

    # start evaluation # TODO: correct the evaluation function
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)

            # forward pass
            outputs = model(inputs)
            classification_output = outputs[0]

            # transform classification output to fit labels
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = classification_output.cpu().numpy().squeeze()
            classification_output = np.argmax(classification_output, axis=-1)

            # transform ground truth labels to fit predictions and sklearn metrics
            classification_ground_truth = labels.cpu().numpy().squeeze()
            classification_ground_truth = np.argmax(classification_ground_truth, axis=-1)

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions_classifier.append(classification_output)
            ground_truth_classifier.append(classification_ground_truth)

        # concatenate all predictions and ground truth labels
        predictions_classifier = np.concatenate(predictions_classifier, axis=0)
        ground_truth_classifier = np.concatenate(ground_truth_classifier, axis=0)

        # calculate evaluation metrics
        evaluation_metrics_classifier = {
            metric: evaluation_metrics_classification[metric](ground_truth_classifier, predictions_classifier) for
            metric in evaluation_metrics_classification}
        # print evaluation metrics
        print('Evaluation metrics for classifier:')
        for metric_name, metric_value in evaluation_metrics_classifier.items():
            print("%s: %.4f" % (metric_name, metric_value))
    # clear RAM from unused variables
    del inputs, labels, outputs, classification_output, classification_ground_truth
    torch.cuda.empty_cache()
    return evaluation_metrics_classifier


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
    # checking input parameters
    if len(criterions) != len(outputs):
        raise ValueError("Number of criterions should be equal to number of outputs of the model.")
    # calculating criterions
    # apply masks for each output of the model and the corresponding ground truth
    if masks is not None:
        ground_truths = [gt[mask] for gt, mask in zip(ground_truths, masks)]
        outputs = [output[mask] for output, mask in zip(outputs, masks)]
    losses = []
    for criterion, output, gt in zip(criterions, outputs, ground_truths):
        losses.append(criterion(output, gt))
    # clear RAM from unused variables
    del outputs, ground_truths
    return losses


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterions: List[torch.nn.Module],
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
    :param criterions: List[torch.nn.Module]
            Loss functions for each output of the model.
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

    running_loss = 0.0
    total_loss = 0.0
    counter = 0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(labels, list):
            labels = [labels]
        # generate masks for each output of the model
        masks = [~torch.isnan(lb) for lb in labels]
        # move data to device
        inputs = [inp.float().to(device) for inp in inputs]
        labels = [lb.to(device) for lb in labels]

        # do train step
        with torch.set_grad_enabled(True):
            # form indecex of labels which should be one-hot encoded
            step_losses = train_step(model, criterions, inputs, labels, masks)
            # normalize losses by number of accumulate gradient steps
            step_losses = [step_loss / accumulate_gradients for step_loss in step_losses] # TODO: make weights for each loss
            # categorical loss has 1.0 weight, while rmse losses has 0.5 weight each
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
    # metaparams
    """metaparams = {
    	"MODEL_WEIGHTS_PATH": model_weights_path,
    	"MODEL_INPUT_SIZE":training_config.MODEL_INPUT_SIZE,
        "device": device,
        # general params
        "architecture": MODEL_TYPE,
        "MODEL_TYPE": MODEL_TYPE,
        "dataset": "AffWild2_Exp",
        "BEST_MODEL_SAVE_PATH": training_config.BEST_MODEL_SAVE_PATH,
        "NUM_WORKERS": training_config.NUM_WORKERS,
        # model architecture
        "NUM_CLASSES": training_config.NUM_CLASSES,
        # training metaparams
        "NUM_EPOCHS": training_config.NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "OPTIMIZER": training_config.OPTIMIZER,
        "AUGMENT_PROB": training_config.AUGMENT_PROB,
        "EARLY_STOPPING_PATIENCE": training_config.EARLY_STOPPING_PATIENCE,
        "WEIGHT_DECAY": training_config.WEIGHT_DECAY,
        # LR scheduller params
        "LR_SCHEDULLER": training_config.LR_SCHEDULLER,
        "ANNEALING_PERIOD": training_config.ANNEALING_PERIOD,
        "LR_MAX_CYCLIC": training_config.LR_MAX_CYCLIC,
        "LR_MIN_CYCLIC": training_config.LR_MIN_CYCLIC,
        "LR_MIN_WARMUP": training_config.LR_MIN_WARMUP,
        "WARMUP_STEPS": training_config.WARMUP_STEPS,
        "WARMUP_MODE": training_config.WARMUP_MODE,
        # gradual unfreezing (if applied)
        "GRADUAL_UNFREEZING": GRADUAL_UNFREEZING,
        "UNFREEZING_LAYERS_PER_EPOCH": training_config.UNFREEZING_LAYERS_PER_EPOCH,
        "LAYERS_TO_UNFREEZE_BEFORE_START": training_config.LAYERS_TO_UNFREEZE_BEFORE_START,
        # discriminative learning
        "DISCRIMINATIVE_LEARNING": DISCRIMINATIVE_LEARNING,
        "DISCRIMINATIVE_LEARNING_INITIAL_LR": training_config.DISCRIMINATIVE_LEARNING_INITIAL_LR,
        "DISCRIMINATIVE_LEARNING_MINIMAL_LR": training_config.DISCRIMINATIVE_LEARNING_MINIMAL_LR,
        "DISCRIMINATIVE_LEARNING_MULTIPLICATOR": training_config.DISCRIMINATIVE_LEARNING_MULTIPLICATOR,
        "DISCRIMINATIVE_LEARNING_STEP": training_config.DISCRIMINATIVE_LEARNING_STEP,
        "DISCRIMINATIVE_LEARNING_START_LAYER": training_config.DISCRIMINATIVE_LEARNING_START_LAYER,
        # loss params
        "loss_multiplication_factor": loss_multiplication_factor,
    }"""
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
    else:
        raise ValueError("Unknown model type: %s" % config.model_type)
    # load model weights
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


def load_config_file(path_to_config_file)->Dict[str, Union[str, int, float, bool]]:
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
