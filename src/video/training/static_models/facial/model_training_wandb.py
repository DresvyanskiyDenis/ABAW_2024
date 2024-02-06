import sys
sys.path.append("/nfs/home/ddresvya/scripts/datatools/")
sys.path.append("/nfs/home/ddresvya/scripts/ABAW_2023_SIU/")
sys.path.append("/nfs/home/ddresvya/scripts/simple-HRNet-master/")

import argparse
from torchinfo import summary
import gc
import os
from functools import partial
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import training_config
from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, \
    Modified_EfficientNet_B4, Modified_ViT_B_16
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping, GradualLayersUnfreezer, gradually_decrease_lr
from pytorch_utils.training_utils.losses import SoftFocalLoss

import wandb

from data_preparation import load_data_and_construct_dataloaders

def evaluate_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device) -> Dict[object, float]:
    evaluation_metrics_classification = {'val_accuracy_classification': accuracy_score,
                                         'val_precision_classification': partial(precision_score, average='macro'),
                                         'val_recall_classification': partial(recall_score, average='macro'),
                                         'val_f1_classification': partial(f1_score, average='macro')
                                         }

    # create arrays for predictions and ground truth labels
    predictions_classifier= []
    ground_truth_classifier = []

    # start evaluation
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


def train_step(model: torch.nn.Module, criterion: torch.nn.Module,
               inputs: Tuple[torch.Tensor, ...], ground_truth: torch.Tensor,
               device: torch.device) -> List:
    """ Performs one training step for a model.

    :param model: torch.nn.Module
            Model to train.
    :param criterion: torch.nn.Module
            Loss functions for each output of the model.
    :param inputs: Tuple[torch.Tensor,...]
            Inputs for the model.
    :param ground_truth: torch.Tensor
            Ground truths for the model. SHould be passed as one-hot encoded tensors
    :param device: torch.device
            Device to use for training.
    :return:
    """
    # forward pass
    output = model(inputs)
    output = output[0]
    # criterion
    classification_criterion = criterion
    # calculate loss based on mask
    ground_truth = ground_truth.to(device)
    loss = classification_criterion(output, ground_truth)

    # clear RAM from unused variables
    del output, ground_truth

    return [loss]


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                device: torch.device, print_step: int = 100,
                accumulate_gradients: Optional[int] = 1,
                warmup_lr_scheduller: Optional[object] = None,
                loss_multiplication_factor:Optional[float]=None) -> float:
    """ Performs one epoch of training for a model.

    :param model: torch.nn.Module
            Model to train.
    :param train_generator: torch.utils.data.DataLoader
            Generator for training data. Note that it should output the ground truths as a tuple of torch.Tensor
            (thus, we have several outputs).
    :param optimizer: torch.optim.Optimizer
            Optimizer for training.
    :param criterion: torch.nn.Module
            Loss functions for each output of the model.
    :param device: torch.device
            Device to use for training.
    :param print_step: int
            Number of mini-batches between two prints of the running loss.
    :param accumulate_gradients: Optional[int]
            Number of mini-batches to accumulate gradients for. If 1, no accumulation is performed.
    :param warmup_lr_scheduller: Optional[torch.optim.lr_scheduler]
            Learning rate scheduller in case we have warmup lr scheduller. In that case, the learning rate is being changed
            after every mini-batch, therefore should be passed to this function.
    :return: float
            Average loss for the epoch.
    """

    running_loss = 0.0
    total_loss = 0.0
    counter = 0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.float()
        inputs = inputs.to(device)

        # do train step
        with torch.set_grad_enabled(True):
            # form indecex of labels which should be one-hot encoded
            step_losses = train_step(model, criterion, inputs, labels, device)
            # normalize losses by number of accumulate gradient steps
            step_losses = [step_loss / accumulate_gradients for step_loss in step_losses]
            # backward pass
            sum_losses = sum(step_losses)
            if loss_multiplication_factor is not None:
                sum_losses = sum_losses * loss_multiplication_factor
            sum_losses.backward()
            # update weights if we have accumulated enough gradients
            if (i + 1) % accumulate_gradients == 0 or (i + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_lr_scheduller is not None:
                    warmup_lr_scheduller.step()

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
                device:str, model_weights_path:str,
                class_weights: torch.Tensor, MODEL_TYPE:str, BATCH_SIZE:int, ACCUMULATE_GRADIENTS:int, GRADUAL_UNFREEZING:Optional[bool]=False,
                DISCRIMINATIVE_LEARNING:Optional[bool]=False,
                loss_multiplication_factor:Optional[float]=None) -> None:
    print("Start of the model training. Gradual_unfreezing:%s, Discriminative_lr:%s" % (GRADUAL_UNFREEZING,
                                                                                       DISCRIMINATIVE_LEARNING))
    # metaparams
    metaparams = {
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
    }
    print("____________________________________________________")
    print("Training params:")
    for key, value in metaparams.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb.init(project="ABAW_2023_Exp_static", config=metaparams)
    config = wandb.config
    wandb.config.update({'BEST_MODEL_SAVE_PATH':wandb.run.dir}, allow_val_change=True)

    # create model
    device = torch.device(device)
    if config.MODEL_TYPE == "EfficientNet-B1":
        model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=config.NUM_CLASSES,
                                         num_regression_neurons=2)
    elif config.MODEL_TYPE == "EfficientNet-B4":
        model = Modified_EfficientNet_B4(embeddings_layer_neurons=256, num_classes=config.NUM_CLASSES,
                                          num_regression_neurons=2)
    elif config.MODEL_TYPE == "ViT_b_16":
        model = Modified_ViT_B_16(embeddings_layer_neurons=256, num_classes=config.NUM_CLASSES,
                                  num_regression_neurons=2)
    else:
        raise ValueError("Unknown model type: %s" % config.MODEL_TYPE)
    # load model weights
    model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH))
    # send model to GPU or CPU
    model = model.to(device)
    # print model architecture
    summary(model, (2, 3, training_config.MODEL_INPUT_SIZE[config.MODEL_TYPE], training_config.MODEL_INPUT_SIZE[config.MODEL_TYPE]))

    # define all model layers (params), which will be used by optimizer
    if config.MODEL_TYPE == "EfficientNet-B1" or config.MODEL_TYPE == "EfficientNet-B4":
        model_layers = [
            *list(list(model.children())[0].features.children()),
            *list(list(model.children())[0].children())[1:],
            *list(model.children())[1:]  # added layers
        ]
    elif config.MODEL_TYPE == "ViT_b_16":
        model_layers = [ list(model.model.children())[0], # first conv layer
                         # encoder
                         list(list(model.model.children())[1].children())[0], # Dropout of encoder
                            # the encoder itself
                            *list(list(list(model.model.children())[1].children())[1].children()), # blocks of encoder
                            # end of the encoder itself
                         list(list(model.model.children())[1].children())[2], # LayerNorm of encoder
                         list(model.model.children())[2], # last linear layer
            *list(model.children())[1:] # added layers
        ]
    else:
        raise ValueError("Unknown model type: %s" % config.MODEL_TYPE)
    # layers unfreezer
    if GRADUAL_UNFREEZING:
        layers_unfreezer = GradualLayersUnfreezer(model=model, layers=model_layers,
                                                  layers_per_epoch=config.UNFREEZING_LAYERS_PER_EPOCH,
                                                  layers_to_unfreeze_before_start=config.LAYERS_TO_UNFREEZE_BEFORE_START,
                                                  input_shape=(config.BATCH_SIZE, 3, training_config.MODEL_INPUT_SIZE[config.MODEL_TYPE], training_config.MODEL_INPUT_SIZE[config.MODEL_TYPE]),
                                                  verbose=True)
    # if discriminative learning is applied
    if DISCRIMINATIVE_LEARNING:
        model_parameters = gradually_decrease_lr(layers=model_layers, initial_lr=config.DISCRIMINATIVE_LEARNING_INITIAL_LR,
                          multiplicator=config.DISCRIMINATIVE_LEARNING_MULTIPLICATOR, minimal_lr=config.DISCRIMINATIVE_LEARNING_MINIMAL_LR,
                          step=config.DISCRIMINATIVE_LEARNING_STEP, start_layer=config.DISCRIMINATIVE_LEARNING_START_LAYER)
        for param_group in model_parameters:
            print("size: {}, lr: {}".format(param_group['params'].shape, param_group['lr']))
        print('The learning rate was changed for each layer according to discriminative learning approach. The new learning rates are:')
    else:
        model_parameters = model.parameters()
    # select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    optimizer = optimizers[config.OPTIMIZER](model_parameters, lr=config.LR_MAX_CYCLIC,
                                             weight_decay=config.WEIGHT_DECAY)
    # Loss functions
    class_weights = class_weights.to(device)
    criterion = SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2)
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ANNEALING_PERIOD,
                                                             eta_min=config.LR_MIN_CYCLIC),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
        'Warmup_cyclic': WarmUpScheduler(optimizer=optimizer,
                                         lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                                 T_max=config.ANNEALING_PERIOD,
                                                                                                 eta_min=config.LR_MIN_CYCLIC),
                                         len_loader=len(train_generator)//ACCUMULATE_GRADIENTS,
                                         warmup_steps=config.WARMUP_STEPS,
                                         warmup_start_lr=config.LR_MIN_WARMUP,
                                         warmup_mode=config.WARMUP_MODE)
    }
    # if we use discriminative learning, we don't need LR scheduler
    if DISCRIMINATIVE_LEARNING:
        lr_scheduller = None
    else:
        lr_scheduller = lr_schedullers[config.LR_SCHEDULLER]
        # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
        if config.LR_SCHEDULLER == 'Warmup_cyclic':
            optimizer.param_groups[0]['lr'] = config.LR_MIN_WARMUP

    # early stopping
    best_val_recall_classification = 0
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.EARLY_STOPPING_PATIENCE,
                                                 save_path=config.BEST_MODEL_SAVE_PATH,
                                                 mode="max")

    # train model
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        train_loss = train_epoch(model, train_generator, optimizer, criterion, device, print_step=100,
                                 accumulate_gradients=ACCUMULATE_GRADIENTS,
                                 warmup_lr_scheduller=lr_scheduller if config.LR_SCHEDULLER == 'Warmup_cyclic' else None,
                                 loss_multiplication_factor= config.loss_multiplication_factor)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metrics_classification = evaluate_model(model, dev_generator, device)

        # update best val metrics got on validation set and log them using wandb
        # also, save model if we got better recall
        if val_metrics_classification['val_recall_classification'] > best_val_recall_classification:
            best_val_recall_classification = val_metrics_classification['val_recall_classification']
            wandb.config.update({'best_val_recall_classification': best_val_recall_classification}, allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model_recall.pth'))

        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log(val_metrics_classification, commit=False)
        wandb.log({'train_loss': train_loss})
        # update LR if needed
        if config.LR_SCHEDULLER == 'ReduceLRonPlateau':
            lr_scheduller.step(best_val_recall_classification)
        elif config.LR_SCHEDULLER == 'Cyclic':
            lr_scheduller.step()

        # check early stopping
        early_stopping_result = early_stopping_callback(best_val_recall_classification, model)
        if early_stopping_result:
            print("Early stopping")
            break
        # unfreeze next n layers
        if GRADUAL_UNFREEZING:
            layers_unfreezer()
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(challenge:str, device:str, model_type, model_weights_path,
         batch_size, accumulate_gradients, gradual_unfreezing, discriminative_learning, loss_multiplication_factor):
    print("Start of the script....")
    # get data loaders
    (train_generator, dev_generator), class_weights = load_data_and_construct_dataloaders(
        model_type=model_type,
        batch_size=batch_size,
        return_class_weights=True,
        challenge=challenge)
    # train the model
    train_model(train_generator=train_generator, dev_generator=dev_generator,class_weights=class_weights,
                MODEL_TYPE=model_type, model_weights_path=model_weights_path,
                BATCH_SIZE=batch_size, ACCUMULATE_GRADIENTS=accumulate_gradients,
                GRADUAL_UNFREEZING=gradual_unfreezing, DISCRIMINATIVE_LEARNING=discriminative_learning,
                loss_multiplication_factor=loss_multiplication_factor,
                device=device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Emotion Recognition model training',
        epilog='Parameters: model_type, batch_size, accumulate_gradients, gradual_unfreezing, discriminative_learning')
    parser.add_argument('--challenge', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_weights_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--accumulate_gradients', type=int, required=True)
    parser.add_argument('--gradual_unfreezing', type=int, required=True)
    parser.add_argument('--discriminative_learning', type=int, required=True)
    parser.add_argument('--loss_multiplication_factor', type=float, required=False, default=1.0)
    args = parser.parse_args()
    # turn passed args from int to bool
    print("Passed args: ", args)
    # check arguments
    if args.challenge not in ['VA', 'Exp']:
        raise ValueError("challenge should be either VA or Exp. Got %s" % args.challenge)
    if args.model_type not in ['EfficientNet-B1', 'EfficientNet-B4', 'ViT_b_16']:
        raise ValueError("model_type should be either EfficientNet-B1, EfficientNet-B4, or ViT_b_16. Got %s" % args.model_type)
    if args.batch_size < 1:
        raise ValueError("batch_size should be greater than 0")
    if args.accumulate_gradients < 1:
        raise ValueError("accumulate_gradients should be greater than 0")
    if args.gradual_unfreezing not in [0,1]:
        raise ValueError("gradual_unfreezing should be either 0 or 1")
    if args.discriminative_learning not in [0,1]:
        raise ValueError("discriminative_learning should be either 0 or 1")
    # convert args to bool
    gradual_unfreezing = True if args.gradual_unfreezing == 1 else False
    discriminative_learning = True if args.discriminative_learning == 1 else False
    model_type = args.model_type
    model_weights_path = args.model_weights_path
    batch_size = args.batch_size
    accumulate_gradients = args.accumulate_gradients
    loss_multiplication_factor = args.loss_multiplication_factor
    device = args.device
    # run main script with passed args
    main(challenge=args.challenge,
         device=device,
         model_type = model_type, model_weights_path=model_weights_path,
         batch_size=batch_size, accumulate_gradients=accumulate_gradients,
         gradual_unfreezing=gradual_unfreezing,
         discriminative_learning=discriminative_learning,
         loss_multiplication_factor=loss_multiplication_factor)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()
