import sys
import os

from src.video.training.dynamic_models.dynamic_models import UniModalTemporalModel
from src.video.training.dynamic_models.multi_task.data_preparation import get_train_dev_dataloaders
from src.video.training.dynamic_models.multi_task.loss import VALoss, SoftFocalLossForSequence
from src.video.training.dynamic_models.multi_task.metrics import np_concordance_correlation_coefficient
from src.video.training.dynamic_models.multi_task.training_utils import train_epoch
from utils.configuration_loading import load_config_file

# infer the path to the project
path_to_the_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir,
                 os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
# dynamic appending of the path to the project to the sys.path (6 folding up)
sys.path.append(path_to_the_project)
# the same, but change "ABAW_2023_SIU" to "datatools"
sys.path.append(path_to_the_project.replace("ABAW_2023_SIU", "datatools"))
sys.path.append(path_to_the_project.replace("ABAW_2023_SIU", "simple-HRNet-master"))

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
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping, GradualLayersUnfreezer, gradually_decrease_lr
from pytorch_utils.training_utils.losses import SoftFocalLoss, RMSELoss

def evaluate_model(model:torch.nn.Module, dev_generator:torch.utils.data.DataLoader, device:torch.device) -> Dict[str, float]:
    classification_metrics = {
        'val_recall': recall_score,
        'val_precision': precision_score,
        'val_f1': f1_score,
        'val_accuracy': accuracy_score
    }
    regression_metrics = {
        'a_val_MAE' : mean_absolute_error,
        'v_val_MAE' : mean_absolute_error,
        'a_val_CCC' : np_concordance_correlation_coefficient,
        'v_val_CCC' : np_concordance_correlation_coefficient,
    }
    model.eval()
    classification_predictions = []
    classification_ground_truths = []
    regression_predictions_a = []
    regression_ground_truths_a = []
    regression_predictions_v = []
    regression_ground_truths_v = []
    with torch.no_grad():
        for i, data in enumerate(dev_generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data  # labels: arousal, valence, one-hot encoded labels
            # generate masks for each output of the model
            masks = ~torch.isnan(labels) # TODO: check the masking generation
            # move data to device
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            # forward
            outputs = model(*inputs) # outputs shape Tuple[(batch_size, num_timesteps, num_classes), (batch_size, num_timesteps, 2)]
            # separate outputs
            classification_output = outputs[0]
            regression_output_a = outputs[1][:,:, 0]
            regression_output_v = outputs[1][:,:, 1]
            # separate labels (ground truths)
            classification_gt = labels[:,:,0] # TODO: check it
            regression_gt_a = labels[:,:,1] # TODO: check it
            regression_gt_v = labels[:,:,2] # TODO: check it
            # apply masks
            classification_output = classification_output[masks[:,:,0]]
            regression_output_a = regression_output_a[masks[:,:,1]]
            regression_output_v = regression_output_v[masks[:,:,2]]
            classification_gt = classification_gt[masks[:,:,0]]
            regression_gt_a = regression_gt_a[masks[:,:,1]]
            regression_gt_v = regression_gt_v[masks[:,:,2]]
            # softmax and argmax for classification
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = torch.argmax(classification_output, dim=-1)
            classification_gt = torch.argmax(classification_gt, dim=-1)
            # append to lists
            classification_predictions.append(classification_output.detach().cpu().numpy())
            classification_ground_truths.append(classification_gt.detach().cpu().numpy())
            regression_predictions_a.append(regression_output_a.detach().cpu().numpy())
            regression_ground_truths_a.append(regression_gt_a.detach().cpu().numpy())
            regression_predictions_v.append(regression_output_v.detach().cpu().numpy())
            regression_ground_truths_v.append(regression_gt_v.detach().cpu().numpy())
    # reshape and concatenate the classification predictions and ground truths
    classification_predictions = np.concatenate([item.reshape((-1,)) for item in classification_predictions], axis=0)
    classification_ground_truths = np.concatenate([item.reshape((-1,)) for item in classification_ground_truths], axis=0)
    # reshape and concatenate the regression predictions and ground truths. Now every element of the list has shape
    # (batch_size, num_timesteps) and we need to reshape them to (num_instances,num_timesteps). where
    # num_instances = sum(element.shape[0] for element in the list)
    tmp_list = []
    for item in regression_predictions_a:
        elements = [item[i, :] for i in range(item.shape[0])]
        tmp_list.extend(elements)
    regression_predictions_a = np.array(tmp_list)
    tmp_list = []
    for item in regression_ground_truths_a:
        elements = [item[i, :] for i in range(item.shape[0])]
        tmp_list.extend(elements)
    regression_ground_truths_a = np.array(tmp_list)
    tmp_list = []
    for item in regression_predictions_v:
        elements = [item[i, :] for i in range(item.shape[0])]
        tmp_list.extend(elements)
    regression_predictions_v = np.array(tmp_list)
    tmp_list = []
    for item in regression_ground_truths_v:
        elements = [item[i, :] for i in range(item.shape[0])]
        tmp_list.extend(elements)
    regression_ground_truths_v = np.array(tmp_list)
    # calculate metrics
    classification_metrics = {key: value(classification_ground_truths, classification_predictions) for key, value in classification_metrics.items()}
    regression_metrics = {key: value(regression_ground_truths_a, regression_predictions_a) for key, value in regression_metrics.items() if 'a' in key}
    regression_metrics_2 = {key: value(regression_ground_truths_v, regression_predictions_v) for key, value in regression_metrics.items() if 'v' in key}
    regression_metrics.update(regression_metrics_2)
    # clear RAM
    del classification_predictions, classification_ground_truths, regression_predictions_a, regression_ground_truths_a, regression_predictions_v, regression_ground_truths_v
    gc.collect()
    torch.cuda.empty_cache()
    return {**classification_metrics, **regression_metrics}


def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader,
                device: torch.device, class_weights: torch.Tensor, training_config:dict):
    training_config['device'] = device
    print("____________________________________________________")
    print("Training params:")
    for key, value in training_config.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb.init(project="ABAW_2023_dynamic_multi_task", config=training_config)
    config = wandb.config
    wandb.config.update({'best_model_save_path': wandb.run.dir}, allow_val_change=True)
    # create model
    if config.model_type == "UniModalTemporalModel":
        model = UniModalTemporalModel(input_shape=(config.window_size, config.num_features), num_classes=config.num_classes,
                                      num_regression_neurons=config.num_regression_neurons)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    # send model to device
    model.to(device)
    # select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    model_parameters = model.parameters()
    optimizer = optimizers[config.optimizer](model_parameters, lr=config.lr_max_cyclic,
                                             weight_decay=config.weight_decay)
    # Loss functions
    class_weights = class_weights.to(device)
    criterions = [SoftFocalLossForSequence(softmax=True, alpha=class_weights, gamma=2, aggregation='mean'),
                  VALoss(0.5, 0.5)]
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
    lr_scheduller = lr_schedullers[config.lr_scheduller]
    # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
    if config.lr_scheduller == 'Warmup_cyclic':
        optimizer.param_groups[0]['lr'] = config.lr_min_warmup

    # early stopping
    best_val_f1 = 0
    best_val_CCC_A = 0
    best_val_CCC_V = 0
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
        val_metrics = evaluate_model(model, dev_generator, device) # TODO: evaluation

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
        if val_metrics['val_CCC_A'] > best_val_CCC_A:
            best_val_CCC_A = val_metrics['val_CCC_A']
            wandb.config.update({'best_val_CCC_A': best_val_CCC_A},
                                allow_val_change=True)
            # save best model
            if not os.path.exists(config.best_model_save_path):
                os.makedirs(config.best_model_save_path)
            torch.save(model.state_dict(), os.path.join(config.best_model_save_path, 'best_model_CCC_A.pth'))
        if val_metrics['val_CCC_V'] > best_val_CCC_V:
            best_val_CCC_V = val_metrics['val_CCC_V']
            wandb.config.update({'best_val_CCC_V': best_val_CCC_V},
                                allow_val_change=True)
            # save best model
            if not os.path.exists(config.best_model_save_path):
                os.makedirs(config.best_model_save_path)
            torch.save(model.state_dict(), os.path.join(config.best_model_save_path, 'best_model_CCC_V.pth'))

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
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()



def main(path_to_config, **params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # parse config file
    config = load_config_file(path_to_config)
    # update config with passed params
    config.update(params)
    config['path_to_fps_file'] = os.path.join(path_to_the_project, config['path_to_fps_file'])
    # get data loaders
    train_loader, dev_loader, class_weights = get_train_dev_dataloaders(config, get_class_weights=True)
    # train model
    train_model(train_loader, dev_loader, device, class_weights, config)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ABAW dynamic uni-modal model training',
        epilog='Parameters: path_to_config_file: str')
    # short name -p, full name --path_to_config_file
    parser.add_argument('--path_to_config_file', '-p', type=str, help='Path to the config file', required=True)
    parser.add_argument('--window_size', '-w', type=int, help='Window size in frames (FPS=5)', required=True)
    args = parser.parse_args()
    # run main script with passed args
    main(args.path_to_config_file, window_size=args.window_size, stride=args.window_size//2)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()

