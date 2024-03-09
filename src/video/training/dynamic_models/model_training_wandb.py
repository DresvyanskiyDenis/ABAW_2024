import os
import random
import sys
from typing import Tuple, Optional, Dict
import argparse
import gc

import numpy as np

# infer the path to the project
path_to_the_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir,
                 os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
# dynamic appending of the path to the project to the sys.path (6 folding up)
sys.path.append(path_to_the_project)
# the same, but change "ABAW_2023_SIU" to "datatools"
sys.path.append(path_to_the_project.replace("ABAW_2023_SIU", "datatools"))
sys.path.append(path_to_the_project.replace("ABAW_2023_SIU", "simple-HRNet-master"))

import wandb
import torch
import pandas as pd

from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping


from src.video.training.dynamic_models.loss import VALoss, SoftFocalLossForSequence
from src.video.training.dynamic_models.training_utils import train_epoch
from utils.configuration_loading import load_config_file
from src.video.training.dynamic_models.data_preparation import get_train_dev_dataloaders, \
    get_dev_resampled_and_full_fps_dicts, load_fps_file
from src.video.training.dynamic_models.evaluation_development import evaluate_on_dev_set_full_fps

from src.video.training.dynamic_models.dynamic_models import UniModalTemporalModel_v1, UniModalTemporalModel_v2, \
    UniModalTemporalModel_v3, UniModalTemporalModel_v4, UniModalTemporalModel_v5


def initialize_model(model_type: str, input_shape: Tuple[int, int],
                     num_classes: Optional[int]=None, num_regression_neurons:Optional[int]=None)-> torch.nn.Module:
    """ Inializes the model depending on the provided model type.

    :param model_type: str
        THe type of the dynamic model. Currently, v1, v2, v3, v4 are supported.
    :param input_shape: Tuple[int, int]
        The shape of the input tensor. First element is the number of frames, the second is the number of features.
    :param num_classes: Optional[int]
        The number of classes. Required for classification models.
    :param num_regression_neurons: Optional[int]
        The number of neurons in the output layer. Required for regression models.
    :return: torch.nn.Module
        The initialized model.
    """
    if model_type == "dynamic_v1":
        model = UniModalTemporalModel_v1(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif model_type == "dynamic_v2":
        model = UniModalTemporalModel_v2(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif model_type == "dynamic_v3":
        model = UniModalTemporalModel_v3(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif model_type == "dynamic_v4":
        model = UniModalTemporalModel_v4(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif model_type == "dynamic_v5":
        model = UniModalTemporalModel_v5(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    # print number of model aprameters
    training_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Model {model_type} has {training_num_params} trainable parameters out of {all_params} total parameters.")

    return model


def train_model(train_generator: torch.utils.data.DataLoader,
                dev_data_resampled:Dict[str, pd.DataFrame], dev_data_full_fps:Dict[str, pd.DataFrame],
                device: torch.device, class_weights: torch.Tensor, training_config:dict):
    training_config['device'] = device
    print("____________________________________________________")
    print("Training params:")
    for key, value in training_config.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb_config = training_config.copy()
    del wandb_config['video_to_fps_dict']
    wandb.init(project=f"ABAW_2023_dynamic_{training_config['challenge']}", config=wandb_config, entity="denisdresvyanskiy")
    config = wandb.config
    wandb.config.update({'best_model_save_path': wandb.run.dir}, allow_val_change=True)
    # create model
    model = initialize_model(training_config['model_type'], input_shape=(training_config['window_size'], training_config['num_features']),
                                num_classes=training_config['num_classes'], num_regression_neurons=training_config['num_regression_neurons'])
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
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if config.challenge == 'VA':
        criterion = VALoss(0.5, 0.5)
    elif config.challenge == 'Exp':
        criterion = SoftFocalLossForSequence(softmax=True, alpha=class_weights, gamma=2, aggregation='mean')
    else:
        raise ValueError(f"Unknown challenge: {config.challenge}")
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
    best_metric_values = {'val_f1_best':0} if config.challenge == 'Exp' else {'val_CCC_A_best':0, 'val_CCC_V_best':0}
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.early_stopping_patience,
                                                 save_path=config.best_model_save_path,
                                                 mode="max")
    # train model
    for epoch in range(config.num_epochs):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        train_loss = train_epoch(model, train_generator, optimizer, criterion, device, print_step=100,
                                 accumulate_gradients=config.accumulate_gradients,
                                 batch_wise_lr_scheduller=lr_scheduller if config.lr_scheduller == 'Warmup_cyclic' else None,
                                 loss_multiplication_factor=config.loss_multiplication_factor,
                                 gradient_clipping_value=config.gradient_clipping)
        print("Train loss: %.10f" % train_loss)
        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metrics = evaluate_on_dev_set_full_fps(dev_set_full_fps=dev_data_full_fps, dev_set_resampled=dev_data_resampled,
                                 video_to_fps=training_config['video_to_fps_dict'], model=model, labels_type=config.challenge,
                                 feature_columns=config.feature_columns,
                                 labels_columns=config.labels_columns,
                                 window_size=config.window_size, device=device,
                                 batch_size=config.batch_size, resampled_fps=config.common_fps)
        print(val_metrics)

        # update best val metrics got on validation set and log them using wandb # TODO: write separate function on updating wandb metrics
        if config.challenge == 'Exp':
            if val_metrics['val_f1'] > best_metric_values['val_f1_best']:
                best_metric_values['val_f1_best'] = val_metrics['val_f1']
                wandb.config.update({'best_val_f1': best_metric_values['val_f1_best']},
                                    allow_val_change=True)
                # save best model
                if not os.path.exists(config.best_model_save_path):
                    os.makedirs(config.best_model_save_path)
                torch.save(model.state_dict(), os.path.join(config.best_model_save_path, 'best_model_f1.pth'))
        else:
            if val_metrics['val_CCC_A'] > best_metric_values['val_CCC_A_best']:
                best_metric_values['val_CCC_A_best'] = val_metrics['val_CCC_A']
                wandb.config.update({'best_val_CCC_A': best_metric_values['val_CCC_A_best']},
                                    allow_val_change=True)
                # save best model
                if not os.path.exists(config.best_model_save_path):
                    os.makedirs(config.best_model_save_path)
                torch.save(model.state_dict(), os.path.join(config.best_model_save_path, 'best_model_CCC_A.pth'))
            if val_metrics['val_CCC_V'] > best_metric_values['val_CCC_V_best']:
                best_metric_values['val_CCC_V_best'] = val_metrics['val_CCC_V']
                wandb.config.update({'best_val_CCC_V': best_metric_values['val_CCC_V_best']},
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
        if config.lr_scheduller == 'Cyclic':
            lr_scheduller.step()
        elif config.lr_scheduller == 'ReduceLRonPlateau':
            raise NotImplementedError("ReduceLRonPlateau is not implemented yet")
        # check early stopping
        metric_for_early_stopping = val_metrics['val_f1'] if config.challenge == 'Exp' else (val_metrics['val_CCC_A'] + val_metrics['val_CCC_V'])/2.
        early_stopping_result = early_stopping_callback(metric_for_early_stopping, model)
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
    # load additional parameters to add to config
    feature_columns = [f'embedding_{i}' for i in range(256)]
    if config['challenge'] == 'Exp':
        labels_columns = [f'category_{i}' for i in range(config['num_classes'])]
    else:
        labels_columns = ['arousal', 'valence']
    # update config with additional loaded parameters
    config['path_to_fps_file'] = os.path.join(path_to_the_project, config['path_to_fps_file'])
    video_to_fps = load_fps_file(config['path_to_fps_file'])
    config['video_to_fps_dict'] = video_to_fps
    config['feature_columns'] = feature_columns
    config['labels_columns'] = labels_columns
    # fixate the seed
    random.seed(config['splitting_seed'])
    os.environ['PYTHONHASHSEED'] = str(config['splitting_seed'])
    np.random.seed(config['splitting_seed'])
    torch.manual_seed(config['splitting_seed'])
    torch.cuda.manual_seed(config['splitting_seed'])
    torch.backends.cudnn.deterministic = True

    # get data loaders
    if config['challenge'] == 'Exp':
        train_loader, dev_loader, class_weights = get_train_dev_dataloaders(config, get_class_weights=True)
    else:
        train_loader, dev_loader = get_train_dev_dataloaders(config, get_class_weights=False)
        class_weights = None
    # get resampled and full fps dev data
    dev_resampled, dev_full_fps = get_dev_resampled_and_full_fps_dicts(config)
    # train model
    train_model(train_loader, dev_resampled, dev_full_fps, device, class_weights, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ABAW dynamic uni-modal model training')
    # short name -p, full name --path_to_config_file
    parser.add_argument('--path_to_config_file', '-p', type=str, help='Path to the config file', required=True)
    parser.add_argument('--model_type', '-m', type=str, help='Type of the model. dynamic_v1, dynamic_v2, dynamic_v3, dynamic_v4', required=True)
    parser.add_argument('--challenge', type=str, help='Challenge. VA or Exp.', required=True)
    parser.add_argument('--window_size', '-w', type=int, help='Window size in frames (FPS=5)', required=True)
    parser.add_argument('--normalization', '-n', type=str, help='Normalization type', required=True)
    args = parser.parse_args()
    # run main script with passed args
    main(args.path_to_config_file, window_size=args.window_size, stride=args.window_size//args.window_size*2, challenge=args.challenge,
         model_type=args.model_type, normalization = args.normalization)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()

