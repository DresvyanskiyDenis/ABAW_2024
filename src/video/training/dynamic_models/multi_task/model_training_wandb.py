import sys
import os

from src.video.training.dynamic_models.dynamic_models import UniModalTemporalModel
from src.video.training.dynamic_models.multi_task.loss import VALoss
from src.video.training.dynamic_models.multi_task.training_utils import train_epoch

# infer the path to the project
path_to_the_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir,
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

def evaluate_model():
    pass


def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader,
                device: torch.device, class_weights: torch.Tensor, training_config):
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
        model = UniModalTemporalModel(input_shape=config.input_shape, num_classes=config.num_classes,
                                      num_regression_neurons=config.num_regression_neurons)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    # send model to device
    model.to(device)
    # print model summary
    print(summary(model, input_size=config.input_shape))
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
    criterions = [SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2),
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


def main():
    pass


if __name__ == "__main__":
    main()

