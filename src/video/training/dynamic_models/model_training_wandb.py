import os
import sys
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
import gc

import wandb
import torch

from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping

from src.video.training.dynamic_models.dynamic_models import UniModalTemporalModel
from src.video.training.dynamic_models.loss import VALoss, SoftFocalLossForSequence
from src.video.training.dynamic_models.training_utils import train_epoch
from utils.configuration_loading import load_config_file
from src.video.training.dynamic_models.data_preparation import get_train_dev_dataloaders
from src.video.training.dynamic_models.evaluation_development import evaluate_on_dev_set_full_fps




def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader,
                device: torch.device, class_weights: torch.Tensor, training_config:dict):
    training_config['device'] = device
    print("____________________________________________________")
    print("Training params:")
    for key, value in training_config.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb.init(project=f"ABAW_2023_dynamic_{training_config['challenge']}", config=training_config, entity="denisdresvyanskiy")
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
                                 loss_multiplication_factor=config.loss_multiplication_factor)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metrics = evaluate_on_dev_set_full_fps() # TODO: evaluation

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
    config['path_to_fps_file'] = os.path.join(path_to_the_project, config['path_to_fps_file'])
    # get data loaders
    if config['challenge'] == 'Exp':
        train_loader, dev_loader, class_weights = get_train_dev_dataloaders(config, get_class_weights=True)
    else:
        train_loader, dev_loader = get_train_dev_dataloaders(config, get_class_weights=False)
        class_weights = None
    # train model
    train_model(train_loader, dev_loader, device, class_weights, config)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ABAW dynamic uni-modal model training',
        epilog='Parameters: path_to_config_file: str')
    # short name -p, full name --path_to_config_file
    parser.add_argument('--path_to_config_file', '-p', type=str, help='Path to the config file', required=True)
    parser.add_argument('--challenge', type=str, help='Challenge. VA or Exp.', required=True)
    parser.add_argument('--window_size', '-w', type=int, help='Window size in frames (FPS=5)', required=True)
    args = parser.parse_args()
    # run main script with passed args
    main(args.path_to_config_file, window_size=args.window_size, stride=args.window_size//2, challenge=args.challenge)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()

