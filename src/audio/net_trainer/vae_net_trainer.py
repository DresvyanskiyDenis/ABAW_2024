import os
import logging
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from audio.utils.accuracy_utils import conf_matrix
from audio.visualization.visualize import plot_conf_matrix
from audio.utils.common_utils import create_logger


class ProblemType(Enum):
    """Problem type Enum
    Used in NetTrainer
    """
    CLASSIFICATION: int = 1
    REGRESSION: int = 2


class VAENetTrainer:
    """Trains the model
       - Performs logging:
            Logs general information (epoch number, phase, loss, performance) in file and console
            Creates tensorboard logs for each phase
            Saves source code in file
            Saves the best model and confusion matrix of this model
       - Runs train/test/devel phases
       - Calculates performance measures
       - Calculates confusion matrix
       - Saves models
        Args:
            log_root (str): Directory for logging
            experiment_name (str): Name of experiments for logging
            c_names (list[str]): Class names to calculate the confusion matrix 
            classification_metrics (list[callable], optional): List of performance measures for classification based on the best results of which the model will be saved. 
                                      The first measure (0) in the list will be used for this, the others provide 
                                      additional information. Defaults to [].
            regression_metrics (list[callable], optional): List of performance measures for regression based on the best results of which the model will be saved. 
                                      The first measure (0) in the list will be used for this, the others provide 
                                      additional information. Defaults to [].
            device (torch.device): Device where the model will be trained
            group_predicts_fn (callable, optional): Function for grouping predicts, f.e. file-wise or windows-wise. 
                                                    It can be used to calculate performance metrics on train/devel/test sets. 
                                                    Defaults to None.
            source_code (str, optional): Source code and configuration for logging. Defaults to None.
            c_names_to_display (list[str], optional): Class names to visualize confuson matrix. Defaults to None.
        """
    def __init__(self, 
                 log_root: str, 
                 experiment_name: str,
                 c_names: list[str], 
                 device: torch.device, 
                 classification_metrics: list[callable] = [], 
                 regression_metrics: list[callable] = [], 
                 group_predicts_fn: callable = None, 
                 source_code: str = None, 
                 c_names_to_display: list[str] = None) -> None:
        self.device = device

        self.model = None
        self.loss = None
        self.optimizer = None
        self.scheduler = None
        self.loss_weights = None

        self.log_root = log_root
        self.exp_folder_name = experiment_name

        self.classification_metrics = classification_metrics
        self.regression_metrics = regression_metrics
        self.c_names = c_names
        self.c_names_to_display = c_names_to_display

        if source_code:
            os.makedirs(os.path.join(self.log_root, self.exp_folder_name, 'logs'), exist_ok=True)
            with open(os.path.join(self.log_root, self.exp_folder_name, 'logs', 'source.log'), 'w') as f:
                f.write(source_code)

        self.group_predicts_fn = group_predicts_fn

        self.logging_paths = None
        self.logger = None

    def create_loggers(self, fold_num: int = None) -> None:
        """Creates folders for logging experiments:
        - general logs (log_path)
        - models folder (model_path)
        - tensorboard logs (tb_log_path)

        Args:
            fold_num (int, optional): Used for cross-validation to specify fold number. Defaults to None.
        """
        fold_name = '' if fold_num is None else 'fold_{0}'.format(fold_num)
        self.logging_paths = {
            'log_path': os.path.join(self.log_root, self.exp_folder_name, 'logs'),
            'model_path': os.path.join(self.log_root, self.exp_folder_name, 'models', fold_name),
            'tb_log_path': os.path.join(self.log_root, self.exp_folder_name, 'tb', fold_name),
        }

        for log_path in self.logging_paths:
            if log_path == 'tb_log_path':
                continue

            os.makedirs(self.logging_paths[log_path], exist_ok=True)

        self.logger = create_logger(os.path.join(self.log_root, self.exp_folder_name, 'logs',
                                                 '{0}.log'.format(fold_name if fold_name else 'logs')),
                                    console_level=logging.NOTSET,
                                    file_level=logging.NOTSET)
    
    def run(self, 
            model: torch.nn.Module, 
            loss: list[torch.nn.modules.loss], 
            optimizer: torch.optim, 
            scheduler: torch.optim.lr_scheduler, 
            num_epochs: int,
            dataloaders: dict[torch.utils.data.dataloader.DataLoader], 
            loss_weights: list[float] = [1, 1],
            log_epochs: list[int] = [], 
            fold_num: int = None, 
            verbose: bool = True) -> None:
        """Iterates over epochs including the following steps:
        - Iterates over phases (train/devel/test phase):
            - Calls `iterate_model` function (as main loop for training/validation/testing)
            - Calculates performance measures (or metrics) using `calc_metrics` function
            - Performs logging
            - Compares performance with previous epochs on phase
            - Calculates confusion matrix
            - Saves better model and confusion matrix
            - Saves epoch/phase/loss/performance statistics in csv file

        Args:
            model (torch.nn.Module): Model instance
            loss (list[torch.nn.modules.loss]): List of loss functions: 0 - va loss, 1 - expr loss
            optimizer (torch.optim): Optimizer
            scheduler (torch.optim.lr_scheduler): Scheduler for dynamicly change LR
            num_epochs (int): Number of training epochs
            dataloaders (dict[torch.utils.data.dataloader.DataLoader]): Train, Development/Validation, Test dataloaders
            loss_weights (list[float], optional): Weights for total loss: 0 - va coefficient, 1 - expr coefficient. Defaults to [1, 1].
            log_epochs (list[int], optional): Exact epoch number for logging. Defaults to [].
            fold_num (int, optional): Used for cross-validation to specify fold number. Defaults to None.
            verbose (bool, optional): Detailed output including tqdm. Defaults to True.
        """
        phases = list(dataloaders.keys())
        self.model = model
        self.loss = loss
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.create_loggers(fold_num)
        d_global_stats = []
        
        summary = {}
        max_perf = {}
        for phase in phases:
            os.makedirs(os.path.join(self.logging_paths['tb_log_path'], phase), exist_ok=True)
            summary[phase] = SummaryWriter(os.path.join(self.logging_paths['tb_log_path'], phase))
            
            classification_main_metric_name = self.classification_metrics[0].__name__
            regression_main_metric_name = self.regression_metrics[0].__name__

            max_perf[phase] = {
                'epoch': 0,
                'performance': {
                    classification_main_metric_name: 0,
                    regression_main_metric_name: 0,
                }
            }
                
        self.logger.info(self.exp_folder_name)
        for epoch in range(1, num_epochs + 1):
            self.logger.info('Epoch {}/{}:'.format(epoch, num_epochs))
            d_epoch_stats = {'epoch': epoch}
            
            for phase, dataloader in dataloaders.items():
                if 'test' in phase and dataloader is None:
                    continue
                
                targets, predicts, sample_info, epoch_va_loss, epoch_expr_loss, epoch_loss = self.iterate_model(phase=phase,
                                                                                                                dataloader=dataloader,
                                                                                                                epoch=epoch,
                                                                                                                verbose=verbose)
                
                if 'va' in phase and 'expr' in phase:
                    self.logger.info(
                        'Epoch: {}. {}. VALoss: {:.4f}, ExprLoss: {:.4f}. LossWeights: [{}]. TotalLoss: {:.4f}.  Performance:'.format(epoch, phase.capitalize(), epoch_va_loss, 
                                                                                                                                      epoch_expr_loss, ' '.join(map(str, loss_weights)), epoch_loss))
                elif 'va' in phase and 'expr' not in phase:
                    self.logger.info(
                        'Epoch: {}. {}. VALoss: {:.4f}. Performance:'.format(epoch, phase.capitalize(), epoch_va_loss))
                elif 'va' not in phase and 'expr' in phase:
                    self.logger.info(
                        'Epoch: {}. {}. ExprLoss: {:.4f}. Performance:'.format(epoch, phase.capitalize(), epoch_expr_loss))
                
                d_epoch_stats['{}_va_loss'.format(phase)] = epoch_va_loss
                d_epoch_stats['{}_expr_loss'.format(phase)] = epoch_expr_loss
                d_epoch_stats['{}_loss'.format(phase)] = epoch_loss
                summary[phase].add_scalar('va_loss', epoch_va_loss, epoch)
                summary[phase].add_scalar('expr_loss', epoch_expr_loss, epoch)
                summary[phase].add_scalar('loss', epoch_loss, epoch)

                if 'va' in phase:
                    self.calc_performance(targets=targets[0], predicts=predicts[0], metrics=self.regression_metrics, problem_type=ProblemType.REGRESSION, 
                                          epoch=epoch, phase=phase, max_perf=max_perf, 
                                          summary=summary, d_epoch_stats=d_epoch_stats, log_epochs=log_epochs, verbose=verbose)
                else:
                    for metric in self.regression_metrics:                    
                        d_epoch_stats['{}_{}'.format(phase, metric.__name__)] = 0

                if 'expr' in phase:
                    self.calc_performance(targets=targets[1], predicts=predicts[1], metrics=self.classification_metrics, problem_type=ProblemType.CLASSIFICATION, 
                                          epoch=epoch, phase=phase, max_perf=max_perf, 
                                          summary=summary, d_epoch_stats=d_epoch_stats, log_epochs=log_epochs, verbose=verbose)
                else:
                    for metric in self.classification_metrics:                    
                        d_epoch_stats['{}_{}'.format(phase, metric.__name__)] = 0
                
            d_global_stats.append(d_epoch_stats)
            pd_global_stats = pd.DataFrame(d_global_stats)
            pd_global_stats.to_csv(os.path.join(self.log_root, 
                                                self.exp_folder_name, 
                                                'logs', 
                                                'stats.csv' if fold_num is None else 'fold_{0}.csv'.format(fold_num)),
                                   sep=';', index=False)
            
            self.logger.info('')
            
        for phase in phases[1:]:
            self.logger.info(phase.capitalize())
            self.logger.info('Epoch: {}, Max performance:'.format(max_perf[phase]['epoch']))
            self.logger.info([metric for metric in max_perf[phase]['performance']])
            self.logger.info([max_perf[phase]['performance'][metric] for metric in max_perf[phase]['performance']])
            self.logger.info('')

        for phase in phases:
            summary[phase].close()
            
        return self.model, max_perf

    def calc_performance(self, targets: torch.Tensor, predicts: torch.Tensor, metrics: list[callable], problem_type: ProblemType, 
                         epoch: int, phase: str, max_perf: dict, 
                         summary: dict[SummaryWriter], d_epoch_stats: dict, log_epochs: list[int] = [], verbose: bool = True) -> None:
        """Calculates performance, confusion matrices
        Saves the best peformance and the best model
        ! Note ! Outside dictionaries are modified, in particular: max_perf, summary, d_epoch_stats
        
        Args:
            targets (torch.Tensor): Targets tensor
            predicts (torch.Tensor): Predicts tensor
            metrics (list[callable]): List of performance measures.
            problem_type (ProblemType): Problem type: for expression challenge - classification, 
                                        for va challenge - regression.
            epoch (int): Epoch number
            phase (str): Name of phase: could be train, devel(valid), test
            max_perf (dict): Dictionary with best performance value
            summary (dict[SummaryWriter]): Summary writer of Tensorboard package
            d_epoch_stats (dict): Dictionary of statistics per epoch
            log_epochs (list[int], optional): Exact epoch number for logging. Defaults to [].
            verbose (bool, optional): Detailed output with tqdm. Defaults to True.
        """
        main_metric_name = metrics[0].__name__
        performance = {}
        for metric in metrics:
            if problem_type == ProblemType.CLASSIFICATION: 
                performance[metric.__name__] = metric(np.hstack(targets), np.asarray(predicts).reshape(-1, len(self.c_names)), average='macro')
            else:
                performance[metric.__name__] = metric(np.stack(targets), np.stack(predicts), average='macro')
        
        if verbose:
            self.logger.info([metric for metric in performance])
            if problem_type == ProblemType.CLASSIFICATION:
                self.logger.info(['{0:.3f}'.format(performance[metric] * 100) for metric in performance])
            else:
                self.logger.info(['{0:.5f}'.format(performance[metric]) for metric in performance])
                
        epoch_score = performance[main_metric_name]
        for metric in performance:
            summary[phase].add_scalar(metric, performance[metric], epoch)                    
            d_epoch_stats['{}_{}'.format(phase, metric)] = performance[metric]

        is_max_performance = (
            ((('test' in phase) or ('devel' in phase)) and (epoch_score > max_perf[phase]['performance'][main_metric_name])) or
            ((('test' in phase) or ('devel' in phase)) and (epoch in log_epochs)))
                
        if is_max_performance:
            if epoch_score > max_perf[phase]['performance'][main_metric_name]:
                max_perf[phase]['performance'] = performance
                max_perf[phase]['epoch'] = epoch
                    
            if problem_type == ProblemType.CLASSIFICATION:
                cm = conf_matrix(np.hstack(targets), np.asarray(predicts).reshape(-1, len(self.c_names)), [i for i in range(len(self.c_names))])
                res_name = 'epoch_{0}_{1}_{2}'.format(epoch, phase, epoch_score)
                plot_conf_matrix(cm, 
                                 labels=self.c_names_to_display if self.c_names_to_display else self.c_names,
                                 xticks_rotation=45,
                                 title='Confusion Matrix. {0}. UAR = {1:.3f}%'.format(phase, epoch_score * 100),
                                 save_path=os.path.join(self.logging_paths['model_path'], '{0}.svg'.format(res_name)))
                    
            self.model.cpu()
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss
            }, os.path.join(self.logging_paths['model_path'], 'epoch_{0}.pth'.format(epoch)))
                    
            self.model.to(self.device)
                
        if problem_type == ProblemType.CLASSIFICATION:
            if os.path.exists(os.path.join(self.logging_paths['model_path'], 'epoch_{0}.pth'.format(epoch))):
                cm = conf_matrix(np.hstack(targets), np.asarray(predicts).reshape(-1, len(self.c_names)), [i for i in range(len(self.c_names))])
                res_name = 'epoch_{0}_{1}_{2}'.format(epoch, phase, epoch_score)
                plot_conf_matrix(cm,
                                 labels=self.c_names_to_display if self.c_names_to_display else self.c_names,
                                 xticks_rotation=45,
                                 title='Confusion Matrix. {0}. UAR = {1:.3f}%'.format(phase, epoch_score * 100),
                                 save_path=os.path.join(self.logging_paths['model_path'], '{0}.svg'.format(res_name)))
    
    def iterate_model(self, 
                      phase: str, 
                      dataloader: torch.utils.data.dataloader.DataLoader, 
                      epoch: int = None, 
                      verbose: bool = True) -> tuple[list[np.ndarray], list[np.ndarray], list[dict], float]:
        """Main training/validation/testing loop:
        ! Note ! This loop needs to be changed if you change scheduler. Default scheduler is CosineAnnealingWarmRestarts
        - Applies softmax funstion on expression predicts

        Args:
            phase (str): Name of phase: could be train, devel(valid), test
            dataloader (torch.utils.data.dataloader.DataLoader): Dataloader of phase
            epoch (int, optional): Epoch number. Defaults to None.
            verbose (bool, optional): Detailed output with tqdm. Defaults to True.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[dict], float]: targets, 
                                                                          predicts, 
                                                                          sample_info for grouping predicts/targets, 
                                                                          epoch_loss
        """
        targets = [[], []]
        predicts = [[], []]
        sample_info = []
        
        if 'train' in phase:
            self.model.train()
        else:
            self.model.eval()
        
        running_va_loss = .0
        running_expr_loss = .0
        running_loss = .0
        iters = len(dataloader)
        
        if 'devel' in phase:
            pass

        # Iterate over data.
        for idx, data in enumerate(tqdm(dataloader, disable=not verbose)):
            inps, labs, s_info = data
            if isinstance(inps, list):
                inps = [d.to(self.device) for d in inps]
            else:
                inps = inps.to(self.device)

            if isinstance(labs, list):
                labs = [d.to(self.device) for d in labs]
            else:
                labs = labs.to(self.device)

            self.optimizer.zero_grad()

            # forward and backward
            preds = None
            with torch.set_grad_enabled('train' in phase):
                preds = self.model(inps)

                va_loss_v = 0 if -5 in labs[0] else self.loss[0](preds[0].reshape(-1, 2), labs[0].reshape(-1, 2)) # if va unlabeled
                expr_loss_v = 0 if -1 in labs[1] else self.loss[1](preds[1].reshape(-1, len(self.c_names)), labs[1].flatten()) # if exp unlabeled
                loss_value = self.loss_weights[0] * va_loss_v + self.loss_weights[1] * expr_loss_v

                # backward + optimize only if in training phase
                if ('train' in phase):
                    loss_value.backward()
                    self.optimizer.step()
                    if self.optimizer:
                        self.scheduler.step(epoch + idx / iters)

            # statistics
            running_va_loss += va_loss_v * inps.size(0) if isinstance(va_loss_v, int) else va_loss_v.item() * inps.size(0)
            running_expr_loss += expr_loss_v * inps.size(0) if isinstance(expr_loss_v, int) else expr_loss_v.item() * inps.size(0)
            running_loss += loss_value * inps.size(0) if isinstance(loss_value, int) else loss_value.item() * inps.size(0)
            
            if isinstance(labs, list):
                labs = [d.cpu().numpy() for d in labs]
            else:
                labs = labs.cpu().numpy()
            
            targets[0].extend(labs[0])
            targets[1].extend(labs[1])
            preds[1] = F.softmax(preds[1], dim=-1) # apply softmax on expression predicts

            if isinstance(labs, list):
                preds = [d.cpu().detach().numpy() for d in preds]
            else:
                preds = preds.cpu().detach().numpy()

            predicts[0].extend(preds[0])
            predicts[1].extend(preds[1])
            sample_info.extend(s_info)

        epoch_va_loss = running_va_loss / iters
        epoch_expr_loss = running_expr_loss / iters
        epoch_loss = running_loss / iters

        if self.group_predicts_fn:
            targets, predicts, sample_info = self.group_predicts_fn(np.asarray(targets), 
                                                                    np.asarray(predicts),
                                                                    sample_info)
       
        return targets, predicts, sample_info, epoch_va_loss, epoch_expr_loss, epoch_loss