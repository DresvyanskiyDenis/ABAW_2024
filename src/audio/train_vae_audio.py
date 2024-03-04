import sys

sys.path.append('../src')

import os
import pprint
import datetime
from copy import deepcopy

import numpy as np

import torch
from torchvision import transforms

from config import config_vae

from augmentation.wave_augmentation import RandomChoice, PolarityInversion, WhiteNoise, Gain

from data.abaw_vae_dataset import AbawVAEDataset, form_train_dataset, VAEGrouping

from net_trainer.vae_net_trainer import VAENetTrainer as NetTrainer

from loss.loss import VALoss, SoftFocalLoss, SoftFocalLossWrapper

from models.audio_vae_models import VAEModelV1, VAEModelV2, VAEModelV3

from utils.data_utils import get_source_code

from utils.accuracy_utils import recall, precision, f1, va_score, v_score, a_score
from utils.common_utils import define_seed
        

def main(config: dict) -> None:
    """Trains with configuration in the following steps:
    - Defines datasets names
    - Defines data augmentations
    - Defines ExprDatasets
    - Defines NetTrainer
    - Defines Dataloaders
    - Defines model
    - Defines weighted loss, optimizer, scheduler
    - Runs NetTrainer 

    Args:
        config (dict): Configuration dictionary
    """
    audio_root = config['FILTERED_WAV_ROOT'] if config['FILTERED'] else config['WAV_ROOT']
    video_root = config['VIDEO_ROOT']
    labels_va_root = config['LABELS_VA_ROOT']
    labels_expr_root = config['LABELS_EXPR_ROOT']
    features_root = config['FEATURES_ROOT']
    
    logs_root = config['LOGS_ROOT']
    model_cls = config['MODEL_PARAMS']['model_cls']
    model_name = config['MODEL_PARAMS']['args']['model_name']
    aug = config['AUGMENTATION']
    num_epochs = config['NUM_EPOCHS']
    batch_size = config['BATCH_SIZE']
    
    c_names = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        
    source_code = 'Configuration:\n{0}\n\nSource code:\n{1}'.format(
        pprint.pformat(config), 
        get_source_code([main, model_cls, AbawVAEDataset, NetTrainer]))    

    ds_names = {
        'va_expr_train': 'train', 
        'expr_devel': 'validation',
        'va_devel': 'validation'
    }
    
    metadata_info = {}
    all_transforms = {}
    for ds in ds_names:
        if 'train' in ds:
            metadata_info[ds] = {
                'label_filenames': form_train_dataset(train_va_root=os.path.join(labels_va_root, 'Train_Set'), 
                                                      dev_va_root=os.path.join(labels_va_root, 'Validation_Set'),
                                                      train_expr_root=os.path.join(labels_expr_root, 'Train_Set'), 
                                                      dev_expr_root=os.path.join(labels_expr_root, 'Validation_Set')),
                'dataset': '{0}_Set'.format(ds_names[ds].capitalize()),
            }

            if aug:
                all_transforms[ds] = [
                    transforms.Compose([
                        RandomChoice([PolarityInversion(), WhiteNoise(), Gain()]),
                    ]),
                ]
            else:
                all_transforms[ds] = [
                    None
                ]
        else:
            if 'va' in ds:
                metadata_info[ds] = {
                    'label_filenames': os.listdir(os.path.join(labels_va_root, '{0}_Set'.format(ds_names[ds].capitalize()))),
                    'dataset': '{0}_Set'.format(ds_names[ds].capitalize()),
                }
            
            if 'expr' in ds:
                metadata_info[ds] = {
                    'label_filenames': os.listdir(os.path.join(labels_expr_root, '{0}_Set'.format(ds_names[ds].capitalize()))),
                    'dataset': '{0}_Set'.format(ds_names[ds].capitalize()),
                }

            all_transforms[ds] = None


    datasets = {}
    for ds in ds_names:
        if 'train' in ds:
            datasets[ds] = torch.utils.data.ConcatDataset([
                AbawVAEDataset(
                    audio_root=audio_root,
                    video_root=video_root,
                    labels_va_root=labels_va_root,
                    labels_expr_root=labels_expr_root,
                    label_filenames=metadata_info[ds]['label_filenames'],
                    dataset=metadata_info[ds]['dataset'],
                    features_root=features_root,
                    va_frames_grouping=VAEGrouping.F2F,
                    expr_frames_grouping=VAEGrouping.F2S,
                    shift=2, min_w_len=2, max_w_len=4, processor_name=model_name,
                    transform=t) for t in all_transforms[ds]
                ]
            )
        else:
            datasets[ds] = AbawVAEDataset(
                    audio_root=audio_root,
                    video_root=video_root,
                    labels_va_root=labels_va_root if 'va' in ds else None,
                    labels_expr_root=labels_expr_root if 'expr' in ds else None,
                    label_filenames=metadata_info[ds]['label_filenames'],
                    dataset=metadata_info[ds]['dataset'],
                    features_root=features_root,
                    va_frames_grouping=VAEGrouping.F2F,
                    expr_frames_grouping=VAEGrouping.F2S,
                    shift=2, min_w_len=2, max_w_len=4, processor_name=model_name,
                    transform=all_transforms[ds],
                )

    
    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    experiment_name = 'MTFLW{0}{1}-{2}'.format('a-' if aug else '-',
                                               model_cls.__name__.replace('-', '_').replace('/', '_'),
                                               datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
        
    net_trainer = NetTrainer(log_root=logs_root,
                             experiment_name=experiment_name,
                             c_names=c_names,
                             classification_metrics=[f1, recall, precision],
                             regression_metrics=[va_score, v_score, a_score],
                             device=device,
                             group_predicts_fn=None,
                             source_code=source_code)
        
    dataloaders = {}
    for ds in ds_names:
        dataloaders[ds] = torch.utils.data.DataLoader(
            datasets[ds],
            batch_size=batch_size,
            shuffle=('train' in ds),
            num_workers=batch_size if batch_size < 9 else 8)
    

    model = model_cls.from_pretrained(model_name)
    
    model.to(device)
    
    class_sample_count = datasets['va_expr_train'].datasets[0].expr_labels_counts
    class_weights = torch.Tensor(max(class_sample_count) / class_sample_count).to(device)
    va_loss = VALoss()
    expr_loss = SoftFocalLossWrapper(focal_loss=SoftFocalLoss(alpha=class_weights), num_classes=len(c_names))
    loss_weights = [1, 1]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10, T_mult=1,
                                                                     eta_min=0.001 * 0.1)

    model, max_perf = net_trainer.run(model=model, loss=[va_loss, expr_loss], loss_weights=loss_weights, optimizer=optimizer, scheduler=scheduler,
                                      num_epochs=num_epochs, dataloaders=dataloaders)

    for phase in ds_names:
        if 'train' in phase:
            continue

        print()
        print(phase.capitalize())
        print('Epoch: {}, Max performance:'.format(max_perf[phase]['epoch']))
        print([metric for metric in max_perf[phase]['performance']])
        print([max_perf[phase]['performance'][metric] for metric in max_perf[phase]['performance']])
        print()


def run_vae_training() -> None:
    """Wrapper for training va/expr challenge
    """
    
    model_cls = [VAEModelV1, VAEModelV2, VAEModelV3]
    
    for augmentation in [True, False]:
        for filtered in [True, False]:
            for m_cls in model_cls:
                cfg = deepcopy(config_vae)
                cfg['FILTERED'] = filtered
                cfg['AUGMENTATION'] = augmentation
                cfg['MODEL_PARAMS']['model_cls'] = m_cls
                
                main(cfg)


if __name__ == '__main__':
    # main(config=config_vae)
    run_vae_training()