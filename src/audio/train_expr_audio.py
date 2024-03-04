import sys

sys.path.append('../src')

import os
import pprint
import datetime
from copy import deepcopy

import numpy as np

import torch
from torchvision import transforms

from config import config_expr

from augmentation.wave_augmentation import RandomChoice, PolarityInversion, WhiteNoise, Gain

from data.abaw_vae_dataset import AbawVAEDataset, VAEGrouping

from net_trainer.net_trainer import NetTrainer, ProblemType

from loss.loss import SoftFocalLoss, SoftFocalLossWrapper

from models.audio_expr_models import ExprModelV1, ExprModelV2, ExprModelV3

from utils.data_utils import get_source_code

from utils.accuracy_utils import recall, precision, f1
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
    labels_root = config['LABELS_ROOT']
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
        'train': 'train', 
        'devel': 'validation'
    }
    
    metadata_info = {}
    all_transforms = {}
    for ds in ds_names:
        metadata_info[ds] = {
            'label_filenames': os.listdir(os.path.join(labels_root, '{0}_Set'.format(ds_names[ds].capitalize()))),
            'dataset': '{0}_Set'.format(ds_names[ds].capitalize()),
        }

        if 'train' in ds:
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
            all_transforms[ds] = None


    datasets = {}
    for ds in ds_names:
        if 'train' in ds:
            datasets[ds] = torch.utils.data.ConcatDataset([                   
                AbawVAEDataset(
                    audio_root=audio_root,
                    video_root=video_root,
                    labels_va_root=None,
                    labels_expr_root=labels_root,
                    label_filenames=metadata_info[ds]['label_filenames'],
                    dataset=metadata_info[ds]['dataset'],
                    features_root=features_root,
                    va_frames_grouping=None,
                    expr_frames_grouping=VAEGrouping.F2S,
                    multitask=False,
                    shift=2, min_w_len=2, max_w_len=4, processor_name=model_name,
                    transform=t) for t in all_transforms[ds]
                ]
            )
        else:
            datasets[ds] = AbawVAEDataset(
                    audio_root=audio_root,
                    video_root=video_root,
                    labels_va_root=None,
                    labels_expr_root=labels_root,
                    label_filenames=metadata_info[ds]['label_filenames'],
                    dataset=metadata_info[ds]['dataset'],
                    features_root=features_root,
                    va_frames_grouping=None,
                    expr_frames_grouping=VAEGrouping.F2S,
                    multitask=False,
                    shift=2, min_w_len=2, max_w_len=4, processor_name=model_name,
                    transform=all_transforms[ds],
                )

    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    experiment_name = 'wCELS{0}{1}-{2}'.format('a-' if aug else '-',
                                          model_cls.__name__.replace('-', '_').replace('/', '_'),
                                          datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
        
    net_trainer = NetTrainer(log_root=logs_root,
                             experiment_name=experiment_name,
                             problem_type=ProblemType.CLASSIFICATION,
                             c_names=c_names,
                             metrics=[f1, recall, precision],
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
    
    class_sample_count = datasets['train'].datasets[0].expr_labels_counts
    class_weights = torch.Tensor(max(class_sample_count) / class_sample_count).to(device)
    loss = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=.2)
    # loss = SoftFocalLossWrapper(focal_loss=SoftFocalLoss(alpha=class_weights), num_classes=len(c_names))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10, T_mult=1,
                                                                     eta_min=0.001 * 0.1)

    model, max_perf = net_trainer.run(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler,
                                      num_epochs=num_epochs, dataloaders=dataloaders, mixup_alpha=None)

    for phase in ds_names:
        if 'train' in phase:
            continue

        print()
        print(phase.capitalize())
        print('Epoch: {}, Max performance:'.format(max_perf[phase]['epoch']))
        print([metric for metric in max_perf[phase]['performance']])
        print([max_perf[phase]['performance'][metric] for metric in max_perf[phase]['performance']])
        print()


def expr_grouping(targets: list[np.ndarray], predicts: list[np.ndarray], sample_info: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Flatten targets ((n, 4) = > (n * 4)), predicts ((n, 4, 8) => (n * 4, 8)), sample_info
    
    Args:
        targets (list[np.ndarray]): List of targets
        predicts (list[np.ndarray]): List of predicts
        sample_info (list[dict]): Sample info

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[dict]]: targets, predicts, sample_info for grouping predicts/targets, 
    """
    targets = np.hstack(targets)
    predicts = np.asarray(predicts).reshape(-1, predicts[0].shape[-1])
    new_sample_info = {}
    for si in sample_info:
        for k, v in si.items():
            if k not in new_sample_info:
                new_sample_info[k] = []
            new_sample_info[k].extend(v)

    return targets, predicts, new_sample_info


def run_expression_training() -> None:
    """Wrapper for training expression challenge
    """
    
    model_cls = [ExprModelV1, ExprModelV2, ExprModelV3]
    
    for augmentation in [True, False]:
        for filtered in [True, False]:
            for m_cls in model_cls:
                cfg = deepcopy(config_expr)
                cfg['FILTERED'] = filtered
                cfg['AUGMENTATION'] = augmentation
                cfg['MODEL_PARAMS']['model_cls'] = m_cls
                
                main(cfg)


if __name__ == '__main__':
    # main(config=config_expr)
    run_expression_training()