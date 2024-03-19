import sys

sys.path.append('src')

import os
import pprint
import datetime
from copy import deepcopy

import numpy as np

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision import transforms

from fusion.config import config_va

from fusion.data.abaw_av_va_dataset import AbawMultimodalVAWithNormDataset

from fusion.net_trainer.net_trainer import NetTrainer, ProblemType

from fusion.models.av_va_models import TestModelSTP, TestModelMean, TestModelConcat
# from fusion.models.fusion_models import final_fusion_model_v1, final_fusion_model_v5, final_fusion_model_v4, \
    # final_fusion_model_v3, final_fusion_model_v2

from audio.loss.loss import VALoss

from fusion.evaluation_fusion import evaluate_model_full_fps

from audio.utils.data_utils import get_source_code

from audio.utils.accuracy_utils import va_score, v_score, a_score
from audio.utils.common_utils import define_seed


def main(config: dict) -> None:
    audio_features_path = config['AUDIO_FEATURES_PATH']
    video_features_path = config['VIDEO_FEATURES_PATH']

    labels_root = config['LABELS_ROOT']
    audio_train_features_path = config['AUDIO_TRAIN_FEATURES_PATH']
    
    logs_root = config['LOGS_ROOT']
    model_cls = config['MODEL_PARAMS']['model_cls']
    model_params = config['MODEL_PARAMS']['model_params']
    aug = config['AUGMENTATION']
    num_epochs = config['NUM_EPOCHS']
    batch_size = config['BATCH_SIZE']
    
    source_code = 'Configuration:\n{0}\n\nSource code:\n{1}'.format(
        pprint.pformat(config), 
        get_source_code([main, model_cls, AbawMultimodalVAWithNormDataset, NetTrainer]))    

    ds_names = {
        'train': 'train', 
        'devel': 'validation'
    }

    metadata_info = {}
    all_transforms = {}
    for ds in ds_names:
        metadata_info[ds] = {
            'audio_features_path': os.path.join(audio_features_path, 'va_{0}.pickle'.format(ds)),
            'label_filenames': os.listdir(os.path.join(labels_root, '{0}_Set'.format(ds_names[ds].capitalize()))),#tts[0] if 'train' in ds else tts[1],
            'dataset': '{0}_Set'.format(ds_names[ds].capitalize()),
        }

        if 'train' in ds:
            if aug:
                all_transforms[ds] = [
                    [None, None, None]
                ]
            else:
                all_transforms[ds] = [
                    [None, None, None]
                ]
        else:
            all_transforms[ds] = [None, None, None]


    datasets = {}
    for ds in ds_names:
        if 'train' in ds:
            datasets[ds] = torch.utils.data.ConcatDataset([                   
                AbawMultimodalVAWithNormDataset(audio_features_path=metadata_info[ds]['audio_features_path'],
                                                  video_features_path=video_features_path,
                                                  labels_root=os.path.join(labels_root, '{0}_Set'.format(ds_names[ds].capitalize())),
                                                  label_filenames=metadata_info[ds]['label_filenames'],
                                                  dataset=metadata_info[ds]['dataset'],
                                                  audio_train_features_path=audio_train_features_path,
                                                  normalizer=[None, None, None] if 'train' in ds else [datasets['train'].datasets[0].a_va_normalizer, 
                                                                                                       datasets['train'].datasets[0].v_v_normalizer,
                                                                                                       datasets['train'].datasets[0].v_a_normalizer],
                                                  shift=2, min_w_len=2, max_w_len=4, transform=t) for t in all_transforms[ds]
                ]
            )
        else:
            datasets[ds] = AbawMultimodalVAWithNormDataset(audio_features_path=metadata_info[ds]['audio_features_path'],
                                                             video_features_path=video_features_path,
                                                             labels_root=os.path.join(labels_root, '{0}_Set'.format(ds_names[ds].capitalize())),
                                                             label_filenames=metadata_info[ds]['label_filenames'],
                                                             dataset=metadata_info[ds]['dataset'],
                                                             audio_train_features_path=audio_train_features_path,
                                                             normalizer=[None, None, None] if 'train' in ds else [datasets['train'].datasets[0].a_va_normalizer, 
                                                                                                                  datasets['train'].datasets[0].v_v_normalizer,
                                                                                                                  datasets['train'].datasets[0].v_a_normalizer],
                                                             shift=2, min_w_len=2, max_w_len=4, transform=all_transforms[ds])

    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    experiment_name = 'Facial-{0}{1}-{2}'.format('a-' if aug else '',
                                                 model_cls.__name__.replace('-', '_').replace('/', '_'),
                                                 datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
        
    net_trainer = NetTrainer(log_root=logs_root,
                             experiment_name=experiment_name,
                             problem_type=ProblemType.REGRESSION,
                             c_names=None,
                             metrics=[va_score, v_score, a_score],
                             device=device,
                             group_predicts_fn=evaluate_model_full_fps, 
                             source_code=source_code)
        
    dataloaders = {}
    for ds in ds_names:
        dataloaders[ds] = torch.utils.data.DataLoader(
            datasets[ds],
            batch_size=batch_size,
            shuffle=('train' in ds),
            num_workers=batch_size if batch_size < 9 else 8)
        
    model = model_cls(**model_params)
    model.to(device)
    
    loss = VALoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
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


def run_va_training() -> None:
    """Wrapper for training va challenge
    """

    model_cls = [TestModelSTP, TestModelMean, TestModelConcat]
    model_params = [{'num_regression_neurons': 2}, {'num_regression_neurons': 2}, {'num_regression_neurons': 2}]

    
    for augmentation in [False]:
        for m_cls, m_params in zip(model_cls, model_params):
            cfg = deepcopy(config_va)
            cfg['AUGMENTATION'] = augmentation
            cfg['MODEL_PARAMS']['model_cls'] = m_cls
            cfg['MODEL_PARAMS']['model_params'] = m_params
            main(cfg)


if __name__ == '__main__':
    # main(config=config_expr)
    run_va_training()