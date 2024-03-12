import sys

sys.path.append('../src')

import os
import pickle
from copy import deepcopy

import numpy as np

import torch

from config import config_expr, config_va

from data.abaw_fe_dataset import AbawFEDataset, VAEGrouping

from net_trainer.net_trainer import NetTrainer, ProblemType

from models.audio_expr_models import ExprModelV3
from models.audio_va_models import VAModelV3

from utils.accuracy_utils import recall, precision, f1
from utils.common_utils import define_seed


def feature_extraction(model_params: dict, config: dict, problem_type: ProblemType) -> None:
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

        all_transforms[ds] = None

    datasets = {}
    for ds in ds_names:
        datasets[ds] = AbawFEDataset(audio_root=audio_root,
                                     video_root=video_root,
                                     labels_va_root=None if problem_type == ProblemType.CLASSIFICATION else labels_root,
                                     labels_expr_root=None if problem_type == ProblemType.REGRESSION else labels_root,
                                     label_filenames=metadata_info[ds]['label_filenames'],
                                     dataset=metadata_info[ds]['dataset'],
                                     features_root=features_root,
                                     va_frames_grouping=None if problem_type == ProblemType.CLASSIFICATION else VAEGrouping.F2F,
                                     expr_frames_grouping=None if problem_type == ProblemType.REGRESSION else VAEGrouping.F2S,
                                     multitask=False,
                                     shift=2, min_w_len=2, max_w_len=4, processor_name=model_name,
                                     transform=None)

    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    net_trainer = NetTrainer(log_root=logs_root,
                             experiment_name='Test',
                             problem_type=ProblemType.CLASSIFICATION if problem_type == ProblemType.CLASSIFICATION else ProblemType.REGRESSION,
                             c_names=c_names if problem_type == ProblemType.CLASSIFICATION else None,
                             metrics=None,
                             device=device,
                             group_predicts_fn=None,
                             source_code=None)
        
    dataloaders = {}
    for ds in ds_names:
        dataloaders[ds] = torch.utils.data.DataLoader(
            datasets[ds],
            batch_size=batch_size,
            shuffle=False,
            num_workers=8)
    
    model = model_cls.from_pretrained(model_name)
    model = model_params['model_cls'].from_pretrained(model_name)
    model.load_state_dict(torch.load(os.path.join(model_params['root_path'], 'epoch_{}.pth'.format(model_params['epoch'])))['model_state_dict'])
    
    model.to(device)
    
    net_trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    net_trainer.model = model
    
    keys_mapping = {
        'fps': 'fps',
        'start_f': 'frame_start',
        'end_f': 'frame_end',
        'start_t': 'timestep_start',
        'end_t': 'timestep_end',
        'mouth_open': 'mouth_open'
    }

    for ds, v in dataloaders.items():
        targets, predicts, features, sample_info = net_trainer.extract_features(phase='test', 
                                                                                dataloader=v,
                                                                                verbose=True)

        new_sample_info = {}
        for s_idx, si in enumerate(sample_info):            
            for idx, fn in enumerate(si['filename']):
                if fn not in new_sample_info:
                    new_sample_info[fn] = {}
                    new_sample_info[fn]['targets'] = []
                    new_sample_info[fn]['predicts'] = []
                    new_sample_info[fn]['features'] = []
                    for k in si.keys():
                        if 'filename' in k:
                            continue

                        new_sample_info[fn][keys_mapping[k]] = []

                new_sample_info[fn]['targets'].append(targets[idx + s_idx * batch_size])
                new_sample_info[fn]['predicts'].append(predicts[idx + s_idx * batch_size])
                new_sample_info[fn]['features'].append(features[idx + s_idx * batch_size])
                for k in si.keys():
                    if 'filename' in k:
                        continue

                    if k in ['start_f', 'end_f']:
                        new_sample_info[fn][keys_mapping[k]].append(int(si[k][idx]))
                    elif k in ['fps', 'start_t', 'end_t']:
                        new_sample_info[fn][keys_mapping[k]].append(float(si[k][idx]))
                    else:
                        new_sample_info[fn][keys_mapping[k]].append(si[k][idx])

        with open(os.path.join(logs_root, 
                               '{0}_{1}.pickle'.format('expr' if problem_type == ProblemType.CLASSIFICATION else 'va', ds)), 
                  'wb') as handle:
            pickle.dump(new_sample_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # EXPR model - wCELSa-ExprModelV3-2024.03.02-09.24.44, spleeter
    # VA model - a-VAModelV3-2024.03.05-02.53.26, spleeter
    model_parameters = [
        {'model_name': 'wCELSa-ExprModelV3-2024.03.02-09.24.44', 'model_cls': ExprModelV3, 'epoch': 98, 'root_path': '/media/maxim/WesternDigitalNew/AbawLogs/EXPR'},
        {'model_name': 'a-VAModelV3-2024.03.05-02.53.26', 'model_cls': VAModelV3, 'epoch': 63, 'root_path': '/media/maxim/WesternDigitalNew/AbawLogs/VA'},
    ]

    # EXPR
    cfg = deepcopy(config_expr)
    cfg['FILTERED'] = True
    cfg['AUGMENTATION'] = False

    m_p = model_parameters[0]
    m_p['root_path'] = os.path.join(m_p['root_path'], m_p['model_name'], 'models')
    feature_extraction(model_params=m_p, config=cfg, problem_type=ProblemType.CLASSIFICATION)

    # VA
    cfg = deepcopy(config_va)
    cfg['FILTERED'] = True
    cfg['AUGMENTATION'] = False

    m_p = model_parameters[1]
    m_p['root_path'] = os.path.join(m_p['root_path'], m_p['model_name'], 'models')
    feature_extraction(model_params=m_p, config=cfg, problem_type=ProblemType.REGRESSION)
