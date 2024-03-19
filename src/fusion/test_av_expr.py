import sys

sys.path.append('src')

from audio.loss.loss import SoftFocalLossWrapper, SoftFocalLoss
from fusion.evaluation_fusion import evaluate_model_full_fps

import os
import pprint
import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision import transforms

from sklearn.model_selection import train_test_split

from fusion.config import config_expr

from fusion.data.abaw_av_expr_dataset import AbawMultimodalExprDataset, AbawMultimodalExprWithNormDataset

from fusion.net_trainer.net_trainer import NetTrainer, ProblemType

from fusion.models.av_expr_models import TestModelV1, TestModelV2
from fusion.models.fusion_models import final_fusion_model_v1, final_fusion_model_v5, final_fusion_model_v4, \
    final_fusion_model_v3, final_fusion_model_v2

from audio.utils.data_utils import get_source_code

from audio.utils.accuracy_utils import recall, precision, f1
from audio.utils.common_utils import define_seed



def extract_predicts(model_params: dict, config: dict, problem_type: ProblemType) -> None:
    audio_features_path = config['AUDIO_FEATURES_PATH']
    video_features_path = config['VIDEO_FEATURES_PATH']

    labels_root = config['LABELS_ROOT']
    audio_train_features_path = config['AUDIO_TRAIN_FEATURES_PATH']
    
    logs_root = config['LOGS_ROOT']
    model_cls = model_params['model_cls']
    model_args = config['MODEL_PARAMS']['model_params']
    aug = config['AUGMENTATION']
    num_epochs = config['NUM_EPOCHS']
    batch_size = config['BATCH_SIZE']
    
    c_names = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']

    ds_names = {
        'train': 'train', 
        'devel': 'validation',
        'test': 'test',
    }

    with open(os.path.join(labels_root, 'expr_test_set_release.txt')) as file:
        test_files = ['{0}.txt'.format(line.rstrip()) for line in file]

    # calc num_frames
    test_metadata = pd.read_csv(os.path.join(labels_root, 'CVPR_6th_ABAW_Expr_test_set_example.txt'), 
                                sep=',', 
                                names=['sample', 'expr'], 
                                skiprows=[0],
                                header=None)

    test_metadata['filename'] = test_metadata['sample'].apply(lambda x: x.split('/')[0])
    test_metadata['frame'] = test_metadata['sample'].apply(lambda x: x.split('/')[1])
    test_num_frames_dict = test_metadata.groupby(['filename']).count()['frame'].to_dict()

    metadata_info = {}
    all_transforms = {}
    for ds in ds_names:
        metadata_info[ds] = {
            'audio_features_path': os.path.join(audio_features_path, 'expr_{0}.pickle'.format(ds)),
            'label_filenames': test_files if ds == 'test' else os.listdir(os.path.join(labels_root, '{0}_Set'.format(ds_names[ds].capitalize()))),
            'total_num_frames': test_num_frames_dict if ds == 'test' else None,
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
                AbawMultimodalExprWithNormDataset(audio_features_path=metadata_info[ds]['audio_features_path'],
                                                  video_features_path=video_features_path,
                                                  labels_root=os.path.join(labels_root, '{0}_Set'.format(ds_names[ds].capitalize())),
                                                  label_filenames=metadata_info[ds]['label_filenames'],
                                                  dataset=metadata_info[ds]['dataset'],
                                                  num_frames_dict=metadata_info[ds]['total_num_frames'],
                                                  audio_train_features_path=audio_train_features_path,
                                                  normalizer=[None, None] if 'train' in ds else [datasets['train'].datasets[0].a_normalizer, datasets['train'].datasets[0].v_normalizer],
                                                  shift=2, min_w_len=2, max_w_len=4, transform=t) for t in all_transforms[ds]
                ]
            )
        else:
            datasets[ds] = AbawMultimodalExprWithNormDataset(audio_features_path=metadata_info[ds]['audio_features_path'],
                                                             video_features_path=video_features_path.replace('_exp.pkl', '_exp_test.pkl') if 'test' in ds else video_features_path,
                                                             labels_root=os.path.join(labels_root, '{0}_Set'.format(ds_names[ds].capitalize())),
                                                             label_filenames=metadata_info[ds]['label_filenames'],
                                                             dataset=metadata_info[ds]['dataset'],
                                                             num_frames_dict=metadata_info[ds]['total_num_frames'],
                                                             audio_train_features_path=audio_train_features_path,
                                                             normalizer=[None, None] if 'train' in ds else [datasets['train'].datasets[0].a_normalizer, datasets['train'].datasets[0].v_normalizer],
                                                             shift=2, min_w_len=2, max_w_len=4, transform=all_transforms[ds])

    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_trainer = NetTrainer(log_root=logs_root,
                             experiment_name='Test',
                             problem_type=problem_type,
                             c_names=c_names if problem_type == ProblemType.CLASSIFICATION else None,
                             metrics=None,
                             device=device,
                             group_predicts_fn=evaluate_model_full_fps,
                             source_code=None)
                   
    dataloaders = {}
    for ds in ds_names:
        dataloaders[ds] = torch.utils.data.DataLoader(
            datasets[ds],
            batch_size=batch_size,
            shuffle=False,
            num_workers=8)
        
    model = model_cls(**model_args)
    model.load_state_dict(torch.load(os.path.join(model_params['root_path'], 'epoch_{}.pth'.format(model_params['epoch'])))['model_state_dict'])

    model.to(device)
    
    net_trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    net_trainer.model = model

    test_metadata.set_index('sample', inplace=True)

    for ds, v in dataloaders.items():
        targets, predicts, sample_info = net_trainer.test_model(phase='test', 
                                                                dataloader=v,
                                                                verbose=True)

        res = []
        for idx, si in enumerate(sample_info):
            res.append({
                'image_location': '{0}/{1:05d}.jpg'.format(si[0], si[1] + 1),
                'category': np.argmax(predicts[idx])
            })

        res_df = pd.DataFrame.from_dict(res).sort_values('image_location')

        res_df["category"] = res_df["category"].astype(int)
        res_df.columns = ["image_location", "Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other"]

        res_df.set_index('image_location', inplace=True)
        if 'test' in ds:
            res_df = res_df.reindex(test_metadata.index)
            res_df.reset_index(inplace=True)
        
        res_df.to_csv(os.path.join(logs_root, 'acoustic_facial_expr_fusion_{}.txt'.format(ds)), sep=',', index=False)

        with open(os.path.join(logs_root, 'acoustic_facial_expr_fusion_{}.txt'.format(ds))) as f:
            lines = f.readlines()

        lines[0] = 'image_location,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n'

        with open(os.path.join(logs_root, 'acoustic_facial_expr_fusion_{}.txt'.format(ds)), "w") as f:
            f.writelines(lines)


if __name__ == '__main__':
    # EXPR model - wCELSa-ExprModelV3-2024.03.02-09.24.44, spleeter
    # VA model - a-VAModelV3-2024.03.05-02.53.26, spleeter
    model_parameters = [
        {'model_name': 'Facial-wFL-final_fusion_model_v4-2024.03.17-04.39.14', 'model_cls': final_fusion_model_v4, 'epoch': 12, 'root_path': '/media/maxim/WesternDigitalNew/AbawLogs/FUSION_EXP'},
    ]

    # EXPR
    cfg = deepcopy(config_expr)

    m_p = model_parameters[0]
    m_p['root_path'] = os.path.join(m_p['root_path'], m_p['model_name'], 'models')
    extract_predicts(model_params=m_p, config=cfg, problem_type=ProblemType.CLASSIFICATION)