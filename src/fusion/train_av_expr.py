import sys

sys.path.append('src')

import os
import pprint
import datetime
from copy import deepcopy

import numpy as np

import torch
from torchvision import transforms

from sklearn.model_selection import train_test_split

from fusion.config import config_expr

from fusion.data.abaw_av_expr_dataset import AbawMultimodalExprDataset

from fusion.net_trainer.net_trainer import NetTrainer, ProblemType

from fusion.models.av_expr_models import TestModel
from fusion.models.fusion_models import final_fusion_model_v1

from audio.utils.data_utils import get_source_code

from fusion.evaluation_fusion import evaluate_model_full_fps

from audio.utils.accuracy_utils import recall, precision, f1
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
    
    c_names = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        
    source_code = 'Configuration:\n{0}\n\nSource code:\n{1}'.format(
        pprint.pformat(config), 
        get_source_code([main, model_cls, AbawMultimodalExprDataset, NetTrainer]))    

    ds_names = {
        'train': 'train', 
        'devel': 'validation'
    }

    validation_files = os.listdir(os.path.join(labels_root, '{0}_Set'.format(ds_names['devel'].capitalize())))
    tts = train_test_split(validation_files, test_size=0.2, random_state=0)
    
    metadata_info = {}
    all_transforms = {}
    for ds in ds_names:
        metadata_info[ds] = {
            'label_filenames': tts[0] if 'train' in ds else tts[1],
            'dataset': '{0}_Set'.format(ds_names[ds].capitalize()),
        }

        if 'train' in ds:
            if aug:
                all_transforms[ds] = [
                    [None, None]
                ]
            else:
                all_transforms[ds] = [
                    [None, None]
                ]
        else:
            all_transforms[ds] = [None, None]


    datasets = {}
    for ds in ds_names:
        if 'train' in ds:
            datasets[ds] = torch.utils.data.ConcatDataset([                   
                AbawMultimodalExprDataset(audio_features_path=audio_features_path,
                                          video_features_path=video_features_path,
                                          labels_root=os.path.join(labels_root, '{0}_Set'.format(ds_names['devel'].capitalize())),
                                          label_filenames=metadata_info[ds]['label_filenames'],
                                          dataset=metadata_info[ds]['dataset'],
                                          audio_train_features_path=audio_train_features_path,
                                          shift=2, min_w_len=2, max_w_len=4, transform=t) for t in all_transforms[ds]
                ]
            )
        else:
            datasets[ds] = AbawMultimodalExprDataset(audio_features_path=audio_features_path,
                                                     video_features_path=video_features_path,
                                                     labels_root=os.path.join(labels_root, '{0}_Set'.format(ds_names['devel'].capitalize())),
                                                     label_filenames=metadata_info[ds]['label_filenames'],
                                                     dataset=metadata_info[ds]['dataset'],
                                                     audio_train_features_path=audio_train_features_path,
                                                     shift=2, min_w_len=2, max_w_len=4, transform=all_transforms[ds])

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
                             group_predicts_fn=None, # TODO
                             source_code=source_code)
        
    dataloaders = {}
    for ds in ds_names:
        dataloaders[ds] = torch.utils.data.DataLoader(
            datasets[ds],
            batch_size=batch_size,
            shuffle=('train' in ds),
            num_workers=batch_size if batch_size < 9 else 8)
        
    # model = model_cls(**model_params)
    model = final_fusion_model_v1(num_classes=len(c_names))
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


def run_expression_training() -> None:
    """Wrapper for training expression challenge
    """
    
    model_cls = [TestModel]
    
    for augmentation in [False]:
        for m_cls in model_cls:
            cfg = deepcopy(config_expr)
            cfg['AUGMENTATION'] = augmentation
            cfg['MODEL_PARAMS']['model_cls'] = m_cls
                
            main(cfg)


if __name__ == '__main__':
    # main(config=config_expr)
    run_expression_training()