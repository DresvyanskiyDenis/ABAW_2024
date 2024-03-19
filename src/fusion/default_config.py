import sys

sys.path.append('src')

from fusion.models.av_expr_models import *
from fusion.models.av_va_models import *


config_expr: dict = {
    'AUDIO_FEATURES_PATH': '',
    'VIDEO_FEATURES_PATH': '',
    'LABELS_ROOT': '',

    'AUDIO_TRAIN_FEATURES_PATH': '',
    
    ###
    'LOGS_ROOT': '',
    'MODEL_PARAMS': {
        'model_cls': TestModel,
        'model_params': {   
        }
    },
    'AUGMENTATION': False,
    'NUM_EPOCHS': 100,
    'BATCH_SIZE': 32,
}

config_va: dict = {
    'AUDIO_FEATURES_PATH': '',
    'VIDEO_FEATURES_PATH': '',
    'LABELS_ROOT': '',

    'AUDIO_TRAIN_FEATURES_PATH': '',
    
    ###
    'LOGS_ROOT': '',
    'MODEL_PARAMS': {
        'model_cls': TestModelSTP,
        'model_params': {   
        }
    },
    'AUGMENTATION': False,
    'NUM_EPOCHS': 100,
    'BATCH_SIZE': 32,
}