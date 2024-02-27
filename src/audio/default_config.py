import sys

sys.path.append('../src')

from models.audio_expr_models import *
from models.audio_va_models import *
from models.audio_vae_models import *

config_expr: dict = {
    'WAV_ROOT': '',
    'FILTERED_WAV_ROOT': '',
    'VIDEO_ROOT': '',
    'LABELS_ROOT': '',
    'FEATURES_ROOT': '',
    
    ###
    'LOGS_ROOT': '',
    'MODEL_PARAMS': {
        'model_cls': ExprModelV1,
        'args': {
            'model_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
        }
    },
    'FILTERED': False,
    'AUGMENTATION': False,
    'NUM_EPOCHS': 100,
    'BATCH_SIZE': 16,
}

config_va: dict = {
    'WAV_ROOT': '',
    'FILTERED_WAV_ROOT': '',
    'VIDEO_ROOT': '',
    'LABELS_ROOT': '',
    'FEATURES_ROOT': '',
    
    ###
    'LOGS_ROOT': '',
    'MODEL_PARAMS': {
        'model_cls': VAModelV1,
        'args': {
            'model_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
        }
    },
    'FILTERED': False,
    'AUGMENTATION': False,
    'NUM_EPOCHS': 100,
    'BATCH_SIZE': 16,
}

config_vae: dict = {
    'WAV_ROOT': '',
    'FILTERED_WAV_ROOT': '',
    'VIDEO_ROOT': '',
    'LABELS_VA_ROOT': '',
    'LABELS_EXPR_ROOT': '',
    'FEATURES_ROOT': '',
    
    ###
    'LOGS_ROOT': '',
    'MODEL_PARAMS': {
        'model_cls': VAEModelV1,
        'args': {
            'model_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
        }
    },
    'FILTERED': False,
    'AUGMENTATION': False,
    'NUM_EPOCHS': 100,
    'BATCH_SIZE': 16,
}