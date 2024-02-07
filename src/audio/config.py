import sys

sys.path.append('../src')

from models.audio_expr_models import *
from models.audio_va_models import *

config_expr: dict = {
    'WAV_ROOT': '/media/maxim/Databases/ABAW2024/data/wavs',
    'FILTERED_WAV_ROOT': '/media/maxim/Databases/ABAW2024/data/vocals',
    'VIDEO_ROOT': '/media/maxim/Databases/ABAW2024/data/videos',
    'LABELS_ROOT': '/media/maxim/Databases/ABAW2024/6th_ABAW_Annotations/EXPR_Classification_Challenge',
    'FEATURES_ROOT': '/media/maxim/Databases/ABAW2024/features/open_mouth',
    
    ###
    'LOGS_ROOT': '/media/maxim/WesternDigital/ABAWLogs/EXPR',
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
    'WAV_ROOT': '/media/maxim/Databases/ABAW2024/data/wavs',
    'FILTERED_WAV_ROOT': '/media/maxim/Databases/ABAW2024/data/vocals',
    'VIDEO_ROOT': '/media/maxim/Databases/ABAW2024/data/videos',
    'LABELS_ROOT': '/media/maxim/Databases/ABAW2024/6th_ABAW_Annotations/VA_Estimation_Challenge',
    'FEATURES_ROOT': '/media/maxim/Databases/ABAW2024/features/open_mouth',
    
    ###
    'LOGS_ROOT': '/media/maxim/WesternDigital/ABAWLogs/VA',
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