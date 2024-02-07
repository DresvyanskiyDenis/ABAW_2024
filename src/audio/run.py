import sys

sys.path.append('../src')

import os
from copy import deepcopy

from models.audio_expr_models import *

from train_expr_audio import main as expr_main

from utils.common_utils import wait_for_it

from config import *

    
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
                
                expr_main(cfg)
    

if __name__ == '__main__':
    run_expression_training()
    