from typing import Dict

import torch
import numpy as np
import pandas as pd




def interpolate_to_full_fps(predictions:np.ndarray, predictions_fps:int, ground_truths_fps:int)->np.ndarray:
    pass




def evaluate_on_dev_set_full_fps(dev_set_full_fps:Dict[str, pd.DataFrame], dev_set_resampled:Dict[str, pd.DataFrame],
                                 video_to_fps:Dict[str, pd.DataFrame], model:torch.nn.Module, labels_type:str)->Dict[str, float]:
    pass


