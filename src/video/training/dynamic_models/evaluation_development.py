from typing import Dict

import torch
import numpy as np
import pandas as pd




def interpolate_to_full_fps(predictions:np.ndarray, predictions_fps:int, ground_truths_fps:int)->np.ndarray:
    """ Interpolates the predictions with predictions_fps to the ground_truths_fps (full fps of the video)
    TO do so, the function performs the following steps:
    1. Calculate the number of frames in the video predictions.shape[0] / predictions_fps * ground_truths_fps
    2. Create a new array with the number of frames from step 1
    3. interpolate the predictions to the new array. Example:
    we have predictions = [0 0 0 0 0 1 1 1 1 1] withf fps = 5 and we want to interpolate it to fps = 15
    the new array firstly should be generated in a way: [0 _ _ 0 _ _ 0 _ _ 0 _ _ 1 _ _ 1 _ _ 1 _ _ 1 _ _ 1 _ _]
    then, we should fill the gaps by interpolated values (linear interpolation):
                                                        [0 0 0 0 0 0 0 0 0 0 0.5 0.5 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    4. return the new array

    :param predictions: np.ndarray
        Array with predictions. Shape: (num_frames, num_classes)
    :param predictions_fps: int
        FPS of the predictions
    :param ground_truths_fps: int
        FPS of the ground truth
    :return: np.ndarray
        New array with interpolated predictions. Shape: (predictions.shape[0] / predictions_fps * ground_truths_fps, num_classes)
    """
    result_num_frames = int(np.ceil(predictions.shape[0] / predictions_fps * ground_truths_fps))
    new_predictions = np.zeros((result_num_frames, predictions.shape[1]))
    # fill the new array with the predictions
    for i in range(predictions.shape[1]): # TODO: check it
        new_predictions[:, i] = np.interp(np.linspace(0, predictions.shape[0], result_num_frames), np.arange(predictions.shape[0]), predictions[:, i])
    return new_predictions




def evaluate_on_dev_set_full_fps(dev_set_full_fps:Dict[str, pd.DataFrame], dev_set_resampled:Dict[str, pd.DataFrame],
                                 video_to_fps:Dict[str, pd.DataFrame], model:torch.nn.Module, labels_type:str)->Dict[str, float]:
    pass


