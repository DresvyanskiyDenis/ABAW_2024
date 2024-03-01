from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.video.training.dynamic_models.metrics import np_concordance_correlation_coefficient


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


def __cut_video_on_windows(video:pd.DataFrame, window_size:int, stride:int)->List[pd.DataFrame]:
    """ Cuts the video on windows with specified window size and stride.

    :param video: pd.DataFrame
        The dataframe with corresponding frames of the video (represented as paths to the frames)
        It has columns ['path', 'frame_number', 'timestep', ...]
    :param window_size: int
        Size of the window. Given in number of frames.
    :param stride: int
        Stride of the window. Given in number of frames.
    :return: List[pd.DataFrame]
        List of dataframes with windows. Each dataframe has the same columns as the input dataframe.
    """
    if len(video) < window_size:
        return [video]
    # create list to store the windows
    windows = []
    # go over the video and cut it on windows
    for i in range(0, len(video) - window_size, stride):
        windows.append(video.iloc[i:i + window_size])
    # last window is always overcut, we should change the last element of the list
    windows[-1] = video.iloc[-window_size:]
    return windows


def average_predictions_on_timesteps(timesteps_with_predictions:List[Tuple[np.ndarray, np.ndarray]])->np.ndarray:
    """ Averages the predictions on the same timesteps. timestamps_with_predictions is a list of tuples
    (timesteps, predictions) where timesteps is an array with timesteps and predictions is an array with model predictions

    :param timesteps_with_predictions: List[Tuple[np.ndarray, np.ndarray]]
        List of tuples with timesteps and predictions.
    :return: np.ndarray
        Array with averaged predictions. Shape: (num_timesteps, predictions.shape[1])
    """
    # get all timesteps
    all_timesteps = np.concatenate([timesteps for timesteps, _ in timesteps_with_predictions])
    # get unique timesteps
    unique_timesteps = np.unique(all_timesteps)
    # create array to store the averaged predictions
    averaged_predictions = np.zeros((len(unique_timesteps), timesteps_with_predictions[0][1].shape[1]))
    # go over unique timesteps and average the predictions
    for i, timestep in enumerate(unique_timesteps):
        # get predictions for the current timestep
        predictions = np.stack([predictions for timesteps, predictions in timesteps_with_predictions if timestep in timesteps])
        # average the predictions
        averaged_predictions[i] = np.mean(predictions, axis=0)
    return averaged_predictions





def evaluate_on_dev_set_full_fps(dev_set_full_fps:Dict[str, pd.DataFrame], dev_set_resampled:Dict[str, pd.DataFrame],
                                 video_to_fps:Dict[str, pd.DataFrame], model:torch.nn.Module, labels_type:str,
                                 feature_columns:List[str], labels_columns:List[str], batch_size:int=32, resampled_fps:int=5)->Dict[str, float]:
    # get unique video names
    video_names = list(dev_set_resampled.keys())
    # create result dictionary
    result = {}
    # go over video names and evaluate the model
    for video_name in video_names:
        # cut on windows the resampled video
        windows = __cut_video_on_windows(dev_set_resampled[video_name], window_size=32, stride=16)
        predictions = []
        for window_idx in range(len(windows), batch_size):
            # extract the batch of windows
            batch_windows = windows[window_idx:window_idx + batch_size]
            timesteps = np.stack([window['timestep'].values for window in batch_windows])
            # extract features from the batch
            batch_windows = [torch.from_numpy(window[feature_columns].values) for window in batch_windows]
            batch_windows = torch.stack(batch_windows)
            # get predictions
            batch_predictions = model(batch_windows).detach().cpu().numpy()
            predictions.append((timesteps, batch_predictions))
        # average predictions on timesteps
        predictions = average_predictions_on_timesteps(predictions)
        # interpolate predictions to the full fps
        ground_truth_fps = video_to_fps[video_name].iloc[0]['fps']
        predictions = interpolate_to_full_fps(predictions, predictions_fps=resampled_fps, ground_truths_fps=ground_truth_fps)
        # get ground truth
        ground_truth = dev_set_full_fps[video_name][labels_columns].values
        # check if the number of frames is the same
        assert predictions.shape[0] == ground_truth.shape[0]
        # calculate the metric
        if labels_type == 'classification':
            result[video_name] = f1_score(ground_truth, predictions, average='macro')
        elif labels_type == 'regression':
            result[video_name] = np_concordance_correlation_coefficient(ground_truth, predictions)
    # get averaged result
    result = np.mean(list(result.values()))
    return result




