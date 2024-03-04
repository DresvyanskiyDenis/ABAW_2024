from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.special import softmax
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
    new_predictions = np.zeros((result_num_frames, predictions.shape[-1]))
    # fill the new array with the predictions
    for i in range(predictions.shape[-1]):
        new_predictions[:, i] = resample(predictions[:, i], result_num_frames)
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


def average_predictions_on_timesteps(timesteps_with_predictions:List[Tuple[np.ndarray, np.ndarray]])->Tuple[np.ndarray, np.ndarray]:
    """ Averages the predictions on the same timesteps. timestamps_with_predictions is a list of tuples
    (timesteps, predictions) where timesteps is an array with timesteps and predictions is an array with model predictions

    :param timesteps_with_predictions: List[Tuple[np.ndarray, np.ndarray]]
        List of tuples with timesteps and predictions.
    :return: Tuple[np.ndarray, np.ndarray]
        Tuple with timesteps and averaged predictions
    """
    # get all timesteps
    num_labels = timesteps_with_predictions[0][1].shape[-1]
    all_timesteps = np.concatenate([timesteps for timesteps, _ in timesteps_with_predictions]).reshape((-1,))
    all_predictions = np.concatenate([predictions for _, predictions in timesteps_with_predictions],
                                     axis=0).reshape((-1,num_labels))
    # get unique timesteps
    unique_timesteps = np.unique(all_timesteps)
    # sort
    unique_timesteps.sort()
    # create array to store the averaged predictions
    averaged_predictions = np.zeros((len(unique_timesteps), num_labels))
    # go over unique timesteps and average the predictions
    for i, timestep in enumerate(unique_timesteps):
        # find indices of the current timestep in the all_timesteps
        indices = np.where(all_timesteps == timestep)
        # take the mean of predictions for the current timestep
        predictions = all_predictions[indices]
        averaged_predictions[i] = predictions.mean(axis=0)
    return unique_timesteps, averaged_predictions


def __apply_hamming_smoothing(array:np.ndarray, smoothing_window_size:int)->np.ndarray:
    """ Smooths the data by applying the Hamming window to every channel of the array.

    :param array: np.ndarray
        Array with data. Shape: (num_timesteps, num_classes)
    :param smoothing_window_size: int
        Size of the smoothing window
    :return: np.ndarray
        Smoothed array. Shape: (num_timesteps, num_classes)
    """
    # uses numpy.hamming function. Remember to apply the mean to every element as the window is not normalized
    for i in range(array.shape[-1]):
        array[:, i] = np.convolve(array[:, i], np.hamming(smoothing_window_size)/np.hamming(smoothing_window_size).sum(), mode='same')
    return array






def evaluate_on_dev_set_full_fps(dev_set_full_fps:Dict[str, pd.DataFrame], dev_set_resampled:Dict[str, pd.DataFrame],
                                 video_to_fps:Dict[str, float], model:torch.nn.Module, labels_type:str,
                                 feature_columns:List[str], labels_columns:List[str],
                                 window_size:int, device:torch.device,
                                 batch_size:int=32, resampled_fps:int=5)->Dict[str, float]:
    # get unique video names
    video_names = list(dev_set_resampled.keys())
    # create result dictionary
    result = {}
    # go over video names and evaluate the model
    for video_name in video_names:
        # cut on windows the resampled video
        windows = __cut_video_on_windows(dev_set_resampled[video_name], window_size=window_size, stride=window_size//5)
        predictions = []
        for window_idx in range(0, len(windows), batch_size):
            # extract the batch of windows
            batch_windows = windows[window_idx:window_idx + batch_size]
            timesteps = np.stack([window['timestamp'].values for window in batch_windows])
            # extract features from the batch
            batch_windows = [torch.from_numpy(window[feature_columns].values) for window in batch_windows]
            batch_windows = torch.stack(batch_windows)
            batch_windows = batch_windows.float().to(device)
            # get predictions
            batch_predictions = model(batch_windows).detach().cpu().numpy()
            predictions.append((timesteps, batch_predictions))
        # average predictions on timesteps
        prediction_timesteps, predictions = average_predictions_on_timesteps(predictions)
        # get ground truth
        ground_truth = dev_set_full_fps[video_name][labels_columns].values
        ground_truth_timesteps = dev_set_full_fps[video_name]['timestamp'].values
        ground_truth_fps = video_to_fps[video_name]
        # interpolate predictions to the full fps
        predictions = interpolate_to_full_fps(predictions, predictions_fps=resampled_fps, ground_truths_fps=ground_truth_fps)
        # Apply hamming smoothing
        predictions = __apply_hamming_smoothing(predictions, smoothing_window_size=int(np.round(ground_truth_fps))//2)
        # check if the number of frames is the same
        if np.abs(predictions.shape[0] - ground_truth.shape[0]) > 10:
            print(f'Video {video_name} has different number of frames in predictions and ground truth. THe number is {np.abs(predictions.shape[0] - ground_truth.shape[0])}')
            #print("Labels timesteps:", ground_truth_timesteps)
            #print("Predictions timesteps:", prediction_timesteps)
            print("This video will be ignored during the evaluation.")
            print("----------------------------------------------")
            continue # TODO: this is a temporary solution. It arises because there are some missing frames in the predictions
                     # as the face has not been identified in some frames. We should fix it in the future.
        # repeat last prediction to match the number of frames
        if ground_truth.shape[0] > predictions.shape[0]:
            repetition = np.repeat(predictions[-1][np.newaxis, :], ground_truth.shape[0] - predictions.shape[0], axis=0)
            predictions = np.concatenate([predictions, repetition], axis=0)
        # OR cut off the last predictions to match the number of frames
        elif ground_truth.shape[0] < predictions.shape[0]:
            predictions = predictions[:ground_truth.shape[0]]
        # calculate the metric
        if labels_type == 'Exp':
            predictions = softmax(predictions, axis=-1)
            predictions = np.argmax(predictions, axis=-1)
            ground_truth = np.argmax(ground_truth, axis=-1)
            result[video_name] = f1_score(ground_truth, predictions, average='macro')
        elif labels_type == 'VA':
            valence = np_concordance_correlation_coefficient(ground_truth[:, 0], predictions[:, 0])
            arousal = np_concordance_correlation_coefficient(ground_truth[:, 1], predictions[:, 1])
            result[video_name] = (valence, arousal)
    # get averaged result
    if labels_type == 'Exp':
        result = {'val_f1':np.array(list(result.values())).mean()}
    elif labels_type == 'VA':
        valence_result = np.array([valence for valence, _ in result.values()]).mean()
        arousal_result = np.array([arousal for _, arousal in result.values()]).mean()
        result = {'val_CCC_V':valence_result,
                  'val_CCC_A':arousal_result}
    return result




