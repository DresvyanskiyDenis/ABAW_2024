from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.special import softmax
from sklearn.metrics import f1_score

from decorators.common_decorators import timer
from src.video.training.dynamic_models.metrics import np_concordance_correlation_coefficient

@timer
def interpolate_to_100_fps(predictions:np.ndarray, predictions_timesteps:np.ndarray)->\
        Tuple[np.ndarray, np.ndarray]:
    """ Interpolates the predictions with predictions_fps to the 100 fps. Thus we will have prediction for every 0.01 second.

    :param predictions: np.ndarray
        Array with predictions. Shape: (num_frames, num_classes)
    :param predictions_fps: int
        FPS of the predictions
    :return: Tuple[np.ndarray, np.ndarray]
        Tuple with interpolated predictions and timesteps.
    """
    new_timesteps = np.arange(0, predictions_timesteps[-1]+0.01, 1/100)
    # round it to 2 decimal places
    new_timesteps = np.round(new_timesteps, 2)
    new_predictions = np.ones((len(new_timesteps), predictions.shape[-1]))*-1
    # fill in the new_predictions array with values from the predictions depending on the timesteps
    mask = np.isin(new_timesteps, predictions_timesteps)
    new_predictions[mask] = predictions
    # We need to fill in the missing values that are denoted with -1. THe problem is that the predictions
    # has not been filled monotonicallz as there were some missing frames, filtered oput frames and so on.
    # However, we need to interpolate somehow the missing values. interp1d requires monotonic array.
    # therefore, we need to do it by hand
    new_predictions_copy = new_predictions.copy()
    for i in range(new_predictions_copy.shape[0]):
        if np.all(new_predictions_copy[i] == -1):
            # find the closest non -1 values with "distance" to it
            left_idx = i-1
            right_idx = i+1
            # find the closest left value with its index
            while left_idx>-1:
                if np.all(new_predictions_copy[left_idx] != -1):
                    break
                left_idx -= 1
            # the same for the right value
            while right_idx<new_predictions_copy.shape[0]:
                if np.all(new_predictions_copy[right_idx] != -1):
                    break
                right_idx += 1
            # interpolate the missing value taking into account the distance to the left and right values
            if left_idx == -1:
                new_predictions[i] = new_predictions_copy[right_idx]
            elif right_idx == new_predictions_copy.shape[0]:
                new_predictions[i] = new_predictions_copy[left_idx]
            else:
                weights = np.array([i - left_idx, right_idx - i])
                # inverse proportional normalized weights
                weights = 1./weights
                weights = weights/weights.sum()
                new_predictions[i] = weights[0]*new_predictions_copy[left_idx] + weights[1]*new_predictions_copy[right_idx]
    return new_predictions, new_timesteps

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
        # pad it with zeros at the start
        zeros = pd.DataFrame(np.zeros((window_size - len(video), len(video.columns))), columns=video.columns)
        video = pd.concat([zeros, video], axis=0)
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

def synchronize_predictions_with_ground_truth(predictions:np.ndarray, predictions_timesteps:np.ndarray,
                                                ground_truth:np.ndarray, ground_truth_timesteps:np.ndarray)->\
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Synchronizes the predictions with ground truth. It means that we take only the predictions
    that have the same timesteps as the ground truth.


    :param predictions: np.ndarray
        The predictions with the shape (num_timesteps1, num_classes or num_regressions)
    :param predictions_timesteps: np.ndarray
        The timesteps of the predictions. Shape: (num_timesteps1,)
    :param ground_truth: np.ndarray
        The ground truth with the shape (num_timesteps2, num_classes or num_regressions)
    :param ground_truth_timesteps: np.ndarray
        The timesteps of the ground truth. Shape: (num_timesteps2,)
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple with synchronized predictions and ground truth. The first two elements are the synchronized predictions
        and timesteps, the last two elements are the synchronized ground truth and timesteps.
    """
    # find common timesteps
    common_timesteps = np.intersect1d(ground_truth_timesteps, predictions_timesteps)
    # find indices of the common timesteps in the predictions and ground truth
    predictions_indices = np.where(np.isin(predictions_timesteps, common_timesteps))[0]
    ground_truth_indices = np.where(np.isin(ground_truth_timesteps, common_timesteps))[0]
    # at this point, the number of common timesteps in the predictions and ground truth should be the same
    new_predictions = predictions[predictions_indices]
    new_predictions_timesteps = predictions_timesteps[predictions_indices]
    new_ground_truth = ground_truth[ground_truth_indices]
    new_ground_truth_timesteps = ground_truth_timesteps[ground_truth_indices]
    # check if the number of timesteps is the same
    assert len(new_predictions_timesteps) == len(new_ground_truth_timesteps)
    return new_predictions, new_predictions_timesteps, new_ground_truth, new_ground_truth_timesteps







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





@timer
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
        # interpolate predictions to the 100 fps
        predictions, predictions_timesteps = interpolate_to_100_fps(predictions, prediction_timesteps)
        # synchronize predictions with ground truth
        predictions, prediction_timesteps, ground_truth, ground_truth_timesteps = \
            synchronize_predictions_with_ground_truth(predictions, predictions_timesteps, ground_truth, ground_truth_timesteps)
        # Apply hamming smoothing
        predictions = __apply_hamming_smoothing(predictions, smoothing_window_size=int(np.round(ground_truth_fps))//2)
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




