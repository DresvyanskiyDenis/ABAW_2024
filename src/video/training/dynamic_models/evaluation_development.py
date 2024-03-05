from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.special import softmax
from sklearn.metrics import f1_score

from decorators.common_decorators import timer
from src.video.training.dynamic_models.metrics import np_concordance_correlation_coefficient


def __interpolate_to_100_fps(predictions:np.ndarray, predictions_timesteps:np.ndarray)->\
        Tuple[np.ndarray, np.ndarray]:
    """ Interpolates the predictions with predictions_fps to the 100 fps using pandas dataframe

    :param predictions:
    :param predictions_timesteps:
    :return:
    """
    # if the first timestep is not 0.0 seconds, than add it to the timesteps and first row of the predictions
    if predictions_timesteps[0] != 0.0:
        predictions_timesteps = np.concatenate([[0.0], predictions_timesteps])
        predictions = np.concatenate([np.expand_dims(predictions[0].copy(), axis=0), predictions], axis=0)
    # create dataframe with predictions
    predictions_df = pd.DataFrame(predictions, columns=[f'pred_{i}' for i in range(predictions.shape[-1])])
    predictions_df['timestep'] = predictions_timesteps
    predictions_df["milliseconds"] = predictions_df["timestep"] * 1000
    # set timestep as index
    predictions_df = predictions_df.set_index('milliseconds')
    # convert index to TimeDeltaIndex
    predictions_df.index = pd.to_timedelta(predictions_df.index, unit='ms')
    # Resample the DataFrame to 100 FPS (0.01 seconds interval)
    predictions_df = predictions_df.resample('10L').asfreq()

    # Interpolate the missing values
    predictions_df = predictions_df.interpolate(method='linear')

    # Reset the index to get the timesteps back as a column
    predictions_df.reset_index(inplace=True)

    predictions_df['timestep'] = predictions_df['milliseconds'].dt.total_seconds().apply("float64")

    # Get the predictions and timesteps
    predictions = predictions_df[[f'pred_{i}' for i in range(predictions.shape[-1])]].values
    predictions_timesteps = predictions_df['timestep'].values
    # round timestep to 2 decimal places
    predictions_timesteps = np.round(predictions_timesteps, 2)
    return predictions, predictions_timesteps


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
    if len(video) <= window_size:
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
    # delete duplicates in the ground truth timesteps and corresponding ground truth
    ground_truth_timesteps, indices = np.unique(ground_truth_timesteps, return_index=True)
    ground_truth = ground_truth[indices]
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
        predictions, predictions_timesteps = __interpolate_to_100_fps(predictions, prediction_timesteps)
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

if __name__=="__main__":
    # small test of the interpolation
    timesteps = np.array([0.01, 0.02, 0.05, 0.06, 0.07, 0.09, 0.1])
    predictions = np.array([[0.1, 0.1],
                              [0.2, 0.2],
                              [0.5, 0.5],
                                [0.6, 0.6],
                                [0.7, 0.7],
                                [0.9, 0.9],
                                [1.0, 1.0]])
    predictions, timesteps = __interpolate_to_100_fps(predictions, timesteps)

    print(timesteps)
    print(predictions)



