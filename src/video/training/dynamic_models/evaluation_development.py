from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.special import softmax
from sklearn.metrics import f1_score

from decorators.common_decorators import timer
from src.evaluation_dynamic import __average_predictions_on_timesteps, evaluate_predictions_on_dev_set_full_fps
from src.video.training.dynamic_models.metrics import np_concordance_correlation_coefficient



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
    # calculate the common frame_num_difference and the reference range of the window
    frame_num_difference = video['frame_num'].diff().round(2).mode().values[0]
    reference_range = np.round(frame_num_difference * (window_size-1), 2)
    # go over the video and cut it on windows. We include only windows with the same range as the reference range
    # (thus, only windows with the monotonically increasing timesteps are included in the result list)
    for i in range(0, len(video) - window_size, stride):
        window = video.iloc[i:i + window_size]
        actual_range = np.round(window["frame_num"].iloc[-1] - window["frame_num"].iloc[0], 2)
        if actual_range == reference_range:
            windows.append(window)
    # sometimes, there are no windows at all (because they are ultra short and most of the labels are missing)
    # then, we will just take the last window
    if len(windows) == 0:
        windows.append(video.iloc[-window_size:])
    # most of the times, the last window is not full, so we will replace it with the window that starts from -window_size
    windows[-1] = video.iloc[-window_size:]
    return windows


@timer
def evaluate_on_dev_set_full_fps(dev_set_full_fps:Dict[str, pd.DataFrame], dev_set_resampled:Dict[str, pd.DataFrame],
                                 video_to_fps:Dict[str, float], model:torch.nn.Module, labels_type:str,
                                 feature_columns:List[str], labels_columns:List[str],
                                 window_size:int, device:torch.device,
                                 batch_size:int=32,
                                 downgrade_to_1_fps:Optional[bool]=None)->Dict[str, float]:
    # get unique video names
    video_names = list(dev_set_resampled.keys())
    # create predictions
    predictions_dict = {}
    # go over video names and evaluate the model
    for video_name in video_names:
        # cut on windows the resampled video
        windows = __cut_video_on_windows(dev_set_resampled[video_name], window_size=window_size, stride=window_size//4)
        predictions = []
        for window_idx in range(0, len(windows), batch_size):
            # extract the batch of windows
            batch_windows = windows[window_idx:window_idx + batch_size]
            if downgrade_to_1_fps is not False:
                timesteps = np.stack([window['timestep'].values[4::5] for window in batch_windows]) # TODO: you can make additional parameter
            else:
                timesteps = np.stack([window['timestep'].values for window in batch_windows])
            # extract features from the batch
            batch_windows = [torch.from_numpy(window[feature_columns].values) for window in batch_windows]
            batch_windows = torch.stack(batch_windows)
            batch_windows = batch_windows.float().to(device)
            # get predictions
            batch_predictions = model(batch_windows).detach().cpu().numpy()
            predictions.append((timesteps, batch_predictions))
        # average predictions on timesteps
        prediction_timesteps, predictions = __average_predictions_on_timesteps(predictions)
        # select the frame_nums depending on the prediction_timesteps
        frame_nums = dev_set_resampled[video_name].loc[dev_set_resampled[video_name]['timestep'].isin(prediction_timesteps)]['frame_num'].values
        # sometimes, the number of frames will be 1 less as the padding has been added to the predictions to make window
        # in this case, there will be timestep equalled 0, but there is no such frame. In this case, we will remove the
        # first timestep and the first prediction
        if len(frame_nums) != len(predictions):
            prediction_timesteps = prediction_timesteps[1:]
            predictions = predictions[1:]
        # combine the frame_nums, timestamps and predictions
        df_data = np.concatenate([frame_nums.reshape(-1, 1), prediction_timesteps.reshape(-1, 1), predictions], axis=1)
        df = pd.DataFrame(df_data, columns=['frame_num', 'timestep'] + labels_columns)
        predictions_dict[video_name] = df
    # evaluate the predictions
    result = evaluate_predictions_on_dev_set_full_fps(predictions=predictions_dict, labels=dev_set_full_fps,
                                             labels_type=labels_type)
    if labels_type == "VA":
        return {"val_CCC_V": result[0], "val_CCC_A": result[1]}
    else:
        return {"val_f1": result}



