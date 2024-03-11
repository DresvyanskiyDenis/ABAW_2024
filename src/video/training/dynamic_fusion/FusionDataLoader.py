from collections import OrderedDict
from typing import Callable, List, Dict, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FusionDataLoader(Dataset):

    def __init__(self, data:Dict[str, pd.DataFrame], modality1_columns:List[str], modality2_columns:List[str],
                 labels_columns:List[str], window_size:Union[int, float], stride:Union[int, float],
                 only_consecutive_windows:Optional[bool]=True):
        """ Loads and cuts on windows the data provided in the data dictionary. The user should provide the
        modality columns as well (expected that in every dataframe there are these columns) that represent the
        respective modalities for every concrete frame.

        :param data: Dict[str, pd.DataFrame]
            The dictionary with the dataframes for every modality and labels. Keys are video filenames, values are
            frames with extracted features of two modalities and labels.
        :param modality1_columns: List[str]
            The list of columns that represent the first modality.
        :param modality2_columns: List[str]
            The list of columns that represent the second modality.
        :param labels_columns: List[str]
            The list of columns that represent the labels.
        :param window_size: int
            The size of the window to cut the data on.
        :param stride: int
            The stride to move the window.
        :param only_consecutive_windows: Optional[bool]
            If True, only consecutive windows will be considered (with monotonic timesteps increase).
            If False, all windows will be considered.
        """
        self.data = data
        self.modality1_columns = modality1_columns
        self.modality2_columns = modality2_columns
        self.labels_columns = labels_columns
        self.window_size = window_size
        self.stride = stride
        self.only_consecutive_windows = only_consecutive_windows

        # cut all data on windows
        self.__cut_all_data_on_windows()

        # to get item every time in the __getitem__ method, we need to create a list of pointers.
        # every pointer points to a concrete window located in self.cut_windows
        # in this way, we can easily shuffle them and get the access to the windows really quickly
        self.pointers = []
        for key, windows in self.cut_windows.items():
            for idx_window, window in enumerate(windows):
                self.pointers.append((key + f"_{idx_window}", window))


    def __len__(self):
        return len(self.pointers)

    def __getitem__(self, idx):
        # get the data and labels using pointer
        _, window = self.pointers[idx]

        embeddings_mod1 = window[self.modality1_columns].values
        embeddings_mod2 = window[self.modality2_columns].values

        # transform embeddings into tensors
        embeddings_mod1 = torch.from_numpy(embeddings_mod1)
        embeddings_mod2 = torch.from_numpy(embeddings_mod2)
        labels = window[self.labels_columns].values
        # change type to float32
        embeddings_mod1 = embeddings_mod1.type(torch.float32)
        embeddings_mod2 = embeddings_mod2.type(torch.float32)
        # turn labels into tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        return embeddings_mod1, embeddings_mod2, labels

    def __cut_all_data_on_windows(self):
        """ Cuts all data on windows. """
        self.cut_windows = OrderedDict()
        for key, frames in self.data.items():
            if self.only_consecutive_windows:
                self.cut_windows[key] = self.__cut_sequence_on_consecutive_windows(frames, self.window_size,
                                                                                   self.stride)
            else:
                self.cut_windows[key] = self.__create_windows_out_of_frames(frames, self.window_size, self.stride)
        # check if there were some sequences with not enough frames to create a window
        # they have been returned as None, so we need to remove them
        self.cut_windows = OrderedDict(
            {key: windows for key, windows in self.cut_windows.items() if windows is not None})

    def __create_windows_out_of_frames(self, frames: pd.DataFrame, window_size: Union[int, float],
                                       stride: Union[int, float]) \
            -> Union[List[pd.DataFrame], None]:
        """ Creates windows of frames out of a pd.DataFrame with frames. Each window is a pd.DataFrame with frames.
        The columns are the same as in the original pd.DataFrame.

        :param frames: pd.DataFrame
                pd.DataFrame with frames. Columns format: ['path', ..., 'label_0', ..., 'label_n']
        :param window_size: Union[int, float]
                Size of the window. If int, it is the number of frames in the window. If float, it is the time in seconds.
        :param stride: Union[int, float]
                Stride of the window. If int, it is the number of frames in the window. If float, it is the time in seconds.
        :return:
        """
        # calculate the number of frames in the window
        num_frames = window_size
        # create windows
        windows = self.__cut_sequence_on_windows(frames, window_size=num_frames, stride=stride)

        return windows

    def __cut_sequence_on_windows(self, sequence: pd.DataFrame, window_size: int, stride: int) -> Union[
        List[pd.DataFrame], None]:
        """ Cuts one sequence of values (represented as pd.DataFrame) into windows with fixed size. The stride is used
        to move the window. If there is not enough values to fill the last window, the window starting from
        sequence_end-window_size is added as a last window.

        :param sequence: pd.DataFrame
                Sequence of values represented as pd.DataFrame
        :param window_size: int
                Size of the window in number of values/frames
        :param stride: int
                Stride of the window in number of values/frames
        :return: List[pd.DataFrame]
                List of windows represented as pd.DataFrames
        """
        # check if the sequence is long enough
        # if not, return None and this sequence will be skipped in the __cut_all_data_on_windows method
        if sequence.shape[0] < window_size:
            return None
        windows = []
        # cut sequence on windows using while and shifting the window every step
        window_start = 0
        window_end = window_start + window_size
        while window_end <= len(sequence):
            windows.append(sequence.iloc[window_start:window_end])
            window_start += stride
            window_end += stride
        # add last window if there is not enough values to fill it
        if window_start < len(sequence):
            windows.append(sequence.iloc[-window_size:])
        return windows

    def __cut_sequence_on_consecutive_windows(self, sequence: pd.DataFrame, window_size: int, stride: int) -> Union[
        List[pd.DataFrame], None]:
        # check if the sequence is long enough
        # if not, return None and this sequence will be skipped in the __cut_all_data_on_windows method
        if sequence.shape[0] < window_size:
            return None
        # the problem is that the timesteps are not monotonically increasing. THerefore, we need to cut the data on windows
        # so that the timesteps within the window increase monotonically (with the same timestep/value).
        # The procedure is the following:
        # 1. Get the value of first timestep
        # 2. Get the value of last timestep
        # 3. Get the difference between the first and the last timestep
        # 4. Calculate the value that should be between the first and the last timestep depending on the size of the window
        # 5. Compare the value from the step 4 with the value from the step 3. If it is not equal, do not take this window
        # 6. If it is equal, take the window
        # !!! IMPORTANT!!! we cut windows in this function based on frame_num column, not on the timestep column
        windows = []
        # cut sequence on windows using while and shifting the window every step
        window_start = 0
        window_end = window_start + window_size
        while window_end <= len(sequence):
            window = sequence.iloc[window_start:window_end]
            # check if the timesteps are monotonically increasing
            first_timestep = window['frame_num'].iloc[0]
            last_timestep = window['frame_num'].iloc[-1]
            # take the most often difference between the timesteps
            timestep_difference = window['frame_num'].diff().round(2).mode().values[0]
            # calculate actual range in timesteps and the value that should be in case we have monotonically increasing timesteps
            actual_range = np.round(last_timestep - first_timestep, 2)
            reference_range = np.round(timestep_difference * (window_size - 1), 2)
            if actual_range == reference_range:
                windows.append(window)
            window_start += stride
            window_end += stride
        # also add the last window as it is usually ignored
        window_start = len(sequence) - window_size
        window_end = len(sequence)
        start_timestamp = sequence['frame_num'].iloc[window_start]
        end_timestamp = sequence['frame_num'].iloc[window_end - 1]
        timestep_difference = end_timestamp - start_timestamp
        timestep_value = start_timestamp + (window_size - 1) * timestep_difference
        if timestep_value == end_timestamp:
            windows.append(sequence.iloc[-window_size:])
        return windows