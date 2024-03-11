import glob
import os
import pickle
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from pytorch_utils.data_loaders.TemporalEmbeddingsLoader import TemporalEmbeddingsLoader
from src.video.training.dynamic_fusion.FusionDataLoader import FusionDataLoader


def load_train_dev(config)-> Tuple[pd.DataFrame, pd.DataFrame]:
    # load train and dev data
    train_facial = pd.read_csv(config['train_embeddings_facial'])
    dev_facial = pd.read_csv(config['dev_embeddings_facial'])
    train_kinesics = pd.read_csv(config['train_embeddings_kinesics'])
    dev_kinesics = pd.read_csv(config['dev_embeddings_kinesics'])
    # create video_name column
    train_facial["video_name"] = train_facial["path"].apply(lambda x: x.split("/")[-2])
    dev_facial["video_name"] = dev_facial["path"].apply(lambda x: x.split("/")[-2])
    train_kinesics["video_name"] = train_kinesics["path"].apply(lambda x: x.split("/")[-2])
    dev_kinesics["video_name"] = dev_kinesics["path"].apply(lambda x: x.split("/")[-2])
    # merge the dataframes. To do so, a new column should be created for video_name+frame_number
    train_facial['video_name_frame_number'] = train_facial['video_name'] + "_" + train_facial['frame_number'].astype(str)
    dev_facial['video_name_frame_number'] = dev_facial['video_name'] + "_" + dev_facial['frame_number'].astype(str)
    train_kinesics['video_name_frame_number'] = train_kinesics['video_name'] + "_" + train_kinesics['frame_number'].astype(str)
    dev_kinesics['video_name_frame_number'] = dev_kinesics['video_name'] + "_" + dev_kinesics['frame_number'].astype(str)
    # rename embeddings columns
    train_facial = train_facial.rename(columns={f'embedding_{i}': f'facial_embedding_{i}' for i in range(256)})
    dev_facial = dev_facial.rename(columns={f'embedding_{i}': f'facial_embedding_{i}' for i in range(256)})
    train_kinesics = train_kinesics.rename(columns={f'embedding_{i}': f'kinesics_embedding_{i}' for i in range(256)})
    dev_kinesics = dev_kinesics.rename(columns={f'embedding_{i}': f'kinesics_embedding_{i}' for i in range(256)})
    # merge the dataframes
    train = train_facial.merge(train_kinesics, on='video_name_frame_number', how='inner')
    dev = dev_facial.merge(dev_kinesics, on='video_name_frame_number', how='inner')
    # drop the video_name_frame_number column and all columns with _y suffix
    train = train.drop(columns=['video_name_frame_number'])
    dev = dev.drop(columns=['video_name_frame_number'])
    train = train.loc[:, ~train.columns.str.endswith('_y')]
    dev = dev.loc[:, ~dev.columns.str.endswith('_y')]
    # rename all columns with _x suffix (there are other columns with _ in them)
    train = train.rename(columns={column_name: column_name.split("_")[0] for column_name in train.columns if column_name.endswith("_x")})
    dev = dev.rename(columns={column_name: column_name.split("_")[0] for column_name in dev.columns if column_name.endswith("_x")})
    # round timestamps to two decimal places
    train['timestamp'] = train['timestamp'].apply(lambda x: round(x, 2))
    dev['timestamp'] = dev['timestamp'].apply(lambda x: round(x, 2))
    # the same for num_frame
    train['frame_num'] = train['frame_num'].apply(lambda x: round(x, 2))
    dev['frame_num'] = dev['frame_num'].apply(lambda x: round(x, 2))
    # rename timestamp to timestep
    train = train.rename(columns={"timestamp": "timestep"})
    dev = dev.rename(columns={"timestamp": "timestep"})
    # normalization
    if config['normalization'] is not None and config['normalization'] in ["minmax", "standard"]:
        # get idx of column "embedding_0"
        embeddings_columns = [column_name for column_name in train.columns if "embedding_" in column_name]
        # normalize the data
        if config['normalization'] == "minmax":
            scaler = MinMaxScaler()
        elif config['normalization'] == "standard":
            scaler = StandardScaler()
        scaler = scaler.fit(train[embeddings_columns])
        train[embeddings_columns] = scaler.transform(train[embeddings_columns])
        dev[embeddings_columns] = scaler.transform(dev[embeddings_columns])
    return train, dev


def separate_data_into_video_sequences(data: pd.DataFrame, video_to_fps:Dict[str, float], common_fps:Optional[int]=None) -> Dict[str, pd.DataFrame]:
    """ Separates dataframe into video sequences based on the 'path' column. Moreover,
    resamples every video to the common_fps by taking every n-th frame.

    :param data: pd.DataFrame
        The dataframe with columns = ['path, ''] # TODO: complete this
    :param video_to_fps: Dict[str, float]
        Dictionary with fps for every video in the data
    :return: Dict[str, pd.DataFrame]
        Dictionary with video sequences. The key is the video name and the value is the dataframe with all frames that
        have been resampled to the common_fps.
    """
    # get unique video names
    video_names = data["video_name"].unique()
    # create dictionary with video sequences
    result = {}
    for video_name in video_names:
        # get the dataframe for the video
        video_data = data[data["video_name"] == video_name]
        # resample the video to the common_fps
        if common_fps is not None:
            every_frame = int(round(video_to_fps[video_name] / common_fps))
        else:
            every_frame = 1
        video_data = video_data.iloc[::every_frame]
        result[video_name] = video_data
    return result

def construct_data_loaders(train_videos:Dict[str, pd.DataFrame], dev_videos:Dict[str, pd.DataFrame], config:dict)\
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Constructs torch.DataLoaders from the provided data.
    The data will be preprocessed firstly using TemporalEmbeddingsLoader() class, which cuts it into windows and
    serves as data loader class. Then, the torch.utils.data.DataLoader will be used to create the final data loaders.

    :param train_videos: Dict[str, pd.DataFrame]
        Dictionary with train video sequences. The key is the video name and the value is the dataframe with corresponding
        video frames from that video resampled to the common_fps.
    :param dev_videos: Dict[str, pd.DataFrame]
        Dictionary with dev video sequences. The key is the video name and the value is the dataframe with corresponding
        video frames from that video resampled to the common_fps.
    :param config: dict
        Dictionary with configuration parameters. It should contain the following keys:
        - window_size: int, size of the window in number of frames
        - stride: int, stride of the window in number of frames
        - batch_size: int, size of the batch
        - num_workers: int, number of workers (threads) for the data loader
    :return: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Data loaders for train and dev data
    """
    # assign columns that will represent features and labels
    if config['challenge'] == "Exp":
        labels_columns = [f'category_{i}' for i in range(config['num_classes'])]
    elif config['challenge'] == "VA":
        labels_columns = ["valence", "arousal"]
    # create data loaders
    modality1_columns = [column_name for column_name in train_videos[list(train_videos.keys())[0]].columns if "facial" in column_name]
    modality2_columns = [column_name for column_name in train_videos[list(train_videos.keys())[0]].columns if "kinesics" in column_name]
    train_loader = FusionDataLoader(data=train_videos, modality1_columns=modality1_columns, modality2_columns=modality2_columns,
                 labels_columns=labels_columns, window_size=config['window_size'], stride=config['stride'],
                 only_consecutive_windows=True)
    dev_loader = FusionDataLoader(data=dev_videos, modality1_columns=modality1_columns, modality2_columns=modality2_columns,
                    labels_columns=labels_columns, window_size=config['window_size'], stride=config['stride'],
                    only_consecutive_windows=True)
    # create torch.utils.data.DataLoader
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                               drop_last=True,shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_loader, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                               shuffle=False)
    return train_loader, dev_loader


def get_train_dev_dataloaders(config:dict, get_class_weights:Optional[bool]=False)\
        ->Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ General function that loads and prepares the data and corresponding dataloaders for the model training.


    :param config: dict
        Dictionary with configuration parameters. It should contain the following keys:
        - train_embeddings: str, path to the train embeddings file
        - dev_embeddings: str, path to the dev embeddings file
        - exp_train_labels_path: str, path to the expression part of the train labels
        - exp_dev_labels_path: str, path to the expression part of the dev labels
        - va_train_labels_path: str, path to the VA part of the train labels
        - va_dev_labels_path: str, path to the VA part of the dev labels
        - metafile_path: str, path to the metafile (of extracted facial frames or pose frames)
        - path_to_data: str, path to the folder with frames
        - window_size: int, size of the window in number of frames
        - stride: int, stride of the window in number of frames
        - batch_size: int, size of the batch
        - num_workers: int, number of workers (threads) for the data loader
        - common_fps: int, common fps for all videos
    :return: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Data loaders for train and dev data.
    """
    # load train and dev data
    train, dev = load_train_dev(config)
    # load fps file
    with open(config['path_to_fps_file'], 'rb') as f:
        video_to_fps = pickle.load(f)
        video_to_fps = {''.join(key.split('.')[:-1]): value for key, value in video_to_fps.items()}
    # separate data into video sequences
    train_video_sequences = separate_data_into_video_sequences(train, video_to_fps, config['common_fps'])
    dev_video_sequences = separate_data_into_video_sequences(dev, video_to_fps, config['common_fps'])
    # apply per-video normalization if needed
    if config['normalization'] in ["per-video-minmax", "per-video-standard"]:
        for video_name in train_video_sequences.keys():
            embedding_columns = [column_name for column_name in train_video_sequences[video_name].columns if "embedding_" in column_name]
            scaler = StandardScaler() if config['normalization'] == "per-video-standard" else MinMaxScaler()
            train_video_sequences[video_name][embedding_columns] = scaler.fit_transform(train_video_sequences[video_name][embedding_columns])
        for video_name in dev_video_sequences.keys():
            embedding_columns = [column_name for column_name in dev_video_sequences[video_name].columns if "embedding_" in column_name]
            scaler = StandardScaler() if config['normalization'] == "per-video-standard" else MinMaxScaler()
            dev_video_sequences[video_name][embedding_columns] = scaler.fit_transform(dev_video_sequences[video_name][embedding_columns])
    # construct data loaders
    train_loader, dev_loader = construct_data_loaders(train_video_sequences, dev_video_sequences, config)
    if get_class_weights:
        if config['challenge'] == "VA":
            raise ValueError("The class weights are not implemented for the VA challenge.")
        labels_columns = [f"category_{i}" for i in range(8)]
        class_weights = __calculate_class_weights(train, labels_columns)
        return train_loader, dev_loader, class_weights
    return train_loader, dev_loader

def get_dev_resampled_and_full_fps_dicts(config:dict)->Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """ Loads dev data and resamples it to the common_fps. Also, loads the dev data with full fps.

    :param config: dict
        Dictionary with configuration parameters. It should contain the following keys:
        - train_embeddings: str, path to the train embeddings file
        - dev_embeddings: str, path to the dev embeddings file
        - exp_train_labels_path: str, path to the expression part of the train labels
        - exp_dev_labels_path: str, path to the expression part of the dev labels
        - va_train_labels_path: str, path to the VA part of the train labels
        - va_dev_labels_path: str, path to the VA part of the dev labels
        - metafile_path: str, path to the metafile (of extracted facial frames or pose frames)
        - path_to_data: str, path to the folder with frames
        - window_size: int, size of the window in number of frames
        - stride: int, stride of the window in number of frames
        - batch_size: int, size of the batch
        - num_workers: int, number of workers (threads) for the data loader
        - common_fps: int, common fps for all videos
    :return: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        Dictionaries with dev data resampled to the common_fps and with full fps.
    """
    # load train and dev data
    train, dev = load_train_dev(config)
    # load fps file
    video_to_fps = load_fps_file(config['path_to_fps_file'])
    # separate data into video sequences
    dev_resampled = separate_data_into_video_sequences(dev, video_to_fps, config['common_fps'])
    dev_full_fps = separate_data_into_video_sequences(dev, video_to_fps, None)
    return dev_resampled, dev_full_fps


def load_fps_file(path_to_fps_file:str)->Dict[str, float]:
    """ Loads the video_to_fps file from the provided path.

    :param path_to_fps_file: str
        Path to the video_to_fps file.
    :return: Dict[str, float]
        Dictionary with video names as keys and fps as values.
    """
    with open(path_to_fps_file, 'rb') as f:
        fps_dict = pickle.load(f)
        fps_dict = {''.join(key.split('.')[:-1]): value for key, value in fps_dict.items()}
    return fps_dict



def __calculate_class_weights(df, labels_columns)->torch.Tensor:
    """ Calculates the class weights for the provided dataframe.

    :param df: pd.DataFrame
        The dataframe with all training labels
    :param labels_columns: List[str]
        The list with the names of the columns that represent the labels
    :return: torch.Tensor
        The array with class weights.
    """
    num_classes = len(labels_columns)
    labels = df[labels_columns]
    labels = labels.dropna()
    class_weights = labels.sum(axis=0)
    class_weights = 1. / (class_weights / class_weights.sum())
    # normalize class weights
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32)
    return class_weights

