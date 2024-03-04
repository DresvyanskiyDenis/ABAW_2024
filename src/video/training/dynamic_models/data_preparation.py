import glob
import os
import pickle
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pytorch_utils.data_loaders.TemporalDataLoader import TemporalDataLoader
from pytorch_utils.data_loaders.TemporalEmbeddingsLoader import TemporalEmbeddingsLoader
from src.video.preprocessing.labels_preprocessing import load_train_dev_AffWild2_labels_with_frame_paths


def generate_fps_file(path_to_videos:str, output_path:str)->None:
    """ Generates a dictionary with fps for all videos in the provided folder and saves it to the output_path
    as a pickle file.

    :param path_to_videos: str
        Path to the folder with videos.
    :param output_path: str
        Path for the result dictionary to be saved. Should end with .pkl
    :return: None
    """
    filenames = glob.glob(os.path.join(path_to_videos, '*'))
    fps_dict = {}
    for filename in tqdm(filenames):
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_dict[os.path.basename(filename)] = fps
        # release the video capture
        cap.release()

    with open(output_path, 'wb') as f:
        pickle.dump(fps_dict, f)


def convert_categories_to_on_hot(df: pd.DataFrame) -> pd.DataFrame:
    """ Converts the categories to one-hot vectors preserving the np.Nans.

    :param df: pd.DataFrame
        The df with the 'category' column
    :return: pd.DataFrame
        The df with the one-hot vectors. They will be concatenated to the original df. The 'category' column will be
        deleted and the one-hot vectors will be called 'category_0', 'category_1', ... '
    """
    # create one-hot vectors for categories
    one_hot = pd.get_dummies(df["category"])
    # apply mask to one-hot vectors so that all np.Nans are in the same rows
    mask = df["category"].isna()
    one_hot[mask] = np.NaN
    # concatenate one-hot vectors to the original df
    df = pd.concat([df, one_hot], axis=1)
    # delete the 'category' column
    df.drop(columns=["category"], inplace=True)
    # rename one-hot vectors using int values
    df.rename(columns={col: f"category_{int(col)}" for col in one_hot.columns}, inplace=True)
    # for those columns, change the type to int
    for col in df.columns:
        if "category" in col:
            df[col] = df[col].astype(np.float32)
    return df




def load_labels(config:dict)->Tuple[pd.DataFrame, pd.DataFrame]:
    """ Loads train and dev labels from the provided paths

    :param config: dict
        Dictionary with configuration parameters. It should contain the following keys:
        - exp_train_labels_path: str, path to the expression part of the train labels
        - exp_dev_labels_path: str, path to the expression part of the dev labels
        - va_train_labels_path: str, path to the VA part of the train labels
        - va_dev_labels_path: str, path to the VA part of the dev labels
        - metafile_path: str, path to the metafile (of extracted facial frames or pose frames)
        - path_to_data: str, path to the folder with frames
    :return: Tuple[pd.DataFrame, pd.DataFrame]
        Dataframes with train and dev labels
    """
    if config['challenge'] == "Exp":
        train_labels, dev_labels = load_train_dev_AffWild2_labels_with_frame_paths(
            paths_to_labels=(config['exp_train_labels_path'], config['exp_dev_labels_path']),
            path_to_metadata=config['metafile_path'],
            challenge="Exp")  # pattern: Dict[filename: frames_with_labels] -> Dict[str, pd.DataFrame]
    elif config['challenge'] == "VA":
        train_labels, dev_labels = load_train_dev_AffWild2_labels_with_frame_paths(
            paths_to_labels=(config['va_train_labels_path'], config['va_dev_labels_path']),
            path_to_metadata=config['metafile_path'],
            challenge="VA")
    else:
        raise ValueError("The challenge should be either 'Exp' or 'VA'.")

    # concat all train labels and dev labels
    train_labels = pd.concat([value for key, value in train_labels.items()], axis=0)
    dev_labels = pd.concat([value for key, value in dev_labels.items()], axis=0)
    # for Exp challenge, delete all -1 values in the "category" column
    if config['challenge'] == "Exp":
        train_labels = train_labels[train_labels["category"] != -1]
        dev_labels = dev_labels[dev_labels["category"] != -1]
    # for VA challenge, delete all values that are not in the range -1<=x<=1
    if config['challenge']== "VA":
        # keep only the frames with arousal and valence -1<=x<=1
        train_labels = train_labels[(train_labels["valence"] >= -1) & (train_labels["valence"] <= 1) &
                                    (train_labels["arousal"] >= -1) & (train_labels["arousal"] <= 1)]
        dev_labels = dev_labels[(dev_labels["valence"] >= -1) & (dev_labels["valence"] <= 1) &
                                (dev_labels["arousal"] >= -1) & (dev_labels["arousal"] <= 1)]
    # change columns names for further work
    train_labels.rename(columns={"path_to_frame": "path"}, inplace=True)
    dev_labels.rename(columns={"path_to_frame": "path"}, inplace=True)
    # convert categories to one-hot vectors
    if config['challenge'] == "Exp":
        train_labels = convert_categories_to_on_hot(train_labels)
        dev_labels = convert_categories_to_on_hot(dev_labels)
    # change absolute paths to relative one up to the second directory + change slashes to the ones that current system uses
    train_labels["path"] = train_labels["path"].apply(lambda x: str(os.path.sep).join(x.split("/")[-2:]))
    dev_labels["path"] = dev_labels["path"].apply(lambda x: str(os.path.sep).join(x.split("/")[-2:]))
    # add config['path_to_data'] to the paths
    train_labels["path"] = config['path_to_data'] + train_labels["path"]
    dev_labels["path"] = config['path_to_data'] + dev_labels["path"]
    return train_labels, dev_labels





def load_train_dev(config)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Loads train and dev data from the provided paths and prepares data for cutting.
    Also, combines data with corresponding labels

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
    :return: Tuple[pd.DataFrame, pd.DataFrame]
        Dataframes with train and dev data (concatenated with labels)
    """
    # load train and dev data
    train = pd.read_csv(config['train_embeddings'])
    dev = pd.read_csv(config['dev_embeddings'])
    # load labels
    train_labels, dev_labels = load_labels(config)
    # combine data with labels based on the path
    train = pd.merge(train, train_labels, on="path")
    dev = pd.merge(dev, dev_labels, on="path")
    # delete all columns with _y and change name of the columns with _x
    train.drop([col for col in train.columns if '_y' in col], axis=1, inplace=True)
    dev.drop([col for col in dev.columns if '_y' in col], axis=1, inplace=True)
    train.rename(columns={col: col.split("_")[0] for col in train.columns if '_x' in col}, inplace=True)
    dev.rename(columns={col: col.split("_")[0] for col in dev.columns if '_x' in col}, inplace=True)
    # reorder columns. First columns are columns from labels, then embeddings
    columns = list(train_labels.columns) + [col for col in train if col not in train_labels.columns]
    train = train[columns]
    dev = dev[columns]
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
    # create video_name column
    data["video_name"] = data["path"].apply(lambda x: x.split("/")[-2])
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
    feature_columns = [f'embedding_{i}' for i in range(256)]
    # create data loaders
    train_loader = TemporalEmbeddingsLoader(embeddings_with_labels=train_videos, window_size=config['window_size'],
                                            stride=config['stride'], consider_timestamps=False,
                                            feature_columns=feature_columns, label_columns=labels_columns,
                                            shuffle = True)
    dev_loader = TemporalEmbeddingsLoader(embeddings_with_labels=dev_videos, window_size=config['window_size'],
                                            stride=config['stride'], consider_timestamps=False,
                                            feature_columns=feature_columns, label_columns=labels_columns)
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


def main():
    # generate fps file just once
    """path_to_videos = '/nfs/scratch/ddresvya/Data/ABAW/'
    output_path = '/nfs/scratch/ddresvya/Data/preprocessed/fps.pkl'
    generate_fps_file(path_to_videos, output_path)"""
    config = {
        "va_train_labels_path": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/",
        "va_dev_labels_path": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
        "exp_train_labels_path": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/",
        "exp_dev_labels_path": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/",
        "metafile_path": "/nfs/scratch/ddresvya/Data/preprocessed/faces/metadata.csv",
        "path_to_data": "/nfs/scratch/ddresvya/Data/preprocessed/faces/",
        "train_embeddings": "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/EfficientNet_b4/b4_facial_features_train.csv",
        "dev_embeddings": "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/EfficientNet_b4/b4_facial_features_dev.csv",
        "path_to_fps_file": "/nfs/home/ddresvya/scripts/ABAW_2023_SIU/src/video/training/dynamic_models/fps.pkl",
        "common_fps":5,
        "window_size": 10,
        "stride": 5,
        "batch_size": 64,
        "num_workers": 4,
        "num_classes":8,
    }
    # check the general function
    train_loader, dev_loader, class_weights = get_train_dev_dataloaders(config, get_class_weights=True)
    for x, y in train_loader:
        print(x.shape, y.shape)






if __name__ == '__main__':
    main()
