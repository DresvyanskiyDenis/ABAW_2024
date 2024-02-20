import glob
import os
import pickle
from typing import Tuple, Dict

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

def merge_exp_with_va_labels(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """ Merges the labels from the Exp challenge with the labels from the VA challenge.

    :param df1: pd.DataFrame
        df with the labels from the Exp challenge.
    :param df2: pd.DataFrame
        df with the labels from the VA challenge.
    :return: pd.DataFrame
        The concatenated df.
    """
    if df1 is None:
        # create dataframe and generate np.Nans in the "category" columns
        df1 = pd.DataFrame({"path_to_frame": df2["path_to_frame"].values})
        df1["category"] = np.NaN
    if df2 is None:
        # create dataframe and generate np.Nans in the "valence" and "arousal" columns
        df2 = pd.DataFrame({"path_to_frame": df1["path_to_frame"].values})
        df2["valence"] = np.NaN
        df2["arousal"] = np.NaN
    # check if the length of the dfs is the same
    assert len(df1) == len(df2), "The length of the dfs should be the same."
    result = pd.merge(df1, df2, on="path_to_frame", suffixes=('', '_y'))
    # delete duplicates in columns
    result.drop([col for col in result.columns if '_y' in col], axis=1, inplace=True)
    return result


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
    train_labels_exp, dev_labels_exp = load_train_dev_AffWild2_labels_with_frame_paths(
        paths_to_labels=(config['exp_train_labels_path'], config['exp_dev_labels_path']),
        path_to_metadata=config['metafile_path'],
        challenge="Exp")  # pattern: Dict[filename: frames_with_labels] -> Dict[str, pd.DataFrame]

    train_labels_va, dev_labels_va = load_train_dev_AffWild2_labels_with_frame_paths(
        paths_to_labels=(config['va_train_labels_path'], config['va_dev_labels_path']),
        path_to_metadata=config['metafile_path'],
        challenge="VA")
    # concatenate train labels from both challenges deleting duplicates
    train_files = list(train_labels_exp.keys()) + list(train_labels_va.keys())
    train_files = set(train_files)
    # concatenate dev labels from both challenges deleting duplicates
    dev_files = list(dev_labels_exp.keys()) + list(dev_labels_va.keys())
    dev_files = set(dev_files)  # 144 files
    # exclude files that are in the dev but also in the train
    train_files = train_files - dev_files  # 353 files
    # merge labels
    train_labels = {}
    for file in train_files:
        train_labels[file] = merge_exp_with_va_labels(train_labels_exp.get(file), train_labels_va.get(file))
    dev_labels = {}
    for file in dev_files:
        dev_labels[file] = merge_exp_with_va_labels(dev_labels_exp.get(file), dev_labels_va.get(file))
    # concat all train labels and dev labels
    train_labels = pd.concat([value for key, value in train_labels.items()], axis=0)
    dev_labels = pd.concat([value for key, value in dev_labels.items()], axis=0)
    # for Exp challenge, substitute -1 with np.NaN
    train_labels["category"] = train_labels["category"].replace(-1, np.NaN)
    dev_labels["category"] = dev_labels["category"].replace(-1, np.NaN)
    # change columns names for further work
    train_labels.rename(columns={"path_to_frame": "path"}, inplace=True)
    dev_labels.rename(columns={"path_to_frame": "path"}, inplace=True)
    # convert categories to one-hot vectors
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
    columns = list(train_labels.columns) + list(train.columns.difference(train_labels.columns))
    train = train[columns]
    dev = dev[columns]
    return train, dev


def separate_data_into_video_sequences(data: pd.DataFrame, video_to_fps:Dict[str, float], common_fps:int) -> Dict[str, pd.DataFrame]:
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
    video_names = data["video_name"].unique().values
    # create dictionary with video sequences
    result = {}
    for video_name in video_names:
        # get the dataframe for the video
        video_data = data[data["video_name"] == video_name]
        # resample the video to the common_fps
        every_frame = int(round(video_to_fps[video_name] / common_fps))
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
    labels_columns = None # TODO: define it
    feature_columns = None # TODO: define it
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


def get_train_dev_dataloaders(config:dict)->Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
    # separate data into video sequences
    train_video_sequences = separate_data_into_video_sequences(train, video_to_fps, config['common_fps'])
    dev_video_sequences = separate_data_into_video_sequences(dev, video_to_fps, config['common_fps'])
    # construct data loaders
    train_loader, dev_loader = construct_data_loaders(train_video_sequences, dev_video_sequences, config)
    return train_loader, dev_loader




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
        "path_to_fps_file": "src/video/training/dynamic_models/fps.pkl",
        "common_fps":5,
        "window_size": 10,
        "stride": 5,
        "batch_size": 64,
        "num_workers": 4
    }
    # check the general function
    train_loader, dev_loader = get_train_dev_dataloaders(config)





if __name__ == '__main__':
    main()
