import os
from functools import partial
from typing import Tuple, Dict, Union, Optional, List, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pytorch_utils.data_loaders.ImageDataLoader_new import ImageDataLoader
from pytorch_utils.data_loaders.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor, \
    ViT_image_preprocessor
from src.video.preprocessing.labels_preprocessing import load_train_dev_AffWild2_labels_with_frame_paths


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
    # rename one-hot vectors
    df.rename(columns={col: f"category_{col}" for col in one_hot.columns}, inplace=True)
    # for those columns, change the type to int
    for col in df.columns:
        if "category" in col:
            df[col] = df[col].astype(np.float32)
    return df




def load_labels_with_frame_paths(config:Dict[str, Union[int, float, str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Loads the labels with the paths to the frames. Both Expression and VA challenges will be loaded.


    :return: Tuple[pd.DataFrame, pd.DataFrame]
        The train and val sets.
    """
    train_labels_exp, dev_labels_exp = load_train_dev_AffWild2_labels_with_frame_paths(
        paths_to_labels=(config['exp_train_labels_path'], config['exp_dev_labels_path']),
        path_to_metadata=config['exp_metadata_path'],
        challenge="Exp") # pattern: Dict[filename: frames_with_labels] -> Dict[str, pd.DataFrame]

    train_labels_va, dev_labels_va = load_train_dev_AffWild2_labels_with_frame_paths(
        paths_to_labels=(config['va_train_labels_path'], config['va_dev_labels_path']),
        path_to_metadata=config['va_metadata_path'],
        challenge="VA")

    # concatenate train labels from both challenges deleting duplicates
    train_files = list(train_labels_exp.keys()) + list(train_labels_va.keys())
    train_files = set(train_files)
    # concatenate dev labels from both challenges deleting duplicates
    dev_files = list(dev_labels_exp.keys()) + list(dev_labels_va.keys())
    dev_files = set(dev_files) # 144 files
    # exclude files that are in the dev but also in the train
    train_files = train_files - dev_files # 353 files
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
    # take every 5th frame
    train_labels = train_labels.iloc[::5, :]
    dev_labels = dev_labels.iloc[::5, :]
    # for Exp challenge, substitute -1 with np.NaN
    train_labels["category"] = train_labels["category"].replace(-1, np.NaN)
    dev_labels["category"] = dev_labels["category"].replace(-1, np.NaN)
    # change columns names for further work
    train_labels.rename(columns={"path_to_frame": "path"}, inplace=True)
    dev_labels.rename(columns={"path_to_frame": "path"}, inplace=True)
    # drop columns frame_num, timestamp, video_name as we do not need them in static model
    train_labels.drop(columns=["frame_num", "timestamp", "video_name"], inplace=True)
    dev_labels.drop(columns=["frame_num", "timestamp", "video_name"], inplace=True)
    # convert categories to one-hot vectors
    train_labels = convert_categories_to_on_hot(train_labels)
    dev_labels = convert_categories_to_on_hot(dev_labels)
    # change absolute paths to relative one up to the third directory + change slashes to the ones that current system uses
    train_labels["path"] = train_labels["path"].apply(lambda x: str(os.path.sep).join(x.split("/")[-3:]))
    dev_labels["path"] = dev_labels["path"].apply(lambda x: str(os.path.sep).join(x.split("/")[-3:]))

    return train_labels, dev_labels

def get_augmentation_function(probability: float) -> Dict[Callable, float]:
    """
    Returns a dictionary of augmentation functions and the probabilities of their application.
    Args:
        probability: float
            The probability of applying the augmentation function.

    Returns: Dict[Callable, float]
        A dictionary of augmentation functions and the probabilities of their application.

    """
    augmentation_functions = {
        pad_image_random_factor: probability,
        grayscale_image: probability,
        partial(collor_jitter_image_random, brightness=0.5, hue=0.3, contrast=0.3,
                saturation=0.3): probability,
        partial(gaussian_blur_image_random, kernel_size=(5, 9), sigma=(0.1, 5)): probability,
        random_perspective_image: probability,
        random_rotation_image: probability,
        partial(random_crop_image, cropping_factor_limits=(0.7, 0.9)): probability,
        random_posterize_image: probability,
        partial(random_adjust_sharpness_image, sharpness_factor_limits=(0.1, 3)): probability,
        random_equalize_image: probability,
        random_horizontal_flip_image: probability,
        random_vertical_flip_image: probability,
    }
    return augmentation_functions

def construct_data_loaders(train: pd.DataFrame, dev: pd.DataFrame,
                           preprocessing_functions: List[Callable],
                           batch_size: int,
                           augmentation_functions: Optional[Dict[Callable, float]] = None,
                           num_workers: int = 8) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Constructs the data loaders for the train, dev sets.

    Args:
        train: pd.DataFrame
            The train set.
        dev: pd.DataFrame
            The dev set.
        preprocessing_functions: List[Callable]
            A list of preprocessing functions to be applied to the images.
        batch_size: int
            The batch size.
        augmentation_functions: Optional[Dict[Callable, float]]
            A dictionary of augmentation functions and the probabilities of their application.
        num_workers: int
            The number of workers to be used by the data loaders.

    Returns: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The data loaders for the train, dev sets.

    """

    train_data_loader = ImageDataLoader(paths_with_labels=train, preprocessing_functions=preprocessing_functions,
                                        augmentation_functions=augmentation_functions, shuffle=True)

    dev_data_loader = ImageDataLoader(paths_with_labels=dev, preprocessing_functions=preprocessing_functions,
                                      augmentation_functions=None, shuffle=False)

    train_dataloader = DataLoader(train_data_loader, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                  shuffle=True)
    dev_dataloader = DataLoader(dev_data_loader, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return (train_dataloader, dev_dataloader)

def load_data_and_construct_dataloaders(config: Dict[str, Union[int, float, str]]) -> \
        Union[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
              Tuple[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], torch.Tensor]]:
    """

    :param config: Dict[str, Union[int, float, str]]
        The configuration dictionary. It contains all needed paths and training parameters.


    Loads the data presented in pd.DataFrames and constructs the data loaders using them. It is a general function
    to assemble all functions defined above.
    :returns Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The train and dev data loaders.
        or
        Tuple[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], torch.Tensor]
        The train, dev data loaders and the class weights calculated based on the training labels.

    """
    if config['model_type'] not in ['EfficientNet-B1', 'EfficientNet-B4', 'ViT_b_16']:
        raise ValueError('The model type should be either "EfficientNet-B1", "EfficientNet-B4", or ViT_b_16.')
    # load pd.DataFrames
    train, dev = load_labels_with_frame_paths(config)
    # define preprocessing functions
    if config['model_type'] == 'EfficientNet-B1':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    elif config['model_type'] == 'EfficientNet-B4':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=380),
                                   EfficientNet_image_preprocessor()]
    elif config['model_type'] == 'ViT_b_16':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=224),
                                   ViT_image_preprocessor()]
    else:
        raise ValueError(f'The model type should be either "EfficientNet-B1", "EfficientNet-B4", or ViT_b_16.'
                         f'Got {config["model_type"]} instead.')
    # define augmentation functions
    augmentation_functions = get_augmentation_function(config['augmentation_probability'])
    # construct data loaders
    train_dataloader, dev_dataloader = construct_data_loaders(train, dev, preprocessing_functions, config['batch_size'],
                                                              augmentation_functions,
                                                              num_workers=config['num_workers'])
    # generate class weights for classification labels # TODO: check it
    num_classes = train.iloc[:, 1:].shape[1]
    labels = train.iloc[:, 1:]
    labels = labels.dropna()
    class_weights = labels.sum(axis=0)
    print(class_weights)
    class_weights = 1. / (class_weights / class_weights.sum())
    # normalize class weights
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32)
    return ((train_dataloader, dev_dataloader), class_weights)

    return (train_dataloader, dev_dataloader)







if __name__ == "__main__":
    print('start')
    config = {
        "va_train_labels_path": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/",
        "va_dev_labels_path": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
        "exp_train_labels_path": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/",
        "exp_dev_labels_path": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/",
        "va_metadata_path": "/nfs/scratch/ddresvya/Data/preprocessed/faces/metadata.csv",
        "exp_metadata_path": "/nfs/scratch/ddresvya/Data/preprocessed/faces/metadata.csv"
    }
    train_labels, dev_labels = load_labels_with_frame_paths(config)
    a=1.+2
    """    # generate random data
    data = pd.DataFrame(columns=["path", "category"])
    data["path"] = [f"file_{i}" for i in range(20)]
    data["category"] = np.random.randint(0, 3, size=20)
    data["category"].iloc[::2] = np.NaN
    mask = data["category"].isna()
    # create one-hot vectors for categories
    one_hot = pd.get_dummies(data["category"])
    # apply mask to one-hot vectors so that all np.Nans are in the same rows
    one_hot[mask] = np.NaN



    a=1+2."""

