from functools import partial
from typing import Tuple, List, Callable, Optional, Dict, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from decorators.common_decorators import timer
from pytorch_utils.data_loaders.ImageDataLoader_new import ImageDataLoader
from pytorch_utils.data_loaders.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor

import training_config


def load_labels_with_frame_paths(challenge:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Loads the labels with the paths to the frames.

    :param challenge: str
        The challenge name. It should be either 'VA' or 'Exp'.

    :return: Tuple[pd.DataFrame, pd.DataFrame]
        The train and val sets.
    """
    if challenge == "VA":
        train_labels = pd.read_csv(training_config.VA_TRAIN_LABELS_PATH)
        dev_labels = pd.read_csv(training_config.VA_DEV_LABELS_PATH)
    elif challenge == "Exp":
        train_labels = pd.read_csv(training_config.Exp_TRAIN_LABELS_PATH)
        dev_labels = pd.read_csv(training_config.Exp_DEV_LABELS_PATH)
    else:
        raise ValueError("The challenge name should be either 'VA' or 'Exp'.")

    # concat all train labels and dev labels
    train_labels = pd.concat([value for key, value in train_labels.items()], axis=0)
    dev_labels = pd.concat([value for key, value in dev_labels.items()], axis=0)

    # take every 5th frame
    train_labels = train_labels.iloc[::5, :]
    dev_labels = dev_labels.iloc[::5, :]

    # delete -1 categories
    if challenge == "Exp":
        train_labels = train_labels[train_labels["category"] != -1]
        dev_labels = dev_labels[dev_labels["category"] != -1]

    # change columns names for further work
    train_labels.rename(columns={"path_to_frame":"path"}, inplace=True)
    dev_labels.rename(columns={"path_to_frame":"path"}, inplace=True)

    # convert categories to one-hot vectors if it is Exp challenge
    if challenge == "Exp":
        # create one-hot vectors
        train_one_hot = pd.get_dummies(train_labels["category"])
        dev_one_hot = pd.get_dummies(dev_labels["category"])
        # concat one-hot vectors to the labels
        train_labels = pd.concat([train_labels, train_one_hot], axis=1)
        dev_labels = pd.concat([dev_labels, dev_one_hot], axis=1)
        # delete category column
        train_labels.drop(columns=["category"], inplace=True)
        dev_labels.drop(columns=["category"], inplace=True)

    return train_labels, dev_labels


def get_augmentation_function(probability:float)->Dict[Callable, float]:
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
        partial(gaussian_blur_image_random, kernel_size=(5, 9), sigma=(0.1, 5)): training_config.AUGMENT_PROB,
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


def construct_data_loaders(train:pd.DataFrame, dev:pd.DataFrame,
                           preprocessing_functions:List[Callable],
                           batch_size:int,
                           augmentation_functions:Optional[Dict[Callable, float]]=None,
                           num_workers:int=8)\
        ->Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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


    train_dataloader = DataLoader(train_data_loader, batch_size=batch_size, num_workers=num_workers, drop_last = True, shuffle=True)
    dev_dataloader = DataLoader(dev_data_loader, batch_size=batch_size, num_workers=num_workers//2, shuffle=False)

    return (train_dataloader, dev_dataloader)


def load_data_and_construct_dataloaders(model_type:str, batch_size:int, challenge:str,
                                        return_class_weights:Optional[bool]=False)->\
        Union[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
              Tuple[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], torch.Tensor]]:
    """
        Args:
            model_type: str
                The type of the model.
            batch_size: int
                The batch size.
            return_class_weights: Optional[bool]
                If True, the function returns the class weights as well.
            challenge: str
                The challenge of the AffWild2. It can be either "Exp" or "VA".

    Loads the data presented in pd.DataFrames and constructs the data loaders using them. It is a general function
    to assemble all functions defined above.
    Returns: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The train and dev data loaders.
        or
        Tuple[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], torch.Tensor]
        The train, dev data loaders and the class weights calculated based on the training labels.

    """
    if model_type not in ['Modified_HRNet']:
        raise ValueError('The model type should be Modified_HRNet.')
    if challenge not in ['Exp', 'VA']:
        raise ValueError('The challenge should be either "Exp" or "VA".')
    # load pd.DataFrames
    train, dev = load_labels_with_frame_paths(challenge=challenge)
    # define preprocessing functions
    if model_type == 'Modified_HRNet':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=256),
                                   convert_image_to_float_and_scale,
                                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]  # From HRNet
    else:
        raise ValueError(
            f'The model type should be "Modified_HRNet". '
            f'Got {model_type} instead.')
    # define augmentation functions
    augmentation_functions = get_augmentation_function(training_config.AUGMENT_PROB)
    # construct data loaders
    train_dataloader, dev_dataloader = construct_data_loaders(train, dev,preprocessing_functions, batch_size,
                                                                    augmentation_functions,
                                                                    num_workers=training_config.NUM_WORKERS)

    if return_class_weights:
        if challenge == 'VA':
            raise ValueError('The class weights cannot be calculated for the VA challenge.')
        num_classes = train.iloc[:, 1:].shape[1]
        labels = train.iloc[:, 1:]
        labels = labels.dropna()
        class_weights = labels.sum(axis=0)
        class_weights = 1. / (class_weights / class_weights.sum())
        # normalize class weights
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights.values, dtype=torch.float32)
        return ((train_dataloader, dev_dataloader), class_weights)

    return (train_dataloader, dev_dataloader)



@timer
def main():
    train_data_loader, dev_data_loader, test_data_loader = load_data_and_construct_dataloaders(challenge='Exp')
    for x, y in train_data_loader:
        print(x.shape, y.shape)
        print("-------------------")

if __name__ == "__main__":
    main()





