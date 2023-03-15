from typing import Tuple, List

import pandas as pd
import torch
import numpy as np





def get_names_of_train_dev_test_videos(path_to_train:str, path_to_dev:str, path_to_test_csv:str)->Tuple[List[str], List[str], List[str]]:
    """ Returns names of the videos that are used for training, validation and testing.

    :param path_to_train_csv: str
            path to the csv file with train videos
    :param path_to_dev_csv: str
            path to the csv file with dev videos
    :param path_to_test_csv: str
            path to the csv file with test videos
    :return: Tuple[List[str], List[str], List[str]]
            names of the videos that are used for training, validation and testing
    """
    # load csv files
    train_names_df = pd.read_csv(path_to_train_csv, header=None)
    dev_names_df = pd.read_csv(path_to_dev_csv, header=None)
    test_names_df = pd.read_csv(path_to_test_csv, header=None)
    # get names of the videos
    # TODO: get done it


    return train_videos, dev_videos, test_videos