import glob
import os
from typing import Tuple, List

import pandas as pd
import torch
import numpy as np


# emotions order: Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other
#                 0         1       2     3       4      5      6       7


def get_names_of_train_dev_test_videos(path_to_train:str, path_to_dev:str, path_to_test_csv:str)->Tuple[List[str], List[str], List[str]]:
    """ Returns names of the videos that are used for training, validation and testing.

    :param path_to_train: str
            path to the dir with train txt annotations
    :param path_to_dev: str
             path to the dir with dev txt annotations
    :param path_to_test_csv: str
            path to the csv file with names of test videos
    :return: Tuple[List[str], List[str], List[str]]
            names of the videos that are used for training, validation and testing
    """
    # create train and dev names
    train_videos = glob.glob(os.path.join(path_to_train, "*.txt"))
    dev_videos = glob.glob(os.path.join(path_to_dev, "*.txt"))
    # get only base names
    train_videos = [os.path.basename(x) for x in train_videos]
    dev_videos = [os.path.basename(x) for x in dev_videos]
    # get rid of .txt extension
    train_videos = [x.split(".")[0] for x in train_videos]
    dev_videos = [x.split(".")[0] for x in dev_videos]
    # get names of the test videos
    test_videos = pd.read_csv(path_to_test_csv, header=None).iloc[:, 0].tolist()

    return train_videos, dev_videos, test_videos