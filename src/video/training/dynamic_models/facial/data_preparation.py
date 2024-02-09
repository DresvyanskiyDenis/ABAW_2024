


"""
What needs to be done:
1. Choose between VA and Exp challenge.
2. The same FPS of every video.
3. Possibility to resample FPS (taking every n-th frame).



"""
import os

import numpy as np
import pandas as pd

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
    # rename one-hot vectors using int values
    df.rename(columns={col: f"category_{int(col)}" for col in one_hot.columns}, inplace=True)
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
        path_to_metadata=config['metafile_path'],
        challenge="Exp") # pattern: Dict[filename: frames_with_labels] -> Dict[str, pd.DataFrame]

    train_labels_va, dev_labels_va = load_train_dev_AffWild2_labels_with_frame_paths(
        paths_to_labels=(config['va_train_labels_path'], config['va_dev_labels_path']),
        path_to_metadata=config['metafile_path'],
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