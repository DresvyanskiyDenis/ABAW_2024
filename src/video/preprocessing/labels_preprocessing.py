import glob
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

emo_categories = {
    "Neutral": 0,
    "Anger": 1,
    "Disgust": 2,
    "Fear": 3,
    "Happiness": 4,
    "Sadness": 5,
    "Surprise": 6,
    "Other": 7,
}


def load_AffWild2_labels(path_to_files: str, challenge: str) -> Dict[str, pd.DataFrame]:
    """ Loads labels from AffWild2 dataset. Originally, the AffWild2 labels are stored as .txt files and
    locate in the path_to_files directory. This function loads them and returns a dictionary with
    video names as keys and DataFrames with labels as values.

    :param path_to_files: str
            path to the directory with .txt files with labels
    :param challenge: str
            name of the challenge ABAW 2023. Can be "VA" or "Exp".
    :return: Dict[str, pd.DataFrame]
            dictionary with video names as keys and DataFrames with labels as values
    """
    if challenge not in ["VA", "Exp"]:
        raise ValueError("challenge should be either 'VA' or 'Exp'")
    # define full paths to the label files
    labels_paths = glob.glob(os.path.join(path_to_files, "*.txt"))
    # get only base names
    video_names = [os.path.basename(x) for x in labels_paths]
    # get rid of .txt extension
    video_names = [x.split(".")[0] for x in video_names]
    # load labels
    labels = [pd.read_csv(x, sep=",") for x in labels_paths]
    if challenge == "Exp":
        # drop all columns except for first one and rename the columns to "category"
        # this is done, because in the Exp challenge as a first row we have just naming of all categories
        labels = [pd.DataFrame(x.iloc[:, 0]) for x in labels]
        labels = [x.rename(columns={"Neutral": "category"}) for x in labels]

    return dict(zip(video_names, labels))


def align_labels_with_extracted_frames(path_to_metadata: str, labels: Dict[str, pd.DataFrame],
                                       challenge: str) -> Dict[str, pd.DataFrame]:
    """ Aligns labels with extracted frames. The problem is that some label files contain less or more values as frames
    in the video. This function aligns labels with already preprocessed, extracted frames. The process is simple:
    * if the number of frames is less than the number of labels, than the path to the last frame is repeated in the
    resulting aligned labels.
    * if the number of frames is more than the number of labels, than we take only len(labels) frames.
    Aligned labels will have both the paths to frames and the labels concatenated in one DataFrame.
    The columns are the following: filename, frame_num, timestamp, category (or valence, arousal)

    :param path_to_metadata: str
            path to the metadata file with extracted labels. This is .csv file with the folliwing columns:
            filename,frame_num,timestamp
    :param labels: Dict[str, pd.DataFrame]
            Dictionary with video names as keys and DataFrames with labels as values
    :param challenge: str
            name of the challenge ABAW 2023. Can be "VA" or "Exp".
    :return: Dict[str, pd.DataFrame]
            Dictionary with video names as keys and DataFrames with labels as values
    """
    pd.options.mode.chained_assignment = None
    if challenge not in ["VA", "Exp"]:
        raise ValueError("challenge should be either 'VA' or 'Exp'")
    result = {}
    # load metadata
    metadata = pd.read_csv(path_to_metadata)
    # form additional columns in metadata with only the name of the video
    metadata["video_name"] = [x.split(os.path.sep)[-2] for x in metadata["filename"]]
    # get video names from labels
    video_names = list(labels.keys())
    # iterate over video names
    for video_name in video_names:
        # take labels for the current video
        labels_for_video = labels[video_name].copy(deep=True)
        # check if this video name exists in metadata. If not, report it and continue
        if video_name not in list(metadata["video_name"].unique()):
            print(f"Video {video_name} is not in extracted data (metadata file). Skipping it.")
            continue
        # take related data from metadata
        metadata_for_video = metadata[metadata["video_name"] == video_name].copy(deep=True)
        # apply the procedure described above
        aligned_labels = labels_for_video
        if len(labels_for_video) < len(metadata_for_video):
            metadata_for_video = metadata_for_video.iloc[:len(labels_for_video)]
        elif len(labels_for_video) > len(metadata_for_video):
            # repeat the last frame path with the frame_num incremented by 1 for every repeat and
            # the timestamp incremented by substraction of the last two timestamps
            # calculate new timestamps and frame numbers (which will be added)
            old_len = labels_for_video.shape[0]
            timestep = metadata_for_video["timestamp"].iloc[-1] - metadata_for_video["timestamp"].iloc[-2]
            new_timestamps = [metadata_for_video["timestamp"].iloc[-1] + timestep * idx for idx in
                              range(1, len(labels_for_video) - len(metadata_for_video) + 1)]
            new_timestamps = [round(x, 2) for x in new_timestamps]
            last_frame_num = metadata_for_video["frame_num"].iloc[-1]
            new_frame_nums = [last_frame_num + idx for idx in
                              range(1, len(labels_for_video) - len(metadata_for_video) + 1)]
            # repeat the last row of metadata_for_video
            repeated_rows = [metadata_for_video.iloc[-1:]] * (len(labels_for_video) - len(metadata_for_video))
            repeated_rows = pd.concat(repeated_rows, ignore_index=True)
            metadata_for_video = pd.concat(
                [metadata_for_video.reset_index(drop=True), repeated_rows.reset_index(drop=True)],
                ignore_index=True)
            metadata_for_video = metadata_for_video.reset_index(drop=True)
            # Change repeated timestamps and frame numbers with created ones
            metadata_for_video["timestamp"].iloc[old_len - len(repeated_rows):] = new_timestamps
            metadata_for_video["frame_num"].iloc[old_len - len(repeated_rows):] = new_frame_nums
        # concat labels and metadata
        aligned_labels = pd.concat([aligned_labels.reset_index(drop=True), metadata_for_video.reset_index(drop=True)],
                                   axis=1, ignore_index=True)
        # add aligned labels to the dictionary
        result[video_name] = aligned_labels
    # del added column in metadata
    metadata = metadata.drop(columns=["video_name"])
    # change the columns names, because as a result of concatenation they are not the same as in the original labels
    # then, reorder the columns
    if challenge == "VA":
        result = {key: value.rename(
            columns={0: "valence", 1: "arousal", 2: "path_to_frame", 3: "frame_num", 4: "timestamp", 5: "video_name"})
                  for key, value in result.items()}
        result = {key: value[["path_to_frame", "frame_num", "timestamp", "video_name", "valence", "arousal"]]
                  for key, value in result.items()}
    elif challenge == "Exp":
        result = {key: value.rename(
            columns={0: "category", 1: "path_to_frame", 2: "frame_num", 3: "timestamp", 4: "video_name"})
                  for key, value in result.items()}
        result = {key: value[["path_to_frame", "frame_num", "timestamp", "video_name", "category", ]]
                  for key, value in result.items()}

    return result


def load_train_dev_AffWild2_labels_with_frame_paths(paths_to_labels: Tuple[str, str], path_to_metadata: str,
                                                    challenge: str) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """ Loads labels for AffWild2 dataset for train and dev sets. The result will be the Dictionary with video names as
    keys and DataFrames with labels as values. The DataFrames will have the following columns:
    Challenge VA: path_to_frame, frame_num, timestamp, video_name, valence, arousal
    Challenge Exp: path_to_frame, frame_num, timestamp, video_name, category

    :param paths_to_labels: Tuple[str, str]
            Paths to the labels for train and dev sets.
    :param path_to_metadata: str
            Path to the metadata file (file with data about extracted frames).
    :param challenge: str
            Challenge name. Can be "VA" or "Exp".
    :return: Dict[str, pd.DataFrame]
            Dictionary with video names as keys and DataFrames with labels as values. Aligned and prepared for training.
    """
    # load labels
    train_labels = load_AffWild2_labels(path_to_files=paths_to_labels[0], challenge=challenge)
    dev_labels = load_AffWild2_labels(path_to_files=paths_to_labels[1], challenge=challenge)
    # load metadata
    metadata = pd.read_csv(path_to_metadata)
    # align labels with extracted frames
    aligned_train_labels = align_labels_with_extracted_frames(path_to_metadata=path_to_metadata,
                                                              labels=train_labels,
                                                              challenge=challenge)
    aligned_dev_labels = align_labels_with_extracted_frames(path_to_metadata=path_to_metadata,
                                                            labels=dev_labels,
                                                            challenge=challenge)

    return aligned_train_labels, aligned_dev_labels


def main():
    challenge = "Exp"
    path_to_train_labels = "/nfs/scratch/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/"
    path_to_dev_labels = "/nfs/scratch/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/"
    path_to_metadata_file = "/nfs/scratch/Data/preprocessed/faces/metadata.csv"

    train_labels, dev_labels = load_train_dev_AffWild2_labels_with_frame_paths(
        paths_to_labels=(path_to_train_labels, path_to_dev_labels),
        path_to_metadata=path_to_metadata_file,
        challenge=challenge)

    print(list(train_labels.values())[-50].head(50))
    print("------------------")
    print(list(dev_labels.values())[-50].head(50))


if __name__ == "__main__":
    main()
