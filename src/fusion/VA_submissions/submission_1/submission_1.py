import pickle

import pandas as pd
import numpy as np
import sys
import os

from video.post_processing.embeddings_extraction_dynamic_test import load_fps_file

path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
sys.path.append(path_to_project)
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "datatools"))
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "simple-HRNet-master"))

def load_test_sample_file_and_preprocess(path_to_sample_file:str, video_to_fps):
    sample_file = pd.read_csv(path_to_sample_file)
    sample_file["video_name"] = sample_file["image_location"].apply(lambda x: x.split("/")[0])
    labels_columns = ["valence", "arousal"]
    sample_file[labels_columns] = np.NaN
    # divide on video names
    video_names = sample_file["video_name"].unique()
    result = {}
    for video_name in video_names:
        result[video_name] = sample_file[sample_file["video_name"] == video_name]
    # generate num_frames and timesteps in every video
    for video_name in result.keys():
        video_info = result[video_name]
        fps = video_to_fps[video_name]
        fps_in_seconds = 1. / fps
        video_info["num_frames"] = np.arange(1, video_info.shape[0]+1, 1).astype('int32')
        video_info["timesteps"] = video_info["num_frames"] * fps_in_seconds
        video_info["timesteps"] = np.round(video_info["timesteps"], 2)
    return result


def main():
    path_to_test_predictions_valence = "/Data/features/test_predictions_dynamic/dynamic_features_facial_valence_test_30.pkl"
    path_to_test_predictions_arousal = "/Data/features/test_predictions_dynamic/dynamic_features_facial_arousal_test_30.pkl"
    path_to_test_sample_file = "/Data/test_set/prediction_files_format/CVPR_6th_ABAW_VA_test_set_sample.txt"
    video_to_fps = load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl"))

    # load pickle files
    with open(path_to_test_predictions_valence, 'rb') as f:
        test_predictions_valence = pickle.load(f)
        # valence is first, therefore we need to filter predictions
        for key in test_predictions_valence.keys():
            test_predictions_valence[key]["predictions"] = test_predictions_valence[key]["predictions"][:, 0]
    with open(path_to_test_predictions_arousal, 'rb') as f:
        test_predictions_arousal = pickle.load(f)
        # arousal is second, therefore we need to filter predictions
        for key in test_predictions_arousal.keys():
            test_predictions_arousal[key]["predictions"] = test_predictions_arousal[key]["predictions"][:, 1]
    # combine two dictionaries
    predictions = {}
    for key in test_predictions_valence.keys():
        predictions[key] = np.stack([test_predictions_valence[key]["predictions"], test_predictions_arousal[key]["predictions"]], axis=1)
    # load sample file and preprocess
    sample_file = load_test_sample_file_and_preprocess(path_to_test_sample_file, video_to_fps)
    submission_result = pd.read_csv(path_to_test_sample_file)
    submission_result["video_name"] = submission_result["image_location"].apply(lambda x: x.split("/")[0])
    submission_result[["valence", "arousal"]] = np.NaN
    # aling test predictions with sample file
    for video_name in sample_file.keys():
        preds = predictions[video_name]
        # duplicate some values if needed
        if preds.shape[0] < sample_file[video_name].shape[0]:
            preds = np.concatenate([preds,
                                    np.repeat(preds[-1][np.newaxis, :], sample_file[video_name].shape[0] - preds.shape[0], axis=0)
                                    ], axis=0)
        elif preds.shape[0] > sample_file[video_name].shape[0]:
            print(f"{video_name} has more predictions than needed. Predictions: {preds.shape[0]}, needed: {sample_file[video_name].shape[0]}")
            preds = preds[:sample_file[video_name].shape[0]]
        assert preds.shape[0] == submission_result[submission_result["video_name"] == video_name].shape[0]
        submission_result.loc[submission_result["video_name"] == video_name, ["valence", "arousal"]] = preds

    # save results
    submission_result.drop(columns=["video_name"], inplace=True)
    # check nans
    assert submission_result.isna().sum().sum() == 0
    output_path = "/Data/test_set/VA/submission_1/submission_1.csv"
    submission_result.to_csv(output_path, index=False)






if __name__ == "__main__":
    main()