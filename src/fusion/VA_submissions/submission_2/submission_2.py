import pickle

import pandas as pd
import numpy as np
import sys
import os

from evaluation_dynamic import __interpolate_to_100_fps, __synchronize_predictions_with_ground_truth, \
    np_concordance_correlation_coefficient
from fusion.exp_submissions.weighted_fusion.submission_2.submission_2 import average_predictions_on_timesteps
from video.preprocessing.labels_preprocessing import load_AffWild2_labels

path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
sys.path.append(path_to_project)
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "datatools"))
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "simple-HRNet-master"))

from video.post_processing.embeddings_extraction_dynamic_test import load_fps_file

def load_video_modality(part:str):
    if part == "dev":
        path_to_test_predictions_valence = "/Data/features/test_predictions_dynamic/VA/dynamic_features_facial_valence_train_dev_30.pkl"
        path_to_test_predictions_arousal = "/Data/features/test_predictions_dynamic/VA/dynamic_features_facial_arousal_train_dev_30.pkl"
    elif part == "test":
        path_to_test_predictions_valence = "/Data/features/test_predictions_dynamic/VA/dynamic_features_facial_valence_test_30.pkl"
        path_to_test_predictions_arousal = "/Data/features/test_predictions_dynamic/VA/dynamic_features_facial_arousal_test_30.pkl"


    # load pickle files
    with open(path_to_test_predictions_valence, 'rb') as f:
        predictions_valence = pickle.load(f)
        # valence is first, therefore we need to filter predictions
        for key in predictions_valence.keys():
            predictions_valence[key]["predictions"] = predictions_valence[key]["predictions"][:, 0]
    with open(path_to_test_predictions_arousal, 'rb') as f:
        predictions_arousal = pickle.load(f)
        # arousal is second, therefore we need to filter predictions
        for key in predictions_arousal.keys():
            predictions_arousal[key]["predictions"] = predictions_arousal[key]["predictions"][:, 1]
    # combine two dictionaries
    predictions = {}
    for key in predictions_valence.keys():
        predictions[key] = np.stack(
            [predictions_valence[key]["predictions"], predictions_arousal[key]["predictions"]], axis=1)
    return predictions


def load_audio_modality(part, video_to_fps):
    if part == "dev":
        path = "/Data/features/test_predictions_dynamic/VA/va_devel.pickle"
    elif part == "test":
        path = "/Data/features/test_predictions_dynamic/VA/va_test.pickle"
    with open(path, 'rb') as f:
        audio_predictions = pickle.load(f)
        audio_predictions = {str(key): audio_predictions[key] for key in audio_predictions.keys()}
        audio_predictions = {key.split(".")[0]: audio_predictions[key] for key in audio_predictions.keys()}
    if part == "dev":
        labels = load_labels(
            path_to_labels="/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
            challenge="VA", video_to_fps=video_to_fps)
    elif part == "test":
        labels = load_test_sample_file_and_preprocess(path_to_sample_file="/Data/test_set/prediction_files_format/CVPR_6th_ABAW_VA_test_set_sample.txt"
                                                       , video_to_fps=video_to_fps)
    audio_predictions = process_dict(audio_predictions, labels)
    return audio_predictions


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







def load_labels(path_to_labels: str, challenge:str, video_to_fps:dict)->pd.DataFrame:
    labels = load_AffWild2_labels(path_to_labels, challenge)
    if '10-60-1280x720_right' in labels.keys(): # nice crutch, but we need it for now
        labels.pop('10-60-1280x720_right')
    # calculate timesteps and num_frames
    for video_name, video_info in labels.items():
        video_fps = video_to_fps[video_name]
        fps_in_seconds = 1. / video_fps
        num_frames = np.arange(1, video_info[["valence", "arousal"]].shape[0]+1, 1).astype('int32')
        timesteps = num_frames * fps_in_seconds
        timesteps = np.round(timesteps, 2)
        video_info['timesteps'] = timesteps
        video_info['num_frames'] = num_frames
    return labels



def process_dict(predictions_dict:dict, labels_dict):
    labels_columns = ["valence","arousal"]
    result_dict = {}
    # go over videos
    for video_name in predictions_dict.keys():
        labels = labels_dict[video_name][labels_columns].values
        labels_timesteps = labels_dict[video_name]["timesteps"].values
        # get predictions
        predictions_timesteps, predictions = average_predictions_on_timesteps(predictions_dict[video_name])
        # before interpolation, add the last timestep of the labels to the current_timesteps and duplicate the prediction for
        # that timestep
        if predictions_timesteps[-1] != labels_timesteps[-1]:
            predictions_timesteps = np.append(predictions_timesteps, labels_timesteps[-1])
            predictions = np.vstack((predictions, predictions[-1]))
        # interpolate predictions to 100 fps
        predictions, predictions_timesteps = __interpolate_to_100_fps(predictions, predictions_timesteps)
        # synchronize predictions and labels
        predictions, predictions_timesteps, labels, labels_timesteps = \
            __synchronize_predictions_with_ground_truth(predictions, predictions_timesteps, labels, labels_timesteps)
        # check if predictions and labels are of the same length
        assert len(predictions) == len(labels)
        result_dict[video_name] = {"predictions": predictions, "predictions_timesteps": predictions_timesteps,
                                     "labels": labels, "labels_timesteps": labels_timesteps}
    return result_dict


def get_best_fusion_weights(audio, video, label_values, num_generations=1000):
    # generate num_generations weights using Dirichlet distribution. We generate both for models and classses
    generated_weights = np.random.dirichlet((1,1), size=(num_generations,audio.shape[-1])).transpose(0,2,1)
    best_CCC = 0
    best_CCC_valence = 0
    best_CCC_arousal = 0
    best_weights = None
    for weights in generated_weights:
        # calculate metrics
        prediction = audio * weights[0][np.newaxis,...] + video * weights[1][np.newaxis,...]
        CCC_valence = np_concordance_correlation_coefficient(label_values[:,0], prediction[:,0])
        CCC_arousal = np_concordance_correlation_coefficient(label_values[:,1], prediction[:,1])
        CCC = (CCC_valence + CCC_arousal) / 2
        if CCC > best_CCC:
            best_CCC = CCC
            best_CCC_valence = CCC_valence
            best_CCC_arousal = CCC_arousal
            best_weights = weights
    return best_weights, best_CCC, best_CCC_valence, best_CCC_arousal


def generate_test_predictions(audio, video, weights, path_to_test_sample, output_path):
    labels_columns = ["valence", "arousal"]
    sample_file = pd.read_csv(path_to_test_sample)
    sample_file["video_name"] = sample_file["image_location"].apply(lambda x: x.split("/")[0])
    sample_file[labels_columns] = np.NaN
    for video_name in video.keys():
        video_preds = video[video_name]
        audio_preds = audio[video_name]["predictions"]
        lbs = sample_file[sample_file["video_name"] == video_name][labels_columns].values

        if video_preds.shape[0] < lbs.shape[0]:
            video_preds = np.concatenate([video_preds, np.repeat(video_preds[-1][np.newaxis, :],
                                                                 lbs.shape[0] - video_preds.shape[0], axis=0)], axis=0)
        elif video_preds.shape[0] > lbs.shape[0]:
            video_preds = video_preds[:lbs.shape[0]]

        if audio_preds.shape[0] < lbs.shape[0]:
            audio_preds = np.concatenate([audio_preds, np.repeat(audio_preds[-1][np.newaxis, :],
                                                                 lbs.shape[0] - audio_preds.shape[0], axis=0)], axis=0)
        elif audio_preds.shape[0] > lbs.shape[0]:
            audio_preds = audio_preds[:lbs.shape[0]]

        assert video_preds.shape[0] == audio_preds.shape[0]
        assert video_preds.shape[0] == lbs.shape[0]
        assert audio_preds.shape[0] == lbs.shape[0]

        final_prediction = audio_preds * weights[0] + video_preds * weights[1]
        sample_file.loc[sample_file["video_name"] == video_name, labels_columns] = final_prediction
    # check on NaNs
    assert sample_file[labels_columns].isna().sum().sum() == 0
    # save results
    sample_file.drop(columns=["video_name"], inplace=True)
    sample_file.to_csv(output_path, index=False)





def main():
    video_to_fps = load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl"))
    video_dev = load_video_modality("dev")
    video_test = load_video_modality("test")
    audio_dev = load_audio_modality("dev", video_to_fps)
    audio_test = load_audio_modality("test", video_to_fps)
    # get keys in video that are only in audio
    video_dev = {key: video_dev[key] for key in video_dev.keys() if key in audio_dev.keys()}
    # load labels
    labels = load_labels(
        path_to_labels="/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
        challenge="VA", video_to_fps=video_to_fps)
    # concat everything
    audio_dev_all = []
    video_dev_all = []
    labels_all = []
    for key in video_dev.keys():
        # check if video has the same length as labels. If not, duplicate the last value
        video_preds = video_dev[key]

        if video_preds.shape[0] < labels[key].shape[0]:
            video_preds = np.concatenate([video_preds, np.repeat(video_preds[-1][np.newaxis, :],
                                                                 labels[key].shape[0] - video_preds.shape[0], axis=0)], axis=0)
        elif video_preds.shape[0] > labels[key].shape[0]:
            video_preds = video_preds[:labels[key].shape[0]]

        video_dev_all.append(video_preds)
        audio_dev_all.append(audio_dev[key]["predictions"])
        labels_all.append(labels[key][["valence", "arousal"]].values)
    audio_dev = np.concatenate(audio_dev_all, axis=0)
    video_dev = np.concatenate(video_dev_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    # some labels have -5, we need to filter them out
    mask = labels[:, 0] != -5
    audio_dev = audio_dev[mask]
    video_dev = video_dev[mask]
    labels = labels[mask]
    mask = labels[:, 1] != -5
    audio_dev = audio_dev[mask]
    video_dev = video_dev[mask]
    labels = labels[mask]
    # get best weights
    best_weights, best_CCC, best_CCC_valence, best_CCC_arousal = get_best_fusion_weights(audio_dev, video_dev, labels)
    print(f"Best weights: {best_weights}")
    print(f"Best CCC: {best_CCC}")
    print(f"Best CCC valence: {best_CCC_valence}")
    print(f"Best CCC arousal: {best_CCC_arousal}")
    # generate predictions
    generate_test_predictions(audio_test, video_test, best_weights,
                               path_to_test_sample="/Data/test_set/prediction_files_format/CVPR_6th_ABAW_VA_test_set_sample.txt",
                               output_path="/Data/test_set/VA/submission_2/submission_2.csv")
    # save best weights
    with open("/Data/test_set/VA/submission_2/best_weights.pkl", 'wb') as f:
        pickle.dump(best_weights, f)





if __name__ == "__main__":
    main()