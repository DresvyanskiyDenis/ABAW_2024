import os
import pickle
import sys

from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import f1_score

path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
sys.path.append(path_to_project)
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "datatools"))
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "simple-HRNet-master"))

import pandas as pd
import numpy as np

from evaluation_dynamic import __average_predictions_on_timesteps, __interpolate_to_100_fps, \
    __synchronize_predictions_with_ground_truth
from video.preprocessing.labels_preprocessing import load_AffWild2_labels


def load_labels(path_to_labels: str, challenge:str, video_to_fps:dict)->pd.DataFrame:
    labels = load_AffWild2_labels(path_to_labels, challenge)
    if '10-60-1280x720_right' in labels.keys(): # nice crutch, but we need it for now
        labels.pop('10-60-1280x720_right')
    # calculate timesteps and num_frames
    for video_name, video_info in labels.items():
        video_fps = video_to_fps[video_name]
        fps_in_seconds = 1. / video_fps
        num_frames = np.arange(1, video_info['category'].shape[0]+1, 1).astype('int32')
        timesteps = num_frames * fps_in_seconds
        timesteps = np.round(timesteps, 2)
        video_info['timesteps'] = timesteps
        video_info['num_frames'] = num_frames
    return labels



def average_predictions_on_timesteps(predictions_dict:dict):
    num_windows = len(predictions_dict["features"])
    array_predictions = []
    for window_idx in range(num_windows):
        timestep_start = predictions_dict["timestep_start"][window_idx]
        timestep_end = predictions_dict["timestep_end"][window_idx]
        frame_start = predictions_dict["frame_start"][window_idx]
        frame_end = predictions_dict["frame_end"][window_idx]
        predictions = predictions_dict["predicts"][window_idx]
        num_predictions = predictions.shape[0]
        # generate timesteps for predictions
        timesteps = np.linspace(timestep_start, timestep_end, num_predictions)
        num_frames = np.linspace(frame_start, frame_end, num_predictions)
        timesteps, num_frames = np.round(timesteps, 2), np.round(num_frames, 0)
        array_predictions.append((num_frames, timesteps, predictions))
    # average predictions on timesteps
    timesteps_with_predictions = [(item[1], item[2]) for item in array_predictions]
    averaged_timesteps, averaged_predictions = __average_predictions_on_timesteps(timesteps_with_predictions)
    return averaged_timesteps, averaged_predictions



def process_dict(predictions_dict:dict, labels_dict):
    labels_columns = ["category"]
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



def get_best_entropy_threshold(audio, video, label_values, alpha):
    weights = [alpha, 1-alpha]
    entropy_thresholds = np.arange(0.05, 2.05, 0.05)
    entropy_thresholds = [1.8] # !!!WARNING!!! Manually found using the distribution of the entropy
    # calculate entropy row-wise for audio
    audio_entropy = entropy(audio, axis=1)
    """import matplotlib.pyplot as plt
    audio_tmp_wrong = np.argmax(audio, axis=-1)
    audio_tmp_wrong = audio_tmp_wrong != label_values
    audio_tmp_right = np.argmax(audio, axis=-1)
    audio_tmp_right = audio_tmp_right == label_values
    # create histogram with two colors
    plt.hist(audio_entropy[audio_tmp_wrong], bins="auto", alpha=0.5, label='wrong')
    plt.hist(audio_entropy[audio_tmp_right], bins="auto", alpha=0.5, label='right')
    plt.legend(loc='upper right')
    plt.show()"""
    #
    best_f1 = 0
    best_threshold = 10000000000000
    # the best threshold is the one that maximizes the f1 score
    for threshold in entropy_thresholds:
        audio_mask = audio_entropy > threshold
        current_audio = audio.copy()
        current_audio[audio_mask] = 0
        # calculate f1 score
        prediction = current_audio * weights[0] + video * weights[1]
        prediction = np.argmax(prediction, axis=-1)
        f1 = f1_score(label_values, prediction, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1







def main():
    path_to_audio_devel = "/home/ddresvya/Data/features/expr_devel.pickle"
    path_to_video = "/home/ddresvya/Data/features/dynamic_features_facial_exp.pkl"
    # load pickle files
    with open(path_to_audio_devel, 'rb') as f:
        audio = pickle.load(f)
        audio = {k.split(".")[0]: audio[k] for k in audio.keys()}
    with open(path_to_video, 'rb') as f:
        video = pickle.load(f)
    # in video, get only keys that are in audio_devel
    video = {k: video[k] for k in audio.keys()}
    # load fps file and labels
    path_to_fps = "src/video/training/dynamic_models/fps.pkl"
    path_to_labels = "/home/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/"
    with open(os.path.join(path_to_project, path_to_fps), 'rb') as f:
        fps = pickle.load(f)
        fps = {k.split(".")[0]: fps[k] for k in fps.keys()}
    # load labels
    labels = load_labels(path_to_labels, "Exp", fps)
    # transform both audio and video to the same format
    audio = process_dict(audio, labels)
    video = process_dict(video, labels)
    # combine predictions of keys of audio and video
    audio_preds = []
    video_preds = []
    labels_values = []
    for video_name in video.keys():
        audio_preds.append(audio[video_name]["predictions"])
        video_preds.append(video[video_name]["predictions"])
        labels_values.append(labels[video_name]["category"])
    audio_preds = np.concatenate(audio_preds, axis=0)
    video_preds = np.concatenate(video_preds, axis=0)
    labels_values = np.concatenate(labels_values)
    # filter out labels that equal -1
    mask = labels_values != -1
    audio_preds = audio_preds[mask]
    video_preds = video_preds[mask]
    labels_values = labels_values[mask]
    # take softmax of video predictions
    video_preds = softmax(video_preds, axis=-1)

    # get best threshold
    best_threshold = 10000000000000
    best_f1 = 0
    best_alpha = 0
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        threshold, f1 = get_best_entropy_threshold(audio_preds, video_preds, labels_values, alpha)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_alpha = alpha
    print("Best threshold: ", best_threshold)
    print("Best alpha: ", best_alpha)
    print("Best f1: ", best_f1)

    # TODO: Dirichlet distribution weights
    # save best weights
    # generate test predictions (submission v3)




if __name__ == "__main__":
    main()