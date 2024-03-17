import os
import pickle
import sys

from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import f1_score

from video.post_processing.embeddings_extraction_dynamic import load_fps_file

path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
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

def filter_out_predictions_with_high_entropy(audio, threshold):
    audio_entropy = entropy(audio, axis=-1)
    audio_mask = audio_entropy > threshold
    current_audio = audio.copy()
    current_audio[audio_mask] = 0
    return current_audio




def get_best_fusion_weights(audio, video, label_values, num_generations=1000):
    # generate num_generations weights using Dirichlet distribution. We generate both for models and classses
    generated_weights = np.random.dirichlet((1,1), size=(num_generations,audio.shape[-1])).transpose(0,2,1)
    best_f1 = 0
    best_weights = None
    for weights in generated_weights:
        # calculate f1 score
        prediction = audio * weights[0][np.newaxis,...] + video * weights[1][np.newaxis,...]
        prediction = np.argmax(prediction, axis=-1)
        f1 = f1_score(label_values, prediction, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
    return best_weights, best_f1



def load_test_sample_file_and_preprocess(path_to_sample_file:str, challenge, video_to_fps):
    sample_file = pd.read_csv(path_to_sample_file)
    sample_file["video_name"] = sample_file["image_location"].apply(lambda x: x.split("/")[0])
    sample_file.drop(columns=["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"],
                     inplace=True)
    if challenge == "Exp":
        labels_columns = ["category"]
    else:
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
    path_to_audio_devel = "/home/ddresvya/Data/features/expr_devel.pickle"
    path_to_video = "/home/ddresvya/Data/features/dynamic_features_facial_exp.pkl"
    path_to_audio_test = "/home/ddresvya/Data/features/test_predictions_dynamic/exp_audio_test_predictions/expr_test.pickle"
    path_to_video_test = "/home/ddresvya/Data/features/test_predictions_dynamic/dynamic_features_facial_exp_test.pkl"
    video_to_fps = load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl"))
    path_to_sample_file = "/home/ddresvya/Data/test_set/prediction_files_format/CVPR_6th_ABAW_Expr_test_set_sample.txt"
    # load test sample file
    test_sample = load_test_sample_file_and_preprocess(path_to_sample_file, "Exp", video_to_fps)
    # load pickle files
    with open(path_to_audio_devel, 'rb') as f:
        audio = pickle.load(f)
        audio = {k.split(".")[0]: audio[k] for k in audio.keys()}
    with open(path_to_video, 'rb') as f:
        video = pickle.load(f)
    # load test pickle files
    with open(path_to_audio_test, 'rb') as f:
        audio_test = pickle.load(f)
        audio_test = {k.split(".")[0]: audio_test[k] for k in audio_test.keys()}
    with open(path_to_video_test, 'rb') as f:
        video_test = pickle.load(f)
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
    audio_test = process_dict(audio_test, test_sample)
    video_test = process_dict(video_test, test_sample)
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

    # filter out predictions with high entropy
    audio_preds = filter_out_predictions_with_high_entropy(audio_preds, best_threshold)
    # find the best weights for the fusion by sampling the Dirichlet distribution
    best_weights, best_f1 = get_best_fusion_weights(audio_preds, video_preds, labels_values)
    print("Best f1 score after fusion with weights generated by Dirichlet distribution: ", best_f1)
    print("Best weights: ", best_weights)

    # prepare test predictions





if __name__ == "__main__":
    main()