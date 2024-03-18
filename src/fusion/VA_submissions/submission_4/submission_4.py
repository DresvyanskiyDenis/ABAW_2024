import pickle
from functools import partial

import pandas as pd
import numpy as np
import sys
import os

from sklearn.ensemble import RandomForestRegressor

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
        path_to_test_predictions_valence = "/home/ddresvya/Data/features/test_predictions_dynamic/VA/dynamic_features_facial_valence_train_dev_30.pkl"
        path_to_test_predictions_arousal = "/home/ddresvya/Data/features/test_predictions_dynamic/VA/dynamic_features_facial_arousal_train_dev_30.pkl"
    elif part == "test":
        path_to_test_predictions_valence = "/home/ddresvya/Data/features/test_predictions_dynamic/VA/dynamic_features_facial_valence_test_30.pkl"
        path_to_test_predictions_arousal = "/home/ddresvya/Data/features/test_predictions_dynamic/VA/dynamic_features_facial_arousal_test_30.pkl"


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
        path = "/home/ddresvya/Data/features/test_predictions_dynamic/VA/va_devel.pickle"
    elif part == "test":
        path = "/home/ddresvya/Data/features/test_predictions_dynamic/VA/va_test.pickle"
    with open(path, 'rb') as f:
        audio_predictions = pickle.load(f)
        audio_predictions = {str(key): audio_predictions[key] for key in audio_predictions.keys()}
        audio_predictions = {key.split(".")[0]: audio_predictions[key] for key in audio_predictions.keys()}
    if part == "dev":
        labels = load_labels(
            path_to_labels="/home/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
            challenge="VA", video_to_fps=video_to_fps)
    elif part == "test":
        labels = load_test_sample_file_and_preprocess(path_to_sample_file="/home/ddresvya/Data/test_set/prediction_files_format/CVPR_6th_ABAW_VA_test_set_sample.txt"
                                                       , video_to_fps=video_to_fps)
    audio_predictions = process_dict(audio_predictions, labels)
    return audio_predictions


def load_statistical_features(part, video_to_fps):
    if part == "dev":
        path = "/home/ddresvya/Data/features/test_predictions_dynamic/VA/VA_dev_efficientB1_mean_max_min_wo_scale_both.pkl"
    elif part == "test":
        path = "/home/ddresvya/Data/features/test_predictions_dynamic/VA/VA_test_efficientB1_mean_max_min_wo_scale_both.pkl"
    with open(path, 'rb') as f:
        predictions = pickle.load(f)
        predictions = {str(key): predictions[key] for key in predictions.keys()}
        predictions = {key.split(".")[0]: predictions[key] for key in predictions.keys()}
    if part == "dev":
        labels = load_labels(
            path_to_labels="/home/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
            challenge="VA", video_to_fps=video_to_fps)
    elif part == "test":
        labels = load_test_sample_file_and_preprocess(path_to_sample_file="/home/ddresvya/Data/test_set/prediction_files_format/CVPR_6th_ABAW_VA_test_set_sample.txt"
                                                       , video_to_fps=video_to_fps)
    predictions = process_dict(predictions, labels)
    return predictions




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


def get_best_fusion_rf_models(audio, video, statistical, label_values):

    features_valence = np.concatenate([audio[:, 0][..., np.newaxis], video[:, 0][..., np.newaxis],
                                       statistical[:, 0][..., np.newaxis]], axis=-1)
    features_arousal = np.concatenate([audio[:, 1][..., np.newaxis], video[:, 1][..., np.newaxis],
                                        statistical[:, 1][..., np.newaxis]], axis=-1)
    n_estimators = [10, 20, 50, 100]
    criterion = ["squared_error", "friedman_mse"]
    best_CCC = 0
    best_CCC_valence = 0
    best_CCC_arousal = 0
    best_model = None
    for n in n_estimators:
        for c in criterion:
            print(f"n_estimators: {n}, criterion: {c}")
            rf_v = RandomForestRegressor(n_estimators=n, criterion=c,
                                        oob_score=np_concordance_correlation_coefficient,
                                        n_jobs=-1)
            rf_a = RandomForestRegressor(n_estimators=n, criterion=c,
                                        oob_score=np_concordance_correlation_coefficient,
                                        n_jobs=-1)
            rf_v = rf_v.fit(features_valence, label_values[:, 0])
            rf_a = rf_a.fit(features_arousal, label_values[:, 1])
            print(f"OOB score valence: {rf_v.oob_score_}")
            print(f"OOB score arousal: {rf_a.oob_score_}")
            if (rf_v.oob_score_+rf_a.oob_score_)/2 > best_CCC:
                best_CCC = (rf_v.oob_score_+rf_a.oob_score_)/2
                best_CCC_valence = rf_v.oob_score_
                best_CCC_arousal = rf_a.oob_score_
                best_model = (rf_v, rf_a)
    return best_model, best_CCC, best_CCC_valence, best_CCC_arousal


def generate_test_predictions(audio, video, statistical, fusion_model, path_to_test_sample, output_path):
    labels_columns = ["valence", "arousal"]
    sample_file = pd.read_csv(path_to_test_sample)
    sample_file["video_name"] = sample_file["image_location"].apply(lambda x: x.split("/")[0])
    sample_file[labels_columns] = np.NaN
    for video_name in video.keys():
        video_preds = video[video_name]
        audio_preds = audio[video_name]["predictions"]
        statistical_preds = statistical[video_name]["predictions"]
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

        if statistical_preds.shape[0] < lbs.shape[0]:
            statistical_preds = np.concatenate([statistical_preds, np.repeat(statistical_preds[-1][np.newaxis, :],
                                                                 lbs.shape[0] - statistical_preds.shape[0], axis=0)], axis=0)
        elif statistical_preds.shape[0] > lbs.shape[0]:
            statistical_preds = statistical_preds[:lbs.shape[0]]

        assert video_preds.shape[0] == audio_preds.shape[0]
        assert video_preds.shape[0] == statistical_preds.shape[0]
        assert video_preds.shape[0] == lbs.shape[0]
        assert audio_preds.shape[0] == lbs.shape[0]
        assert statistical_preds.shape[0] == lbs.shape[0]

        fusion_model_valence, fusion_model_arousal = fusion_model
        valence_predictions = np.concatenate([audio_preds[:,0][...,np.newaxis],
                                            video_preds[:,0][...,np.newaxis],
                                            statistical_preds[:,0][...,np.newaxis]], axis=-1)
        arousal_predictions = np.concatenate([audio_preds[:,1][...,np.newaxis],
                                            video_preds[:,1][...,np.newaxis],
                                            statistical_preds[:,1][...,np.newaxis]], axis=-1)
        final_valence_prediction = fusion_model_valence.predict(valence_predictions)
        final_arousal_prediction = fusion_model_arousal.predict(arousal_predictions)
        final_prediction = np.concatenate([final_valence_prediction.reshape((-1,1)), final_arousal_prediction.reshape((-1,1))], axis=-1)
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
    statistical_dev = load_statistical_features("dev", video_to_fps)
    statistical_test = load_statistical_features("test", video_to_fps)
    # load labels
    labels = load_labels(
        path_to_labels="/home/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
        challenge="VA", video_to_fps=video_to_fps)
    # get keys in video that are only in statistical
    dev_keys = list(statistical_dev.keys())
    video_dev = {key: video_dev[key] for key in dev_keys}
    audio_dev = {key: audio_dev[key] for key in dev_keys}
    labels = {key: labels[key] for key in dev_keys}
    # concat everything
    audio_dev_all = []
    video_dev_all = []
    statistical_dev_all = []
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
        statistical_dev_all.append(statistical_dev[key]["predictions"])
        labels_all.append(labels[key][["valence", "arousal"]].values)
    audio_dev = np.concatenate(audio_dev_all, axis=0)
    video_dev = np.concatenate(video_dev_all, axis=0)
    statistical_dev = np.concatenate(statistical_dev_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    # some labels have -5, we need to filter them out
    mask = labels[:, 0] != -5
    audio_dev = audio_dev[mask]
    video_dev = video_dev[mask]
    statistical_dev = statistical_dev[mask]
    labels = labels[mask]
    mask = labels[:, 1] != -5
    audio_dev = audio_dev[mask]
    video_dev = video_dev[mask]
    statistical_dev = statistical_dev[mask]
    labels = labels[mask]
    # get best weights
    best_fusion_models, best_CCC, best_CCC_valence, best_CCC_arousal = get_best_fusion_rf_models(audio_dev, video_dev,
                                                                                                    statistical_dev, labels)
    print(f"Best models:", best_fusion_models)
    print(f"Best CCC: {best_CCC}")
    print(f"Best CCC valence: {best_CCC_valence}")
    print(f"Best CCC arousal: {best_CCC_arousal}")
    # generate predictions
    generate_test_predictions(audio_test, video_test, statistical_test, best_fusion_models,
                              "/home/ddresvya/Data/test_set/prediction_files_format/CVPR_6th_ABAW_VA_test_set_sample.txt",
                              "/home/ddresvya/Data/test_set/VA/submission_4/submission_4.csv")
    # save best models
    with open("/home/ddresvya/Data/test_set/VA/submission_4/best_fusion_model_valence.pkl", 'wb') as f:
        pickle.dump(best_fusion_models[0], f)
    with open("/home/ddresvya/Data/test_set/VA/submission_4/best_fusion_model_arousal.pkl", 'wb') as f:
        pickle.dump(best_fusion_models[1], f)





if __name__ == "__main__":
    main()