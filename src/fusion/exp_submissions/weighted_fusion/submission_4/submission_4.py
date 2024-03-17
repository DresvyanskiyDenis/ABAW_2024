import os
import pickle
import sys
from functools import partial

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

from fusion.exp_submissions.weighted_fusion.submission_2.submission_2 import load_test_sample_file_and_preprocess, load_labels, \
    process_dict
from video.post_processing.embeddings_extraction_dynamic import load_fps_file

path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
sys.path.append(path_to_project)
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "datatools"))
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "simple-HRNet-master"))


def get_best_fusion_rf_model(audio, video, statistical, label_values):
    features = np.concatenate([audio, video, statistical], axis=1)
    n_estimators = [10, 50, 100,]
    criterion = ["gini", "entropy"]
    best_f1 = 0
    best_model = None
    best_oob = -10000000
    for n in n_estimators:
        for c in criterion:
            print(f"n_estimators: {n}, criterion: {c}")
            rf = RandomForestClassifier(n_estimators=n, criterion=c, class_weight="balanced",
                                        oob_score=partial(f1_score, average="macro"),
                                        n_jobs=-1)
            rf = rf.fit(features, label_values)
            f1 = f1_score(label_values, rf.predict(features), average="macro")
            print(f"F1 score: {f1}")
            print(f"OOB score: {rf.oob_score_}")
            if rf.oob_score_ > best_f1:
                best_f1 = rf.oob_score_
                best_model = rf
    return best_model, best_f1







def generate_test_predictions(audio, video, statistical, path_to_test_sample, fusion_model,
                              output_path):
    labels_columns = ["category"]
    sample_file = pd.read_csv(path_to_test_sample)
    sample_file["video_name"] = sample_file["image_location"].apply(lambda x: x.split("/")[0])
    sample_file.drop(columns=["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"],
                     inplace=True)
    sample_file["category"] = np.NaN
    for video_name in video.keys():
        audio_predictions = audio[video_name]["predictions"]
        video_predictions = video[video_name]["predictions"]
        staatistical_predictions = statistical[video_name]["predictions"]
        video_predictions = softmax(video_predictions, axis=-1)

        # duplicate last predictions for audio and video predictions if there are less predictions than in sample_file
        if audio_predictions.shape[0] < sample_file.loc[sample_file["video_name"]==video_name, labels_columns].shape[0]:
            difference = sample_file.loc[sample_file["video_name"] == video_name, labels_columns].shape[0] - \
                         audio_predictions.shape[0]
            audio_predictions = np.concatenate([audio_predictions,
                                                np.repeat(np.array(audio_predictions[-1]).reshape((1,-1)), difference, axis=0)], axis=0)
        if video_predictions.shape[0] < sample_file.loc[sample_file["video_name"] == video_name, labels_columns].shape[0]:
            difference = sample_file.loc[sample_file["video_name"] == video_name, labels_columns].shape[0] - \
                         video_predictions.shape[0]
            video_predictions = np.concatenate([video_predictions,
                                                np.repeat(np.array(video_predictions[-1]).reshape((1,-1)), difference,
                                                          axis=0)], axis=0)
        if staatistical_predictions.shape[0] < sample_file.loc[sample_file["video_name"] == video_name, labels_columns].shape[0]:
            difference = sample_file.loc[sample_file["video_name"] == video_name, labels_columns].shape[0] - \
                         staatistical_predictions.shape[0]
            staatistical_predictions = np.concatenate([staatistical_predictions,
                                                np.repeat(np.array(staatistical_predictions[-1]).reshape((1,-1)), difference,
                                                          axis=0)], axis=0)

        assert sample_file.loc[sample_file["video_name"]==video_name, labels_columns].shape[0] == audio_predictions.shape[0]
        assert sample_file.loc[sample_file["video_name"] == video_name, labels_columns].shape[0] == video_predictions.shape[0]
        assert sample_file.loc[sample_file["video_name"] == video_name, labels_columns].shape[0] == staatistical_predictions.shape[0]
        # generate predictions by filtering out high entropy and summing up with weights
        predictions = fusion_model.predict_proba(np.concatenate([audio_predictions, video_predictions, staatistical_predictions], axis=1))
        predictions = np.argmax(predictions, axis=-1)
        sample_file.loc[sample_file["video_name"]==video_name, "category"] = predictions
    # check on NaN values
    assert sample_file[labels_columns].isna().sum().sum() == 0
    sample_file.drop(columns=["video_name"], inplace=True)
    sample_file["category"] = sample_file["category"].astype(int)
    sample_file.columns = ["image_location", "Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other"]
    sample_file.to_csv(output_path, index=False)


def main():
    path_to_audio_devel = "/home/ddresvya/Data/features/expr_devel.pickle"
    path_to_video = "/home/ddresvya/Data/features/dynamic_features_facial_exp.pkl"
    path_statistical_devel = "/home/ddresvya/Data/features/test_predictions_dynamic/Exp_dev_statistical/Expr_dev_ViT_mean_max_min_wo_scale_both.pkl"
    path_to_audio_test = "/home/ddresvya/Data/features/test_predictions_dynamic/exp_audio_test_predictions/expr_test.pickle"
    path_to_video_test = "/home/ddresvya/Data/features/test_predictions_dynamic/dynamic_features_facial_exp_test.pkl"
    path_statistical_test = "/home/ddresvya/Data/features/test_predictions_dynamic/Exp_dev_statistical/Expr_test_ViT_mean_max_min_wo_scale_both.pkl"
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
    with open(path_statistical_devel, 'rb') as f:
        statistical = pickle.load(f)
        statistical = {str(k): statistical[k] for k in statistical.keys()}
        statistical = {k.split(".")[0]: statistical[k] for k in statistical.keys()}
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
    statistical = process_dict(statistical, labels)
    # combine predictions of keys of audio and video
    audio_preds = []
    video_preds = []
    statistical_preds = []
    labels_values = []
    for video_name in video.keys():
        audio_preds.append(audio[video_name]["predictions"])
        video_preds.append(video[video_name]["predictions"])
        statistical_preds.append(statistical[video_name]["predictions"])
        labels_values.append(labels[video_name]["category"])
    audio_preds = np.concatenate(audio_preds, axis=0)
    video_preds = np.concatenate(video_preds, axis=0)
    statistical_preds = np.concatenate(statistical_preds, axis=0)
    labels_values = np.concatenate(labels_values)
    # filter out labels that equal -1
    mask = labels_values != -1
    audio_preds = audio_preds[mask]
    video_preds = video_preds[mask]
    statistical_preds = statistical_preds[mask]
    labels_values = labels_values[mask]
    # take softmax of video predictions
    video_preds = softmax(video_preds, axis=-1)

    # find the best weights for the fusion by sampling the Dirichlet distribution
    best_rf_model, best_f1 = get_best_fusion_rf_model(audio_preds, video_preds, statistical_preds, labels_values)
    print("Best f1 score after fusion with weights generated by Dirichlet distribution: ", best_f1)
    print("Best model: ", best_rf_model)

    # load test pickle files
    with open(path_to_audio_test, 'rb') as f:
        audio_test = pickle.load(f)
        audio_test = {k.split(".")[0]: audio_test[k] for k in audio_test.keys()}
    with open(path_to_video_test, 'rb') as f:
        video_test = pickle.load(f)
    with open(path_statistical_test, 'rb') as f:
        statistical_test = pickle.load(f)
        statistical_test = {str(k): statistical_test[k] for k in statistical_test.keys()}
        statistical_test = {k.split(".")[0]: statistical_test[k] for k in statistical_test.keys()}
    # prepare test predictions
    audio_test = process_dict(audio_test, test_sample)
    video_test = video_test
    statistical_test = process_dict(statistical_test, test_sample)
    generate_test_predictions(audio_test, video_test, statistical_test, path_to_sample_file, best_rf_model,
                              "/home/ddresvya/Data/test_set/Exp/submission_4/submission_4.csv")
    # save best weights and threshold
    with open("/home/ddresvya/Data/test_set/Exp/submission_4/best_rf_model.pickle", "wb") as f:
        pickle.dump(best_rf_model, f)





if __name__ == "__main__":
    main()