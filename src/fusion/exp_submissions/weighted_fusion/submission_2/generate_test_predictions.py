import glob
import os
import sys
from typing import List, Optional

from sklearn.preprocessing import MinMaxScaler, StandardScaler

path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir,os.path.pardir,os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
sys.path.append(path_to_project)
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "datatools"))
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "simple-HRNet-master"))



import pandas as pd
import numpy as np
import pickle

from scipy.special import softmax
from tqdm import tqdm
import torch

from src.evaluation_dynamic import __average_predictions_on_timesteps, __interpolate_to_100_fps, \
    __synchronize_predictions_with_ground_truth, __apply_hamming_smoothing
from src.video.post_processing.embeddings_extraction_dynamic import __initialize_static_feature_extractor, \
    __initialize_face_detector, __initialize_pose_detector, process_one_video_static, align_labels_with_metadata, \
    __cut_video_on_windows, load_fps_file, __initialize_dynamic_model



def generate_test_predictions_one_video(df_video:pd.DataFrame, challenge, window_size,  stride, model,
                                        feature_columns, device, original_fps, needed_fps,
                                        batch_size:Optional[int]=32):
    # take avery n frame depending on the original_fps and needed_fps
    needed_num_predictions = df_video.shape[0]
    every_n_frame = int(original_fps / needed_fps)
    df_video_resampled = df_video.iloc[::every_n_frame]
    # cut on windows
    windows = __cut_video_on_windows(df_video_resampled, window_size=window_size, stride=stride)
    predictions_all = []
    pred_timesteps_all = []
    for window_idx in range(0, len(windows), batch_size):
        # extract the batch of windows
        batch_windows = windows[window_idx:window_idx + batch_size]
        timesteps = np.stack([window['timestep'].values for window in batch_windows])
        # extract features from the batch
        batch_windows = [torch.from_numpy(window[feature_columns].values) for window in batch_windows]
        batch_windows = torch.stack(batch_windows).float().to(device)
        # get predictions
        _ , batch_predictions = model(batch_windows)
        batch_predictions = batch_predictions.detach().cpu().numpy()
        # append everythign to the corresponding lists
        predictions_all.append(batch_predictions)
        pred_timesteps_all.append(timesteps)
    # average predictions on timesteps (however, we need to prepare them to the format List[Tuple[np.ndarray, np.ndarray]])
    # now they are List[np.ndarray]
    preds_with_timesteps = list(zip(pred_timesteps_all, predictions_all))
    pred_timesteps_all, predictions_all = __average_predictions_on_timesteps(preds_with_timesteps)
    # select the frame_nums depending on the prediction_timesteps
    pred_frame_nums = df_video.loc[df_video['timestep'].isin(pred_timesteps_all)]['frame_num'].values
    # round to two decimal places timesteps and frame_nums
    pred_timesteps_all = np.round(pred_timesteps_all, 2)
    pred_frame_nums = np.round(pred_frame_nums, 2)
    # before interpolation, add the last timestep of the df_video to the pred_timesteps_all (duplicating the prediction)
    # as it can be that the last timestep has been removed due to the resampling
    if pred_timesteps_all[-1] != df_video['timestep'].values[-1]:
        pred_timesteps_all = np.append(pred_timesteps_all, np.array([df_video['timestep'].values[-1]]), axis=0)
        predictions_all = np.append(predictions_all, [predictions_all[-1]], axis=0)
    # interpolate predictions to the 100 fps
    predictions_all, pred_timesteps_all = __interpolate_to_100_fps(predictions_all, pred_timesteps_all)
    # get "ground truth" timesteps and "labels" (we need them for function, but since there are no labels, we fill it with NaN)
    original_timesteps = np.round(df_video['timestep'].values, 2)
    original_labels = np.zeros((len(original_timesteps), predictions_all.shape[-1])) * np.NaN
    # synchronize predictions with the original timesteps
    predictions_all, pred_timesteps_all, original_labels, original_timesteps = \
        __synchronize_predictions_with_ground_truth(predictions_all, pred_timesteps_all, original_labels,
                                                    original_timesteps)
    # apply hamming window
    predictions_all = __apply_hamming_smoothing(predictions_all, smoothing_window_size=original_fps//2)
    # check all parameters that we have the same number of frames
    assert predictions_all.shape[0] == df_video.shape[0]

    # take the frame nums corresponding to timesteps
    pred_frame_nums = df_video.loc[df_video['timestep'].isin(pred_timesteps_all)]['frame_num'].values

    return pred_frame_nums, pred_timesteps_all, predictions_all







def generate_test_predictions_all_videos(dynamic_model_type, path_to_weights, normalization, embeddings_columns,
                               input_shape, num_classes, num_regression_neurons, video_to_fps,
                               challenge, path_to_extracted_features:str, window_size:int, stride:int, device:torch.device,
                               output_path:str,
                               batch_size:int=32):
    dynamic_model = __initialize_dynamic_model(dynamic_model_type=dynamic_model_type, path_to_weights=path_to_weights,
                                               input_shape=input_shape, num_classes=num_classes,
                                               num_regression_neurons=num_regression_neurons, challenge=challenge)
    dynamic_model = dynamic_model.to(device)
    if normalization == "min_max":
        normalizer = MinMaxScaler()
    elif normalization == "standard":
        normalizer = StandardScaler()
    else:
        normalizer = None
    # load metadata
    metadata_static = glob.glob(os.path.join(path_to_extracted_features, "*.csv"))
    metadata_static = {os.path.basename(file).split(".")[0]: pd.read_csv(file) for file in metadata_static}
    # assign column names
    columns = (['video_name', 'frame_num', 'timestep'] + [f"facial_embedding_{i}" for i in range(256)] +
               [f"pose_embedding_{i}" for i in range(256)])
    if challenge == "Exp":
        columns = columns + ["category"]
    else:
        columns = columns + ["valence", "arousal"]
    # assign column names to every dataframe
    for video in metadata_static.keys():
        metadata_static[video].columns = columns
    # fit normalizer
    # concatenate embeddings columns if tuple
    feature_columns = embeddings_columns if not isinstance(embeddings_columns, tuple) else embeddings_columns[0] + \
                                                                                           embeddings_columns[1]
    if normalizer is not None:
        features = np.concatenate(
            [metadata_static[video][feature_columns].dropna().values for video in metadata_static.keys()], axis=0)
        normalizer = normalizer.fit(features)
    # process all videos
    result = {}
    for video in tqdm(metadata_static.keys()):
        df = metadata_static[video]
        # normalize features
        if normalization in ['per-video-minmax', 'per-video-standard']:
            normalizer = MinMaxScaler() if normalization == "per-video-minmax" else StandardScaler()
            normalizer = normalizer.fit(df[feature_columns].values)
        df.loc[:, feature_columns] = normalizer.transform(df[feature_columns].values)
        predictions = generate_test_predictions_one_video(df_video=df, challenge=challenge, window_size=window_size,
                                                            stride=stride, model=dynamic_model, feature_columns=feature_columns,
                                                            device=device, original_fps=video_to_fps[video], needed_fps=5,
                                                            batch_size=batch_size)
        result[video] = {"frame_nums": predictions[0], "timesteps": predictions[1], "predictions": predictions[2]}
    # save predictions to the
    with open(output_path, 'wb') as file:
        pickle.dump(result, file)










def main():
    # make dynamic predictions
    config_dynamic_face_uni_modal_exp = {
        "dynamic_model_type": "dynamic_v3",
        "embeddings_columns": [f"facial_embedding_{i}" for i in range(256)],
        "normalization": "min_max",
        "path_to_weights": "/Data/weights_best_models/fine_tuned_dynamic/uni_modal_face_best.pth",
        "input_shape": (20, 256),
        "num_classes": 8,
        "num_regression_neurons": None,
        "device": torch.device("cuda"),
        "window_size": 20,
        "stride": 10,
        "batch_size": 32,
        "challenge": "Exp",
        'video_to_fps': load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl")),
        'output_path': "/Data/features/dynamic_features_facial_exp_test.pkl",
        'path_to_extracted_features': "/Data/features/Exp_test/",
    }
    generate_test_predictions_all_videos(**config_dynamic_face_uni_modal_exp)

    """# valence window size 30
    config_dynamic_face_uni_modal_valence_30 = {
        "dynamic_model_type": "dynamic_v3",
        "embeddings_columns": [f"facial_embedding_{i}" for i in range(256)],
        "normalization": "per-video-minmax",
        "path_to_weights": "/Data/weights_best_models/fine_tuned_dynamic_VA/uni_modal_face_valence_best.pth",
        "input_shape": (30, 256),
        "num_classes": None,
        "num_regression_neurons": 2,
        "device": torch.device("cuda"),
        "window_size": 30,
        "stride": 15,
        "batch_size": 32,
        "challenge": "VA",
        'video_to_fps': load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl")),
        'output_path': "/Data/features/test_predictions_dynamic/dynamic_features_facial_valence_test_30.pkl",
        'path_to_extracted_features': "/Data/features/VA_test/",
    }
    generate_test_predictions_all_videos(**config_dynamic_face_uni_modal_valence_30)

    # arousal window size 30
    config_dynamic_face_uni_modal_arousal_30 = {
        "dynamic_model_type": "dynamic_v3",
        "embeddings_columns": [f"facial_embedding_{i}" for i in range(256)],
        "normalization": "per-video-minmax",
        "path_to_weights": "/Data/weights_best_models/fine_tuned_dynamic_VA/uni_modal_face_arousal_best.pth",
        "input_shape": (30, 256),
        "num_classes": None,
        "num_regression_neurons": 2,
        "device": torch.device("cuda"),
        "window_size": 30,
        "stride": 15,
        "batch_size": 32,
        "challenge": "VA",
        'video_to_fps': load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl")),
        'output_path': "/Data/features/test_predictions_dynamic/dynamic_features_facial_arousal_test_30.pkl",
        'path_to_extracted_features': "/Data/features/VA_test/",
    }
    generate_test_predictions_all_videos(**config_dynamic_face_uni_modal_arousal_30)

    # valence window size 20
    config_dynamic_face_uni_modal_valence_20 = {
        "dynamic_model_type": "dynamic_v3",
        "embeddings_columns": [f"facial_embedding_{i}" for i in range(256)],
        "normalization": "per-video-minmax",
        "path_to_weights": "/Data/weights_best_models/fine_tuned_dynamic_VA/uni_modal_face_best_valence_20.pth",
        "input_shape": (20, 256),
        "num_classes": None,
        "num_regression_neurons": 2,
        "device": torch.device("cuda"),
        "window_size": 20,
        "stride": 10,
        "batch_size": 32,
        "challenge": "VA",
        'video_to_fps': load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl")),
        'output_path': "/Data/features/test_predictions_dynamic/dynamic_features_facial_valence_test_20.pkl",
        'path_to_extracted_features': "/Data/features/VA_test/",
    }
    generate_test_predictions_all_videos(**config_dynamic_face_uni_modal_valence_20)

    # arousal window size 20
    config_dynamic_face_uni_modal_arousal_20 = {
        "dynamic_model_type": "dynamic_v3",
        "embeddings_columns": [f"facial_embedding_{i}" for i in range(256)],
        "normalization": "per-video-minmax",
        "path_to_weights": "/Data/weights_best_models/fine_tuned_dynamic_VA/uni_modal_face_best_arousal_20.pth",
        "input_shape": (20, 256),
        "num_classes": None,
        "num_regression_neurons": 2,
        "device": torch.device("cuda"),
        "window_size": 20,
        "stride": 10,
        "batch_size": 32,
        "challenge": "VA",
        'video_to_fps': load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl")),
        'output_path': "/Data/features/test_predictions_dynamic/dynamic_features_facial_arousal_test_20.pkl",
        'path_to_extracted_features': "/Data/features/VA_test/",
    }
    generate_test_predictions_all_videos(**config_dynamic_face_uni_modal_arousal_20)"""


    # generatopn of dev predictions
    # valence window size 30
    config_dynamic_face_uni_modal_valence_30 = {
        "dynamic_model_type": "dynamic_v3",
        "embeddings_columns": [f"facial_embedding_{i}" for i in range(256)],
        "normalization": "per-video-minmax",
        "path_to_weights": "/Data/weights_best_models/fine_tuned_dynamic_VA/uni_modal_face_valence_best.pth",
        "input_shape": (30, 256),
        "num_classes": None,
        "num_regression_neurons": 2,
        "device": torch.device("cuda"),
        "window_size": 30,
        "stride": 15,
        "batch_size": 32,
        "challenge": "VA",
        'video_to_fps': load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl")),
        'output_path': "/Data/features/test_predictions_dynamic/dynamic_features_facial_valence_train_dev_30.pkl",
        'path_to_extracted_features': "/Data/features/VA/",
    }
    generate_test_predictions_all_videos(**config_dynamic_face_uni_modal_valence_30)

    # arousal window size 30
    config_dynamic_face_uni_modal_arousal_30 = {
        "dynamic_model_type": "dynamic_v3",
        "embeddings_columns": [f"facial_embedding_{i}" for i in range(256)],
        "normalization": "per-video-minmax",
        "path_to_weights": "/Data/weights_best_models/fine_tuned_dynamic_VA/uni_modal_face_arousal_best.pth",
        "input_shape": (30, 256),
        "num_classes": None,
        "num_regression_neurons": 2,
        "device": torch.device("cuda"),
        "window_size": 30,
        "stride": 15,
        "batch_size": 32,
        "challenge": "VA",
        'video_to_fps': load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl")),
        'output_path': "/Data/features/test_predictions_dynamic/dynamic_features_facial_arousal_train_dev_30.pkl",
        'path_to_extracted_features': "/Data/features/VA/",
    }
    generate_test_predictions_all_videos(**config_dynamic_face_uni_modal_arousal_30)



if __name__ == "__main__":
    main()