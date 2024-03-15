import glob
import os
import sys
from typing import List, Optional

from sklearn.preprocessing import MinMaxScaler, StandardScaler

path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
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


def process_all_videos_static_test(config, videos:List[str]):
    # initialize face detector and pose detector
    face_detector = __initialize_face_detector()
    pose_detector = __initialize_pose_detector()
    # initialize static feature extractor
    facial_feature_extractor, facial_preprocessing_functions = __initialize_static_feature_extractor(
        static_model_type=config['static_model_type'],
        path_to_weights=config['path_to_static_weights'],
        num_classes=config['num_classes'],
        num_regression_neurons=config['num_regression_neurons'],
        path_hrnet_weights=config['path_hrnet_weights'],
        challenge=config["challenge"]
    )
    pose_feature_extractor, pose_preprocessing_functions = __initialize_static_feature_extractor(
        static_model_type=config['pose_model_type'],
        path_to_weights=config['path_to_pose_weights'],
        num_classes=config['num_classes'],
        num_regression_neurons=config['num_regression_neurons'],
        path_hrnet_weights=config['path_hrnet_weights'],
        challenge=config["challenge"]
    )
    facial_feature_extractor = facial_feature_extractor.to(config["device"])
    pose_feature_extractor = pose_feature_extractor.to(config["device"])
    # get only videos that are in labels
    path_to_data = config["path_to_videos"]
    videos = [os.path.basename(video).split(".")[0] for video in videos]
    # go over videos
    for video in tqdm(videos):
        videofile_name = video+".mp4" if video+".mp4" in os.listdir(path_to_data) else video+".avi"
        if os.path.exists(os.path.join(config["output_static_features"], f"{os.path.basename(video)}.csv")):
            continue
        # process one video with static models
        metadata_static = process_one_video_static(path_to_video=os.path.join(path_to_data, videofile_name),
                                                   face_detector=face_detector, pose_detector=pose_detector,
                                                    facial_feature_extractor=(facial_feature_extractor, facial_preprocessing_functions),
                                                    pose_feature_extractor=(pose_feature_extractor, pose_preprocessing_functions),
                                                    device=config["device"])
        current_labels = pd.DataFrame([np.NaN]*len(metadata_static), columns=["category"])
        metadata_static = align_labels_with_metadata(metadata_static, current_labels, challenge=config["challenge"])
        # save extracted features
        metadata_static.to_csv(os.path.join(config["output_static_features"], f"{os.path.basename(video)}.csv"), index=False)





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
    # transform predictions if needed
    if challenge=="Exp":
        predictions_all = softmax(predictions_all, axis=-1)
        predictions_all = np.argmax(predictions_all, axis=-1)
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
    # TODO: TBD as I do not know now the final format (test set has not been reliased yet)
    dynamic_model = __initialize_dynamic_model(dynamic_model_type=dynamic_model_type, path_to_weights=path_to_weights,
                                               input_shape=input_shape, num_classes=num_classes,
                                               num_regression_neurons=num_regression_neurons, challenge=challenge)
    dynamic_model = dynamic_model.to(device)
    normalizer = MinMaxScaler() if normalization == "min_max" else StandardScaler()
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
    features = np.concatenate(
        [metadata_static[video][feature_columns].dropna().values for video in metadata_static.keys()], axis=0)
    normalizer = normalizer.fit(features)
    # process all videos
    result = {}
    for video in tqdm(metadata_static.keys()):
        df = metadata_static[video]
        # normalize features
        df.loc[:, feature_columns] = normalizer.transform(df[feature_columns].values)
        predictions = generate_test_predictions_one_video(df_video=df, challenge=challenge, window_size=window_size,
                                                            stride=stride, model=dynamic_model, feature_columns=feature_columns,
                                                            device=device, original_fps=video_to_fps[video], needed_fps=5,
                                                            batch_size=batch_size)
        result[video] = predictions
    # save predictions to the
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for video in result.keys():
        path_to_save = os.path.join(output_path, f"{video}.csv")
        df_to_write = pd.DataFrame(np.concatenate([item.reshape(-1,1) for item in result[video]], axis=1),
                                   columns=["frame_num", "timestep"] +
                                           ["category"] if challenge=="Exp" else ["valence", "arousal"])
        if challenge == "Exp":
            df_to_write = df_to_write["category"].astype("int32")
            # write first row as string "Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other" just using some writer
            with open(path_to_save, 'w') as f:
                f.write("Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n")
        elif challenge == "VA":
            df_to_write = df_to_write[["valence", "arousal"]]
            # write first row as string "valence,arousal" just using some writer
            with open(path_to_save, 'w') as f:
                f.write("valence,arousal\n")

        df_to_write.to_csv(path_to_save, mode='a', index=False, header=False)








def main():
    # static Exp
    config_static_exp = {
        "static_model_type": "ViT_b_16",
        "pose_model_type": "HRNet",
        "path_to_static_weights": "/home/ddresvya/Data/weights_best_models/fine_tuned/Exp_challenge/AffWild2_static_exp_best_ViT.pth",
        "path_to_pose_weights": "/home/ddresvya/Data/weights_best_models/fine_tuned/Exp_challenge/AffWild2_static_exp_best_HRNet.pth",
        "path_hrnet_weights": "/home/ddresvya/PhD/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
        "output_static_features": "/home/ddresvya/Data/features/Exp_test/",
        "num_classes": 8,
        "num_regression_neurons": None,
        "device": torch.device("cuda"),
        "batch_size": 32,
        "challenge": "Exp",
        "path_to_videos": "/home/ddresvya/Data/ABAW/"
    }
    # read test filenames from txt
    videos_test_exp = list(np.loadtxt("/home/ddresvya/Data/test_set/names_of_videos_in_each_test_set/Expression_Recognition_Challenge_test_set_release.txt", dtype=str).flatten())
    # create output folder if not exists
    if not os.path.exists(config_static_exp["output_static_features"]):
        os.makedirs(config_static_exp["output_static_features"])
    process_all_videos_static_test(config=config_static_exp, videos=videos_test_exp)


    # static VA
    """config_static_VA = {
        "static_model_type": "EfficientNet-B1",
        "pose_model_type": "HRNet",
        "path_to_static_weights": "/home/ddresvya/Data/weights_best_models/fine_tuned/VA_challenge/AffWild2_static_va_best_b1.pth",
        "path_to_pose_weights": "/home/ddresvya/Data/weights_best_models/fine_tuned/VA_challenge/AffWild2_static_va_best_HRNet.pth",
        "path_hrnet_weights": "/home/ddresvya/PhD/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
        "output_static_features": "/nfs/scratch/ddresvya/Data/features/VA_test/",
        "num_classes": None,
        "num_regression_neurons": 2,
        "device": torch.device("cuda"),
        "challenge": "VA",
        "batch_size": 32,
    }
    videos_test_VA = glob.glob(None) # TODO: add path to videos
    process_all_videos_static_test(config=config_static_VA, videos=videos_test_VA)"""


    """# make dynamic predictions
    config_dynamic_face_uni_modal_exp = {
        "dynamic_model_type": "dynamic_v3",
        "embeddings_columns": [f"facial_embedding_{i}" for i in range(256)],
        "normalization": "min_max",
        "path_to_weights": "/home/ddresvya/Data/weights_best_models/fine_tuned_dynamic/uni_modal_face_best.pth",
        "input_shape": (20, 256),
        "num_classes": 8,
        "num_regression_neurons": None,
        "device": torch.device("cuda"),
        "window_size": 20,
        "stride": 10,
        "batch_size": 32,
        "challenge": "Exp",
        'video_to_fps': load_fps_file(os.path.join(path_to_project, "src/video/training/dynamic_models/fps.pkl")),
        'output_path': None, # TODO: add path to the output
        'path_to_extracted_features': None # TODO: add path to the extracted features
    }
    generate_test_predictions_all_videos(**config_dynamic_face_uni_modal_exp)"""



if __name__ == "__main__":
    main()