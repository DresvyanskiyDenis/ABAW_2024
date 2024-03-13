import gc
import glob
from functools import partial
from typing import Tuple, Optional, List, Callable, Dict
import sys
import torch
import os
import pickle



path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
sys.path.append(path_to_project)
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "datatools"))
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "simple-HRNet-master"))


import cv2
import numpy as np
import pandas as pd


import torchvision.transforms as T
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from SimpleHRNet import SimpleHRNet
from pytorch_utils.data_preprocessing import convert_image_to_float_and_scale
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B4, Modified_EfficientNet_B1, Modified_ViT_B_16
from pytorch_utils.models.Pose_estimation.HRNet import Modified_HRNet
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor, \
    ViT_image_preprocessor
from src.video.preprocessing.face_extraction_utils import recognize_faces_bboxes, get_bbox_closest_to_previous_bbox, \
    get_most_confident_person, extract_face_according_bbox, load_and_prepare_detector_retinaFace_mobileNet
from src.video.preprocessing.labels_preprocessing import load_train_dev_AffWild2_labels_with_frame_paths, \
    load_AffWild2_labels
from src.video.preprocessing.pose_extraction_utils import get_bboxes_for_frame, apply_bbox_to_frame
from src.video.training.dynamic_fusion.models import VisualFusionModel_v1, VisualFusionModel_v2
from src.video.training.dynamic_models.dynamic_models import UniModalTemporalModel_v1, UniModalTemporalModel_v2, \
    UniModalTemporalModel_v3, UniModalTemporalModel_v4, UniModalTemporalModel_v5, UniModalTemporalModel_v6_1_fps, \
    UniModalTemporalModel_v7_1_fps, UniModalTemporalModel_v8_1_fps




def load_fps_file(path_to_fps_file:str)->Dict[str, float]:
    """ Loads the video_to_fps file from the provided path.

    :param path_to_fps_file: str
        Path to the video_to_fps file.
    :return: Dict[str, float]
        Dictionary with video names as keys and fps as values.
    """
    with open(path_to_fps_file, 'rb') as f:
        fps_dict = pickle.load(f)
        fps_dict = {''.join(key.split('.')[:-1]): value for key, value in fps_dict.items()}
    return fps_dict

class hook_model(nn.Module):
    def __init__(self, model, hook_layer, challenge:str):
        super(hook_model, self).__init__()
        self.model = model
        self.hook = hook_layer
        self.features = None
        self.challenge = challenge
        def get_features():
            def hook(model, input, output):
                self.features = output.detach()

            return hook

        self.hook.register_forward_hook(get_features())
    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            output = self.model(*x)
        else:
            output = self.model(x)
        if isinstance(output, tuple) or isinstance(output, list):
            if len(output) == 1:
                output = output[0]
            elif self.challenge == "Exp":
                output = output[0].detach()
            elif self.challenge == "VA":
                output = output[1].detach()
        return self.features, output


def __cut_video_on_windows(video:pd.DataFrame, window_size:int, stride:int)->List[pd.DataFrame]:
    """ Cuts the video on windows with specified window size and stride.

    :param video: pd.DataFrame
        The dataframe with corresponding frames of the video (represented as paths to the frames)
        It has columns ['path', 'frame_num', 'timestep', ...]
    :param window_size: int
        Size of the window. Given in number of frames.
    :param stride: int
        Stride of the window. Given in number of frames.
    :return: List[pd.DataFrame]
        List of dataframes with windows. Each dataframe has the same columns as the input dataframe.
    """
    if len(video) <= window_size:
        # pad it with zeros at the start
        zeros = pd.DataFrame(np.zeros((window_size - len(video), len(video.columns))), columns=video.columns)
        video = pd.concat([zeros, video], axis=0)
        return [video]
    # create list to store the windows
    windows = []
    # calculate the common frame_num_difference and the reference range of the window
    frame_num_difference = video['frame_num'].diff().round(2).mode().values[0]
    reference_range = np.round(frame_num_difference * (window_size-1), 2)
    # go over the video and cut it on windows. We include only windows with the same range as the reference range
    # (thus, only windows with the monotonically increasing timesteps are included in the result list)
    for i in range(0, len(video) - window_size, stride):
        window = video.iloc[i:i + window_size]
        actual_range = np.round(window["frame_num"].iloc[-1] - window["frame_num"].iloc[0], 2)
        if actual_range == reference_range:
            windows.append(window)
    # sometimes, there are no windows at all (because they are ultra short and most of the labels are missing)
    # then, we will just take the last window
    if len(windows) == 0:
        windows.append(video.iloc[-window_size:])
    # most of the times, the last window is not full, so we will replace it with the window that starts from -window_size
    windows[-1] = video.iloc[-window_size:]
    return windows

def __initialize_face_detector():
    detector = load_and_prepare_detector_retinaFace_mobileNet()
    return detector

def __initialize_pose_detector():
    detector = SimpleHRNet(c=48, nof_joints=17, multiperson=True,
                yolo_version='v3',
                yolo_model_def=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/",
                                            "models_/detectors/yolo/config/yolov3.cfg"),
                yolo_class_path=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/",
                                             "models_/detectors/yolo/data/coco.names"),
                yolo_weights_path=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/",
                                               "models_/detectors/yolo/weights/yolov3.weights"),
                checkpoint_path=r"/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w48_384x288.pth",
                return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1, device=torch.device("cuda"))
    return detector





def __initialize_static_feature_extractor(challenge, static_model_type, path_to_weights, num_classes, num_regression_neurons,
                                          path_hrnet_weights:Optional[str]=None):
    if static_model_type=='EfficientNet-B4':
        model = Modified_EfficientNet_B4(embeddings_layer_neurons=256, num_classes=num_classes,
                                         num_regression_neurons=num_regression_neurons)
        # cut off last two layers
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=380),
                                   EfficientNet_image_preprocessor()]
    elif static_model_type=='EfficientNet-B1':
        model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=num_classes,
                                         num_regression_neurons=num_regression_neurons)
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    elif static_model_type=='ViT_b_16':
        model = Modified_ViT_B_16(embeddings_layer_neurons=256, num_classes=num_classes,
                                  num_regression_neurons=2)
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=224),
                                   ViT_image_preprocessor()]

    elif static_model_type=='HRNet':
        model = Modified_HRNet(pretrained=True,
                               path_to_weights=path_hrnet_weights,
                               embeddings_layer_neurons=256, num_classes=num_classes,
                               num_regression_neurons=num_regression_neurons,
                               consider_only_upper_body=True)
        # define preprocessing functions
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=256),
                                   convert_image_to_float_and_scale,
                                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]  # From HRNet
        #model.classifier = nn.Identity()
    # load weights
    model.load_state_dict(torch.load(path_to_weights))
    # make hook to get embeddings
    hook_layer = model.activation_embeddings
    model = hook_model(model, hook_layer, challenge=challenge)
    # freeze model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model, preprocessing_functions


def __initialize_dynamic_model(dynamic_model_type, path_to_weights, input_shape, num_classes, num_regression_neurons,
                               challenge):
    if dynamic_model_type == "dynamic_v1":
        model = UniModalTemporalModel_v1(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif dynamic_model_type == "dynamic_v2":
        model = UniModalTemporalModel_v2(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif dynamic_model_type == "dynamic_v3":
        model = UniModalTemporalModel_v3(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif dynamic_model_type == "dynamic_v4":
        model = UniModalTemporalModel_v4(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif dynamic_model_type == "dynamic_v5":
        model = UniModalTemporalModel_v5(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif dynamic_model_type == "dynamic_v6_1_fps":
        model = UniModalTemporalModel_v6_1_fps(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif dynamic_model_type == "dynamic_v7_1_fps":
        model = UniModalTemporalModel_v7_1_fps(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif dynamic_model_type == "dynamic_v8_1_fps":
        model = UniModalTemporalModel_v8_1_fps(input_shape=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif dynamic_model_type == "fusion_v1":
        model = VisualFusionModel_v1(input_size=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    elif dynamic_model_type == "fusion_v2":
        model = VisualFusionModel_v2(input_size=input_shape, num_classes=num_classes, num_regression_neurons=num_regression_neurons)
    else:
        raise ValueError(f"Unknown dynamic model type: {dynamic_model_type}")
    # load weights
    model.load_state_dict(torch.load(path_to_weights))
    # make hook to get
    hook_layer = list(model.named_children())[-2][1]
    model = hook_model(model, hook_layer, challenge=challenge)
    # freeze model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def __recognize_face(frame, face_detector, previous_face, previous_bbox)->Tuple[np.ndarray, np.ndarray]:
    bboxes_face = recognize_faces_bboxes(frame, face_detector, conf_threshold=0.8)
    # if face is not recognized, note it as NaN
    if bboxes_face is None:
        # if there were no face before
        if previous_face is None:
            face = np.zeros((224, 224, 3), dtype=np.uint8)
            bbox = None
        else:
            # save previous face
            face = previous_face
            bbox = previous_bbox
    else:
        # otherwise, extract the face and save it
        if previous_bbox is not None:
            # if we want to keep the same person, then we need to calculate the distances between the
            # center of the previous bbox and the current ones. Then, choose the closest one.
            bbox = get_bbox_closest_to_previous_bbox(bboxes_face, previous_bbox)
        else:
            # otherwise, take the most confident one
            bbox = get_most_confident_person(bboxes_face)
        # extract face according to bbox
        face = extract_face_according_bbox(frame, bbox)
    return face, bbox


def __recognize_pose(frame, pose_detector, previous_pose, previous_bbox)->Tuple[np.ndarray, np.ndarray]:
    # recognize the pose
    bboxes = get_bboxes_for_frame(extractor=pose_detector, frame=frame)
    # if not recognized, note it as NaN
    if bboxes is None:
        # if there were no face before
        if previous_pose is None:
            pose = np.zeros((224, 224, 3), dtype=np.uint8)
            bbox = None
        else:
            # save previous face
            pose = previous_pose
            bbox = previous_bbox
    else:
        # otherwise, extract the face and save it
        if previous_bbox is not None:
            # if we want to keep the same person, then we need to calculate the distances between the
            # center of the previous bbox and the current ones. Then, choose the closest one.
            bbox = get_bbox_closest_to_previous_bbox(bboxes, previous_bbox)
        else:
            # otherwise, take the first one, since we do not have confidences
            bbox = bboxes[0]
        # extract face according to bbox
        pose = apply_bbox_to_frame(frame, bbox)
    return pose, bbox


def process_one_video_static(path_to_video:str, face_detector:object, pose_detector:object,
                             facial_feature_extractor:Tuple[nn.Module, List[Callable]],
                             pose_feature_extractor:Tuple[nn.Module, List[Callable]],
                             device)->pd.DataFrame:
    # extract feature extractors and preprocessing functions
    facial_feature_extractor, facial_preprocessing_functions = facial_feature_extractor
    pose_feature_extractor, pose_preprocessing_functions = pose_feature_extractor
    video_name = os.path.basename(path_to_video).split(".")[0]
    # metadata
    columns = (["video_name", "frame_num", "timestep"] +
               [f"facial_embedding_{i}" for i in range(256)] + [f"pose_embedding_{i}" for i in range(256)])
    metadata = pd.DataFrame(columns=columns)
    # load video file
    video = cv2.VideoCapture(path_to_video)
    # get FPS
    FPS = video.get(cv2.CAP_PROP_FPS)
    FPS_in_seconds = 1. / FPS
    # go through all frames
    counter = 1
    previous_face = None
    previous_pose = None
    last_bbox_face = None
    last_bbox_pose = None
    pbar = tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    while video.isOpened():
        ret, frame = video.read()
        pbar.update(1)
        if ret:
            # convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # calculate timestamp
            timestep = counter * FPS_in_seconds
            # round it to 2 digits to make it readable
            timestep = round(timestep, 2)
            # recognize the face
            face, face_bbox = __recognize_face(frame, face_detector, previous_face, last_bbox_face)
            previous_face = face
            last_bbox_face = last_bbox_face
            # recognize the pose
            pose, pose_bbox = __recognize_pose(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), # HRNet requires BGR format of frame
                                    pose_detector, previous_pose, last_bbox_pose)
            pose = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB) # transform back to RGB
            previous_pose = pose
            last_bbox_pose = last_bbox_pose
            # extract facial embeddings
            face = torch.from_numpy(face).permute(2, 0, 1)
            for func in facial_preprocessing_functions:
                face = func(face)
            face = face.unsqueeze(0).to(device)
            facial_embeddings, _ = facial_feature_extractor(face)
            facial_embeddings = facial_embeddings.detach().cpu().numpy().squeeze()
            # extract pose embeddings
            pose = torch.from_numpy(pose).permute(2, 0, 1)
            for func in pose_preprocessing_functions:
                pose = func(pose)
            pose = pose.unsqueeze(0).to(device)
            pose_embeddings, _ = pose_feature_extractor(pose)
            pose_embeddings = pose_embeddings.detach().cpu().numpy().squeeze()
            # save everything to metadata. Generate row
            metadata_row = {"video_name": video_name, "frame_num": counter, "timestep": timestep}
            metadata_row.update({f"facial_embedding_{i}": facial_embeddings[i] for i in range(256)})
            metadata_row.update({f"pose_embedding_{i}": pose_embeddings[i] for i in range(256)})
            metadata = pd.concat([metadata, pd.DataFrame(metadata_row, index=[0])], axis=0, ignore_index=True)
            # update counter
            counter += 1
        else:
            break
    metadata.reset_index(drop=True, inplace=True)
    return metadata


def process_one_video_dynamic(df_video, window_size, stride, dynamic_model, feature_columns, labels_columns,
                              device, original_fps, needed_fps,
                              batch_size:Optional[int]=32)->pd.DataFrame:
    # take avery n frame depending on the original_fps and needed_fps
    every_n_frame = int(original_fps / needed_fps)
    df_video = df_video.iloc[::every_n_frame]
    # cut on windows
    windows = __cut_video_on_windows(df_video, window_size=window_size, stride=stride)
    predictions = []
    for window_idx in range(0, len(windows), batch_size):
        # extract the batch of windows
        batch_windows = windows[window_idx:window_idx + batch_size]
        timesteps = np.stack([window['timestep'].values for window in batch_windows])
        num_frames = np.stack([window['frame_num'].values for window in batch_windows])
        labels = np.stack([window[labels_columns].values for window in batch_windows])
        # extract features from the batch # TODO: take into account bi-modal model
        if isinstance(feature_columns, tuple):
            batch_windows = [
                [torch.from_numpy(window[feature_columns[i]].values) for window in batch_windows] for i in range(len(feature_columns))
            ]
            batch_windows = [torch.stack(batch_windows[i]).float().to(device) for i in range(len(feature_columns))]
        else:
            batch_windows = [torch.from_numpy(window[feature_columns].values) for window in batch_windows]
            batch_windows = torch.stack(batch_windows).float().to(device)
        # get predictions
        batch_features, batch_predictions = dynamic_model(batch_windows) if not isinstance(dynamic_model, tuple) else dynamic_model(*batch_windows)
        batch_features = batch_features.detach().cpu().numpy()
        batch_predictions = batch_predictions.detach().cpu().numpy()
        predictions.append((num_frames, timesteps, labels, batch_features, batch_predictions))
    # flatten all arrays. Now we have List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    # need to get List[np.ndarray], where np.ndarray is 1D array
    num_frames = np.concatenate([item[0] for item in predictions], axis=0)
    num_frames = [num_frames[idx] for idx in range(num_frames.shape[0])]
    timesteps = np.concatenate([item[1] for item in predictions], axis=0)
    timesteps = [timesteps[idx] for idx in range(timesteps.shape[0])]
    labels = np.concatenate([item[2] for item in predictions], axis=0)
    labels = [labels[idx] for idx in range(labels.shape[0])]
    features = np.concatenate([item[3] for item in predictions], axis=0)
    features = [features[idx] for idx in range(features.shape[0])]
    preds = np.concatenate([item[4] for item in predictions], axis=0)
    preds = [preds[idx] for idx in range(preds.shape[0])]
    return num_frames, timesteps, labels, features, preds



def align_labels_with_metadata(metadata:pd.DataFrame, labels:pd.DataFrame, challenge)->pd.DataFrame:
    # merge metadata with labels
    labels.columns = ["category"] if challenge == "Exp" else ["valence", "arousal"]
    if metadata.shape[0] == labels.shape[0]:
        metadata = pd.concat([metadata, labels], axis=1, ignore_index=True)
    else:
        # if metadata has more frames than labels, then we need to duplicate the last label
        if metadata.shape[0] > labels.shape[0]:
            last_label = labels.iloc[-1]
            labels = pd.concat([labels, pd.DataFrame([last_label]* (metadata.shape[0] - labels.shape[0]))], axis=0, ignore_index=True)
            metadata = pd.concat([metadata, labels], axis=1, ignore_index=True)
        else:
            # if metadata has less frames than labels, then we need to cut the labels
            labels = labels.iloc[:metadata.shape[0]]
            metadata = pd.concat([metadata, labels], axis=1, ignore_index=True)
    metadata.reset_index(drop=True, inplace=True)
    return metadata




def process_all_videos_static(config, videos:List[str]):
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
    # load labels
    train_labels = load_AffWild2_labels(path_to_files=config["path_to_train_labels"], challenge=config["challenge"])
    dev_labels = load_AffWild2_labels(path_to_files=config["path_to_dev_labels"], challenge=config["challenge"])
    # unite labels (they are two dicts)
    labels = {**train_labels, **dev_labels}
    # get only videos that are in labels
    path_to_data = config["path_to_data"]
    videos = [os.path.basename(video).split(".")[0] for video in videos]
    # get only videos that are in labels
    videos = [video for video in videos if video in labels.keys()]
    # go over videos
    for video in tqdm(videos):
        videofile_name = video+".mp4" if video+".mp4" in os.listdir(path_to_data) else video+".avi"
        # process one video with static models
        metadata_static = process_one_video_static(path_to_video=os.path.join(path_to_data, videofile_name),
                                                   face_detector=face_detector, pose_detector=pose_detector,
                                                    facial_feature_extractor=(facial_feature_extractor, facial_preprocessing_functions),
                                                    pose_feature_extractor=(pose_feature_extractor, pose_preprocessing_functions),
                                                    device=config["device"])
        current_labels = labels[video]
        metadata_static = align_labels_with_metadata(metadata_static, current_labels, challenge=config["challenge"])
        # save extracted features
        metadata_static.to_csv(os.path.join(config["output_static_features"], f"{os.path.basename(video)}.csv"), index=False)




def process_all_videos_dynamic(dynamic_model_type, path_to_weights, normalization, embeddings_columns,
                               input_shape, num_classes, num_regression_neurons, video_to_fps,
                                 challenge, path_to_extracted_features:str, window_size:int, stride:int, device:torch.device,
                                    batch_size:int=32):
    # initialize dynamic models
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
    feature_columns = embeddings_columns if not isinstance(embeddings_columns, tuple) else embeddings_columns[0] + embeddings_columns[1]
    features = np.concatenate([metadata_static[video][feature_columns].dropna().values for video in metadata_static.keys()], axis=0)
    normalizer = normalizer.fit(features)
    # process all videos
    result = {}
    labels_columns = ["category"] if challenge == "Exp" else ["valence", "arousal"]
    for video in tqdm(metadata_static.keys()):
        df = metadata_static[video]
        # drop nan values
        df = df.dropna()
        # normalize features
        df.loc[:, feature_columns] = normalizer.transform(df[feature_columns].values)
        predictions = process_one_video_dynamic(df_video=df, original_fps=video_to_fps[video], needed_fps=5,
                                                window_size=window_size,
                                                        stride=stride, dynamic_model=dynamic_model,
                                                        feature_columns=embeddings_columns,
                                                        labels_columns=labels_columns,
                                                        device=device, batch_size=batch_size)
        # predictions -> (num_frames, timesteps, labels, batch_features, batch_predictions)
        # form the "value" of result dict
        # features -> [[...], [...], ...]
        num_frames, timesteps, labels, features, preds = predictions
        values = {
            'features' : features,
            'frame_start' : [item[0] for item in num_frames],
            'frame_end' : [item[-1] for item in num_frames],
            'timestep_start' : [item[0] for item in timesteps],
            'timestep_end' : [item[-1] for item in timesteps],
            'predicts' : preds,
            'targets' : labels,
        }
        result[video] = values
    return result







if __name__ == "__main__":
    # static extraction (exp and VA)
    """config_static_exp = {
        "static_model_type": "ViT_b_16",
        "pose_model_type": "HRNet",
        "path_to_static_weights": "/home/ddresvya/Data/weights_best_models/fine_tuned/Exp_challenge/AffWild2_static_exp_best_ViT.pth",
        "path_to_pose_weights": "/home/ddresvya/Data/weights_best_models/fine_tuned/Exp_challenge/AffWild2_static_exp_best_HRNet.pth",
        "path_hrnet_weights": "/home/ddresvya/PhD/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
        "output_static_features": "/home/ddresvya/Data/features/Exp/",
        "dynamic_model_facial": "dynamic_v3",
        "dynamic_model_pose": "dynamic_v3",
        "dynamic_model_fusion": "fusion_v1",
        "normalization_face": "min_max",
        "normalization_pose": "min_max",
        "normalization_fusion": None,
        "path_dynamic_model_facial": "/home/ddresvya/Data/weights_best_models/fine_tuned_dynamic/uni_modal_facial_best.pth",
        "path_dynamic_model_pose": "/home/ddresvya/Data/weights_best_models/fine_tuned_dynamic/uni_modal_pose_best.pth",
        "path_dynamic_model_fusion": "/home/ddresvya/Data/weights_best_models/fine_tuned_dynamic/fusion_best.pth",
        "input_shape": (20, 256),
        "num_classes": 8,
        "num_regression_neurons": None,
        "device": torch.device("cuda"),
        "window_size": 20,
        "stride": 10,
        "batch_size": 32,
        "path_to_train_labels": "/home/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/",
        "path_to_dev_labels": "/home/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/",
        "challenge": "Exp",
    }
    videos = glob.glob("/home/ddresvya/Data/ABAW/*")

    metadata_dynamic = process_all_videos_static(config_static_exp, videos)"""

    """config_static_VA = {
        "static_model_type": "EfficientNet-B1",
        "pose_model_type": "HRNet",
        "path_to_static_weights": "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/VA_challenge/AffWild2_static_va_best_b1.pth",
        "path_to_pose_weights": "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/VA_challenge/AffWild2_static_va_best_HRNet.pth",
        "path_hrnet_weights": "/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
        "output_static_features": "/nfs/scratch/ddresvya/Data/features/VA/",
        "num_classes": None,
        "num_regression_neurons": 2,
        "device": torch.device("cuda"),
        "path_to_train_labels": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/",
        "path_to_dev_labels": "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
        "challenge": "VA",
        "path_to_data": "/nfs/home/ddresvya/scripts/ABAW/",
    }
    videos = glob.glob("/nfs/home/ddresvya/scripts/ABAW/*")
    metadata_dynamic = process_all_videos_static(config_static_VA, videos)"""


    # dynamic extraction (exp)
    config_dynamic_face_uni_modal_exp = {
        "dynamic_model_facial": "dynamic_v3",
        "dynamic_model_pose": "dynamic_v3",
        "dynamic_model_fusion": "fusion_v1",
        "normalization_face": "min_max",
        "normalization_pose": "min_max",
        "normalization_fusion": "standard",
        "path_dynamic_model_facial": "/home/ddresvya/Data/weights_best_models/fine_tuned_dynamic/uni_modal_face_best.pth",
        "path_dynamic_model_pose": "/home/ddresvya/Data/weights_best_models/fine_tuned_dynamic/uni_modal_pose_best.pth",
        "path_dynamic_model_fusion": "/home/ddresvya/Data/weights_best_models/fine_tuned_dynamic/bi_modal_face_pose_best.pth",
        "input_shape": (20, 256),
        "num_classes": 8,
        "num_regression_neurons": None,
        "device": torch.device("cuda"),
        "window_size": 20,
        "stride": 10,
        "batch_size": 32,
        "path_to_train_labels": "/home/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/",
        "path_to_dev_labels": "/home/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/",
        "challenge": "Exp",
        'video_to_fps': load_fps_file(os.path.join(path_to_project,"src/video/training/dynamic_models/fps.pkl"))
    }

    result_uni_modal_face_exp = process_all_videos_dynamic(dynamic_model_type=config_dynamic_face_uni_modal_exp["dynamic_model_facial"],
                                        path_to_weights=config_dynamic_face_uni_modal_exp["path_dynamic_model_facial"],
                                        normalization=config_dynamic_face_uni_modal_exp["normalization_face"],
                                        embeddings_columns=[f"facial_embedding_{i}" for i in range(256)],
                                        input_shape= config_dynamic_face_uni_modal_exp["input_shape"],
                                        num_classes=config_dynamic_face_uni_modal_exp["num_classes"],
                                        num_regression_neurons=config_dynamic_face_uni_modal_exp["num_regression_neurons"],
                                        video_to_fps=config_dynamic_face_uni_modal_exp["video_to_fps"],
                                        challenge=config_dynamic_face_uni_modal_exp["challenge"],
                                        path_to_extracted_features="/home/ddresvya/Data/features/Exp/",
                                        window_size=config_dynamic_face_uni_modal_exp["window_size"],
                                        stride=config_dynamic_face_uni_modal_exp["stride"],
                                        device=config_dynamic_face_uni_modal_exp["device"],
                                        batch_size=config_dynamic_face_uni_modal_exp["batch_size"])
    # save using pickle
    with open("/home/ddresvya/Data/dynamic_features_facial_exp.pkl", "wb") as file:
        pickle.dump(result_uni_modal_face_exp, file)


    result_uni_modal_pose_exp = process_all_videos_dynamic(dynamic_model_type=config_dynamic_face_uni_modal_exp["dynamic_model_pose"],
                                        path_to_weights=config_dynamic_face_uni_modal_exp["path_dynamic_model_pose"],
                                        normalization=config_dynamic_face_uni_modal_exp["normalization_pose"],
                                        embeddings_columns=[f"pose_embedding_{i}" for i in range(256)],
                                        input_shape= config_dynamic_face_uni_modal_exp["input_shape"],
                                        num_classes=config_dynamic_face_uni_modal_exp["num_classes"],
                                        num_regression_neurons=config_dynamic_face_uni_modal_exp["num_regression_neurons"],
                                        video_to_fps=config_dynamic_face_uni_modal_exp["video_to_fps"],
                                        challenge=config_dynamic_face_uni_modal_exp["challenge"],
                                        path_to_extracted_features="/home/ddresvya/Data/features/Exp/",
                                        window_size=config_dynamic_face_uni_modal_exp["window_size"],
                                        stride=config_dynamic_face_uni_modal_exp["stride"],
                                        device=config_dynamic_face_uni_modal_exp["device"],
                                        batch_size=config_dynamic_face_uni_modal_exp["batch_size"])
    # save using pickle
    with open("/home/ddresvya/Data/dynamic_features_pose_exp.pkl", "wb") as file:
        pickle.dump(result_uni_modal_pose_exp, file)

    result_bi_modal_face_pose_exp = process_all_videos_dynamic(dynamic_model_type=config_dynamic_face_uni_modal_exp["dynamic_model_fusion"],
                                        path_to_weights=config_dynamic_face_uni_modal_exp["path_dynamic_model_fusion"],
                                        normalization=config_dynamic_face_uni_modal_exp["normalization_fusion"],
                                        embeddings_columns=([f"facial_embedding_{i}" for i in range(256)], [f"pose_embedding_{i}" for i in range(256)]),
                                        input_shape= config_dynamic_face_uni_modal_exp["input_shape"],
                                        num_classes=config_dynamic_face_uni_modal_exp["num_classes"],
                                        num_regression_neurons=config_dynamic_face_uni_modal_exp["num_regression_neurons"],
                                        video_to_fps=config_dynamic_face_uni_modal_exp["video_to_fps"],
                                        challenge=config_dynamic_face_uni_modal_exp["challenge"],
                                        path_to_extracted_features="/home/ddresvya/Data/features/Exp/",
                                        window_size=config_dynamic_face_uni_modal_exp["window_size"],
                                        stride=config_dynamic_face_uni_modal_exp["stride"],
                                        device=config_dynamic_face_uni_modal_exp["device"],
                                        batch_size=config_dynamic_face_uni_modal_exp["batch_size"])
    # save using pickle
    with open("/home/ddresvya/Data/dynamic_features_fusion_exp.pkl", "wb") as file:
        pickle.dump(result_bi_modal_face_pose_exp, file)


    # dynamic extraction (VA)
    """config_dynamic_face_uni_modal_va = {
        "dynamic_model_facial": None,
        "dynamic_model_pose": None,
        "dynamic_model_fusion": None,
        "normalization_face": None,
        "normalization_pose": None,
        "normalization_fusion": None,
        "path_dynamic_model_facial": None,
        "path_dynamic_model_pose": None,
        "path_dynamic_model_fusion": None,
        "input_shape": (20, 256),
        "num_classes": None,
        "num_regression_neurons": 2,
        "device": torch.device("cuda"),
        "window_size": 20,
        "stride": 10,
        "batch_size": 32,
        "path_to_train_labels": "/home/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/",
        "path_to_dev_labels": "/home/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
        "challenge": "VA",
        'video_to_fps': load_fps_file(os.path.join(path_to_project,"src/video/training/dynamic_models/fps.pkl"))
    }
    
    result_uni_modal_face_va = process_all_videos_dynamic(dynamic_model_type=config_dynamic_face_uni_modal_va["dynamic_model_facial"],
                                        path_to_weights=config_dynamic_face_uni_modal_va["path_dynamic_model_facial"],
                                        normalization=config_dynamic_face_uni_modal_va["normalization_face"],
                                        embeddings_columns=[f"facial_embedding_{i}" for i in range(256)],
                                        input_shape= config_dynamic_face_uni_modal_va["input_shape"],
                                        num_classes=config_dynamic_face_uni_modal_va["num_classes"],
                                        num_regression_neurons=config_dynamic_face_uni_modal_va["num_regression_neurons"],
                                        video_to_fps=config_dynamic_face_uni_modal_va["video_to_fps"],
                                        challenge=config_dynamic_face_uni_modal_va["challenge"],
                                        path_to_extracted_features="/home/ddresvya/Data/features/VA/",
                                        window_size=config_dynamic_face_uni_modal_va["window_size"],
                                        stride=config_dynamic_face_uni_modal_va["stride"],
                                        device=config_dynamic_face_uni_modal_va["device"],
                                        batch_size=config_dynamic_face_uni_modal_va["batch_size"])
    # save using pickle
    with open("/home/ddresvya/Data/dynamic_features_facial_va.pkl", "wb") as file:
        pickle.dump(result_uni_modal_face_va, file)
        
    result_uni_modal_pose_va = process_all_videos_dynamic(dynamic_model_type=config_dynamic_face_uni_modal_va["dynamic_model_pose"],
                                        path_to_weights=config_dynamic_face_uni_modal_va["path_dynamic_model_pose"],
                                        normalization=config_dynamic_face_uni_modal_va["normalization_pose"],
                                        embeddings_columns=[f"pose_embedding_{i}" for i in range(256)],
                                        input_shape= config_dynamic_face_uni_modal_va["input_shape"],
                                        num_classes=config_dynamic_face_uni_modal_va["num_classes"],
                                        num_regression_neurons=config_dynamic_face_uni_modal_va["num_regression_neurons"],
                                        video_to_fps=config_dynamic_face_uni_modal_va["video_to_fps"],
                                        challenge=config_dynamic_face_uni_modal_va["challenge"],
                                        path_to_extracted_features="/home/ddresvya/Data/features/VA/",
                                        window_size=config_dynamic_face_uni_modal_va["window_size"],
                                        stride=config_dynamic_face_uni_modal_va["stride"],
                                        device=config_dynamic_face_uni_modal_va["device"],
                                        batch_size=config_dynamic_face_uni_modal_va["batch_size"])
    # save using pickle
    with open("/home/ddresvya/Data/dynamic_features_pose_va.pkl", "wb") as file:
        pickle.dump(result_uni_modal_pose_va, file)
        
    result_bi_modal_face_pose_va = process_all_videos_dynamic(dynamic_model_type=config_dynamic_face_uni_modal_va["dynamic_model_fusion"],
                                        path_to_weights=config_dynamic_face_uni_modal_va["path_dynamic_model_fusion"],
                                        normalization=config_dynamic_face_uni_modal_va["normalization_fusion"],
                                        embeddings_columns=([f"facial_embedding_{i}" for i in range(256)], [f"pose_embedding_{i}" for i in range(256)]),
                                        input_shape= config_dynamic_face_uni_modal_va["input_shape"],
                                        num_classes=config_dynamic_face_uni_modal_va["num_classes"],
                                        num_regression_neurons=config_dynamic_face_uni_modal_va["num_regression_neurons"],
                                        video_to_fps=config_dynamic_face_uni_modal_va["video_to_fps"],
                                        challenge=config_dynamic_face_uni_modal_va["challenge"],
                                        path_to_extracted_features="/home/ddresvya/Data/features/VA/",
                                        window_size=config_dynamic_face_uni_modal_va["window_size"],
                                        stride=config_dynamic_face_uni_modal_va["stride"],
                                        device=config_dynamic_face_uni_modal_va["device"],
                                        batch_size=config_dynamic_face_uni_modal_va["batch_size"])
    # save using pickle
    with open("/home/ddresvya/Data/dynamic_features_fusion_va.pkl", "wb") as file:
        pickle.dump(result_bi_modal_face_pose_va, file)"""


