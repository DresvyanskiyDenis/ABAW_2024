from functools import partial
from typing import Tuple

import pandas as pd
import numpy as np
import os
import sys
path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)) + os.path.sep
sys.path.append(path_to_project)
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "datatools"))
sys.path.append(path_to_project.replace("ABAW_2023_SIU", "simple-HRNet-master"))

import torch

from feature_extraction.pytorch_based.embeddings_extraction_torch import EmbeddingsExtractor
from pytorch_utils.data_preprocessing import convert_image_to_float_and_scale
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B4, Modified_EfficientNet_B1, Modified_ViT_B_16
from pytorch_utils.models.Pose_estimation.HRNet import Modified_HRNet
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor, \
    ViT_image_preprocessor
from src.video.training.static_models.multi_task.data_preparation import load_labels_with_frame_paths
import torchvision.transforms as T
from src.video.preprocessing.labels_preprocessing import load_train_dev_AffWild2_labels_with_frame_paths

def load_labels_with_frame_paths(config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if config['challenge'] == "VA":
        train_labels, dev_labels = load_train_dev_AffWild2_labels_with_frame_paths(
            paths_to_labels=(config['VA_train_labels_path'], config['VA_dev_labels_path']),
            path_to_metadata=config['metafile_path'],
            challenge=config['challenge'])
    elif config['challenge'] == "Exp":
        train_labels, dev_labels = load_train_dev_AffWild2_labels_with_frame_paths(
            paths_to_labels=(config['Exp_train_labels_path'], config['Exp_dev_labels_path']),
            path_to_metadata=config['metafile_path'],
            challenge=config['challenge'])
    else:
        raise ValueError("The challenge name should be either 'VA' or 'Exp'.")

    # concat all train labels and dev labels
    train_labels = pd.concat([value for key, value in train_labels.items()], axis=0)
    dev_labels = pd.concat([value for key, value in dev_labels.items()], axis=0)

    if config['challenge'] == "VA":
        # keep only the frames with arousal and valence -1<=x<=1
        train_labels = train_labels[(train_labels["valence"] >= -1) & (train_labels["valence"] <= 1) &
                                    (train_labels["arousal"] >= -1) & (train_labels["arousal"] <= 1)]
        dev_labels = dev_labels[(dev_labels["valence"] >= -1) & (dev_labels["valence"] <= 1) &
                                (dev_labels["arousal"] >= -1) & (dev_labels["arousal"] <= 1)]

    # delete -1 categories
    if config['challenge'] == "Exp":
        train_labels = train_labels[train_labels["category"] != -1]
        dev_labels = dev_labels[dev_labels["category"] != -1]

    # change columns names for further work
    train_labels.rename(columns={"path_to_frame": "path"}, inplace=True)
    dev_labels.rename(columns={"path_to_frame": "path"}, inplace=True)


    # convert categories to one-hot vectors if it is Exp challenge
    if config['challenge'] == "Exp":
        # create one-hot vectors
        train_one_hot = pd.get_dummies(train_labels["category"])
        dev_one_hot = pd.get_dummies(dev_labels["category"])
        # concat one-hot vectors to the labels
        train_labels = pd.concat([train_labels, train_one_hot], axis=1)
        dev_labels = pd.concat([dev_labels, dev_one_hot], axis=1)
        # delete category column
        train_labels.drop(columns=["category"], inplace=True)
        dev_labels.drop(columns=["category"], inplace=True)


    return train_labels, dev_labels





def initialize_model_and_preprocessing_fucntions(config)->Tuple[torch.nn.Module, list]:
    """ Initializes the model and preprocessing functions based on the model type.
    Config should contain the following keys:
    - model_type: str, name of the model
    - path_to_weights: str, path to the weights of the model
    - num_classes: int, number of classes in the classification task
    - in case of HRNet, path_hrnet_weights: str, path to the weights of the HRNet model

    :param config: dict
        Configuration dictionary.
    :return: Tuple[torch.nn.Module, list]
        Model and preprocessing functions.
    """
    if config['model_type']=='EfficientNet-B4':
        model = Modified_EfficientNet_B4(embeddings_layer_neurons=256, num_classes=config['num_classes'],
                                         num_regression_neurons=config['num_regression_neurons'])
        if config['challenge'] == "VA": model.classifier = torch.nn.Identity()
        model.load_state_dict(torch.load(config['path_to_weights']))
        # cut off last two layers
        model = torch.nn.Sequential(*list(model.children())[:-2])
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=380),
                                   EfficientNet_image_preprocessor()]
    elif config['model_type']=='EfficientNet-B1':
        model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=config['num_classes'],
                                         num_regression_neurons=config['num_regression_neurons'])
        if config['challenge'] == "VA": model.classifier = torch.nn.Identity()
        model.load_state_dict(torch.load(config['path_to_weights']))
        # cut off last two layers
        model = torch.nn.Sequential(*list(model.children())[:-2])
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    elif config['model_type']=='ViT_b_16':
        model = Modified_ViT_B_16(embeddings_layer_neurons=256, num_classes=config['num_classes'],
                                  num_regression_neurons=config['num_regression_neurons'])
        if config['challenge'] == "VA": model.classifier = torch.nn.Identity()
        model.load_state_dict(torch.load(config['path_to_weights']))
        # cut off last two layers
        model = torch.nn.Sequential(*list(model.children())[:-2])
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=224),
                                   ViT_image_preprocessor()]
    elif config['model_type']=='HRNet':
        model = Modified_HRNet(pretrained=True,
                               path_to_weights=config['path_hrnet_weights'],
                               embeddings_layer_neurons=256, num_classes=config['num_classes'],
                               num_regression_neurons=config['num_regression_neurons'] if config['challenge'] == "VA" else None,
                               consider_only_upper_body=True)
        if config['challenge'] == "VA": model.classifier = torch.nn.Identity()
        model.load_state_dict(torch.load(config['path_to_weights']))
        # cut off last two layers
        if config['challenge'] == "Exp": model.classifier = torch.nn.Identity()
        if config['challenge'] == "VA":
            model.regression = torch.nn.Identity()
        # freeze model
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        # define preprocessing functions
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=256),
                                   convert_image_to_float_and_scale,
                                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]  # From HRNet
        # wrapper class to modify the output of the Modified_HRNet model (as it will output Tuple of outputs and we
        # need only the first one)
        if config['challenge'] == "VA":
            class Wrapper(torch.nn.Module):
                def __init__(self, model):
                    super(Wrapper, self).__init__()
                    self.model = model
                def forward(self, x):
                    return self.model(x)[0]
            model = Wrapper(model)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    # turn model to evaluation mode
    model.eval()

    return model, preprocessing_functions






def extract_features(config):
    # load train and dev data
    train, dev = load_labels_with_frame_paths(config)
    # rename columns
    if config["challenge"] == "Exp":
        train.rename(columns={i: f"category_{i}" for i in range(8)}, inplace=True)
        dev.rename(columns={i: f"category_{i}" for i in range(8)}, inplace=True)
    # initialize model and preprocessing functions
    extractor_model, preprocessing_functions = initialize_model_and_preprocessing_fucntions(config)
    # initialize embeddings extractor
    embeddings_extractor = EmbeddingsExtractor(extractor_model, preprocessing_functions=preprocessing_functions,
                                               output_shape=256)
    # extract embeddings
    embeddings_extractor.extract_embeddings(train, output_path=config['output_path_train'],
                                            batch_size=130, num_workers=4, verbose=True,
                                            labels_columns=config['labels_columns'])
    embeddings_extractor.extract_embeddings(dev, output_path=config['output_path_dev'],
                                            batch_size=130, num_workers=4, verbose=True,
                                            labels_columns=config['labels_columns'])





def main():
    print("With timestamps")
    config_face_extraction_b1_exp = {
        'challenge': "Exp",
        'Exp_train_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/',
        'Exp_dev_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/',
        'metafile_path': "/nfs/scratch/ddresvya/Data/preprocessed/faces/metadata.csv",
        'va_train_labels_path': None,
        'va_dev_labels_path': None,
        'labels_columns': ['timestamp', 'frame_num']+[f"category_{i}" for i in range(8)],
        'path_to_data': "/nfs/scratch/ddresvya/Data/preprocessed/faces/",
        'model_type': 'EfficientNet-B1',
        'path_to_weights': "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/Exp_challenge/AffWild2_static_exp_best_B1.pth",
        'num_classes': 8,
        'num_regression_neurons': 2,
        'output_path_train': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/Exp_challenge/EfficientNet_b1/b1_facial_features_train.csv",
        'output_path_dev': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/Exp_challenge/EfficientNet_b1/b1_facial_features_dev.csv",
    }

    config_face_extraction_ViT_exp = {
        'challenge': "Exp",
        'Exp_train_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/',
        'Exp_dev_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/',
        'metafile_path': "/nfs/scratch/ddresvya/Data/preprocessed/faces/metadata.csv",
        'va_train_labels_path': None,
        'va_dev_labels_path': None,
        'labels_columns': ['timestamp', 'frame_num']+[f"category_{i}" for i in range(8)],
        'path_to_data': "/nfs/scratch/ddresvya/Data/preprocessed/faces/",
        'model_type': 'ViT_b_16',
        'path_to_weights': "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/Exp_challenge/AffWilf2_static_exp_best_ViT.pth",
        'num_classes': 8,
        'num_regression_neurons': 2,
        'output_path_train': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/Exp_challenge/ViT_b_16/ViT_b_16_facial_features_train.csv",
        'output_path_dev': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/Exp_challenge/ViT_b_16/ViT_b_16_facial_features_dev.csv",
    }


    config_pose_extraction_HRNet = {
        'challenge': "Exp",
        'Exp_train_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/',
        'Exp_dev_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/',
        'metafile_path': "/nfs/scratch/ddresvya/Data/preprocessed/pose/metadata.csv",
        'va_train_labels_path': None,
        'va_dev_labels_path': None,
        'labels_columns': ['timestamp', 'frame_num']+[f"category_{i}" for i in range(8)],
        'path_to_data': "/nfs/scratch/ddresvya/Data/preprocessed/pose/",
        'model_type': 'HRNet',
        'path_hrnet_weights': "/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
        'path_to_weights': "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/Exp_challenge/AffWild2_static_exp_best_HRNet.pth",
        'num_classes': 8,
        'num_regression_neurons': 2,
        'output_path_train': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/Exp_challenge/HRNet/HRNet_kinesics_features_train.csv",
        'output_path_dev': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/Exp_challenge/HRNet/HRNet_kinesics_features_dev.csv",
    }


    # check if the output path exists
    folder_b1 = os.path.dirname(config_face_extraction_b1_exp['output_path_train'])
    folder_vit = os.path.dirname(config_face_extraction_ViT_exp['output_path_train'])
    folder_hrnet = os.path.dirname(config_pose_extraction_HRNet['output_path_train'])
    # create dirs
    os.makedirs(folder_b1, exist_ok=True)
    os.makedirs(folder_vit, exist_ok=True)
    os.makedirs(folder_hrnet, exist_ok=True)
    # extract features
    extract_features(config_pose_extraction_HRNet)
    extract_features(config_face_extraction_b1_exp)
    extract_features(config_face_extraction_ViT_exp)



    # VA challenge
    config_face_extraction_b1_VA = {
        'challenge': "VA",
        'Exp_train_labels_path': None,
        'Exp_dev_labels_path': None,
        'metafile_path': "/nfs/scratch/ddresvya/Data/preprocessed/faces/metadata.csv",
        'VA_train_labels_path': "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/",
        'VA_dev_labels_path': "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
        'path_to_data': "/nfs/scratch/ddresvya/Data/preprocessed/face/",
        'labels_columns': ['timestamp', 'frame_num']+['arousal', 'valence'],
        'model_type': 'EfficientNet-B1',
        'path_to_weights': "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/VA_challenge/AffWild2_static_va_best_b1.pth",
        'num_classes': 8,
        'num_regression_neurons':2,
        'output_path_train': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/VA_challenge/EfficientNet_b1/b1_facial_features_train.csv",
        'output_path_dev': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/VA_challenge/EfficientNet_b1/b1_facial_features_dev.csv",
    }

    config_face_extraction_b4_VA = {
        'challenge': "VA",
        'Exp_train_labels_path': None,
        'Exp_dev_labels_path': None,
        'metafile_path': "/nfs/scratch/ddresvya/Data/preprocessed/faces/metadata.csv",
        'VA_train_labels_path': "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/",
        'VA_dev_labels_path': "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
        'path_to_data': "/nfs/scratch/ddresvya/Data/preprocessed/face/",
        'labels_columns': ['timestamp', 'frame_num']+['arousal', 'valence'],
        'model_type': 'EfficientNet-B4',
        'path_to_weights': "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/VA_challenge/AffWild2_static_va_best_b4.pth",
        'num_classes': 8,
        'num_regression_neurons':2,
        'output_path_train': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/VA_challenge/EfficientNet_b4/b4_facial_features_train.csv",
        'output_path_dev': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/VA_challenge/EfficientNet_b4/b4_facial_features_dev.csv",
    }

    config_face_extraction_HRNet_VA = {
        'challenge': "VA",
        'Exp_train_labels_path': None,
        'Exp_dev_labels_path': None,
        'metafile_path': "/nfs/scratch/ddresvya/Data/preprocessed/pose/metadata.csv",
        'VA_train_labels_path': "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/",
        'VA_dev_labels_path': "/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/",
        'path_to_data': "/nfs/scratch/ddresvya/Data/preprocessed/pose/",
        'labels_columns': ['timestamp', 'frame_num']+['arousal', 'valence'],
        'model_type': 'HRNet',
        'path_hrnet_weights': "/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
        'path_to_weights': "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/VA_challenge/AffWild2_static_va_best_HRNet.pth",
        'num_classes': 8,
        'num_regression_neurons': 2,
        'output_path_train': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/VA_challenge/HRNet/HRNet_facial_features_train.csv",
        'output_path_dev': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/VA_challenge/HRNet/HRNet_facial_features_dev.csv",
    }

    # check if the output path exists
    folder_b1 = os.path.dirname(config_face_extraction_b1_VA['output_path_train'])
    folder_vit = os.path.dirname(config_face_extraction_b4_VA['output_path_train'])
    folder_hrnet = os.path.dirname(config_face_extraction_HRNet_VA['output_path_train'])
    # create dirs
    os.makedirs(folder_b1, exist_ok=True)
    os.makedirs(folder_vit, exist_ok=True)
    os.makedirs(folder_hrnet, exist_ok=True)
    # extract features
    extract_features(config_face_extraction_HRNet_VA)
    extract_features(config_face_extraction_b4_VA)
    extract_features(config_face_extraction_b1_VA)







if __name__ == "__main__":
    main()