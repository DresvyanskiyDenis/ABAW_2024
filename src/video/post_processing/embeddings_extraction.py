from functools import partial
from typing import Tuple

import pandas as pd
import numpy as np
import os
import sys

path_to_project = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir,
                 os.path.pardir, os.path.pardir)) + os.path.sep
sys.path.append(path_to_project)

import torch

from feature_extraction.pytorch_based.embeddings_extraction_torch import EmbeddingsExtractor
from pytorch_utils.data_preprocessing import convert_image_to_float_and_scale
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B4, Modified_EfficientNet_B1, Modified_ViT_B_16
from pytorch_utils.models.Pose_estimation.HRNet import Modified_HRNet
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor, \
    ViT_image_preprocessor
from src.video.training.static_models.multi_task.data_preparation import load_labels_with_frame_paths
import torchvision.transforms as T


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
                                         num_regression_neurons=2)
        model.load_state_dict(torch.load(config['path_to_weights']))
        # cut off last two layers
        model = torch.nn.Sequential(*list(model.children())[:-2])
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=380),
                                   EfficientNet_image_preprocessor()]
    elif config['model_type']=='EfficientNet-B1':
        model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=config['num_classes'],
                                         num_regression_neurons=2)
        model.load_state_dict(torch.load(config['path_to_weights']))
        # cut off last two layers
        model = torch.nn.Sequential(*list(model.children())[:-2])
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    elif config['model_type']=='ViT_b_16':
        model = Modified_ViT_B_16(embeddings_layer_neurons=256, num_classes=config['num_classes'],
                                  num_regression_neurons=2)
        model.load_state_dict(torch.load(config['path_to_weights']))
        # cut off last two layers
        model = torch.nn.Sequential(*list(model.children())[:-2])
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=224),
                                   ViT_image_preprocessor()]
    elif config['model_type']=='HRNet':
        model = Modified_HRNet(pretrained=True,
                               path_to_weights=config['path_hrnet_weights'],
                               embeddings_layer_neurons=256, num_classes=config['num_classes'],
                               num_regression_neurons=2,
                               consider_only_upper_body=True)
        model.load_state_dict(torch.load(config['path_to_weights']))
        # cut off last two layers
        model.classifier = torch.nn.Identity()
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
        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)[0]
        model = Wrapper(model)
        model.eval()

    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    return model, preprocessing_functions






def extract_features(config):
    # load train and dev data
    train, dev = load_labels_with_frame_paths(config, take_every_n_frame=1)
    # initialize model and preprocessing functions
    extractor_model, preprocessing_functions = initialize_model_and_preprocessing_fucntions(config)
    # initialize embeddings extractor
    embeddings_extractor = EmbeddingsExtractor(extractor_model, preprocessing_functions=preprocessing_functions,
                                               output_shape=256)
    # extract embeddings
    embeddings_extractor.extract_embeddings(train, output_path=config['output_path_train'],
                                                               batch_size=64, num_workers=4, verbose=True)
    embeddings_extractor.extract_embeddings(dev, output_path=config['output_path_dev'],
                                                                batch_size=64, num_workers=4, verbose=True)





def main():
    config_face_extraction_b4 = {
        'exp_train_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/',
        'exp_dev_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/',
        'metafile_path': "/nfs/scratch/ddresvya/Data/preprocessed/faces/metadata.csv",
        'va_train_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/',
        'va_dev_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/',
        'path_to_data': "/nfs/scratch/ddresvya/Data/preprocessed/faces/",
        'model_type': 'EfficientNet-B4',
        'path_to_weights': "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/best_efficientNet_b4.pth",
        'num_classes': 8,
        'output_path_train': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/EfficientNet_b4/b4_facial_features_train.csv",
        'output_path_dev': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/EfficientNet_b4/b4_facial_features_dev.csv",
    }

    config_pose_extraction_HRNet = {
        'exp_train_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/',
        'exp_dev_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/',
        'metafile_path': "/nfs/scratch/ddresvya/Data/preprocessed/pose/metadata.csv",
        'va_train_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/',
        'va_dev_labels_path': '/nfs/scratch/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/',
        'path_to_data': "/nfs/scratch/ddresvya/Data/preprocessed/pose/",
        'model_type': 'HRNet',
        'path_hrnet_weights': "/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
        'path_to_weights': "/nfs/scratch/ddresvya/Data/weights_best_models/ABAW/fine_tuned/best_hrnet.pth",
        'num_classes': 8,
        'output_path_train': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/HRNet/HRNet_kinesics_features_train.csv",
        'output_path_dev': "/nfs/scratch/ddresvya/Data/preprocessed/extracted_features/HRNet/HRNet_kinesics_features_dev.csv",
    }

    # check if the output path exists
    folder_path_b4 = os.path.join(*config_face_extraction_b4['output_path_train'].split('/')[:-1])
    folder_path_HRNet = os.path.join(*config_pose_extraction_HRNet['output_path_train'].split('/')[:-1])
    os.makedirs('/'+folder_path_b4, exist_ok=True)
    os.makedirs('/'+folder_path_HRNet, exist_ok=True)
    # extract features
    extract_features(config_pose_extraction_HRNet)
    extract_features(config_face_extraction_b4)





if __name__ == "__main__":
    main()