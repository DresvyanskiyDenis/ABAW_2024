import sys

sys.path.append('src')

import os
import pickle

import numpy as np
import pandas as pd

import cv2
import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from sklearn.model_selection import train_test_split


class AbawMultimodalExprDataset(Dataset):
    """Multimodal dataset for EXPR
    Preprocesses labels and features during initialization

    Args:
        audio_features_path (str): Path to audio features
        video_features_path (str): Path to video features
        labels_root (str): Labels root dir
        label_filenames (str): Filenames of labels
        dataset (str): Dataset type. Can be 'Train' or 'Validation'
        audio_train_features_path (str): Path to train set audio features
        sr (int, optional): Sample rate of audio files. Defaults to 16000.
        shift (int, optional): Window shift in seconds. Defaults to 4.
        min_w_len (int, optional): Minimum window length in seconds. Defaults to 2.
        max_w_len (int, optional): Maximum window length in seconds. Defaults to 4.
        transform (torchvision.transforms.transforms.Compose, optional): transform object. Defaults to None.
    """
    def __init__(self, 
                 audio_features_path: str, 
                 video_features_path: str, 
                 labels_root: str, 
                 label_filenames: str, 
                 dataset: str,
                 audio_train_features_path: str, 
                 sr: int = 16000, 
                 shift: int = 4, 
                 min_w_len: int = 2, 
                 max_w_len: int = 4, 
                 transform: list[torchvision.transforms.transforms.Compose] = [None, None]) -> None:
        self.audio_features_path = audio_features_path
        self.video_features_path = video_features_path
        self.labels_root = labels_root
        self.label_filenames = [l_fn.replace('.txt', '') for l_fn in label_filenames]
        if '10-60-1280x720_right' in self.label_filenames:
            self.label_filenames.remove('10-60-1280x720_right') # remove this file

        self.dataset = dataset

        self.audio_train_features_path = audio_train_features_path
        
        self.sr = sr
        self.shift = shift
        self.min_w_len = min_w_len
        self.max_w_len = max_w_len
        
        self.a_transform, self.v_transform, self.av_transform = transform
        
        self.audio_data = None
        self.audio_meta = []

        self.video_data = None
        self.video_meta = []
        self.new_fps = 5 # downsampling to fps per second
        
        self.audio_mean_features_value = None # mean features on speech segments of train set
        
        self.expr_labels = []
        self.expr_labels_counts = []
                
        self.prepare_audio_data()
        self.prepare_video_data()

    def prepare_video_data(self) -> None:
        self.video_data = None
        with open(self.video_features_path, 'rb') as handle:
            temp = pickle.load(handle)
            temp = dict(sorted(temp.items()))
            temp = {l: temp[l] for l in self.label_filenames}
            self.video_data = dict(sorted(temp.items())) # sort by filename
        
        for fn in self.video_data.keys():
            for idx, targets in enumerate(self.audio_data[fn]['targets']): # iterate over audio, because it has less num of windows
                self.video_meta.append({'filename': fn, 'idx': idx})
                self.expr_labels.extend(self.video_data[fn]['targets'][idx])
                    
        self.expr_labels = np.asarray(self.expr_labels)
        self.expr_labels_counts = np.unique(self.expr_labels[self.expr_labels != -1], return_counts=True)[1] # remove -1
    
    def prepare_audio_data(self) -> None:
        # Calculate mean audio features across train dataset
        train_audio_features = []
        with open(self.audio_train_features_path, 'rb') as handle:
            a_train_data = pickle.load(handle)
        
        for fn in a_train_data.keys():
            for idx, m_o in enumerate(a_train_data[fn]['mouth_open']):
                mouth_open_w = np.split(m_o, np.arange(self.new_fps, len(m_o), self.new_fps))
                mouth_open = np.asarray([max(set(i), key=list(i).count) for i in mouth_open_w]).T

                mouth_open_index = (mouth_open == 1)
                train_audio_features.append(a_train_data[fn]['features'][idx][mouth_open_index, :])

        self.audio_mean_features_value = np.repeat(np.concatenate(train_audio_features).mean(axis=0)[np.newaxis, :], 4, axis=0)
        
        self.audio_data = None
        with open(self.audio_features_path, 'rb') as handle:
            temp = pickle.load(handle)
            temp = {l.replace('.txt', ''): temp['{0}.txt'.format(l)] for l in self.label_filenames}
            self.audio_data = dict(sorted(temp.items())) # sort by filename

        for fn in self.audio_data.keys():    
            for idx, mouth_open in enumerate(self.audio_data[fn]['mouth_open']):
                mouth_open_w = np.split(mouth_open, np.arange(self.new_fps, len(mouth_open), self.new_fps))
                mouth_open = np.asarray([max(set(i), key=list(i).count) for i in mouth_open_w]).T
                
                mouth_close_index = (mouth_open == 0)
                non_zeros = np.count_nonzero(mouth_open)
                
                self.audio_meta.append({'filename': fn, 'idx': idx})
                if non_zeros == 4:
                    continue
                
                if non_zeros == 0:
                    self.audio_data[fn]['features'][idx] = np.copy(self.audio_mean_features_value)
                else:
                    window_mean = self.audio_data[fn]['features'][idx][~mouth_close_index,:].mean(axis=0)
                    self.audio_data[fn]['features'][idx][mouth_close_index] = np.copy(window_mean)
                

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor, torch.Tensor], torch.LongTensor, list[dict]]:
        """Gets features from dataset
        
        Args:
            index (int): Index of sample from metadata

        Returns:
            tuple[list[torch.FloatTensor, torch.FloatTensor], torch.LongTensor, list[dict]]: [a_f, v_f], Y, sample_info as list for dataloader
        """
        a_meta = self.audio_meta[index]
        a_features = self.audio_data[a_meta['filename']]['features'][a_meta['idx']]
        
        v_meta = self.video_meta[index]
        v_features = self.video_data[v_meta['filename']]['features'][v_meta['idx']][:20] #TODO
        v_features = np.pad(v_features, ((0, min(20, abs(len(v_features) - 20))), (0, 0)), mode='edge') #TODO

        if self.a_transform:
            a_features = self.a_transform(a_features)
        
        if self.v_transform:
            v_features = self.v_transform(v_features)

        if self.av_transform:
            a_features, v_features = self.av_transform(a_features, v_features)

        sample_info = {k: self.video_data[v_meta['filename']][k] if 'fps' in k \
            else self.video_data[v_meta['filename']][k][v_meta['idx']] \
                for k in self.video_data[v_meta['filename']].keys()}
        
        del sample_info['targets']
        del sample_info['predicts']
        del sample_info['features']
        
        sample_info['challenge'] = 'Exp'
        sample_info['path_to_labels'] = self.labels_root
        sample_info['video_name'] = a_meta['filename']
        sample_info['fps'] = self.audio_data[a_meta['filename']]['fps'][a_meta['idx']]
        sample_info['mouth_open'] = self.audio_data[a_meta['filename']]['mouth_open'][a_meta['idx']]

        y = self.video_data[v_meta['filename']]['targets'][v_meta['idx']][:20] #TODO
        y = np.pad(y.squeeze(), (0, min(20, abs(len(y) - 20))), mode='edge') #TODO

        return [torch.FloatTensor(a_features), torch.FloatTensor(v_features)], torch.LongTensor(y), [sample_info]
            
    def __len__(self) -> int:
        """Return number of all samples in dataset

        Returns:
            int: Length of meta list
        """
        return len(self.audio_meta)


if __name__ == "__main__":
    labels_root = '/media/maxim/Databases/ABAW2024/6th_ABAW_Annotations/EXPR_Classification_Challenge'
     
    ds_names = {
        'train': 'train', 
        'devel': 'validation'
    }
    
    validation_files = os.listdir(os.path.join(labels_root, '{0}_Set'.format(ds_names['devel'].capitalize())))
    
    metadata_info = {}
    for ds in ds_names:
        metadata_info[ds] = {
            'label_filenames': os.listdir(os.path.join(labels_root, '{0}_Set'.format(ds_names[ds].capitalize()))),
            'dataset': '{0}_Set'.format(ds_names[ds].capitalize()),
        }
        
    # EXPR
    for ds in ds_names:
        amed = AbawMultimodalExprDataset(audio_features_path='/media/maxim/WesternDigital/ABAWLogs/EXPR/audio_features/expr_{0}.pickle'.format(ds),
                                         video_features_path='/media/maxim/WesternDigital/ABAWLogs/EXPR/Dynamic_features_exp_new2/dynamic_features_facial_exp.pkl',
                                         labels_root=labels_root,
                                         label_filenames=metadata_info[ds]['label_filenames'],
                                         dataset=metadata_info[ds]['dataset'],
                                         audio_train_features_path='/media/maxim/WesternDigital/ABAWLogs/EXPR/audio_features/expr_train.pickle',
                                         shift=2, min_w_len=2, max_w_len=4)

        dl = torch.utils.data.DataLoader(amed, batch_size=8, shuffle=False, num_workers=8)

        for d in dl:
            pass

        print('{0}, {1}, OK'.format(ds, len(amed.audio_meta)))
