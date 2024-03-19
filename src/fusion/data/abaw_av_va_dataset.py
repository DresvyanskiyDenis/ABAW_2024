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
from sklearn.preprocessing import MinMaxScaler


class AbawMultimodalVAWithNormDataset(Dataset):
    """Multimodal dataset for EXPR
    Preprocesses labels and features during initialization
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
                 normalizer: list[MinMaxScaler] = [None, None, None],
                 num_frames_dict: dict = None,
                 transform: list[torchvision.transforms.transforms.Compose] = [None, None, None]) -> None:
        self.audio_features_path = audio_features_path
        self.video_features_path = video_features_path
        self.labels_root = labels_root
        self.label_filenames = [l_fn.replace('.txt', '') for l_fn in label_filenames]

        files_to_remove = [
            # train
            '10-60-1280x720_right', '100-29-1080x1920', '110-30-270x480', '112', '116-30-1280x720', '120', '28-30-1280x720-1', '288', '290', 
            '297', '299', '30-30-1920x1080_left', '30-30-1920x1080_right', '306', '308', '155', '160', '167', '169', '176', '178', '421', 
            '426', '428', '430', '225', '233', '348', '350', '359', '373', '374', '386', '388', '39-25-424x240', '399', '402', '69-25-854x480', 
            '71-30-1920x1080', '78-30-960x720', '87-25-1920x1080', '96-30-1280x720', '127', '129', '131', '136', '138-30-1280x720', '138', 
            '143', '144', '183', '184', '192', '195', '201', '206', '208', '210', '486', '488', '490', '494', '499', '5-60-1920x1080-3', 
            '5-60-1920x1080-4', '51-30-1280x720', '52-30-1280x720_left', '52-30-1280x720_right', '57-25-426x240', '246', '248', '250', '257', 
            '259', '262', '273', '317', '319', '325', '334', '34-25-1920x1080', '341', '346',
            # devel
            '118-30-640x480', '12-24-1920x1080', '13-30-1920x1080', '25-25-600x480', '48-30-720x1280', '8-30-1280x720', 
            'video54', 'video61', 'video66', 'video77', 'video79', 'video82', 'video85', 'video86_2', 
            'video93', 'video94'
        ]
            
        for f in files_to_remove:
            if f in self.label_filenames:
                self.label_filenames.remove(f) # remove this file

        self.dataset = dataset

        self.audio_train_features_path = audio_train_features_path
        self.num_frames_dict = num_frames_dict
        
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
        
        self.prepare_audio_data()
        self.prepare_video_data()

        self.a_va_normalizer = None
        self.v_v_normalizer = None
        self.v_a_normalizer = None

        if normalizer[0]:
            self.a_va_normalizer = normalizer[0]
        else:
            self.a_va_normalizer = self.train_minmax_scaler(self.audio_meta, self.audio_data)
        
        if normalizer[1]:
            self.v_v_normalizer = normalizer[1]
        else:
            self.v_v_normalizer = self.train_minmax_scaler(self.video_meta, self.video_data, idx=0)

        if normalizer[2]:
            self.v_a_normalizer = normalizer[2]
        else:
            self.v_a_normalizer = self.train_minmax_scaler(self.video_meta, self.video_data, idx=1)

    def train_minmax_scaler(self, meta, data, idx=None) -> None:
        all_features = []
        for m in meta:
            if idx is not None:
                all_features.append(data[m['filename']]['features'][m['idx']][idx]) # last idx: 0 -> valence, 1 -> arousal
            else:
                all_features.append(data[m['filename']]['features'][m['idx']])
            

        all_features = np.concatenate(all_features)
        scaler = MinMaxScaler()
        scaler = scaler.fit(all_features)
        return scaler

    def prepare_video_data(self) -> None:
        self.video_data = None
        with open(self.video_features_path, 'rb') as handle:
            temp = pickle.load(handle)
            temp = dict(sorted(temp.items()))
            temp = {l: temp[l] for l in self.label_filenames}
            self.video_data = dict(sorted(temp.items())) # sort by filename

        with open(self.video_features_path.replace('valence', 'arousal'), 'rb') as handle:
            temp = pickle.load(handle)
            temp = dict(sorted(temp.items()))
            temp = {l: temp[l] for l in self.label_filenames}
            for k in self.video_data.keys():
                for idx, _ in enumerate(self.video_data[k]['features']):
                    self.video_data[k]['features'][idx] = (self.video_data[k]['features'][idx], temp[k]['features'][idx])
        
        for fn in self.video_data.keys():
            for idx, targets in enumerate(self.audio_data[fn]['targets']): # iterate over audio, because it has less num of windows
                self.video_meta.append({'filename': fn, 'idx': idx})
    
    def prepare_audio_data(self) -> None:
        # Calculate mean audio features across train dataset
        train_audio_features = []
        with open(self.audio_train_features_path, 'rb') as handle:
            a_train_data = pickle.load(handle)
        
        for fn in a_train_data.keys():
            for idx, mouth_open in enumerate(a_train_data[fn]['mouth_open']):
                mouth_open_index = (mouth_open == 1)
                train_audio_features.append(a_train_data[fn]['features'][idx][mouth_open_index, :])

        self.audio_mean_features_value = np.repeat(np.concatenate(train_audio_features).mean(axis=0)[np.newaxis, :], 20, axis=0)
        
        self.audio_data = None
        with open(self.audio_features_path, 'rb') as handle:
            temp = pickle.load(handle)
            temp = {l.replace('.txt', ''): temp['{0}.txt'.format(l)] for l in self.label_filenames}
            self.audio_data = dict(sorted(temp.items())) # sort by filename

        for fn in self.audio_data.keys():    
            for idx, mouth_open in enumerate(self.audio_data[fn]['mouth_open']):                
                mouth_close_index = (mouth_open == 0)
                non_zeros = np.count_nonzero(mouth_open)
                
                self.audio_meta.append({'filename': fn, 'idx': idx})
                if non_zeros == 20:
                    continue
                
                if non_zeros == 0:
                    self.audio_data[fn]['features'][idx] = np.copy(self.audio_mean_features_value)
                else:
                    w = self.audio_data[fn]['features'][idx][~mouth_close_index,:]
                    window_mean = w.mean(axis=0)
                    self.audio_data[fn]['features'][idx][mouth_close_index] = np.copy(window_mean)
                

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor, torch.Tensor], torch.LongTensor, list[dict]]:
        """Gets features from dataset
        
        Args:
            index (int): Index of sample from metadata

        Returns:
            tuple[list[torch.FloatTensor, torch.FloatTensor], torch.LongTensor, list[dict]]: [a_f, v_f], Y, sample_info as list for dataloader
        """
        a_meta = self.audio_meta[index]
        a_va_features = self.audio_data[a_meta['filename']]['features'][a_meta['idx']]
        
        v_meta = self.video_meta[index]
        v_va_features = np.stack(self.video_data[v_meta['filename']]['features'][v_meta['idx']])[:, :20, :] #TODO
        v_va_features = np.pad(v_va_features, ((0, 0), (0, min(20, abs(len(v_va_features[0]) - 20))), (0, 0)), mode='edge')

        # norm a&v features
        if self.a_va_normalizer:
            a_va_features = self.a_va_normalizer.transform(a_va_features)

        if self.v_v_normalizer:
            v_v_features = self.v_v_normalizer.transform(v_va_features[0])
            
        if self.v_a_normalizer:
            v_a_features = self.v_a_normalizer.transform(v_va_features[1])

        v_va_features = np.stack((v_v_features, v_a_features))

        if self.a_transform:
            a_va_features = self.a_transform(a_va_features)
        
        if self.v_transform:
            v_va_features = self.v_transform(v_va_features)

        if self.av_transform:
            a_va_features, v_va_features = self.av_transform(a_va_features, v_va_features)

        sample_info = {k: self.video_data[v_meta['filename']][k] if 'fps' in k \
            else self.video_data[v_meta['filename']][k][v_meta['idx']] \
                for k in self.video_data[v_meta['filename']].keys()}
        
        del sample_info['targets']
        del sample_info['predicts']
        del sample_info['features']
        
        sample_info['challenge'] = 'VA'
        sample_info['path_to_labels'] = self.labels_root
        sample_info['video_name'] = a_meta['filename']
        sample_info['fps'] = self.audio_data[a_meta['filename']]['fps'][a_meta['idx']]
        sample_info['total_num_frames'] = self.num_frames_dict[a_meta['filename']] if self.num_frames_dict else -1
        sample_info['mouth_open'] = self.audio_data[a_meta['filename']]['mouth_open'][a_meta['idx']]

        y = self.video_data[v_meta['filename']]['targets'][v_meta['idx']][:20] #TODO
        y = np.pad(y, ((0, min(20, abs(len(y) - 20))), (0, 0)), mode='edge') #TODO

        return [torch.FloatTensor(a_va_features), torch.FloatTensor(v_va_features)], torch.FloatTensor(y), [sample_info]
            
    def __len__(self) -> int:
        """Return number of all samples in dataset

        Returns:
            int: Length of meta list
        """
        return len(self.audio_meta)


if __name__ == "__main__":
    labels_root = '/6th_ABAW_Annotations/VA_Estimation_Challenge'
     
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
    datasets = {}
    for ds in ds_names:
        datasets[ds] = AbawMultimodalVAWithNormDataset(audio_features_path='/extracted_av_feats/VA/audio_features/va_{0}.pickle'.format(ds),
                                         video_features_path='/extracted_av_feats/VA/video_features/dynamic_features_facial_valence_20.pkl',
                                         labels_root=labels_root,
                                         label_filenames=metadata_info[ds]['label_filenames'],
                                         dataset=metadata_info[ds]['dataset'],
                                         audio_train_features_path='/extracted_av_feats/VA/audio_features/va_train.pickle',
                                         normalizer=[None, None, None] if 'train' in ds else [datasets['train'].a_va_normalizer, 
                                                                                              datasets['train'].v_v_normalizer,
                                                                                              datasets['train'].v_a_normalizer],
                                         shift=2, min_w_len=2, max_w_len=4)

        dl = torch.utils.data.DataLoader(datasets[ds], batch_size=8, shuffle=False, num_workers=8)

        for d in dl:
            pass

        print('{0}, {1}, OK'.format(ds, len(datasets[ds].audio_meta)))
