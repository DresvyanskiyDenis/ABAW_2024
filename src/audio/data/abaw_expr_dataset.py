import sys

sys.path.append('src/audio')

import os
import pandas as pd

import cv2
import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from config import config_expr


class AbawExprDataset(Dataset):
    """Expression dataset
    Preprocesses labels and features during initialization

    Args:
        audio_root (str): Wavs root dir
        video_root (str): Videos root dir
        labels_root (str): Labels root dir
        features_root (str): Features root dir
        sr (int, optional): Sample rate of audio files. Defaults to 16000.
        shift (int, optional): Window shift in seconds. Defaults to 4.
        min_w_len (int, optional): Minimum window length in seconds. Defaults to 2.
        max_w_len (int, optional): Maximum window length in seconds. Defaults to 4.
        transform (torchvision.transforms.transforms.Compose, optional): transform object. Defaults to None.
        processor_name (str, optional): Name of model in transformers library. Defaults to 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'.
    """
    def __init__(self, 
                 audio_root: str, 
                 video_root: str, 
                 labels_root: str, 
                 features_root: str, 
                 sr: int = 16000, 
                 shift: int = 4, 
                 min_w_len: int = 2, 
                 max_w_len: int = 4, 
                 transform: torchvision.transforms.transforms.Compose = None, 
                 processor_name: str = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim') -> None:
        self.audio_root = audio_root
        self.video_root = video_root
        self.labels_root = labels_root
        self.features_root = features_root
        
        self.sr = sr
        self.shift = shift
        self.min_w_len = min_w_len
        self.max_w_len = max_w_len
        
        self.transform = transform
        
        self.meta = []
        self.labels = []
        self.threshold = .5 # num of seconds with open mouth for threshold. 0 - default, without threshold
        
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        
        self.prepare_data()
    
    def parse_features(self, 
                       lab_feat_df: pd.core.frame.DataFrame, 
                       lab_filename: str, 
                       frame_rate: float = 30.0) -> tuple[list[dict], list]:
        """Creates windows with `shift`, `max_w_len`, `min_w_len` in the following steps:
        - Filters frames with mouth_open, and with labels using threshold
        - Splits data on consecutive row values (based on lab_id = frame_id - 1):
            [0, 1, 2, 6, 10, 11, 12, 14, 15, 16] -> [[0, 1, 2], [6], [10, 11, 12], [14, 15, 16]]
        - Splits obtained sequences with `shift`, `max_w_len`, `min_w_len`:
            skips sequence with length less than `min_w_len`
            or
            splits sequence on windows:
                `seg` - pointer of label index where window is started, 
                        iterates from 0 to len of sequence (or labels) with step of `shift`
                `temp` - windowed labels with indexes from `seg` to `seg + max_w_len`
                `start` - pointer for first frame number of window (frame number + `seg`)
                `end` - pointer for last frame number of window without last element 
                        (frame number + `seg` + len(`temp`) - 1)
                
                if length of obtained windowed labels (`temp`) less than `min_w_len`
                forms window from the end of sequence:
                    `temp` - windowed labels from end to start with length of `max_w_len`
                    `start` - 0 if `max_w_len` greater than labels length 
                              (it means that in sequence there is only one segment with length less than `max_w_len`)
                              len(labels) - max_w_len) else
                    `end` - last frame number        

        Args:
            lab_feat_df (pd.core.frame.DataFrame): Features with labels dataframe
            lab_filename (str): Lab filename
            frame_rate (float, optional): Frame rate of video. Defaults to 30.0.

        Returns:
            tuple[list[dict], list]: Created list of window info (lab_filename, start_t, end_t, start_f, end_f, label), and list of labels
        """
        shift = round(self.shift * frame_rate)
        max_w_len = round(self.max_w_len * frame_rate)
        min_w_len = round(self.min_w_len * frame_rate)

        # filter mouth_open and mislabeled
        mouth_open_threshold = round(self.threshold * frame_rate)
        lab_feat_df['mouth_closed'] = 1 - lab_feat_df['mouth_open']
        s = lab_feat_df['mouth_closed'].diff().ne(0).cumsum()
        lab_feat_df = lab_feat_df[(lab_feat_df['label'] != -1) & ((s.groupby(s).transform('size') < mouth_open_threshold)  | (lab_feat_df['mouth_open'] == 1))]
        
        # Split the data frame based on consecutive row values differences
        sequences = dict(tuple(lab_feat_df.groupby(lab_feat_df['lab_id'].diff().gt(1).cumsum())))
        
        timings = []
        all_labs = []
        for idx, s in sequences.items():
            frames = s['lab_id'].to_list()
            labels = s['label'].to_list()

            if len(labels) < min_w_len: # less than min_w_len
                continue

            for seg in range(0, len(labels), shift):
                temp = labels[seg: seg + max_w_len]
                start = frames[0] + seg
                end = frames[0] + seg + len(temp) - 1 # skip last frame
                
                if len(temp) < min_w_len: # if less than min_w_len: get last -max_w_len elements
                    temp = labels[-max_w_len:]
                    start = frames[max(0, len(labels) - max_w_len)] # 0 or frame[-max_w_len]
                    end = frames[-1]

                    if len(labels) <= max_w_len:
                        continue

                final_label = max(set(temp), key=temp.count)

                timings.append({
                    'lab_filename': lab_filename,
                    'start_t': start / frame_rate,
                    'end_t': end / frame_rate,
                    'start_f': start,
                    'end_f': end,
                    'label': final_label
                })

                all_labs.append(final_label)
        
        return timings, all_labs
        
    def find_corresponding_video_fps(self, lab_filename: str) -> float:
        """Finds video fps with corresponding label file in the following steps:
        - Removes extension of file, '_left', '_right' prefixes from label filename
        - Forms list with corresponding video files (with duplicates)
        - Picks first video from video files candidates
        - Gets FPS of video file

        Args:
            lab_filename (str): Label filename

        Returns:
            float: FPS value
        """
        lab_filename = lab_filename.split('.')[0]
        lab_fns = [lab_filename.split(postfix)[0] for postfix in ['_right', '_left']]
        res = []
        for l_fn in lab_fns:
            res.extend([v for v in os.listdir(self.video_root) if l_fn == v.split('.')[0]])

        vidcap = cv2.VideoCapture(os.path.join(self.video_root, list(set(res))[0]))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        return fps
        
    def prepare_data(self) -> None:
        """
            Iterates over labels_root, and prepare data:
            - Reads labels
            - Reads features
            - Merges labels and features
            - Finds video FPS to create windows
            - Joins obtained data and splits it into windows
            - Forms metadata and labels
        """
        for dp, _, fps in os.walk(self.labels_root):
            if not fps:
                continue
                
            for fp in fps:
                filename = os.path.join(dp, fp)
                if '.DS_Store' in filename:
                    continue
                
                labs = pd.read_csv(filename, sep='.', names=['label'], header=0)
                labs['lab_id'] = labs.index + 1
                
                features = pd.read_csv(os.path.join(self.features_root, fp.replace('txt', 'csv')), 
                                       sep=',', 
                                       names=['feat_id', 'frame', 'surface_area_mouth', 'mouth_open'], 
                                       header=0)

                labs_and_feats = labs.merge(features, how='left', left_on='lab_id', right_on='frame')
                labs_and_feats[['mouth_open']] = labs_and_feats[['mouth_open']].fillna(value=0.0)
                v_fps = self.find_corresponding_video_fps(fp)

                timings, all_labs = self.parse_features(lab_feat_df=labs_and_feats, 
                                                        lab_filename=fp, 
                                                        frame_rate=v_fps)
                self.meta.extend(timings)
                self.labels.extend(all_labs)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, list[dict]]:
        """Gets sample from dataset:
        - Reads audio
        - Selects indexes of audio according to metadata
        - Pads the obtained values to `max_w_len` seconds
        - Augments the obtained window
        - Extracts preliminary wav2vec features
        - Drops channel dimension

        Args:
            index (int): Index of sample from metadata

        Returns:
            tuple[torch.FloatTensor, int, list[dict]]: x, Y, sample_info as list for dataloader
        """
        data = self.meta[index]

        wav_path = data['lab_filename'].replace('_right', '').replace('_left', '').replace('txt', 'wav')        
        a_data, a_data_sr = torchaudio.load(os.path.join(self.audio_root, wav_path))

        a_data = a_data[:, round(a_data_sr * data['start_t']): round(a_data_sr * data['end_t'])]
        print('NON ZERO ', torch.count_nonzero(a_data))
        
        a_data = torch.nn.functional.pad(a_data, 
                                         (0, max(0, self.max_w_len * a_data_sr - a_data.shape[1])), 
                                         mode='constant')
        
        if self.transform:
            a_data = self.transform(a_data)

        wave = self.processor(a_data, sampling_rate=a_data_sr)
        wave = wave['input_values'][0].squeeze()
        
        sample_info = {
            'filename': os.path.basename(data['lab_filename']),
            'start_t': data['start_t'],
            'end_t': data['end_t'],
            'start_f': data['start_f'],
            'end_f': data['end_f'],
        }

        y = data['label']
        
        return torch.FloatTensor(wave), y, [sample_info]
            
    def __len__(self) -> int:
        """Return number of all samples in dataset

        Returns:
            int: Length of meta list
        """
        return len(self.meta)
    
    
if __name__ == "__main__":
    aed = AbawExprDataset(audio_root=config_expr['FILTERED_WAV_ROOT'],
                          video_root=config_expr['VIDEO_ROOT'],
                          labels_root=config_expr['LABELS_ROOT'],
                          features_root=config_expr['FEATURES_ROOT'],
                          shift=2, min_w_len=2, max_w_len=4)
    
    print(len(aed))    
    print(aed[1][0].shape)
    print(aed[1][1])
    
    