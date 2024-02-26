import sys

sys.path.append('src/audio')

import os
import numpy as np
import pandas as pd

import cv2
import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from config import config_va

from utils.common_utils import round_math, bytes_to_array, array_to_bytes


class AbawVADataset(Dataset):
    """Valence\Arousal dataset
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
        self.threshold = 0 # num of seconds with open mouth for threshold. 0 - default, without threshold
        self.new_fps = 5 # downsampling to fps per second
        
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        
        self.prepare_data()
    
    def parse_features(self, 
                       lab_feat_df: pd.core.frame.DataFrame, 
                       lab_filename: str) -> list[dict]:
        """Creates windows with `shift`, `max_w_len`, `min_w_len` in the following steps:
        - Gets FPS and number of frames 
        - Filters frames with mouth_open, and with va using threshold
        - Splits data on consecutive row values (based on lab_id = frame_id - 1):
            [0, 1, 2, 6, 10, 11, 12, 14, 15, 16] -> [[0, 1, 2], [6], [10, 11, 12], [14, 15, 16]]
        
        - Splits obtained sequences with `shift`, `max_w_len`, `min_w_len`:
            skips sequence with length less than `min_w_len`
            or
            splits sequence on windows:
                `seg` - pointer of frame index where window is started, 
                        iterates from 0 to len of sequence (or frame) with step of `shift`
                `va_window` - window with indexes from `seg` to `seg + max_w_len`
                `start` - pointer for first frame number of window (frame number + `seg`)
                `end` - pointer for last frame number of window without last element 
                        (frame number + `seg` + len(`va_window`) - 1)
                
                if length of obtained window (`va_window`) less than `min_w_len`
                forms window from the end of sequence:
                    `va_window` - window with indexes from end to start with length of `max_w_len`
                    `start` - 0 if `max_w_len` greater than frames length 
                              (it means that in sequence there is only one segment with length less than `max_w_len`)
                              len(`frames`) - `max_w_len`) else
                    `end` - last frame number
                    Drop this window if len(`frames`) < `max_w_len` to avoid duplicates:
                        f.e. frame_rate = 30, len(seq) = 76, max_w_len = 4 * 30. In this case we 
                        will have only 3 seconds of VA.
                        seg 0: frames 0 - 60 extended to 4 * 30 and converted to 0 - 76
                        seg 1: frames 60 - 76 extended to 4 * 30 and converted to 0 - 76
        - Pads labels values to `max_w_len` seconds
                
        Args:
            lab_feat_df (pd.core.frame.DataFrame): Features with labels dataframe
            lab_filename (str): Lab filename

        Returns:
            (list[dict]): Created list of window info (lab_filename, start_t, end_t, start_f, end_f, va)
        """
        frame_rate, num_frames = self.find_corresponding_video_info(lab_filename)
        
        shift = self.shift * round_math(frame_rate)
        max_w_len = self.max_w_len * round_math(frame_rate)
        min_w_len = self.min_w_len * round_math(frame_rate)

        # filter mouth_open and mislabeled
        mouth_open_threshold = self.threshold * round_math(frame_rate)
        lab_feat_df['mouth_closed'] = 1 - lab_feat_df['mouth_open']
        s = lab_feat_df['mouth_closed'].diff().ne(0).cumsum()
        lab_feat_df = lab_feat_df[((s.groupby(s).transform('size') < mouth_open_threshold) | (lab_feat_df['mouth_open'] == 1))]
        lab_feat_df = lab_feat_df[(lab_feat_df['valence'] != -5) & (lab_feat_df['arousal'] != -5)].reset_index(drop=True)

        # select frames with index
        lab_feat_df['frame_id'] = (lab_feat_df['lab_id'] - 1) # Work with lab_id instead of frame, frame contains Nan
        downsampled_frames = list(map(round_math, np.arange(0, max_w_len - 1, round_math(frame_rate) / self.new_fps, dtype=float)))
        
        # Split the data frame based on consecutive row values differences
        sequences = dict(tuple(lab_feat_df.groupby(lab_feat_df['lab_id'].diff().gt(1).cumsum())))
        timings = []
        for idx, s in sequences.items():
            frames = s['frame'].astype(int).to_list()
            va_values = s[['valence', 'arousal']].values
            
            if len(frames) < min_w_len: # less than min_w_len
                continue

            for seg in range(0, len(frames), shift):
                va_window = va_values[seg: seg + max_w_len, :]
                start = frames[seg]
                end_idx = seg + len(va_window)
                end = frames[end_idx - 1] if end_idx > len(frames) - 1 else frames[end_idx] # skip last frame
                
                if len(va_window) < max_w_len: # if less than max_w_len: get last -max_w_len elements
                    va_window = va_values[-max_w_len:, :]
                    start = frames[max(0, len(frames) - max_w_len)] # 0 or frame[-max_w_len]
                    end = frames[-1]
                    
                    val = np.pad(va_window, 
                                 [(0, max(0, max_w_len - len(va_window))), (0, 0)], 
                                 'edge')
                    val = val[downsampled_frames, :]
                    
                    timings.append({
                        'lab_filename': lab_filename,
                        'start_t': start / round_math(frame_rate),
                        'end_t': end / round_math(frame_rate),
                        'start_f': start,
                        'end_f': end,
                        'va': array_to_bytes(val) # make val hashable
                    })

        # check duplicates
        # f.e. frame_rate = 30, len(seq) = 76, max_w_len = 4 * 30. In this case we will have only 3 seconds of VA.
        # seg 0: frames 0 - 60 extended to 4 * 30 and converted to 0 - 76
        # seg 1: frames 60 - 76 extended to 4 * 30 and converted to 0 - 76
        timings = [dict(t) for t in {tuple(d.items()) for d in timings}]
        
        for t in timings:
            t['va'] = bytes_to_array(t['va'])
        
        return timings
    
    def find_corresponding_video_info(self, lab_filename: str) -> tuple[float, float]:
        """Finds video info with corresponding label file in the following steps:
        - Removes extension of file, '_left', '_right' prefixes from label filename
        - Forms list with corresponding video files (with duplicates)
        - Picks first video from video files candidates
        - Gets FPS and total number of frames of video file

        Args:
            lab_filename (str): Label filename

        Returns:
            tuple[float, float]: FPS value and total number of frames
        """
        lab_filename = lab_filename.split('.')[0]
        lab_fns = [lab_filename.split(postfix)[0] for postfix in ['_right', '_left']]
        res = []
        for l_fn in lab_fns:
            res.extend([v for v in os.listdir(self.video_root) if l_fn == v.split('.')[0]])

        vidcap = cv2.VideoCapture(os.path.join(self.video_root, list(set(res))[0]))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        num_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        return fps, num_frames
        
    def prepare_data(self):
        """
            Iterates over labels_root, and prepare data:
            - Reads labels
            - Reads features
            - Merges labels and features
            - Finds video FPS
            - Joins obtained data and splits it into windows
            - Forms metadata
        """
        for dp, _, fps in os.walk(self.labels_root):
            if not fps:
                continue
        
            for fp in fps:
                filename = os.path.join(dp, fp)
                if '.DS_Store' in filename:
                    continue
                
                labs = pd.read_csv(filename, sep=',', names=['valence', 'arousal'], header=0)
                labs['lab_id'] = labs.index + 1

                features = pd.read_csv(os.path.join(self.features_root, fp.replace('txt', 'csv')), 
                                       sep=',', 
                                       names=['feat_id', 'frame', 'surface_area_mouth', 'mouth_open'], 
                                       header=0)

                labs_and_feats = labs.merge(features, how='left', left_on='lab_id', right_on='frame')
                labs_and_feats[['mouth_open']] = labs_and_feats[['mouth_open']].fillna(value=0.0)

                timings = self.parse_features(lab_feat_df=labs_and_feats, 
                                              lab_filename=fp)
                self.meta.extend(timings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray, list[dict]]:
        """Gets sample from dataset:
        - Reads all audio and converts stereo to mono
        - Selects indexes of audio according to metadata
        - Pads the obtained values to `max_w_len` seconds
        - Augments the obtained window
        - Extracts preliminary wav2vec features
        - Drops channel dimension

        Args:
            index (int): Index of sample from metadata

        Returns:
            tuple[torch.FloatTensor, np.ndarray, list[dict]]: x, Y (2 x `max_w_len`), sample_info as list for dataloader
        """
        data = self.meta[index]

        wav_path = data['lab_filename'].replace('_right', '').replace('_left', '').replace('txt', 'wav')        
        a_data, a_data_sr = torchaudio.load(os.path.join(self.audio_root, wav_path))
        a_data = a_data[:, round(a_data_sr * data['start_t']): min(round(a_data_sr * data['end_t']), 
                                                                   a_data_sr * (data['end_t'] + self.max_w_len))] # Due to rounding error fps - cut off window end
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

        y = data['va']

        return torch.FloatTensor(wave), y, [sample_info]
            
    def __len__(self) -> int:
        """Return number of all samples in dataset

        Returns:
            int: Length of meta list
        """
        return len(self.meta)
    
    
if __name__ == "__main__":
    avad = AbawVADataset(audio_root=config_va['FILTERED_WAV_ROOT'],
                         video_root=config_va['VIDEO_ROOT'],
                         labels_root=os.path.join(config_va['LABELS_ROOT'], 'Train_Set'),
                         features_root=config_va['FEATURES_ROOT'],
                         shift=2, min_w_len=2, max_w_len=4)
        
    dl = torch.utils.data.DataLoader(
        avad,
        batch_size=8,
        shuffle=False,
        num_workers=8)

    for d in dl:
        pass
            
    print('OK')
