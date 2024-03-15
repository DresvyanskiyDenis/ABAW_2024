import sys

sys.path.append('src')

import os
from enum import Enum

import numpy as np
import pandas as pd

import cv2
import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from audio.config import *
from audio.utils.common_utils import round_math, array_to_bytes, bytes_to_array


class VAEGrouping(Enum):
    """Logic for grouping labels
    """
    F2F: int = 1 # Frame to frame => N fps -> N labels
    F2S: int = 2 # Frame to second => N fps -> 1 label
    F2W: int = 3 # Frame to windows => N fps * max_w_len -> 1 label


class AbawFEDataset(Dataset):
    """Dataset for feature extraction
    Preprocesses labels and features during initialization

    Args:
        audio_root (str): Wavs root dir
        video_root (str): Videos root dir
        labels_va_root (str): Valence\Arousal labels root dir
        labels_expr_root (str): Expression labels root dir
        label_filenames (str): Filenames of labels
        dataset (str): Dataset type. Can be 'Train' or 'Validation'
        features_root (str): Features root dir
        sr (int, optional): Sample rate of audio files. Defaults to 16000.
        shift (int, optional): Window shift in seconds. Defaults to 4.
        min_w_len (int, optional): Minimum window length in seconds. Defaults to 2.
        max_w_len (int, optional): Maximum window length in seconds. Defaults to 4.
        transform (torchvision.transforms.transforms.Compose, optional): transform object. Defaults to None.
        va_frames_grouping (VAEGrouping, optional): Grouping method for VA. Defaults to VAEGrouping.F2F.
        expr_frames_grouping (VAEGrouping, optional): Grouping method for VA. Defaults to VAEGrouping.F2W.
        multitask (bool, optional): Is multitask dataset?. Defaults to True.
        processor_name (str, optional): Name of model in transformers library. Defaults to 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'.

        Raises:
            ValueError: Raises error if both labels_va_root and labels_expr_root are not null, and multitask is False.
    """
    def __init__(self, 
                 audio_root: str, 
                 video_root: str, 
                 labels_va_root: str, 
                 labels_expr_root: str, 
                 label_filenames: str, 
                 dataset: str,
                 features_root: str, 
                 sr: int = 16000, 
                 shift: int = 4, 
                 min_w_len: int = 2, 
                 max_w_len: int = 4, 
                 transform: torchvision.transforms.transforms.Compose = None, 
                 va_frames_grouping: VAEGrouping = VAEGrouping.F2F, 
                 expr_frames_grouping: VAEGrouping = VAEGrouping.F2W, 
                 multitask: bool = True,
                 processor_name: str = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim') -> None:
        self.audio_root = audio_root
        self.video_root = video_root
        self.labels_va_root = labels_va_root
        self.labels_expr_root = labels_expr_root
        self.label_filenames = label_filenames
        self.dataset = dataset
        self.features_root = features_root
        
        self.sr = sr
        self.shift = shift
        self.min_w_len = min_w_len
        self.max_w_len = max_w_len
        
        self.transform = transform
        self.va_frames_grouping = va_frames_grouping
        self.expr_frames_grouping = expr_frames_grouping
        self.multitask = multitask
        if not self.multitask and self.labels_va_root and self.labels_expr_root:
            raise ValueError('This dataset shold be multitask')
        
        self.meta = []
        self.expr_labels = []
        self.new_fps = 5 # downsampling to fps per second
        self.expr_labels_counts = []

        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        
        self.prepare_data()
    
    def parse_features(self, 
                       lab_feat_df: pd.core.frame.DataFrame, 
                       lab_filename: str) -> tuple[list[dict], list[np.ndarray]]:
        """Creates windows with `shift`, `max_w_len`, `min_w_len` in the following steps:
        - Gets FPS and number of frames 
        - Pads labels values to `max_w_len` seconds
        - Downsamples to `self.new_fps`. Removes several labels
        - Drops `timings` duplicates:
            f.e. frame_rate = 30, len(seq) = 76, max_w_len = 4 * 30. In this case we 
            will have only 3 seconds of VA.
            seg 0: frames 0 - 60 extended to 4 * 30 and converted to 0 - 76
            seg 1: frames 60 - 76 extended to 4 * 30 and converted to 0 - 76
                
        Args:
            lab_feat_df (pd.core.frame.DataFrame): Features with labels dataframe
            lab_filename (str): Lab filename

        Returns:
            (list[dict], list[np.ndarray]): Created list of window info (lab_filename, start_t, end_t, start_f, end_f, va, expr) and expression labels
        """
        frame_rate, num_frames = self.find_corresponding_video_info(lab_filename)
        
        shift = self.shift * round_math(frame_rate)
        max_w_len = self.max_w_len * round_math(frame_rate)
        min_w_len = self.min_w_len * round_math(frame_rate)
        
        timings = []
        
        frames = lab_feat_df['lab_id'].astype(int).to_list() # lab_id is the same as frame
        mouth_open = lab_feat_df['mouth_open'].astype(int).values # lab_id is the same as frame

        if self.labels_va_root:
            va_values = lab_feat_df[['valence', 'arousal']].values
        else:
            va_values = np.full((len(frames), 2), -5)

        if self.labels_expr_root:
            exprs = lab_feat_df['expr'].values
        else:
            exprs = np.full(len(frames), -1)

        for seg in range(0, len(frames), shift):
            va_window = va_values[seg: seg + max_w_len, :]
            expr_window = exprs[seg: seg + max_w_len]
            mouth_open_window = mouth_open[seg: seg + max_w_len]
                
            start = frames[seg]
            end_idx = seg + len(expr_window)
            end = frames[end_idx - 1] if end_idx > len(frames) - 1 else frames[end_idx] # skip last frame
                
            if len(expr_window) < max_w_len: # if less than max_w_len: get last -max_w_len elements
                va_window = va_values[-max_w_len:, :]
                expr_window = exprs[-max_w_len:]
                mouth_open_window = mouth_open[-max_w_len:]

                start = frames[max(0, len(frames) - max_w_len)] # 0 or frame[-max_w_len]
                end = frames[-1]
                    
            val, m_o = self.pad_labels(va_window, mouth_open_window, frame_rate)
            exprl, m_o = self.pad_labels(expr_window, mouth_open_window, frame_rate)
            
            timings.append({
                'lab_filename': lab_filename,
                'fps': frame_rate,
                'start_t': start / round_math(frame_rate),
                'end_t': end / round_math(frame_rate),
                'start_f': start,
                'end_f': end,
                'mouth_open': array_to_bytes(m_o),
                'va': array_to_bytes(val),
                'expr': array_to_bytes(exprl)
            })

        # check duplicates
        # f.e. frame_rate = 30, len(seq) = 76, max_w_len = 4 * 30. In this case we will have only 3 seconds of VA.
        # seg 0: frames 0 - 60 extended to 4 * 30 and converted to 0 - 76
        # seg 1: frames 60 - 76 extended to 4 * 30 and converted to 0 - 76
        timings = [dict(t) for t in {tuple(d.items()) for d in timings}]
        timings = sorted(timings, key=lambda d: d['start_t'])
        
        expr_labels = []
        for t in timings:
            t['va'] = bytes_to_array(t['va'])
            t['expr'] = bytes_to_array(t['expr'])
            t['mouth_open'] = bytes_to_array(t['mouth_open'])
            expr_labels.append(t['expr'])
        
        return timings, expr_labels

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
        
    def prepare_data(self) -> None:
        """
            Iterates over labels_root, and prepare data:
            - Reads labels
            - Reads features
            - Merges labels and features
            - Joins obtained data and splits it into windows
            - Forms metadata and labels
            - Forms expr label statistics
        """
        for label_file_name in sorted(self.label_filenames):
            if '.DS_Store' in label_file_name:
                continue
            
            if self.labels_va_root:
                va_label_file_path = os.path.join(self.labels_va_root, self.dataset, label_file_name)
                va_labs = pd.read_csv(va_label_file_path, sep=',', names=['valence', 'arousal'], header=0)
            else:
                va_labs = pd.DataFrame()

            if self.labels_expr_root:
                expr_label_file_path = os.path.join(self.labels_expr_root, self.dataset, label_file_name)
                expr_labs = pd.read_csv(expr_label_file_path, sep='.', names=['expr'], header=0)
            else:
                expr_labs = pd.DataFrame()

            labs = pd.concat([va_labs, expr_labs], axis=1)
            labs['lab_id'] = labs.index + 1

            features = pd.read_csv(os.path.join(self.features_root, label_file_name.replace('txt', 'csv')), 
                                   sep=',', 
                                   names=['feat_id', 'frame', 'surface_area_mouth', 'mouth_open'], 
                                   header=0)

            labs_and_feats = labs.merge(features, how='left', left_on='lab_id', right_on='frame')
            labs_and_feats[['mouth_open']] = labs_and_feats[['mouth_open']].fillna(value=0.0)
            
            timings, expr_processed_labs = self.parse_features(lab_feat_df=labs_and_feats, 
                                                               lab_filename=label_file_name)
            self.meta.extend(timings)
            self.expr_labels.extend(expr_processed_labs)

        self.expr_labels_counts = np.unique(np.asarray(self.expr_labels), return_counts=True)[1]

    def pad_labels(self, targets: np.ndarray, mouth_open: np.ndarray, frame_rate: float) -> torch.LongTensor:
        """Pads targets for va or expr with mouth open and applies downsampling with `self.new_fps`
        VAEGrouping.F2F: 
            Pads va or expr to `round_math(frame_rate)` * `self.max_w_len` length
            Transform va and expr values
        VAEGrouping.F2S: 
            Pads va or expr to `round_math(frame_rate)` * `self.max_w_len` length
            Splits targets on windows with sizes of `round_math(frame_rate)`. 
            Calculates frame-wise mean for va. Calculates frame-wise moda for expr.
            Reshape va to (2, -1, `round_math(frame_rate)`). Transform expr values
        VAEGrouping.F2W: 
            Pads va or expr to `round_math(frame_rate)` * `self.max_w_len` length
            Splits targets on windows with sizes of `round_math(frame_rate)`. 
            Calculates frame-wise mean for va. Transform va values.
            Calculates second-wise moda for expr.
        Args:
            targets (np.ndarray): Input targets
            mouth_open (np.ndarray): Mouth open array
            frame_rate (float): FPS value

        Raises:
            ValueError: Raise if ndim of targets more than 2

        Returns:
            torch.LongTensor: Padded targets
        """
        downsampled_frames = list(map(round_math, np.arange(0, 
                                                            round_math(frame_rate) * self.max_w_len - 1, 
                                                            round_math(frame_rate) / self.new_fps, dtype=float)))
        
        mouth_open = np.pad(mouth_open, (0, max(0, round_math(frame_rate) * self.max_w_len - len(mouth_open))), 'edge')
        mouth_open = mouth_open[downsampled_frames]

        if targets.ndim == 2:
            targets = np.pad(targets, 
                             [(0, max(0, round_math(frame_rate) * self.max_w_len - len(targets))), (0, 0)], 
                             'edge')

            targets = targets[downsampled_frames, :]
            if self.va_frames_grouping == VAEGrouping.F2S:
                targets_w = np.split(targets, np.arange(self.new_fps, len(targets), self.new_fps))
                targets = np.asarray(targets).reshape(2, -1, self.new_fps)
            elif self.va_frames_grouping == VAEGrouping.F2W:
                targets_w = np.split(targets, np.arange(self.new_fps, len(targets), self.new_fps))
                targets = np.asarray([np.mean(i, axis=0) for i in targets_w]).T
                
        elif targets.ndim == 1:
            tar_v, tar_c = np.unique(targets, return_counts=True)
            targets = np.pad(targets, 
                             (0, max(0, round_math(frame_rate) * self.max_w_len - len(targets))), 
                             'constant', 
                             constant_values=tar_v[np.argmax(tar_c)])

            targets = targets[downsampled_frames]
            if self.expr_frames_grouping == VAEGrouping.F2S:
                targets_w = np.split(targets, np.arange(self.new_fps, len(targets), self.new_fps))
                targets = np.asarray([max(set(i), key=list(i).count) for i in targets_w]).T
            elif self.expr_frames_grouping == VAEGrouping.F2W:
                targets = max(set(targets), key=list(targets).count)
        else:
            raise ValueError('Targets dimension > 2')
        
        return targets, mouth_open

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[np.ndarray, np.ndarray], list[dict]]:
        """Gets sample from dataset:
        - Reads audio
        - Selects indexes of audio according to metadata
        - Pads the obtained wav
        - Augments the obtained window
        - Extracts preliminary wav2vec features
        - Drops channel dimension

        Args:
            index (int): Index of sample from metadata

        Returns:
            tuple[torch.FloatTensor, list[np.ndarray, np.ndarray], list[dict]]: x, Y, sample_info as list for dataloader
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
            'fps': data['fps'],
            'start_t': data['start_t'],
            'end_t': data['end_t'],
            'start_f': data['start_f'],
            'end_f': data['end_f'],
            'mouth_open': data['mouth_open']
        }

        y_va = torch.FloatTensor(data['va'])
        y_expr = torch.LongTensor(data['expr'])
    
        if self.multitask:
            y = [y_va, y_expr]
        else:
            if self.labels_va_root and not self.labels_expr_root:
                y = y_va
            elif not self.labels_va_root and self.labels_expr_root:
                y = y_expr

        return torch.FloatTensor(wave), y, [sample_info]
            
    def __len__(self) -> int:
        """Return number of all samples in dataset

        Returns:
            int: Length of meta list
        """
        return len(self.meta)


if __name__ == "__main__":
    ds_names = {
        'train': 'train', 
        'devel': 'validation'
    }

    metadata_info = {}
    for ds in ds_names:
        metadata_info[ds] = {
            'va_label_filenames': os.listdir(os.path.join(config_va['LABELS_ROOT'], '{0}_Set'.format(ds_names[ds].capitalize()))),
            'expr_label_filenames': os.listdir(os.path.join(config_expr['LABELS_ROOT'], '{0}_Set'.format(ds_names[ds].capitalize()))),
            'dataset': '{0}_Set'.format(ds_names[ds].capitalize()),
        }
    
    # EXPR
    for ds in ds_names:
        afed = AbawFEDataset(audio_root=config_expr['FILTERED_WAV_ROOT'],
                             video_root=config_expr['VIDEO_ROOT'],
                             labels_va_root=None,
                             labels_expr_root=config_expr['LABELS_ROOT'],
                             label_filenames=metadata_info[ds]['expr_label_filenames'],
                             dataset=metadata_info[ds]['dataset'],
                             features_root=config_expr['FEATURES_ROOT'],
                             va_frames_grouping=None,
                             expr_frames_grouping=VAEGrouping.F2S,
                             multitask=False,
                             shift=2, min_w_len=2, max_w_len=4)

        dl = torch.utils.data.DataLoader(afed, batch_size=8, shuffle=False, num_workers=8)

        for d in dl:
            pass

        print('{0}, {1}, OK'.format(ds, len(afed.meta)))

    # VA
    for ds in ds_names:
        afed = AbawFEDataset(audio_root=config_va['FILTERED_WAV_ROOT'],
                             video_root=config_va['VIDEO_ROOT'],
                             labels_va_root=config_va['LABELS_ROOT'],
                             labels_expr_root=None,
                             label_filenames=metadata_info[ds]['va_label_filenames'],
                             dataset=metadata_info[ds]['dataset'],
                             features_root=config_va['FEATURES_ROOT'],
                             va_frames_grouping=VAEGrouping.F2F,
                             expr_frames_grouping=None,
                             multitask=False,
                             shift=2, min_w_len=2, max_w_len=4)

        dl = torch.utils.data.DataLoader(afed, batch_size=8, shuffle=False, num_workers=8)

        for d in dl:
            pass

        print('{0}, {1}, OK'.format(ds, len(afed.meta)))
            