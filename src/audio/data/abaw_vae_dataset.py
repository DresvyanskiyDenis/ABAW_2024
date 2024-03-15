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


class AbawVAEDataset(Dataset):
    """Valence\Arousal\Expression dataset
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
        self.threshold = .5 # num of seconds with open mouth for threshold. 0 - default, without threshold
        self.new_fps = 5 # downsampling to fps per second
        self.expr_labels_counts = []

        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        
        self.prepare_data()
    
    def parse_features(self, 
                       lab_feat_df: pd.core.frame.DataFrame, 
                       lab_filename: str) -> tuple[list[dict], list[np.ndarray]]:
        """Creates windows with `shift`, `max_w_len`, `min_w_len` in the following steps:
        - Gets FPS and number of frames 
        - Filters frames with mouth_open, and with va/expr using threshold
        - Splits data on consecutive row values (based on lab_id):
            [0, 1, 2, 6, 10, 11, 12, 14, 15, 16] -> [[0, 1, 2], [6], [10, 11, 12], [14, 15, 16]]
        
        - Splits obtained sequences with `shift`, `max_w_len`, `min_w_len`:
            skips sequence with length less than `min_w_len`
            or
            splits sequence on windows:
                `seg` - pointer of frame index where window is started, 
                        iterates from 0 to len of sequence (or frame) with step of `shift`
                `va_window`, `expr_window` - windows with indexes from `seg` to `seg + max_w_len`
                `start` - pointer for first frame number of window (frame number + `seg`)
                `end` - pointer for last frame number of window without last element 
                        (frame number + `seg` + len(`expr_window`) - 1)
                
                if length of obtained window (`expr_window`) less than `min_w_len`
                forms window from the end of sequence:
                    `va_window`, `expr_window` - windows with indexes from end to start with length of `max_w_len`
                    `start` - 0 if `max_w_len` greater than frames length 
                              (it means that in sequence there is only one segment with length less than `max_w_len`)
                              len(`frames`) - `max_w_len`) else
                    `end` - last frame number
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

        # filter mouth_open and mislabeled
        mouth_open_threshold = self.threshold * round_math(frame_rate)
        lab_feat_df['mouth_closed'] = 1 - lab_feat_df['mouth_open']
        s = lab_feat_df['mouth_closed'].diff().ne(0).cumsum()
        lab_feat_df = lab_feat_df[((s.groupby(s).transform('size') < mouth_open_threshold) | (lab_feat_df['mouth_open'] == 1))]      

        if self.labels_va_root:
            lab_feat_df = lab_feat_df[(lab_feat_df['valence'] != -5) & (lab_feat_df['arousal'] != -5)]
        
        if self.labels_expr_root:
            lab_feat_df = lab_feat_df[lab_feat_df['expr'] != -1]
        
        # select frames with index
        lab_feat_df['frame_id'] = (lab_feat_df['lab_id'] - 1) # Work with lab_id instead of frame, frame contains Nan            
        
        # Split the data frame based on consecutive row values differences
        sequences = dict(tuple(lab_feat_df.groupby(lab_feat_df['lab_id'].diff().gt(1).cumsum())))
        timings = []
        for idx, s in sequences.items():
            frames = s['lab_id'].astype(int).to_list() # lab_id is the same as frame
            if self.labels_va_root:
                va_values = s[['valence', 'arousal']].values
            else:
                va_values = np.full((len(frames), 2), -5)

            if self.labels_expr_root:
                exprs = s['expr'].values
            else:
                exprs = np.full(len(frames), -1)
            
            if len(frames) < min_w_len: # less than min_w_len
                continue
            
            for seg in range(0, len(frames), shift):
                va_window = va_values[seg: seg + max_w_len, :]
                expr_window = exprs[seg: seg + max_w_len]
                
                start = frames[seg]
                end_idx = seg + len(expr_window)
                end = frames[end_idx - 1] if end_idx > len(frames) - 1 else frames[end_idx] # skip last frame
                
                if len(expr_window) < max_w_len: # if less than max_w_len: get last -max_w_len elements
                    va_window = va_values[-max_w_len:, :]
                    expr_window = exprs[-max_w_len:]
                    start = frames[max(0, len(frames) - max_w_len)] # 0 or frame[-max_w_len]
                    end = frames[-1]
                    
                val = self.pad_labels(va_window, frame_rate)
                exprl = self.pad_labels(expr_window, frame_rate)

                timings.append({
                    'lab_filename': lab_filename,
                    'start_t': start / round_math(frame_rate),
                    'end_t': end / round_math(frame_rate),
                    'start_f': start,
                    'end_f': end,
                    'va': array_to_bytes(val),
                    'expr': array_to_bytes(exprl)
                })

        # check duplicates
        # f.e. frame_rate = 30, len(seq) = 76, max_w_len = 4 * 30. In this case we will have only 3 seconds of VA.
        # seg 0: frames 0 - 60 extended to 4 * 30 and converted to 0 - 76
        # seg 1: frames 60 - 76 extended to 4 * 30 and converted to 0 - 76
        timings = [dict(t) for t in {tuple(d.items()) for d in timings}]
        
        expr_labels = []
        for t in timings:
            t['va'] = bytes_to_array(t['va'])
            t['expr'] = bytes_to_array(t['expr'])
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

    def pad_labels(self, targets: np.ndarray, frame_rate: float) -> torch.LongTensor:
        """Pads targets for va or expr and applies downsampling with `self.new_fps`
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
            frame_rate (float): FPS value

        Raises:
            ValueError: Raise if ndim of targets more than 2

        Returns:
            torch.LongTensor: Padded targets
        """
        downsampled_frames = list(map(round_math, np.arange(0, 
                                                            round_math(frame_rate) * self.max_w_len - 1, 
                                                            round_math(frame_rate) / self.new_fps, dtype=float)))
        
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
        
        return targets

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
            'start_t': data['start_t'],
            'end_t': data['end_t'],
            'start_f': data['start_f'],
            'end_f': data['end_f'],
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
    

def form_train_dataset(train_va_root: str, dev_va_root: str, train_expr_root: str, dev_expr_root: str) -> list[str]:
    """Finds intersection in train between VA dataset and Exp dataset and exclude validation files

    Args:
        train_va_root (str): Train VA dataset root
        dev_va_root (str): Validation VA dataset root
        train_expr_root (str): Train Exp dataset root
        dev_expr_root (str): Validation Exp dataset root

    Returns:
        list[str]: Train files for both VA dataset and Exp dataset withou validation
    """
    return list((set(os.listdir(train_va_root)) & set(os.listdir(train_expr_root))) - (set(os.listdir(dev_va_root)) & set(os.listdir(dev_expr_root))))


def form_dev_dataset(dev_va_root: str, dev_expr_root: str) -> list[str]:
    """Finds intersection in validation between VA dataset and Exp dataset

    Args:
        dev_va_root (str): Validation VA dataset root
        dev_expr_root (str): Validation Exp dataset root

    Returns:
        list[str]: Validation files for both VA dataset and Exp dataset
    """
    return list(set(os.listdir(dev_va_root)) & set(os.listdir(dev_expr_root)))


def form_vae_dev_dataset(dev_va_root: str, dev_expr_root: str) -> tuple[list[str]]:
    """Finds difference in validation between VA dataset and Exp dataset

    Args:
        dev_va_root (str): Validation VA dataset root
        dev_expr_root (str): Validation Exp dataset root

    Returns:
        tuple[list[str]]: Validation files for only VA dataset and for only Exp dataset
    """
    return list(set(os.listdir(dev_va_root)) - set(os.listdir(dev_expr_root))), list(set(os.listdir(dev_expr_root)) - set(os.listdir(dev_va_root)))


if __name__ == "__main__":
    ds_names = {
        'train': 'train', 
        'devel': 'validation'
    }

    vae_train_files = form_train_dataset(train_va_root=os.path.join(config_vae['LABELS_VA_ROOT'], 'Train_Set'), 
                                         dev_va_root=os.path.join(config_vae['LABELS_VA_ROOT'], 'Validation_Set'),
                                         train_expr_root=os.path.join(config_vae['LABELS_EXPR_ROOT'], 'Train_Set'), 
                                         dev_expr_root=os.path.join(config_vae['LABELS_EXPR_ROOT'], 'Validation_Set'))
                                        
    va_dev_files = os.listdir(os.path.join(config_vae['LABELS_VA_ROOT'], 'Validation_Set'))
    expr_dev_files = os.listdir(os.path.join(config_vae['LABELS_EXPR_ROOT'], 'Validation_Set'))

    for va_g in [VAEGrouping.F2F, VAEGrouping.F2S, VAEGrouping.F2S]:
        for expr_g in [VAEGrouping.F2F, VAEGrouping.F2S, VAEGrouping.F2S]:
            avad = AbawVAEDataset(audio_root=config_vae['FILTERED_WAV_ROOT'],
                                  video_root=config_vae['VIDEO_ROOT'],
                                  labels_va_root=config_vae['LABELS_VA_ROOT'],
                                  labels_expr_root=config_vae['LABELS_EXPR_ROOT'],
                                  label_filenames=vae_train_files,
                                  dataset='{0}_Set'.format(ds_names['train'].capitalize()),
                                  features_root=config_vae['FEATURES_ROOT'],
                                  va_frames_grouping=va_g,
                                  expr_frames_grouping=expr_g,
                                  shift=2, min_w_len=2, max_w_len=4)

            dl = torch.utils.data.DataLoader(
                avad,
                batch_size=8,
                shuffle=False,
                num_workers=8)

            for d in dl:
                pass
            
            print(va_g, expr_g, 'OK')
            
    for va_g in [VAEGrouping.F2F, VAEGrouping.F2S, VAEGrouping.F2S]:
        for expr_g in [VAEGrouping.F2F, VAEGrouping.F2S, VAEGrouping.F2S]:
            avad = AbawVAEDataset(audio_root=config_vae['FILTERED_WAV_ROOT'],
                                  video_root=config_vae['VIDEO_ROOT'],
                                  labels_va_root=config_vae['LABELS_VA_ROOT'],
                                  labels_expr_root=None,
                                  label_filenames=va_dev_files,
                                  dataset='{0}_Set'.format(ds_names['devel'].capitalize()),
                                  features_root=config_vae['FEATURES_ROOT'],
                                  va_frames_grouping=va_g,
                                  expr_frames_grouping=expr_g,
                                  shift=2, min_w_len=2, max_w_len=4)

            dl = torch.utils.data.DataLoader(
                avad,
                batch_size=8,
                shuffle=False,
                num_workers=8)

            for d in dl:
                pass
            
            print(va_g, expr_g, 'OK')
            
    for va_g in [VAEGrouping.F2F, VAEGrouping.F2S, VAEGrouping.F2S]:
        for expr_g in [VAEGrouping.F2F, VAEGrouping.F2S, VAEGrouping.F2S]:
            avad = AbawVAEDataset(audio_root=config_vae['FILTERED_WAV_ROOT'],
                                  video_root=config_vae['VIDEO_ROOT'],
                                  labels_va_root=None,
                                  labels_expr_root=config_vae['LABELS_EXPR_ROOT'],
                                  label_filenames=expr_dev_files,
                                  dataset='{0}_Set'.format(ds_names['devel'].capitalize()),
                                  features_root=config_vae['FEATURES_ROOT'],
                                  va_frames_grouping=va_g,
                                  expr_frames_grouping=expr_g,
                                  shift=2, min_w_len=2, max_w_len=4)

            dl = torch.utils.data.DataLoader(
                avad,
                batch_size=8,
                shuffle=False,
                num_workers=8)

            for d in dl:
                pass
            
            print(va_g, expr_g, 'OK')
