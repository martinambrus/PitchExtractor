#coding: utf-8
import os
import os.path as osp
import time
import random
import json
import numpy as np
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import pyworld as pw

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

DEFAULT_MEL_PARAMS = {
    "sample_rate": 24000,
    "n_mels": 80,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 300,
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=DEFAULT_MEL_PARAMS["sample_rate"],
                 mel_params=None,
                 data_augmentation=False,
                 validation=False,
                 verbose=True
                 ):

        self.verbose = verbose
        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [d[0] for d in _data_list]

        mel_params = mel_params or {}
        if 'win_len' in mel_params and 'win_length' not in mel_params:
            mel_params['win_length'] = mel_params.pop('win_len')

        self.mel_params = DEFAULT_MEL_PARAMS.copy()
        self.mel_params.update(mel_params)

        if sr is not None:
            self.sr = sr
        else:
            self.sr = self.mel_params.get('sample_rate', DEFAULT_MEL_PARAMS['sample_rate'])

        # ensure mel spectrogram uses the dataset sample rate
        self.mel_params['sample_rate'] = self.sr

        if self.verbose:
            print(f"[MelDataset] Using mel-spectrogram parameters: {self.mel_params}")
        logger.info("Using mel-spectrogram parameters: %s", self.mel_params)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**self.mel_params)

        # cache management helpers
        self._mel_cache_suffix = "_mel.npy"
        self._mel_meta_suffix = "_mel_meta.json"
        self._mel_cache_invalidated = False
        self._cache_enabled = True

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        self.mean, self.std = -4, 4

        # for silence detection
        self.zero_value = -10 # what the zero value is
        self.bad_F0 = 5 # if less than 5 frames are non-zero, it's a bad F0, try another algorithm

    def __len__(self):
        return len(self.data_list)

    def path_to_mel_and_label(self, path):
        wave_tensor, wave_sr = self._load_tensor(path)

        # use pyworld to get F0
        output_file = path + "_f0.npy"
        # check if the file exists
        if os.path.isfile(output_file): # if exists, load it directly
            f0 = np.load(output_file)
        else: # if not exist, create F0 file
            if self.verbose:
                print('Computing F0 for ' + path + '...')
            x = wave_tensor.numpy().astype("double")
            frame_period = self.mel_params['hop_length'] * 1000 / self.sr
            _f0, t = pw.harvest(x, self.sr, frame_period=frame_period)
            if sum(_f0 != 0) < self.bad_F0: # this happens when the algorithm fails
                _f0, t = pw.dio(x, self.sr, frame_period=frame_period) # if harvest fails, try dio
            f0 = pw.stonemask(x, _f0, t, self.sr)
            # save the f0 info for later use
            np.save(output_file, f0)
        
        f0 = torch.from_numpy(f0).float()
        
        if self.data_augmentation:
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        expected_metadata = self._build_mel_metadata(wave_tensor, wave_sr)
        mel_tensor = self._load_cached_mel(path, expected_metadata)
        if mel_tensor is None:
            mel_tensor = self.to_melspec(wave_tensor)
            if self._cache_enabled and not self.data_augmentation:
                self._save_mel_cache(path, mel_tensor, expected_metadata)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)
        
        f0_zero = (f0 == 0)
        
        #######################################
        # You may want your own silence labels here
        # The more accurate the label, the better the resultss
        is_silence = torch.zeros(f0.shape)
        is_silence[f0_zero] = 1
        #######################################
        
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
            f0 = f0[random_start:random_start + self.max_mel_length]
            is_silence = is_silence[random_start:random_start + self.max_mel_length]
        
        if torch.any(torch.isnan(f0)): # failed
            f0[torch.isnan(f0)] = self.zero_value # replace nan value with 0
        
        return mel_tensor, f0, is_silence


    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, f0, is_silence = self.path_to_mel_and_label(data)
        return mel_tensor, f0, is_silence

    def _load_tensor(self, data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor, sr

    def _build_mel_metadata(self, wave_tensor, wave_sr):
        num_samples = int(wave_tensor.shape[0]) if wave_tensor.ndim > 0 else int(wave_tensor.numel())
        num_channels = int(wave_tensor.shape[1]) if wave_tensor.ndim > 1 else 1

        def _serialize(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.generic,)):
                return value.item()
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                return value.item() if value.numel() == 1 else value.tolist()
            return value

        serialized_params = {k: _serialize(v) for k, v in self.mel_params.items()}

        return {
            "audio_sample_rate": int(wave_sr),
            "audio_num_samples": num_samples,
            "audio_num_channels": num_channels,
            "dataset_sample_rate": int(self.sr),
            "mel_params": serialized_params,
        }

    def _mel_cache_paths(self, path):
        return path + self._mel_cache_suffix, path + self._mel_meta_suffix

    def _load_cached_mel(self, path, expected_metadata):
        if not self._cache_enabled or self.data_augmentation:
            return None

        mel_cache_path, meta_cache_path = self._mel_cache_paths(path)

        if not os.path.isfile(mel_cache_path):
            # no cached mel available
            # remove stray metadata file if present to avoid stale comparisons later
            if os.path.isfile(meta_cache_path) and not self._mel_cache_invalidated:
                self._invalidate_mel_cache(meta_cache_path, reason="metadata_without_mel")
            return None

        if not os.path.isfile(meta_cache_path):
            # stale cache without metadata; invalidate the entire cache once
            self._invalidate_mel_cache(meta_cache_path, reason="missing_metadata")
            return None

        try:
            with open(meta_cache_path, "r", encoding="utf-8") as meta_file:
                cached_metadata = json.load(meta_file)
        except (OSError, json.JSONDecodeError):
            self._invalidate_mel_cache(meta_cache_path, reason="unreadable_metadata")
            return None

        if cached_metadata != expected_metadata:
            self._invalidate_mel_cache(meta_cache_path, reason="metadata_mismatch")
            return None

        try:
            mel_numpy = np.load(mel_cache_path)
        except (OSError, ValueError):
            self._invalidate_mel_cache(mel_cache_path, reason="unreadable_cache")
            return None

        return torch.from_numpy(mel_numpy)

    def _invalidate_mel_cache(self, reference_path, reason="unknown"):
        if self._mel_cache_invalidated:
            # ensure the reference file is removed even on subsequent calls
            self._remove_file_safely(reference_path)
            return

        self._mel_cache_invalidated = True
        if self.verbose:
            print(f"[MelDataset] Mel cache invalidation triggered ({reason}). Clearing cached spectrograms...")
        logger.info("Mel cache invalidation triggered (%s). Clearing cached spectrograms.", reason)

        for audio_path in self.data_list:
            mel_cache_path, meta_cache_path = self._mel_cache_paths(audio_path)
            self._remove_file_safely(mel_cache_path)
            self._remove_file_safely(meta_cache_path)

    @staticmethod
    def _remove_file_safely(path):
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Failed to remove cache file %s: %s", path, exc)

    def _save_mel_cache(self, path, mel_tensor, metadata):
        mel_cache_path, meta_cache_path = self._mel_cache_paths(path)
        mel_numpy = mel_tensor.detach().cpu().numpy()
        try:
            np.save(mel_cache_path, mel_numpy)
            with open(meta_cache_path, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, sort_keys=True)
        except OSError as exc:
            logger.warning("Failed to save mel cache for %s: %s", path, exc)

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        f0s = torch.zeros((batch_size, self.max_mel_length)).float()
        is_silences = torch.zeros((batch_size, self.max_mel_length)).float()

        for bid, (mel, f0, is_silence) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            f0s[bid, :mel_size] = f0
            is_silences[bid, :mel_size] = is_silence

        if self.max_mel_length > self.min_mel_length:
            random_slice = np.random.randint(
                self.min_mel_length//self.mel_length_step,
                1+self.max_mel_length//self.mel_length_step) * self.mel_length_step + self.min_mel_length
            mels = mels[:, :, :random_slice]
            f0 = f0[:, :random_slice]

        mels = mels.unsqueeze(1)
        return mels, f0s, is_silences


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    
    dataset = MelDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
