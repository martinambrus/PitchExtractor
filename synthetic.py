"""Synthetic speech generation utilities for perfectly-labeled training data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class _GeneratorConfig:
    sample_rate: int
    hop_length: int
    min_duration: float
    max_duration: float
    min_f0: float
    max_f0: float
    num_harmonics: int
    harmonic_decay: float
    vibrato_semitones: float
    vibrato_rate_hz: float
    max_glide_semitones: float
    noise_std: float
    breath_noise_std: float
    unvoiced_probability: float
    unvoiced_min_duration: float
    unvoiced_max_duration: float
    attack_time: float
    release_time: float
    amplitude_min: float
    amplitude_max: float
    jitter_semitones: float
    seed: int


class SyntheticSpeechGenerator:
    """Generate synthetic speech-like waveforms with known F0 contours.

    The generator produces harmonic stacks with smooth glides, vibrato, and
    optional unvoiced regions. The ground-truth F0 contour is returned alongside
    the waveform so the training pipeline can mix perfectly labeled examples
    into the dataset, following the approach used by SwiftF0.
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        mel_transform,
        mean: float,
        std: float,
        config: Optional[Dict] = None,
    ):
        del mel_transform  # Currently unused but kept for compatibility.
        del mean, std
        cfg = dict(config or {})
        self._config = _GeneratorConfig(
            sample_rate=int(sample_rate),
            hop_length=int(hop_length),
            min_duration=float(cfg.get("min_duration", 0.7)),
            max_duration=float(cfg.get("max_duration", 2.5)),
            min_f0=float(cfg.get("min_f0", 60.0)),
            max_f0=float(cfg.get("max_f0", 600.0)),
            num_harmonics=max(1, int(cfg.get("num_harmonics", 6))),
            harmonic_decay=float(cfg.get("harmonic_decay", 1.6)),
            vibrato_semitones=float(cfg.get("vibrato_semitones", 0.35)),
            vibrato_rate_hz=float(cfg.get("vibrato_rate_hz", 5.5)),
            max_glide_semitones=float(cfg.get("max_glide_semitones", 5.0)),
            noise_std=float(cfg.get("noise_std", 0.0025)),
            breath_noise_std=float(cfg.get("breath_noise_std", 0.0015)),
            unvoiced_probability=float(cfg.get("unvoiced_probability", 0.2)),
            unvoiced_min_duration=float(cfg.get("unvoiced_min_duration", 0.04)),
            unvoiced_max_duration=float(cfg.get("unvoiced_max_duration", 0.18)),
            attack_time=float(cfg.get("attack_time", 0.02)),
            release_time=float(cfg.get("release_time", 0.04)),
            amplitude_min=float(cfg.get("amplitude_min", 0.55)),
            amplitude_max=float(cfg.get("amplitude_max", 0.9)),
            jitter_semitones=float(cfg.get("jitter_semitones", 0.05)),
            seed=int(cfg.get("seed", 1729)),
        )

        if self._config.max_duration < self._config.min_duration:
            self._config = self._config.__class__(
                sample_rate=self._config.sample_rate,
                hop_length=self._config.hop_length,
                min_duration=self._config.max_duration,
                max_duration=self._config.min_duration,
                min_f0=self._config.min_f0,
                max_f0=self._config.max_f0,
                num_harmonics=self._config.num_harmonics,
                harmonic_decay=self._config.harmonic_decay,
                vibrato_semitones=self._config.vibrato_semitones,
                vibrato_rate_hz=self._config.vibrato_rate_hz,
                max_glide_semitones=self._config.max_glide_semitones,
                noise_std=self._config.noise_std,
                breath_noise_std=self._config.breath_noise_std,
                unvoiced_probability=self._config.unvoiced_probability,
                unvoiced_min_duration=self._config.unvoiced_min_duration,
                unvoiced_max_duration=self._config.unvoiced_max_duration,
                attack_time=self._config.attack_time,
                release_time=self._config.release_time,
                amplitude_min=self._config.amplitude_min,
                amplitude_max=self._config.amplitude_max,
                jitter_semitones=self._config.jitter_semitones,
                seed=self._config.seed,
            )

        self._rng = np.random.RandomState(self._config.seed)

    def _spawn_rng(self, index: Optional[int]) -> np.random.RandomState:
        if index is None:
            seed = self._rng.randint(0, 2**31 - 1)
        else:
            seed = (self._config.seed + int(index)) % (2**31 - 1)
        return np.random.RandomState(seed)

    def generate(self, index: Optional[int] = None) -> Dict[str, np.ndarray]:
        cfg = self._config
        rng = self._spawn_rng(index)

        duration = rng.uniform(cfg.min_duration, cfg.max_duration)
        num_samples = max(int(duration * cfg.sample_rate), cfg.hop_length * 2)
        t = np.arange(num_samples, dtype=np.float64) / cfg.sample_rate

        start_f0 = rng.uniform(cfg.min_f0, cfg.max_f0)
        glide = rng.uniform(-cfg.max_glide_semitones, cfg.max_glide_semitones)
        end_f0 = np.clip(start_f0 * (2.0 ** (glide / 12.0)), cfg.min_f0, cfg.max_f0)
        log_start = math.log2(start_f0)
        log_end = math.log2(end_f0)
        base_f0 = 2.0 ** np.linspace(log_start, log_end, num_samples)

        vibrato_depth = rng.uniform(0.0, cfg.vibrato_semitones)
        vibrato_phase = rng.uniform(0.0, 2.0 * math.pi)
        vibrato = 2.0 ** (
            (vibrato_depth / 12.0)
            * np.sin(2.0 * math.pi * cfg.vibrato_rate_hz * t + vibrato_phase)
        )

        jitter = 2.0 ** (rng.normal(0.0, cfg.jitter_semitones, size=num_samples) / 12.0)
        inst_freq = base_f0 * vibrato * jitter

        phase_increment = 2.0 * math.pi * inst_freq / cfg.sample_rate
        phase = np.cumsum(phase_increment)

        signal = np.zeros(num_samples, dtype=np.float64)
        for harmonic in range(1, cfg.num_harmonics + 1):
            amplitude = 1.0 / (harmonic ** cfg.harmonic_decay)
            phase_offset = rng.uniform(0.0, 2.0 * math.pi)
            signal += amplitude * np.sin(harmonic * phase + phase_offset)

        attack_samples = max(1, int(cfg.attack_time * cfg.sample_rate))
        release_samples = max(1, int(cfg.release_time * cfg.sample_rate))
        envelope = np.ones(num_samples, dtype=np.float64)
        envelope[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)
        envelope[-release_samples:] *= np.linspace(1.0, 0.0, release_samples)

        signal *= envelope
        signal += rng.normal(0.0, cfg.noise_std, size=num_samples)

        voiced_mask_samples = np.ones(num_samples, dtype=bool)
        if rng.random() < cfg.unvoiced_probability:
            max_segments = max(1, int(duration / cfg.unvoiced_min_duration))
            num_segments = rng.randint(1, max(2, min(4, max_segments + 1)))
            for _ in range(num_segments):
                segment_length = rng.uniform(cfg.unvoiced_min_duration, cfg.unvoiced_max_duration)
                start_time = rng.uniform(0.0, max(1e-3, duration - segment_length))
                start_idx = int(start_time * cfg.sample_rate)
                end_idx = min(num_samples, start_idx + int(segment_length * cfg.sample_rate))
                if end_idx <= start_idx:
                    continue
                voiced_mask_samples[start_idx:end_idx] = False
                signal[start_idx:end_idx] = rng.normal(
                    0.0, cfg.breath_noise_std, size=end_idx - start_idx
                )

        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak
        amplitude = rng.uniform(cfg.amplitude_min, cfg.amplitude_max)
        signal = np.clip(signal * amplitude, -1.0, 1.0)

        frame_indices = np.arange(0, num_samples, cfg.hop_length)
        frame_indices = np.clip(frame_indices, 0, num_samples - 1)
        frame_f0 = inst_freq[frame_indices]

        frame_voiced = []
        for start in frame_indices:
            end = min(num_samples, start + cfg.hop_length)
            voiced_ratio = np.mean(voiced_mask_samples[start:end])
            frame_voiced.append(1.0 if voiced_ratio >= 0.5 else 0.0)
        frame_voiced = np.asarray(frame_voiced, dtype=np.float32)
        frame_f0 = np.asarray(frame_f0, dtype=np.float32)
        frame_f0[frame_voiced < 0.5] = 0.0

        return {
            "waveform": signal.astype(np.float32),
            "f0": frame_f0,
            "voiced_mask": frame_voiced,
            "mel": None,
        }
