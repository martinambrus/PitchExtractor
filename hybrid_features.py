"""Hybrid DSP and neural feature utilities for pitch extraction."""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def _align_length(values: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample ``values`` to ``target_frames`` using linear interpolation."""

    values = np.asarray(values, dtype=np.float64)
    if target_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    if values.size == target_frames:
        return values.astype(np.float32)
    if values.size == 0:
        return np.zeros((target_frames,), dtype=np.float32)

    original_indices = np.linspace(0.0, values.size - 1, num=values.size)
    target_indices = np.linspace(0.0, values.size - 1, num=target_frames)
    resampled = np.interp(target_indices, original_indices, values)

    zero_mask = values == 0.0
    if np.any(zero_mask):
        nearest_indices = np.clip(np.round(target_indices).astype(int), 0, values.size - 1)
        resampled[zero_mask[nearest_indices]] = 0.0

    return resampled.astype(np.float32)


class HybridFeatureBuilder:
    """Compute auxiliary DSP pitch tracks for hybrid DSP/ML training."""

    DEFAULT_ALGORITHMS: Sequence[str] = ("autocorr", "cepstrum", "harmonic", "harvest")

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        frame_length: Optional[int] = None,
        config: Optional[Dict] = None,
        verbose: bool = False,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.config = dict(config or {})
        self.verbose = verbose

        derived_frame_length = frame_length or int(self.config.get("frame_length", 0))
        if not derived_frame_length:
            derived_frame_length = max(int(self.config.get("analysis_window", 0)), self.hop_length * 4)
        if derived_frame_length <= 0:
            raise ValueError("frame_length must be a positive integer")
        self.frame_length = int(derived_frame_length)

        self.window = np.hanning(self.frame_length)
        self.fmin = float(self.config.get("fmin", 50.0))
        self.fmax = float(self.config.get("fmax", 1100.0))
        if self.fmin <= 0 or self.fmax <= 0:
            raise ValueError("fmin and fmax must be positive")
        if self.fmin >= self.fmax:
            raise ValueError("fmin must be smaller than fmax")

        raw_algorithms: Iterable[str] = self.config.get("algorithms", self.DEFAULT_ALGORITHMS)
        algorithms: List[str] = []
        for name in raw_algorithms:
            normalized = str(name).strip().lower()
            if not normalized:
                continue
            if normalized not in {"autocorr", "cepstrum", "harmonic", "harvest"}:
                LOGGER.warning("Unknown DSP feature algorithm '%s'; skipping.", name)
                continue
            algorithms.append(normalized)

        self._pyworld = None
        if "harvest" in algorithms:
            try:
                import pyworld as pw  # type: ignore

                self._pyworld = pw
            except ImportError:
                algorithms.remove("harvest")
                LOGGER.warning("pyworld not available; disabling Harvest DSP feature")
                if verbose:
                    print("[HybridFeatureBuilder] pyworld not available; disabling Harvest feature")

        self.feature_names: List[str] = list(dict.fromkeys(algorithms))
        self.feature_count: int = len(self.feature_names)

        self._harmonic_bins = int(self.config.get("harmonic_bins", 256))
        if self._harmonic_bins <= 0:
            self._harmonic_bins = 256
        self._harmonic_partials = int(self.config.get("harmonic_partials", 5))
        if self._harmonic_partials <= 0:
            self._harmonic_partials = 5
        self._autocorr_threshold = float(self.config.get("autocorr_threshold", 0.1))
        self._cepstrum_threshold = float(self.config.get("cepstrum_threshold", 0.1))

    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, int]:
        return {"algorithms": list(self.feature_names), "frame_length": self.frame_length}

    # ------------------------------------------------------------------
    def compute_aligned(self, waveform: np.ndarray, target_frames: int) -> Optional[np.ndarray]:
        features = self.compute(waveform)
        if not features:
            return None
        aligned: List[np.ndarray] = []
        for name in self.feature_names:
            values = features.get(name)
            if values is None:
                values = np.zeros((0,), dtype=np.float32)
            aligned.append(_align_length(values, target_frames))
        return np.stack(aligned, axis=0) if aligned else None

    # ------------------------------------------------------------------
    def compute(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        if self.feature_count == 0:
            return {}

        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        frames = self._frame_signal(waveform)
        num_frames = frames.shape[0]

        results: Dict[str, np.ndarray] = {}
        for name in self.feature_names:
            if name == "autocorr":
                results[name] = self._autocorrelation_pitch(frames)
            elif name == "cepstrum":
                results[name] = self._cepstrum_pitch(frames)
            elif name == "harmonic":
                results[name] = self._harmonic_summation_pitch(frames)
            elif name == "harvest":
                results[name] = self._harvest_pitch(waveform, num_frames)

        return results

    # ------------------------------------------------------------------
    def _frame_signal(self, waveform: np.ndarray) -> np.ndarray:
        frame_length = self.frame_length
        hop = self.hop_length
        if waveform.size < frame_length:
            pad = frame_length - waveform.size
        else:
            remainder = (waveform.size - frame_length) % hop
            pad = (hop - remainder) % hop
        if pad:
            waveform = np.pad(waveform, (0, pad), mode="reflect")

        num_frames = 1 + (waveform.size - frame_length) // hop
        if num_frames <= 0:
            num_frames = 1
            padded = np.zeros((frame_length,), dtype=waveform.dtype)
            padded[: waveform.size] = waveform[: frame_length]
            frames = padded.reshape(1, -1)
        else:
            try:
                frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_length)[::hop]
            except AttributeError:
                frames = np.stack(
                    [waveform[i : i + frame_length] for i in range(0, waveform.size - frame_length + 1, hop)],
                    axis=0,
                )
        windowed = frames * self.window
        return windowed.astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    def _autocorrelation_pitch(self, frames: np.ndarray) -> np.ndarray:
        min_period = max(1, int(math.floor(self.sample_rate / self.fmax)))
        max_period = max(min_period + 1, int(math.ceil(self.sample_rate / self.fmin)))
        pitches = np.zeros((frames.shape[0],), dtype=np.float32)
        for idx, frame in enumerate(frames):
            autocorr = np.correlate(frame, frame, mode="full")[frame.size - 1 :]
            if autocorr.size <= max_period:
                continue
            autocorr[:min_period] = 0.0
            autocorr[max_period + 1 :] = 0.0
            peak = autocorr.argmax()
            max_value = autocorr[peak]
            if max_value <= 0:
                continue
            if self._autocorr_threshold > 0:
                normalised = max_value / (autocorr[min_period:max_period + 1].max() + 1e-8)
                if normalised < self._autocorr_threshold:
                    continue
            pitches[idx] = float(self.sample_rate) / float(max(peak, 1))
        return pitches

    # ------------------------------------------------------------------
    def _cepstrum_pitch(self, frames: np.ndarray) -> np.ndarray:
        min_period = max(1, int(math.floor(self.sample_rate / self.fmax)))
        max_period = max(min_period + 1, int(math.ceil(self.sample_rate / self.fmin)))
        pitches = np.zeros((frames.shape[0],), dtype=np.float32)
        for idx, frame in enumerate(frames):
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)
            magnitude[magnitude == 0] = 1e-12
            cepstrum = np.fft.irfft(np.log(magnitude))
            if cepstrum.size <= max_period:
                continue
            search_region = cepstrum[min_period : max_period + 1]
            peak = np.argmax(search_region) + min_period
            peak_value = cepstrum[peak]
            if self._cepstrum_threshold > 0 and peak_value < self._cepstrum_threshold:
                continue
            pitches[idx] = float(self.sample_rate) / float(max(peak, 1))
        return pitches

    # ------------------------------------------------------------------
    def _harmonic_summation_pitch(self, frames: np.ndarray) -> np.ndarray:
        frequencies = np.fft.rfftfreq(self.frame_length, d=1.0 / self.sample_rate)
        candidate_freqs = np.linspace(self.fmin, self.fmax, num=self._harmonic_bins)
        pitches = np.zeros((frames.shape[0],), dtype=np.float32)
        for idx, frame in enumerate(frames):
            spectrum = np.abs(np.fft.rfft(frame))
            if not np.any(spectrum):
                continue
            best_score = 0.0
            best_freq = 0.0
            for candidate in candidate_freqs:
                score = 0.0
                for harmonic in range(1, self._harmonic_partials + 1):
                    target = candidate * harmonic
                    if target > frequencies[-1]:
                        break
                    bin_index = np.argmin(np.abs(frequencies - target))
                    score += spectrum[bin_index] / harmonic
                if score > best_score:
                    best_score = score
                    best_freq = candidate
            pitches[idx] = best_freq if best_score > 0 else 0.0
        return pitches

    # ------------------------------------------------------------------
    def _harvest_pitch(self, waveform: np.ndarray, target_frames: int) -> np.ndarray:
        if self._pyworld is None:
            return np.zeros((target_frames,), dtype=np.float32)
        frame_period = self.hop_length * 1000.0 / float(self.sample_rate)
        f0, _ = self._pyworld.harvest(
            waveform.astype("double", copy=False),
            self.sample_rate,
            frame_period=frame_period,
        )
        return _align_length(f0.astype(np.float32), target_frames)


def fuse_f0_predictions(
    neural_f0: torch.Tensor,
    dsp_curves: Optional[torch.Tensor],
    *,
    feature_names: Optional[Sequence[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    neural_weight: float = 1.5,
    octave_tolerance_cents: float = 120.0,
) -> torch.Tensor:
    """Fuse neural and DSP F0 trajectories into a single robust estimate."""

    if dsp_curves is None or dsp_curves.numel() == 0:
        return neural_f0

    if neural_f0.dim() == 3 and neural_f0.size(-1) == 1:
        neural = neural_f0.squeeze(-1)
    else:
        neural = neural_f0

    dsp = dsp_curves
    if dsp.dim() != 3:
        raise ValueError("dsp_curves must be a 3-D tensor of shape (batch, features, frames)")

    batch, features, frames = dsp.shape
    if features == 0:
        return neural

    device = neural.device
    dtype = neural.dtype
    if neural.dim() != 2:
        raise ValueError("neural_f0 must have shape (batch, frames) or (batch, frames, 1)")

    if neural.shape != (batch, frames):
        raise ValueError("neural_f0 and dsp_curves must share batch and frame dimensions")

    stack = torch.cat([neural.unsqueeze(1), dsp], dim=1)

    weight_tensor = torch.ones((batch, features + 1, frames), device=device, dtype=dtype)
    weight_tensor[:, 0, :] = neural_weight

    dsp_weights = torch.ones((features,), device=device, dtype=dtype)
    if weights and feature_names:
        name_to_index = {name: idx for idx, name in enumerate(feature_names)}
        for name, value in weights.items():
            idx = name_to_index.get(name)
            if idx is not None:
                dsp_weights[idx] = float(value)
    weight_tensor[:, 1:, :] = dsp_weights.view(1, -1, 1)

    valid = stack > 0.0
    weight_tensor = weight_tensor * valid.to(dtype)

    if octave_tolerance_cents > 0:
        eps = torch.finfo(dtype).eps
        neural_expanded = neural.unsqueeze(1).expand_as(dsp)
        ratio = torch.where(
            (dsp > 0.0) & (neural_expanded > 0.0),
            dsp / torch.clamp(neural_expanded, min=eps),
            torch.ones_like(dsp),
        )
        cents = 1200.0 * torch.abs(torch.log2(torch.clamp(ratio, min=eps)))
        penalty = torch.exp(-cents / float(octave_tolerance_cents))
        weight_tensor[:, 1:, :] = weight_tensor[:, 1:, :] * penalty

    weighted_sum = (stack * weight_tensor).sum(dim=1)
    weight_sum = weight_tensor.sum(dim=1)
    fused = torch.where(weight_sum > 0, weighted_sum / torch.clamp(weight_sum, min=torch.finfo(dtype).eps), neural)
    fused = torch.where(weight_sum > 0, fused, neural)
    return fused


__all__ = ["HybridFeatureBuilder", "fuse_f0_predictions"]

