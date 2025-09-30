"""Runtime-selectable F0 extraction backends.

This module centralises all fundamental F0 (pitch) extraction utilities and
provides a single ``F0Extractor`` facade that can iterate over multiple
backends until one succeeds.  The intent is to make it trivial to experiment
with different pitch trackers (speed vs. accuracy trade-offs) without having to
modify the training or evaluation pipelines.

Each backend is wrapped in a light-weight class that hides the dependency
surface and exposes a consistent ``compute`` method returning a NumPy array
containing Hertz values.  Optional dependencies are imported lazily and
reported via ``BackendUnavailableError`` so callers can gracefully fall back to
another backend defined in the configuration.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Dict, List, Optional

import numpy as np


LOGGER = logging.getLogger(__name__)


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot be constructed due to missing deps."""


class BackendComputationError(RuntimeError):
    """Raised when a backend fails to compute an F0 trajectory."""


@dataclasses.dataclass
class BackendResult:
    f0: np.ndarray
    backend_name: str
    details: Optional[str] = None


class BaseF0Backend:
    """Base class for all backends."""

    backend_type: str = "base"

    def __init__(
        self,
        name: str,
        sr: int,
        hop_length: int,
        config: Optional[Dict] = None,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.sample_rate = sr
        self.hop_length = hop_length
        self.config = config or {}
        self.verbose = verbose

    @property
    def frame_period_ms(self) -> float:
        return float(self.config.get("frame_period_ms", self.hop_length * 1000.0 / self.sample_rate))

    @property
    def cache_key(self) -> str:
        suffix = self.config.get("cache_key_suffix")
        if suffix:
            return f"{self.name}-{suffix}"
        return self.name

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {message}")
        LOGGER.debug("[%s] %s", self.name, message)

    # ------------------------------------------------------------------
    # API surface expected from subclasses
    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class PyWorldBackend(BaseF0Backend):
    backend_type = "pyworld"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import pyworld as pw  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailableError("pyworld is not installed") from exc

        self._pw = pw
        self.algorithm = self.config.get("algorithm", "harvest")
        self.fallback_algorithm = self.config.get("fallback", "dio")
        self.use_stonemask = bool(self.config.get("stonemask", True))

    def _run_algorithm(self, algorithm: str, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        frame_period = self.frame_period_ms
        if algorithm == "harvest":
            return self._pw.harvest(audio, sr, frame_period=frame_period)
        if algorithm == "dio":
            return self._pw.dio(audio, sr, frame_period=frame_period)
        if algorithm == "stonemask":
            # stonemask is a refinement step and expects the initial F0 curve
            f0, t = self._pw.harvest(audio, sr, frame_period=frame_period)
            return self._pw.stonemask(audio, f0, t, sr), t
        raise ValueError(f"Unsupported PyWorld algorithm: {algorithm}")

    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = int(sr or self.sample_rate)
        signal = audio.astype("double", copy=False)
        algorithm = self.algorithm
        f0, t = self._run_algorithm(algorithm, signal, sr)
        if np.count_nonzero(f0) < self.config.get("min_voiced_frames", 5) and self.fallback_algorithm:
            self.log(
                f"Primary algorithm '{algorithm}' returned too few voiced frames; switching to '{self.fallback_algorithm}'."
            )
            f0, t = self._run_algorithm(self.fallback_algorithm, signal, sr)
        if self.use_stonemask and algorithm != "stonemask":
            f0 = self._pw.stonemask(signal, f0, t, sr)
        return f0.astype(np.float64)


class CrepeBackend(BaseF0Backend):
    backend_type = "crepe"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import crepe  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailableError("crepe is not installed") from exc

        self._crepe = crepe
        self.model = self.config.get("model", "full")
        self.viterbi = bool(self.config.get("viterbi", True))
        self.center = bool(self.config.get("center", True))
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.05))
        self.step_size_ms = float(self.config.get("step_size_ms", self.frame_period_ms))

    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = int(sr or self.sample_rate)
        waveform = audio.astype(np.float32, copy=False)
        time, frequency, confidence, _ = self._crepe.predict(
            waveform,
            sr,
            model=self.model,
            step_size=self.step_size_ms,
            center=self.center,
            viterbi=self.viterbi,
            verbose=0,
        )
        f0 = frequency.astype(np.float64)
        if self.confidence_threshold > 0:
            f0[confidence < self.confidence_threshold] = 0.0
        self.log(f"CREPE analysed {len(time)} frames with mean confidence {confidence.mean():.3f}.")
        return f0


class RMVPEBackend(BaseF0Backend):
    backend_type = "rmvpe"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import torch
            from rmvpe import RMVPE  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailableError("rmvpe is not installed") from exc

        self._torch = torch
        device = self.config.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = self.config.get("model_path")
        is_half = bool(self.config.get("is_half", device == "cuda"))
        self._model = RMVPE(model_path, is_half=is_half, device=device) if model_path else RMVPE(is_half=is_half, device=device)
        self._model.eval()

    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = int(sr or self.sample_rate)
        waveform = audio.astype(np.float32, copy=False)
        tensor = self._torch.from_numpy(waveform).unsqueeze(0)
        with self._torch.no_grad():  # pragma: no cover - heavy dependency
            f0 = self._model.infer_from_audio(tensor, sr)
        return f0.squeeze(0).cpu().numpy().astype(np.float64)


class PraatBackend(BaseF0Backend):
    backend_type = "praat"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import parselmouth  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailableError("parselmouth (Praat bindings) is not installed") from exc

        self._parselmouth = parselmouth
        self.min_pitch = float(self.config.get("min_pitch", 40.0))
        self.max_pitch = float(self.config.get("max_pitch", 1100.0))
        self.silence_threshold = float(self.config.get("silence_threshold", 0.03))
        self.voicing_threshold = float(self.config.get("voicing_threshold", 0.45))
        self.octave_cost = float(self.config.get("octave_cost", 0.01))
        self.pitch_unit = self.config.get("unit", "Hertz")

    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = int(sr or self.sample_rate)
        sound = self._parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch(
            time_step=self.frame_period_ms / 1000.0,
            pitch_floor=self.min_pitch,
            pitch_ceiling=self.max_pitch,
            method=self.config.get("method", "ac"),
            silence_threshold=self.silence_threshold,
            voicing_threshold=self.voicing_threshold,
            octave_cost=self.octave_cost,
            unit=self.pitch_unit,
        )
        f0 = pitch.selected_array[self.pitch_unit.lower()].astype(np.float64)
        return f0


class ReaperBackend(BaseF0Backend):
    backend_type = "reaper"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import pyreaper  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailableError("pyreaper is not installed") from exc

        self._reaper = pyreaper
        self.min_f0 = float(self.config.get("min_pitch", 40.0))
        self.max_f0 = float(self.config.get("max_pitch", 600.0))
        self.do_high_pass = bool(self.config.get("do_high_pass", True))

    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = int(sr or self.sample_rate)
        signal = audio.astype(np.float64, copy=False)
        frame_period_sec = self.frame_period_ms / 1000.0
        times, f0, _, vuv = self._reaper.reaper(
            signal,
            sr,
            minf0=self.min_f0,
            maxf0=self.max_f0,
            frame_period=frame_period_sec,
            do_high_pass=self.do_high_pass,
        )
        f0 = f0.astype(np.float64)
        f0[vuv == 0] = 0.0
        return f0


class ParselmouthBackend(PraatBackend):
    """Alias backend for clarity when users explicitly select 'parselmouth'."""

    backend_type = "parselmouth"


BACKEND_REGISTRY = {
    "pyworld": PyWorldBackend,
    "crepe": CrepeBackend,
    "rmvpe": RMVPEBackend,
    "praat": PraatBackend,
    "reaper": ReaperBackend,
    "parselmouth": ParselmouthBackend,
}


def _normalise_backend_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class F0Extractor:
    """Facade for computing F0 with configurable backend fallbacks."""

    DEFAULT_SEQUENCE = [
        {"name": "pyworld_harvest", "type": "pyworld", "config": {"algorithm": "harvest", "fallback": "dio"}},
        {"name": "pyworld_dio", "type": "pyworld", "config": {"algorithm": "dio", "fallback": None}},
    ]

    def __init__(
        self,
        sr: int,
        hop_length: int,
        config: Optional[Dict] = None,
        verbose: bool = False,
    ) -> None:
        self.sample_rate = sr
        self.hop_length = hop_length
        self.verbose = verbose
        config = config or {}
        self.bad_f0_threshold = int(config.get("bad_f0_threshold", 5))
        self.zero_fill_value = float(config.get("zero_fill_value", 0.0))

        sequence = config.get("backend_order") or [entry["name"] for entry in self.DEFAULT_SEQUENCE]
        backends_config = config.get("backends", {})

        # Merge defaults with user configuration so that the default sequence
        # always works even when users omit explicit backend definitions.
        defaults: Dict[str, Dict] = {entry["name"]: entry for entry in self.DEFAULT_SEQUENCE}
        merged_sequence: List[Dict] = []
        for raw_name in sequence:
            if isinstance(raw_name, dict):
                merged_sequence.append(raw_name)
                continue
            name = str(raw_name)
            backend_cfg = backends_config.get(name, {})
            default_entry = defaults.get(name, {"name": name, "type": name})
            merged_entry = {**default_entry, **backend_cfg}
            merged_entry.setdefault("name", name)
            merged_entry.setdefault("type", merged_entry.get("backend", merged_entry.get("type", name)))
            merged_entry.setdefault("enabled", True)
            merged_sequence.append(merged_entry)

        self.backends: List[BaseF0Backend] = []
        errors: List[str] = []
        for entry in merged_sequence:
            if not entry.get("enabled", True):
                continue
            name = entry.get("name") or entry.get("type")
            backend_type = (entry.get("type") or entry.get("backend") or "pyworld").lower()
            backend_cls = BACKEND_REGISTRY.get(backend_type)
            if backend_cls is None:
                errors.append(f"Unknown backend type '{backend_type}' (entry: {name})")
                continue
            backend_name = _normalise_backend_name(str(name))
            backend_config = entry.get("config") or {k: v for k, v in entry.items() if k not in {"name", "type", "backend", "enabled"}}
            try:
                instance = backend_cls(
                    name=backend_name,
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                    config=backend_config,
                    verbose=verbose,
                )
            except BackendUnavailableError as exc:
                message = f"Skipping backend '{backend_name}': {exc}"
                errors.append(message)
                LOGGER.warning(message)
                continue
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"Failed to initialise backend '{backend_name}': {exc}")
                LOGGER.exception("Failed to initialise backend '%s'", backend_name)
                continue
            self.backends.append(instance)

        if not self.backends:
            error_message = "No usable F0 backends are configured."
            if errors:
                error_message += " Details: " + "; ".join(errors)
            raise RuntimeError(error_message)

        cache_tag_components = [_normalise_backend_name(backend.cache_key) for backend in self.backends]
        self.cache_identifier = "-" + "_".join(cache_tag_components) if cache_tag_components else ""

    # ------------------------------------------------------------------
    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> BackendResult:
        sr = int(sr or self.sample_rate)
        for backend in self.backends:
            try:
                f0 = backend.compute(audio, sr)
            except BackendUnavailableError as exc:
                LOGGER.warning("Backend '%s' became unavailable: %s", backend.name, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Backend '%s' failed with error", backend.name)
                continue

            if f0 is None:
                continue
            f0 = np.asarray(f0, dtype=np.float64)
            if np.count_nonzero(f0) < self.bad_f0_threshold:
                LOGGER.warning(
                    "Backend '%s' returned only %d voiced frames; attempting next backend.",
                    backend.name,
                    int(np.count_nonzero(f0)),
                )
                continue
            return BackendResult(f0=f0, backend_name=backend.name)

        raise BackendComputationError("All configured F0 backends failed to produce a valid contour.")

    # ------------------------------------------------------------------
    def align_length(self, values: np.ndarray, target_frames: int) -> np.ndarray:
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

    # ------------------------------------------------------------------
    def describe_backends(self) -> List[str]:
        return [backend.name for backend in self.backends]


def build_f0_extractor(
    sr: int,
    hop_length: int,
    config: Optional[Dict] = None,
    verbose: bool = False,
) -> F0Extractor:
    return F0Extractor(sr=sr, hop_length=hop_length, config=config, verbose=verbose)

