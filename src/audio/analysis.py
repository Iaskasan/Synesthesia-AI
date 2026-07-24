"""High-level audio analysis used by the application."""

from dataclasses import asdict, dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioFeatures:
    """A compact, serializable description of an audio excerpt."""

    tempo_bpm: float
    rms_energy: float
    spectral_centroid_hz: float
    spectral_contrast: float
    zero_crossing_rate: float
    chroma: tuple[float, ...]
    mfcc: tuple[float, ...]

    def to_dict(self) -> dict[str, float | list[float]]:
        result = asdict(self)
        result["chroma"] = list(self.chroma)
        result["mfcc"] = list(self.mfcc)
        return result


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def analyze_audio(y: np.ndarray, sr: int, n_mfcc: int = 13) -> AudioFeatures:
    """Extract fixed-size features suitable for inference and model training."""
    if y.size == 0:
        raise ValueError("The audio excerpt is empty.")
    if sr <= 0:
        raise ValueError("The sample rate must be positive.")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return AudioFeatures(
        tempo_bpm=float(np.asarray(tempo).squeeze()),
        rms_energy=_mean(librosa.feature.rms(y=y)),
        spectral_centroid_hz=_mean(
            librosa.feature.spectral_centroid(y=y, sr=sr)
        ),
        spectral_contrast=_mean(
            librosa.feature.spectral_contrast(y=y, sr=sr)
        ),
        zero_crossing_rate=_mean(librosa.feature.zero_crossing_rate(y)),
        chroma=tuple(float(value) for value in np.mean(chroma, axis=1)),
        mfcc=tuple(float(value) for value in np.mean(mfcc, axis=1)),
    )

