import numpy as np
import pytest

from src.audio.analysis import analyze_audio
from src.audio.extract_features import extract_tempo


def test_analyze_audio_returns_fixed_size_finite_features():
    sample_rate = 22_050
    time = np.arange(sample_rate * 2) / sample_rate
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    features = analyze_audio(audio, sample_rate)

    assert len(features.chroma) == 12
    assert len(features.mfcc) == 13
    assert all(np.isfinite(value) for value in features.to_dict()["mfcc"])
    assert features.rms_energy > 0


def test_analyze_audio_rejects_empty_excerpt():
    with pytest.raises(ValueError, match="empty"):
        analyze_audio(np.array([], dtype=np.float32), 22_050)


def test_extract_tempo_always_returns_float():
    audio = np.zeros(22_050, dtype=np.float32)
    assert isinstance(extract_tempo(audio, 22_050), float)
