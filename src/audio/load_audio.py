#!/usr/bin/env python3
"""
Module for loading audio files.
"""

from pathlib import Path
from typing import BinaryIO

import librosa


def load_audio(
    source: str | Path | BinaryIO,
    duration: float = 30,
    sample_rate: int = 22_050,
):
    """
    Load an audio file.

    Args:
        source: Path or uploaded file containing supported audio.
        duration (int): Number of seconds to load (default: 30s).
        sample_rate: Target sampling rate, for consistent model features.

    Returns:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
    """
    if duration <= 0:
        raise ValueError("Duration must be positive.")

    y, sr = librosa.load(source, sr=sample_rate, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError("The audio excerpt is empty.")
    return y, sr


# TODO: add error handling (e.g., file not found, unsupported format)
