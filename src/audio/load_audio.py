#!/usr/bin/env python3
"""
Module for loading audio files.
"""

import librosa


def load_audio(file_path: str, duration: int = 30):
    """
    Load an audio file.

    Args:
        file_path (str): Path to the audio file (e.g., .mp3, .wav).
        duration (int): Number of seconds to load (default: 30s).

    Returns:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
    """
    y, sr = librosa.load(file_path, duration=duration)
    return y, sr


# TODO: add error handling (e.g., file not found, unsupported format)
