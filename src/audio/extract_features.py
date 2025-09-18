#!/usr/bin/env python3
"""
Module for extracting features from audio signals.
"""

import librosa
import numpy as np


def extract_tempo(y, sr):
    """
    Estimate tempo (BPM) of an audio signal.
    """
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo


def extract_mfcc(y, sr, n_mfcc=13):
    """
    Extract MFCC features from audio.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
        n_mfcc (int): Number of MFCCs to return.

    Returns:
        np.ndarray: MFCC feature matrix (n_mfcc x frames).
    """
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


# TODO: add more features (spectral centroid, chroma, etc.)
