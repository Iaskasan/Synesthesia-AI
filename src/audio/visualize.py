#!/usr/bin/env python3
"""
Visualization utilities for audio data.
"""

import librosa.display
import matplotlib.pyplot as plt


def plot_waveform(y, sr):
    """
    Plot the waveform of an audio signal.
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_mfcc(mfccs, sr=22050):
    """
    Plot MFCCs as a heatmap.
    """

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis="time")
    plt.colorbar()
    plt.title("MFCC")
    plt.ylabel("MFCC Coefficient")
    plt.yticks(range(mfccs.shape[0]), range(1, mfccs.shape[0] + 1))  # Label 1–13
    plt.tight_layout()
    plt.show()

