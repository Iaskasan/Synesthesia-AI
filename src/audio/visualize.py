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
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
    return fig


def plot_mfcc(mfccs):
    """
    Plot MFCCs as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    image = librosa.display.specshow(mfccs, x_axis="time", ax=ax)
    fig.colorbar(image, ax=ax)
    ax.set_title("MFCC")
    fig.tight_layout()
    return fig
