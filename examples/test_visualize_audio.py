from src.audio.load_audio import load_audio
from src.audio.extract_features import extract_mfcc
from src.audio.visualize import plot_waveform, plot_mfcc

audio_file = "data/raw/soe_temple.mp3"

# Load 20s of audio
y, sr = load_audio(audio_file, duration=20)

# Plot waveform
plot_waveform(y, sr)

# Extract MFCCs
mfccs = extract_mfcc(y, sr)

# Plot MFCC heatmap
plot_mfcc(mfccs)
