from src.audio.load_audio import load_audio

# Replace with a real mp3/wav file you have in your project
audio_file = "data/raw/soe_temple.mp3"

y, sr = load_audio(audio_file, duration=20)

print("Waveform shape:", y.shape)
print("Sampling rate:", sr)
print("First 10 samples:", y[:10])
print("Min amplitude:", y.min())
print("Max amplitude:", y.max())
