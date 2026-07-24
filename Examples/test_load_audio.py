from src.audio.load_audio import load_audio

# Replace with a real mp3/wav file you have in your project
audio_file = "/home/iaskasan/Synesthesia-AI-1/dataset/autotagging_moodtheme_audio-low-00.tar"

y, sr = load_audio(audio_file, duration=10)

print("Waveform shape:", y.shape)
print("Sampling rate:", sr)
print("First 10 samples:", y[:10])
