import numpy as np
import matplotlib.pyplot as plt

# Parameters
sr = 22_050  # sampling rate (samples per second)
duration = 0.01  # 10 ms of audio
freq = 440  # A4 note (440 Hz)

# Continuous time (high resolution)
t_cont = np.linspace(0, duration, 1000)
wave_cont = np.sin(2 * np.pi * freq * t_cont)

# Sampled time (22,050 samples per second)
t_samples = np.linspace(0, duration, int(sr * duration))
wave_samples = np.sin(2 * np.pi * freq * t_samples)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t_cont, wave_cont, label="Continuous wave")
plt.stem(t_samples, wave_samples, linefmt="r-", markerfmt="ro", basefmt=" ", label="Samples")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Continuous wave vs digital samples")
plt.show()
