import numpy as np
import matplotlib.pyplot as plt

fs = 44100
t_step = 1/fs
f0 = 100
f1 = 400

t = np.arange(0, 0.01, t_step)
y = np.sin(2*np.pi*f0*t) + np.sin(2*np.pi*f1*t)

plt.plot(t, y)
plt.show()

fft_original = np.fft.fft(y)
freq_original = np.fft.fftfreq(len(y), d=1/fs)

idx = freq_original >= 0
X_mag = np.abs(fft_original[idx]) * 2 / len(y)   # Normalize magnitude
freqs_pos = freq_original[idx]
print(len(y))
print(len(freqs_pos))

plt.figure(figsize=(10, 4))
plt.plot(freqs_pos, X_mag)
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()