import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# params
fs = 300
t_step = 1/fs
f0 = 5
f1 = 20

# time domain
t = np.arange(0, 1, t_step)

# signals
y_1 = np.sin(2*np.pi*f0*t) + np.sin(2*np.pi*f1*t)
part_1=np.sin(2*np.pi*f0*t[0:len(t)//2])
part_2=np.sin(2*np.pi*f1*t[len(t)//2:])
y_2 = np.concatenate((part_1, part_2))

# fft
fft_original = np.fft.fft(y_1)
freq_original = np.fft.fftfreq(len(t), d=1/fs)

idx = freq_original >= 0
X_mag = np.abs(fft_original[idx]) * 2 / len(t)   # Normalize magnitude
freqs_pos = freq_original[idx]

plt.plot(t, y_1, label="f(t)=sin(2π5t) + sin(2π20y)")
plt.plot(t, y_2, label="g(t)={'sin(2π5t); t<0.5, sin(2π20t); t>0.5'}")
plt.grid()
plt.legend()
plt.title("Przebieg czasowy sygnałów f(t) oraz g(t)")
plt.xlabel("t [s]")
plt.ylabel("Amplituda")
plt.show()

plt.plot(freqs_pos, X_mag)
plt.title("Pokrywające się widmo Fouriera sygnałów f(t) oraz g(t)")
plt.grid()
plt.xlabel("f [Hz]")
plt.show()

# STFT parameters
nperseg = 64  # Window length
noverlap = 48  # Overlap between windows
nfft = 256     # FFT length

# Calculate STFT for both signals
f1, t1, Zxx1 = signal.stft(y_1, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
f2, t2, Zxx2 = signal.stft(y_2, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

# Plot spectrograms
plt.figure(figsize=(12, 10))
plt.suptitle('Spektrogramy STFT sygnałów f(t) oraz g(t)', fontsize=16)
# Signal y_1 spectrogram
plt.subplot(2, 1, 1)
plt.pcolormesh(t1, f1, np.abs(Zxx1), shading='gouraud', cmap='viridis')
plt.colorbar(label='Amplituda')
plt.title('f(t)=sin(2π5t) + sin(2π20t)')
plt.ylabel('f [Hz]')
plt.xlabel('t [s]')
plt.ylim([0, 50])  # Limit y-axis to focus on our frequencies of interest

# Signal y_2 spectrogram
plt.subplot(2, 1, 2)
plt.pcolormesh(t2, f2, np.abs(Zxx2), shading='gouraud', cmap='viridis')
plt.colorbar(label='Amplituda')
plt.title('g(t)={sin(2π5t); t<0.5, sin(2π20t); t>0.5}')
plt.ylabel('f [Hz]')
plt.xlabel('t [s]')
plt.ylim([0, 50])  # Limit y-axis to focus on our frequencies of interest

plt.tight_layout()
plt.show()

