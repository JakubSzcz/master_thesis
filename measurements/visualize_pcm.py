import numpy as np
import matplotlib.pyplot as plt

# Generate a smooth sinusoidal signal
t = np.linspace(0, 0.2, 500, endpoint=False)
n = range(1,501)
frequency = 5  # 5 Hz
frequency1 = 15  # 5 Hz
amplitude = 1
signal = amplitude * np.sin(2 * np.pi * frequency * t) + np.sin(2 * np.pi * frequency1 * t)

# Simulate PCM quantization
bd = 8
num_levels = 2**bd
signal_min = signal.min()
signal_max = signal.max()
quantization_levels = np.linspace(signal_min, signal_max, num_levels)
quantized_signal = np.round((signal - signal_min) / (signal_max - signal_min) * (num_levels - 1))
quantized_signal = quantization_levels[quantized_signal.astype(int)]

# Plot the original and quantized signals
plt.figure(figsize=(6, 5))
plt.plot(n, signal, label='Sygnał oryginalny', linewidth=2)
plt.step(n, quantized_signal, label='Sygnał PCM', where='mid', linestyle='--', linewidth=2)
plt.title(f'Sygnał oryginalny oraz skwantyzowany sygnał PCM\ndla rozdzielczości bitowej równej {bd}')
plt.xlabel('Numer próbki')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()