import matplotlib.pyplot as plt
import numpy as np
import random
import util.math as mymath
from util.wavFile import read_wav_file
import sounddevice as sd

# parameters
n_samples = 50000 # samples in base signal
block_size = 10 # size of range block (domain block = *2)
n_domains = 1000 # number of Domains blocks
n_range = int(n_samples / block_size) # number of range block
decoding_iterations = 1 # iterations number while decoding
wave_offset = 100000
d_threshold = 0.01


# generating base image
t = np.linspace(0, 1, n_samples)
audio_meta_data, X = read_wav_file("../resources/example.wav")
audio_samplerate = audio_meta_data["fs"]
X = X[0][wave_offset:n_samples+wave_offset]
#X = np.sin(t * 2 * np.pi)


# range blocks, covering all the signal, no overlapping allowed
R = [X[i:i + block_size] for i in range(0, n_samples, block_size)]
# domain blocks, overlapping allowed
D = []
for _ in range(n_domains):
    ind = random.randint(0, n_samples - (block_size * 2))
    D.append(X[ind:ind + block_size * 2])

# CODING
# unique D used in coding
d_unique = set()
# d_i, alpha, beta,
codded = []
print("start coding")
for r_i, r in enumerate(R):
    min_rms = 10000
    fit_d_i = 0
    fit_alpha = 1
    fit_beta = 0
    if r_i % 5 == 0:
        print(f"{r_i+1}/{n_range}")
    for d_i, d in enumerate(D):
        d_down = mymath.downsample(d)
        alpha, beta = mymath.calculate_alpha_beta(d_down, r)
        d_rms_cal = mymath.d_rms(d_down, mymath.transform(alpha, beta, d_down))
        if d_rms_cal < min_rms:
            fit_d_i = d_i
            d_unique.add(d_i)
            fit_alpha = alpha
            fit_beta = beta
            min_rms = d_rms_cal
        if min_rms < d_rms_cal:
            break

    codded.append((fit_d_i, fit_alpha, fit_beta))
print("end coding")
print(f"Number of unique d used in codding process {len(d_unique)}/{n_domains}")

#DECODING
print("start decoding")
R_reconstructed = []
for i in range(n_range):
    temp = mymath.downsample(D[codded[i][0]])
    for _ in range(decoding_iterations):
        temp = mymath.transform(codded[i][1], codded[i][2], temp)
    R_reconstructed.append(temp)

R_reconstructed = np.array(R_reconstructed)
print("end decoding")
print(f"RMS between reconstructed and original data = {mymath.d_rms(R_reconstructed.flatten(), X)}")


# PLOTTING
plt.grid()
plt.title("Original signal")
for ind in range(n_range):
    plt.plot(t[ind * block_size: (ind + 1) * block_size], R[ind])
plt.show()


plt.grid()
plt.title("Reconstructed signal")
for ind in range(n_range):
    plt.plot(t[ind * block_size: (ind + 1) * block_size], R_reconstructed[ind])
plt.show()

sd.play(np.array(R).flatten(), samplerate=audio_samplerate, blocking=True)

sd.play(np.array(R_reconstructed).flatten(), samplerate=audio_samplerate, blocking=True)
