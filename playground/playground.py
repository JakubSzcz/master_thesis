import matplotlib.pyplot as plt
import numpy as np
import random
import util.math as mymath
from util.wavFile import read_wav_file

# parameters
n_samples = 10000 # samples in base signal
block_size = 100 # size of range block (domain block = *2)
n_domains = 1000 # number of Domains blocks
n_range = int(n_samples / block_size) # number of range block
decoding_iterations = 1 # iterations number while decoding


# generating base image
t = np.linspace(0, 1, n_samples)
X = read_wav_file("../resources/example.wav")[1][0][1000:n_samples+1000]
#X = np.sin(t * 2 * np.pi)


# range blocks, covering all the signal, no overlapping allowed
R = [X[i:i + block_size] for i in range(0, n_samples, block_size)]
# domain blocks, overlapping allowed
D = []
for _ in range(n_domains):
    ind = random.randint(0, n_samples - (block_size * 2))
    D.append(X[ind:ind + block_size * 2])

# CODING
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
            fit_alpha = alpha
            fit_beta = beta
            min_rms = d_rms_cal

    codded.append((fit_d_i, fit_alpha, fit_beta))
print("end coding")

# DECODING
R_reconstructed = []
for i in range(n_range):
    temp = mymath.downsample(D[codded[i][0]])
    for _ in range(decoding_iterations):
        temp = mymath.transform(codded[i][1], codded[i][2], temp)
    R_reconstructed.append(temp)

R_reconstructed = np.array(R_reconstructed)
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
