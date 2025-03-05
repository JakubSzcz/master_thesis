import matplotlib.pyplot as plt
import numpy as np
import random
import util.math as mymath
from scipy.signal import chirp

# parameters
n_samples = 10000  # samples in base signal
block_size = 10  # size of range block (domain block = *2)
n_domains = 500  # number of Domains blocks
n_range = int(n_samples / block_size)  # number of range block
decoding_iterations = 10  # iterations number while decoding

if n_samples % block_size != 0:
    raise ValueError("n_samples must be a multiple of block_size")

# signals
t = np.linspace(0, 1, n_samples)
original = np.sin(t * 2 * np.pi)
#original = chirp(t, f0=1, f1=30, t1=n_samples, method='linear')
random_vector = np.random.uniform(0, 1, n_samples)

# range blocks
R = [original[i:i + block_size] for i in range(0, n_samples, block_size)]

# unique domains blocks
#D = [original[i:i + block_size*2] for i in range(0, n_samples, block_size*2)] # no overlapping
D = []
D_indexes = []  # temp, for unique indexes, possible overlapping
while len(D) < n_domains:
    ind = random.randint(0, n_samples - (block_size * 2))
    if ind in D_indexes:
        continue
    D_indexes.append(ind)
    D.append(original[ind:ind + (block_size * 2)])

# d_i, alpha, beta,
codded = []
print("start coding")
for r_i, r in enumerate(R):
    min_rms = 10000
    fit_d_i = 0
    fit_alpha = 1
    fit_beta = 0
    if r_i % 5 == 0:
        print(f"{r_i + 1}/{n_range}")
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
print(codded)

# decoding
decoded = [random_vector[i:i + block_size] for i in range(0, n_samples, block_size)]
for _ in range(decoding_iterations):
    temp = np.array(decoded).flatten().tolist()
    for ind, w in enumerate(codded):
        decoded[ind] = mymath.transform(w[1], w[2], mymath.downsample(temp[w[0]:w[0] + (block_size * 2)]))

result = np.array(decoded).flatten()
plt.figure()
plt.grid()
plt.title("Attractor vs Original function")
plt.plot(t, result, label="Attractor")
plt.plot(t, original, label="Original function")
plt.legend()
plt.show()

print(f"RMS = {mymath.d_rms(result, original)}")