import matplotlib.pyplot as plt
import numpy as np
import IFS.IFS_1d as ifs
import util.math as mymath
from scipy.signal import chirp

# parameters
n_samples = 1000  # samples in base signal
block_size = 20  # size of range block (domain block = *2)
n_domains = 200  # number of Domains blocks
n_range = int(n_samples / block_size)  # number of range block
decoding_iterations = 10  # iterations number while decoding

# signals
t = np.linspace(0, 1, n_samples)
original = np.sin(t * 2 * np.pi)
#original = chirp(t, f0=1, f1=30, t1=n_samples, method='linear')
random_vector = np.random.uniform(0, 1, n_samples)

R, D, D_start_sample = ifs.generate_range_domain_blocks(block_size, n_domains, original)

# d_i, alpha, beta,
print("start encoding")
codded = ifs.encode(R, D, D_start_sample)
print("end encoding")

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