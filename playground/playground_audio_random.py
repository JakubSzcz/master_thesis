import numpy as np
import IFS.IFS_1d as ifs
import util.common as common

# parameters
n_samples = 1000  # samples in base signal
block_size = 10  # size of range block (domain block = *2)
n_domains = 200  # number of Domains blocks

# signals
t = np.linspace(0, 1, n_samples)
original = np.sin(t * 2 * np.pi)
#original = chirp(t, f0=1, f1=30, t1=n_samples, method='linear')
random_vector = np.random.uniform(0, 1, n_samples)

print("starting partitioning...")
R, D, D_start_sample = ifs.generate_range_domain_blocks(block_size, n_domains, original)
print("partitioning finished.")

print("starting encoding...")
codded = ifs.encode(R, D, D_start_sample)
print("encoding finished.")

# decoding
print("starting decoding...")
decoded = ifs.decode(codded)
print("decoding finished.")

common.print_attr_vs_orig(original, decoded)