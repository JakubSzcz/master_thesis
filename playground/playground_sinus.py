import numpy as np
from IFS.IFS_1d import IFS
import util.common as common
import matplotlib.pyplot as plt
import pprint as pp
import pywt

# parameters
n_samples = 1024  # samples in base signal
n_domains = 4  # number of domains blocks

# signals
t = np.linspace(0, 1, n_samples)
random_vector = np.random.uniform(0, 1, n_samples)
original = np.sin(t * 2 * np.pi) + np.sin(t * 5 * np.pi) * np.sin(t * 10 * np.pi)
#original = original + random_vector

lev = 5
results = pywt.wavedec(original, 'db1', level=lev) # [low freq -> b_00, high_1 (deepest) --> a_0, high_2, ... high_n (shallowest) a_3]
count = len(results[4])
print(count)
#pp.pprint(results)

# Reconstruct the original signal
reconstructed = pywt.waverec(results, 'db1')

plt.plot(results[0])
plt.show()
# ifs = IFS()
#
# R, D, D_start_sample = ifs.generate_range_domain_blocks(n_domains, original)
#
# codded = ifs.encode(R, D, D_start_sample)
#
# decoded = ifs.decode(codded)
#
# common.print_attr_vs_orig(decoded, original, n_domains=n_domains, n_range=10)
