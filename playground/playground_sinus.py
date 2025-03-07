import numpy as np
from IFS.IFS_1d import IFS
import util.common as common

# parameters
n_samples = 1000  # samples in base signal
n_domains = 20  # number of domains blocks

# signals
t = np.linspace(0, 1, n_samples)
original = np.sin(t * 2 * np.pi)
random_vector = np.random.uniform(0, 1, n_samples)

ifs = IFS()

R, D, D_start_sample = ifs.generate_range_domain_blocks(n_domains, original)

codded = ifs.encode(R, D, D_start_sample)

decoded = ifs.decode(codded)

common.print_attr_vs_orig(decoded, original, n_domains=n_domains, n_range=10)
