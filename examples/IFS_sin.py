import numpy as np
from IFS.IFS_1d import IFS
import util.common as common

# parameters
n_samples = 1000  # samples in base signal
n_domains = 4  # number of domains blocks
n_range = 10  # for printing only

# sinus generation
t = np.linspace(0, 1, n_samples)
original_signal = np.sin(t * 2 * np.pi)

ifs = IFS()

R, D, D_start_sample = ifs.generate_range_domain_blocks(n_domains, original_signal)

codded_data = ifs.encode(R, D, D_start_sample)

decoded_signal = ifs.decode(codded_data)

common.print_attr_vs_orig(decoded_signal, original_signal, n_domains=n_domains, n_range=n_range)
