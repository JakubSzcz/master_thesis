import numpy as np
from IFS.IFS_1d import IFS
import util.common as common

# parameters
n_samples = 1000  # samples in base signal
n_range = 100 # for printing only
n_domains = int(n_range/2)  # number of domains blocks # number of domains blocks

# sinus generation
t = np.linspace(0, 1, n_samples)
original_signal = np.sin(t * 2 * np.pi)

ifs = IFS()

#common.print_signal(original_signal, "Bloki źródłowe", plot_ranges_size=int(n_samples/n_range)*2)

R, D, D_start_sample = ifs.generate_range_domain_blocks(n_domains, original_signal, domains_overlaps=False)

codded_data = ifs.encode(R, D, D_start_sample)

decoded_signal = ifs.decode(codded_data, printing_flag=False)

common.print_attr_vs_orig(decoded_signal, original_signal, n_domains=n_domains, n_range=n_range, n_samples=n_samples)
