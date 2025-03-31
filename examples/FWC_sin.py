import numpy as np
import compression.fractal_wavelet_copression as fwc
import util.common as common

# PARAMETERS
DECOMPOSITION_LEVEL = 3
BLOCK_HEIGHT = 2
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT
n_samples = 1000  # samples in base signal
WAVELET_FAMILY = 'db2'

# sinus generation
t = np.linspace(0, 1, n_samples)
original_signal = np.sin(t *4 * np.pi)

# DWT on signal (wavelet decomposition)
wavelet_coefficients = fwc.wavelet_decomposition(original_signal, WAVELET_FAMILY, DECOMPOSITION_LEVEL)

# ENCODING
codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT)

# DECODING
decoded_signal = fwc.decode(codded_data, WAVELET_FAMILY, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, n_samples)

# PRINTING
common.print_attr_vs_orig(decoded_signal, original_signal)