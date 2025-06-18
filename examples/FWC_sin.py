import numpy as np
import compression.fractal_wavelet_compression_core as fwc
import util.common as common

# PARAMETERS
DECOMPOSITION_LEVEL = 4
BLOCK_HEIGHT = 3
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT

WAVELET_FAMILY = 'db2'

# sin signal generation
f_0 = 10
f_1 = 4
f_2 = 30
n_samples = 1000
t = np.linspace(0, 1, n_samples)
original_signal = np.sin(t * f_0 * np.pi) + np.sin(t * f_1 * np.pi) + np.sin(t * f_2 * np.pi)

# DWT on signal (wavelet decomposition)
wavelet_coefficients = fwc.wavelet_decomposition(original_signal, WAVELET_FAMILY, DECOMPOSITION_LEVEL)

# ENCODING
codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT)

# DECODING
decoded_signal = fwc.decode(codded_data, WAVELET_FAMILY, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, n_samples,
                            original_signal=original_signal)

# PRINTING
common.print_attr_vs_orig(decoded_signal, original_signal)
