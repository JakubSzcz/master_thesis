import compression.fractal_wavelet_copression as fwc
import numpy as np
import sounddevice as sd
import util.common as common

# PARAMETERS
DECOMPOSITION_LEVEL = 6
BLOCK_HEIGHT = 4
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT
WAVELET_FAMILY = 'coif12'
file_types = ("sound", "speech")
file_type = file_types[0]
title_appendix = f"for {file_type} with {WAVELET_FAMILY}"

n_samples = 2 ** 16

# READING ORIGINAL SIGNAL
original_signal, audio_samplerate = common.read_example_file(n_samples=n_samples, file_type=file_type)

# DWT on signal (wavelet decomposition)
wavelet_coefficients = fwc.wavelet_decomposition(original_signal, WAVELET_FAMILY, DECOMPOSITION_LEVEL)

# ENCODING
# codded_data = (to_be_stored, coded_blocks)
codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT)

# DECODING
decoded_signal = fwc.decode(codded_data, WAVELET_FAMILY, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, n_samples)

# PRINTING
common.print_signal(original_signal, "original signal")
common.print_signal(decoded_signal, "reconstructed signal", title_appendix=title_appendix)
common.print_attr_vs_orig(decoded_signal, original_signal, title_appendix=title_appendix)

# PLAYING
sd.play(original_signal, samplerate=audio_samplerate, blocking=True)
sd.play(np.array(decoded_signal), samplerate=audio_samplerate, blocking=True)
