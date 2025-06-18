import os

import compression.fractal_wavelet_compression_core as fwc
import util.common as common
import pandas as pd
import util.math as mymath

# PARAMETERS
DECOMPOSITION_LEVEL = 4
BLOCK_HEIGHT = 3
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT  # NOT COUNTING B_0
WAVELET_FAMILY = 'db24'
file_types = ["sound", "speech", "confutatis", "badinerie", "rondo-alla-turca"]
STORES_ONLY_ALPHA = False
n_samples = [i for i in range(10, 18)]
csv_path = "different_n_DL3.csv"

all_results = []
for file_type in file_types:
    print(f"Processing file type {file_type}")
    results_per_file = []
    for n_sample in n_samples:
        # READING ORIGINAL SIGNAL
        original_signal, _, _ = common.read_example_file(n_samples=2**n_sample, file_type=file_type, suppress_logs=True)

        # DWT on signal (wavelet decomposition)
        wavelet_coefficients = fwc.wavelet_decomposition(original_signal, WAVELET_FAMILY, DECOMPOSITION_LEVEL,
                                                         suppress_logs=True)

        # ENCODING
        # codded_data = (to_be_stored, coded_blocks)
        codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT,
                                          store_only_alpha=STORES_ONLY_ALPHA, suppress_logs=True)

        # DECODING
        decoded_signal = fwc.decode(codded_data, WAVELET_FAMILY, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, 2**n_sample,
                                    suppress_logs=True)

        mse = mymath.calculate_mse(decoded_signal, original_signal)
        psnr = mymath.calculate_psnr(mse)
        all_results.append({
            'file_type': file_type,
            'n_sample': n_sample,
            'psnr': psnr
        })

# Create one big DataFrame
df = pd.DataFrame(all_results)

# Find row with highest PSNR for each file_type
best_per_file_type = df.loc[df.groupby('file_type')['psnr'].idxmax()]
print(best_per_file_type)
file_exists = os.path.isfile(csv_path)
df.to_csv(csv_path, mode='a', header=not file_exists, index=False)