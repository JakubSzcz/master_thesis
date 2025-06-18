import time
import os

import compression.fractal_wavelet_compression_core as fwc
import util.common as common
import util.math as mymath
import pandas as pd

# PARAMETERS
WAVELET_FAMILY = 'db1'
file_type = "sound"
STORES_ONLY_ALPHA = False
n_samples = [i for i in range(10, 20)]
csv_path = f"n_vs_bh.csv"
# (DL,BH)
decomp_params = [(i + 1, i) for i in range(2, 7)]

# warmup
common.testing_warmup()

all_results = []
for iteration in range(5):
    print(f"Processing iteration {iteration}")

    for n_sample in n_samples:

        for DECOMPOSITION_LEVEL, BLOCK_HEIGHT in decomp_params:
            RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT

            # READING ORIGINAL SIGNAL
            original_signal, _, _ = common.read_example_file(n_samples=2 ** n_sample, file_type=file_type,
                                                             suppress_logs=True)

            # DWT on signal (wavelet decomposition)
            wavelet_coefficients = fwc.wavelet_decomposition(original_signal, WAVELET_FAMILY, DECOMPOSITION_LEVEL,
                                                             suppress_logs=True)

            # ENCODING
            start_time_compression = time.time()
            codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT,
                                              store_only_alpha=STORES_ONLY_ALPHA, suppress_logs=True)

            end_time_compression = time.time()

            start_time_decompression = time.time()
            decoded_signal = fwc.decode(codded_data, WAVELET_FAMILY, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, 2 ** n_sample,
                                        suppress_logs=True)
            end_time_decompression = time.time()
            mse = mymath.calculate_mse(original_signal, decoded_signal)

            print(
                f"N: {n_sample} - BH: {BLOCK_HEIGHT} - time: {round(end_time_decompression - start_time_compression, 5)}")
            all_results.append({
                'iteration': iteration,
                'n': n_sample,
                'bh': BLOCK_HEIGHT,
                'exec_time_comp': end_time_compression - start_time_compression,
                'exec_time_decomp': end_time_decompression - start_time_compression,
                'psnr': mymath.calculate_psnr(mse),
                'mse': mse,
                'e_distance': mymath.distance(original_signal, decoded_signal),
                'mre': mymath.calculate_mre(original_signal, decoded_signal),
            })

# Create one big DataFrame
df = pd.DataFrame(all_results)

# save
print(f"finished")
print(df)
file_exists = os.path.isfile(csv_path)
df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
