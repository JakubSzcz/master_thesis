import time
import os

import compression.fractal_wavelet_compression_core as fwc
import util.common as common
import pandas as pd

# PARAMETERS
DECOMPOSITION_LEVEL = 4
BLOCK_HEIGHT = 3
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT
WAVELET_FAMILY = 'db4'
file_type = "sound"
STORES_ONLY_ALPHA = False
n_samples = [i for i in range(10, 18)]

type_of_computation = "faiss_v2"
#type_of_computation = "bf_v2"
csv_path = f"faiss_vs_bf_{type_of_computation}.csv"

# warmup
common.testing_warmup()

all_results = []
for iteration in range(5):
    print(f"Processing iteration {iteration}")

    for n_sample in n_samples:
        # READING ORIGINAL SIGNAL
        original_signal, _, _ = common.read_example_file(n_samples=2 ** n_sample, file_type=file_type,
                                                         suppress_logs=True)

        # DWT on signal (wavelet decomposition)
        wavelet_coefficients = fwc.wavelet_decomposition(original_signal, WAVELET_FAMILY, DECOMPOSITION_LEVEL,
                                                         suppress_logs=True)

        # ENCODING
        start_time = time.time()
        codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT,
                                          store_only_alpha=STORES_ONLY_ALPHA, suppress_logs=True)

        end_time = time.time()
        print(f"samples: {n_sample} - time: {round(end_time - start_time, 5)}")
        all_results.append({
            'iteration': iteration,
            'n_sample': n_sample,
            "exec_time": end_time - start_time,
        })

# Create one big DataFrame
df = pd.DataFrame(all_results)

# save
print(f"results for {type_of_computation} finished")
print(df)
file_exists = os.path.isfile(csv_path)
df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
