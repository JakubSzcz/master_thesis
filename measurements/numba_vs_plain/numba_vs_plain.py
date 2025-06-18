import time
import os

import compression.fractal_wavelet_compression_core as fwc
import util.common as common
import pandas as pd

# PARAMETERS
WAVELET_FAMILY = 'db4'
file_type = "sound"
STORES_ONLY_ALPHA = False
n_samples = [i for i in range(10, 18)]
type_of_computation = "numba_v2"
#type_of_computation = "plain_v2"
csv_path = f"numba_vs_plain_{type_of_computation}.csv"
decomposition_levels = [(3, 2), (4, 3), (5, 4)]


# warmup
common.testing_warmup()

all_results = []
for iteration in range(5):
    print(f"Processing iteration {iteration}")

    for DECOMPOSITION_LEVEL, BLOCK_HEIGHT in decomposition_levels:
        print(f"Processing DL {DECOMPOSITION_LEVEL}")
        RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT

        for n_sample in n_samples:
            # READING ORIGINAL SIGNAL
            original_signal, _, _ = common.read_example_file(n_samples=2 ** n_sample, file_type=file_type,
                                                             suppress_logs=True, samples_offset=10_000)

            start_time = time.time()
            # DWT on signal (wavelet decomposition)
            wavelet_coefficients = fwc.wavelet_decomposition(original_signal, WAVELET_FAMILY, DECOMPOSITION_LEVEL,
                                                             suppress_logs=True)

            # ENCODING
            # codded_data = (to_be_stored, coded_blocks)
            codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT,
                                              store_only_alpha=STORES_ONLY_ALPHA, suppress_logs=True)

            # DECODING
            decoded_signal = fwc.decode(codded_data, WAVELET_FAMILY, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, 2 ** n_sample,
                                        suppress_logs=True)

            end_time = time.time()
            print(f"samples: {n_sample} - time: {round(end_time-start_time,5)}")
            all_results.append({
                'file_type': iteration,
                'n_sample': n_sample,
                'block_height': BLOCK_HEIGHT,
                "exec_time": end_time - start_time,
            })

# Create one big DataFrame
df = pd.DataFrame(all_results)

# save
print(f"results for {type_of_computation} finished")
print(df)
file_exists = os.path.isfile(csv_path)
df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
