import time
import os

import numpy as np
import pywt

import compression.fractal_wavelet_compression_core as fwc
import util.common as common
import util.math as mymath
import pandas as pd

from util.wavFile import read_wav_file

# PARAMETERS
wavelets = pywt.wavelist(kind='discrete')
DECOMPOSITION_LEVEL = 4
BLOCK_HEIGHT = 3
STORES_ONLY_ALPHA = False
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT
n_samples = 2 ** 12
csv_path = f"wavelet_occurrences.csv"
source_folder_prefix = r"C:\magisterka\master_thesis\resources\testing_data"

print("Reading files...")
data_files = []
for file_name in os.listdir(source_folder_prefix):
    data_files.append(source_folder_prefix + "\\" + file_name)
print(f"Files to process: {len(data_files)}")

# warmup
common.testing_warmup()

all_results = []
pre_offset = 10 ** 5
offset = 10 ** 4
samples_to_take = 2 ** 12
file_id = 0
frames_to_take = 3
skipped = 0
progress_incrementor = 1 if int(0.05 * 300) == 0 else int(0.05 * 300)
for index, file_path in enumerate(data_files):

    if index % progress_incrementor == 0:
        print(f"\rProgress: {round(index * 100 / 300, 2)}%.", end="", flush=True)
    
    # read file
    _, channels = read_wav_file(file_path)
    channel = channels[0]

    channel = np.trim_zeros(channel)
    channel = channel[pre_offset:]
    # ensure separation between frames
    all_samples = len(channel)
    if all_samples < (offset + samples_to_take) * frames_to_take:
        offset = 10 ** 3
        print(f"WRONG LEN {file_path}")
    if all_samples < (offset + samples_to_take) * frames_to_take:
        offset = 10 ** 2
        print(f"AGAIN: {file_path}")
    if all_samples < (offset + samples_to_take) * frames_to_take:
        print(f"SKIP: {file_path}")
        skipped += 1
        continue

    previous_len = 0
    for i in range(frames_to_take):
        best_psnr = -100000
        best_mre = 100000
        best_wavelet_psnr = None
        best_wavelet_mre = None
        for WAVELET_FAMILY in wavelets:
            original_signal = channel[(offset + samples_to_take) * i:(offset + samples_to_take) * i + samples_to_take]
            # validation
            if i != 0 and (previous_len != len(original_signal)):
                print(f"INVALID PREVIOUS LENGTH: {file_path}")


            if len(original_signal) != samples_to_take:
                print(f"INVALID LENGTH: {file_path}")
            previous_len = len(original_signal)

            wavelet_coefficients = fwc.wavelet_decomposition(original_signal, WAVELET_FAMILY, DECOMPOSITION_LEVEL,
                                                             suppress_logs=True)

            start_time = time.time()
            codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT,
                                              store_only_alpha=STORES_ONLY_ALPHA, suppress_logs=True)

            decoded_signal = fwc.decode(codded_data, WAVELET_FAMILY, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, samples_to_take,
                                        suppress_logs=True)
            end_time = time.time()
            mse = mymath.calculate_mse(original_signal, decoded_signal)
            mre = mymath.calculate_mre(original_signal, decoded_signal)
            psnr = mymath.calculate_psnr(mse)

            if mre < best_mre:
                best_mre = mre
                best_wavelet_mre = WAVELET_FAMILY
            if psnr > best_psnr:
                best_psnr = psnr
                best_wavelet_psnr = WAVELET_FAMILY

        all_results.append({
            'file_id': file_id,
            'offset': i*offset,
            'best_psnr': best_psnr,
            'best_wavelet_psnr': best_wavelet_psnr,
            'best_mre': best_mre,
            'best_wavelet_mre': best_wavelet_mre,
        })
        file_id += 1

# Create one big DataFrame
df = pd.DataFrame(all_results)

# save
print(df)
print(f"finished, files skipped: {skipped}")
file_exists = os.path.isfile(csv_path)
df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
