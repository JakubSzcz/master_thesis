import numpy as np
import time
import pandas as pd

import compression.fractal_wavelet_copression as fwc
from util.wavFile import read_wav_file
import util.math as mymath


def read_signals(files: list):
    signals = {}
    for file in files:
        signal_id = file.split("/")[-1].split(".")[0]
        metadata, signal = read_wav_file(file)
        if metadata["channels"] > 1:
            for channel_n, channel in enumerate(signal):
                signals[f"{signal_id}_{channel_n}"] = channel[:2**12]
        else:
            signals[f"{signal_id}"] = signal[0][:2**12]
    return signals


def create_datasets(wavelets_coefficients: list, r_blocks_level: int, block_height: int, signal_id: str):
    print(f"Creating datasets for {signal_id}...")
    start_time_enc = time.time()
    # prepare blocks
    a_coeffs = wavelets_coefficients[1:]
    r_matrix, d_matrix = fwc.generate_r_d(r_blocks_level, block_height, a_coeffs)
    n_range, _ = r_matrix.shape
    # encode
    progress_incrementor = 1 if int(0.05 * n_range) == 0 else int(0.05 * n_range)
    # (signal_ind, r_ind, d_ind, mean, var, std, skew, energy)
    dataset = []
    for r_i, r in enumerate(r_matrix):
        # progress logging
        if r_i % progress_incrementor == 0:
            print(f"\rProgress: {round(r_i * 100 / n_range, 2)}%.", end="", flush=True)
        # parameters to encode
        distance_min = 1000000
        d_index = 0
        # find the best base domain from domain pool to transform into range block with min d_rms
        for d_i, d in enumerate(d_matrix):
            alpha, beta = mymath.calculate_alpha_beta(d, r)
            transformed = mymath.transform(alpha, beta, d)
            distance_calc = mymath.distance(d, transformed)

            if distance_calc < distance_min:
                d_index = d_i
                distance_min = distance_calc

        # save parameters for each range block
        attr = mymath.compute_features(r)
        dataset.append([signal_id, r_i, d_index, attr[0], attr[1], attr[2], attr[3], attr[4]])
    print("\rProgress: 100%.", flush=True)
    print(f"encoding finished with {round(time.time() - start_time_enc, 2)}s for {signal_id}")
    return pd.DataFrame(dataset,
                        columns=["signal_id", "r_ind", "d_ind", "mean", "variance", "std", "skewness", "energy"])


# PARAMETERS
DECOMPOSITION_LEVEL = 5
BLOCK_HEIGHT = 4
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT
WAVELET_FAMILY = 'coif12'

files = ["../resources/en_speech.wav", "../resources/sound.wav"]

signals = read_signals(files)
wavelet_coefficients = {}
for signal_id, signal in signals.items():
    wavelet_coefficients[signal_id] = fwc.wavelet_decomposition(signal, WAVELET_FAMILY, DECOMPOSITION_LEVEL)
dfs = {}
for signal_id, wavelet_coefficients in wavelet_coefficients.items():
    df = create_datasets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, signal_id)
    df.to_csv(f"./datasets/{signal_id}.csv", index=False)

