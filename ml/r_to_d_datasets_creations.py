import time
import pandas as pd

import compression.fractal_wavelet_copression as fwc
import util.math as mymath
from util.matching import brute_force_r_to_d_matching
from util.wavFile import read_wav_file


def read_signals(files: list) -> dict:
    """
    Reads files samples from each channel
    :param files: list of files to read
    :return: dict with signal_id as key (e.g. sound_n where n is channel number) and samples as values
    """
    signals = {}
    for file in files:
        signal_id = file.split("/")[-1].split(".")[0]
        metadata, signal = read_wav_file(file)
        if metadata["channels"] > 1:
            for channel_n, channel in enumerate(signal):
                signals[f"{signal_id}_{channel_n}"] = channel[:len(channel) // 8]  # too many samples, takes too long
        else:
            signals[f"{signal_id}"] = signal[0]
    return signals


def create_r_to_d_datasets(wavelets_coefficients: list, r_blocks_level: int, block_height: int, signal_id: str):
    """
    Saves datasets with metric for each best possible (with L2 metric) match for r to d to the .csv file
    :param wavelets_coefficients: coefficients of wavelets decomposition of signal
    :param r_blocks_level: level at which range blocks roots are
    :param block_height: how big the range block (tree) is
    :param signal_id: signal identifier e.g. sound_n
    """
    print(f"Creating dataset for {signal_id}...")
    start_time_enc = time.time()
    # prepare blocks
    a_coeffs = wavelets_coefficients[1:]
    r_matrix, d_matrix = fwc.generate_r_d(r_blocks_level, block_height, a_coeffs)
    n_range, _ = r_matrix.shape

    progress_incrementor = 1 if int(0.05 * n_range) == 0 else int(0.05 * n_range)
    dataset = []

    # encode
    for r_i, r in enumerate(r_matrix):
        # progress logging
        if r_i % progress_incrementor == 0:
            print(f"\rProgress: {round(r_i * 100 / n_range, 2)}%.", end="", flush=True)

        d_index, _, _ = brute_force_r_to_d_matching(r, d_matrix, False)

        # compute features for matched blocks
        r_features = mymath.compute_features(r)
        d_features = mymath.compute_features(d_matrix[d_index])

        # save parameters for each matched pair
        # [signal_id, r_index, r_mean, r_variance, r_std, r_skewness, r_energy, d_index, d_mean, d_variance,
        # d_std, d_skewness, d_energy,]
        dataset.append([signal_id, r_i, r_features[0], r_features[1], r_features[2], r_features[3], r_features[4],
                        d_index, d_features[0], d_features[1], d_features[2], d_features[3], d_features[4]])

    print("\rProgress: 100%.", flush=True)
    print(f"encoding finished with {round(time.time() - start_time_enc, 2)}s for {signal_id}")
    df = pd.DataFrame(dataset,
                      columns=["signal_id", "r_ind", "r_mean", "r_variance", "r_std", "r_skewness", "r_energy",
                               "d_ind", "d_mean", "d_variance", "d_std", "d_skewness", "d_energy"])

    # save dataset file
    df.to_csv(f"./datasets/{signal_id}.csv", index=False)
    print(f"{signal_id}.csv saved.")


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
    create_r_to_d_datasets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, signal_id)
