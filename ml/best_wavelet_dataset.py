import os

import asyncio
import numpy as np
import librosa
import antropy as ant
import pandas as pd
import pywt
from numba import njit, prange

import compression.fractal_wavelet_compression_core as fwc
from util.wavFile import read_wav_file
import util.math as mymath

folder_path = '../resources/'
folder_save_path = '../ml/datasets/best_wavelet'

# List only files
files = [f for f in os.listdir(folder_path)]


def feature_extractor(signal, metadata):
    # time based
    n_samples = len(signal)
    signal = np.array(signal)
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(signal.astype(float)))
    mean = np.mean(signal)
    variance = np.var(signal)
    std = np.std(signal)
    skewness = np.mean((signal - mean) ** 3) / (std ** 3 + 1e-8)  # Skewness
    energy = np.sum(signal ** 2)

    # frequency based
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=metadata["fs"])
    mean_bandwidth = np.mean(bandwidth)
    variance_bandwidth = np.var(bandwidth)
    std_bandwidth = np.std(bandwidth)
    min_bandwidth = np.min(bandwidth)
    max_bandwidth = np.max(bandwidth)
    spectral_entropy = ant.spectral_entropy(signal, sf=metadata["fs"], method='fft', normalize=True)

    features = {
        "n_samples": n_samples,
        "zero_cross_rate": zcr_mean,
        "mean": mean,
        "variance": variance,
        "std": std,
        "skewness": skewness,
        "energy": energy,
        "mean_bandwidth": mean_bandwidth,
        "min_bandwidth": min_bandwidth,
        "max_bandwidth": max_bandwidth,
        "variance_bandwidth": variance_bandwidth,
        "std_bandwidth": std_bandwidth,
        "spectral_entropy": spectral_entropy
    }

    return features
@njit
def split_into_frames(signal: np.ndarray) -> list:
    return [signal[i: i + FRAME_SIZE] for i in range(0, len(signal), FRAME_SIZE)]


DECOMPOSITION_LEVEL = 3
BLOCK_HEIGHT = 2
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT
FRAME_SIZE = 2 ** 15


data = []
wavelets = pywt.wavelist(kind='discrete')
# datasets
for file in files:
    print("Processing", file)
    metadata, channels = read_wav_file("../resources/" + file)
    for channel in channels:
        frames = split_into_frames(channel)
        for ind_frame, frame in enumerate(frames):
            if np.all(frame == 0):
                continue
            try:
                features = feature_extractor(frame, metadata)
                max_psnr = -10000
                best_wavelet = None
                for ind_wave, wavelet in enumerate(wavelets):
                    wavelet_coefficients = fwc.wavelet_decomposition(frame, wavelet, DECOMPOSITION_LEVEL, suppress_logs=True)
                    codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, suppress_logs=True)
                    decoded_signal = fwc.decode(codded_data, wavelet, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, len(frame), suppress_logs=True)

                    psnr = mymath.calculate_psnr(mymath.calculate_mse(decoded_signal, frame))
                    if psnr > max_psnr:
                        max_psnr = psnr
                        best_wavelet = wavelet
                        print(f"psnr: {psnr} for wavelet: {wavelet}")
                    if ind_wave % 5 == 0:
                        print(f"{ind_wave} wavelets checked for {ind_frame} frames procesed of {len(frames)} for file {file}")
                if ind_frame % 5 == 0:
                    print(f"{ind_frame} frames procesed of {len(frames)} for file {file}")
                data.append([features, best_wavelet])
            except Exception as e:
                print(e)

df = pd.DataFrame(data, columns=['features', 'wavelet'])
df.to_csv(folder_save_path + '/best_wavelet.csv', index=False)