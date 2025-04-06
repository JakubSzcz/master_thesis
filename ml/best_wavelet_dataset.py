import os
import numpy as np
import librosa
import antropy as ant
import pandas as pd
from numba import njit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import functools

import compression.fractal_wavelet_compression_core as fwc
from util.wavFile import read_wav_file
import util.math as mymath


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
    if n_samples < 2 ** 11:
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=metadata["fs"], n_fft=n_samples // 4)
    else:
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
    FRAME_SIZE = 2 ** 15
    if len(signal) < FRAME_SIZE:
        return [signal]
    else:
        return [signal[i: i + FRAME_SIZE] for i in range(0, len(signal), FRAME_SIZE)]

def find_best_wavelet_per_frame(frame, wavelets, dl, rbl, bh, frame_ind):
    print(f"Finding best wavelet for frame {frame_ind}...")
    max_psnr = -10000
    best_wavelet = None

    for wavelet in wavelets:
        wavelet_coefficients = fwc.wavelet_decomposition(frame, wavelet, dl, suppress_logs=True)
        codded_data = fwc.encode_wavelets(wavelet_coefficients, rbl, bh, suppress_logs=True)
        decoded_signal = fwc.decode(codded_data, wavelet, rbl, bh, len(frame), suppress_logs=True)

        psnr = mymath.calculate_psnr(mymath.calculate_mse(decoded_signal, frame))
        if psnr > max_psnr:
            max_psnr = psnr
            best_wavelet = wavelet

    print(f"Best PSNR = {max_psnr} for wavelet: {best_wavelet} for frame {frame_ind}")
    return best_wavelet


def process_frame(frame, frame_ind, metadata, wavelets, dl, rbl, bh):

    # Skip empty frames
    if np.all(frame == 0):
        print(f"Frame {frame_ind} skipped (all samples == 0).")
        return None

    # Extract features
    features = feature_extractor(frame, metadata)

    # Find best wavelet
    try:
        best_wavelet = find_best_wavelet_per_frame(frame, wavelets, dl, rbl, bh, frame_ind)
        features["wavelet"] = best_wavelet
        return features
    except ValueError:
        print(f"Frame {frame_ind} skipped (transformation failed).")
        return None
    except Exception:
        print(f"Frame {frame_ind} skipped (couldn't find best wavelet).")
        return None


def process_channel(channel, channel_ind, metadata, total_channels, wavelets, dl, rbl, bh):
    # channel, channel_ind, metadata, total_channels, wavelets, dl, rbl, bh = channel_data

    print(f"Processing channel {channel_ind + 1}/{total_channels}...")
    channel = np.trim_zeros(channel, "fb")
    frames = split_into_frames(channel)

    # Create a partial function with fixed parameters
    process_frame_partial = functools.partial(
        process_frame,
        metadata=metadata,
        wavelets=wavelets,
        dl=dl,
        rbl=rbl,
        bh=bh
    )

    # Process frames in parallel using ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(process_frame_partial, frame, i) for i, frame in enumerate(frames)]
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)
    print(f"Finished processing channel {channel_ind + 1}/{total_channels}.")
    return results


def create_dataset(files):
    DECOMPOSITION_LEVEL = 3
    BLOCK_HEIGHT = 2
    RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT
    wavelets = ['bior1.1', 'bior1.3', 'bior1.5', 'bior4.4', 'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif11', 'coif12',
                'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db15', 'db16',
                'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29',
                'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1',
                'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4',
                'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']

    for file in files:
        file_results = []
        print(f"Processing {file}...")
        metadata, channels = read_wav_file("../resources/" + file)
        #
        # # Prepare data for parallel processing
        # channel_data = [
        #     (channel, i, metadata, len(channels), wavelets, DECOMPOSITION_LEVEL, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT)
        #     for i, channel in enumerate(channels)
        # ]
        #
        # # Process channels in parallel
        # with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        #     channel_results = list(executor.map(process_channel, channel_data))

        for channel_ind, channel in enumerate(channels):
            file_results.extend(
                process_channel(channel, channel_ind, metadata, len(channels), wavelets, DECOMPOSITION_LEVEL,
                                RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT))

        # Create and save DataFrame
        if file_results:
            df = pd.DataFrame(file_results,
                              columns=["n_samples", "zero_cross_rate", "mean", "variance", "std", "skewness", "energy",
                                       "mean_bandwidth", "min_bandwidth", "max_bandwidth", "variance_bandwidth",
                                       "std_bandwidth", "spectral_entropy", "wavelet"])
            df.to_csv(folder_save_path + f'/best_wavelet_{file.split(".")[0]}.csv', index=False)
            print(f"Results saved for file {file}.")
        else:
            print("No valid frames were processed.")


if __name__ == "__main__":
    # Replace with your file list
    folder_path = '../resources/'
    folder_save_path = '../ml/datasets/best_wavelet'

    # List only files
    files = [f for f in os.listdir(folder_path)]
    create_dataset(files)
