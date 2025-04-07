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


def feature_extractor(signal: np.ndarray, metadata: dict) -> dict:
    """
    Extracts statistical data from whole signal frame
    :param signal: 1d array with signal samples
    :param metadata: signal metadata ("fs")
    :return: dictionary with extracted statistics
    """
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
    """
    Splits long singal into frames
    :param signal: 1d array with signal samples
    :return: list of separated frames from signal of size equals to FRAME_SIZE
    """
    FRAME_SIZE = 2 ** 15
    if len(signal) < FRAME_SIZE:
        return [signal]
    else:
        return [signal[i: i + FRAME_SIZE] for i in range(0, len(signal), FRAME_SIZE)]


def find_best_wavelet_per_frame(frame: np.ndarray, wavelets: list, dl: int, rbl: int, bh: int, frame_ind: int) -> str:
    """
    Finds best (highest PSNR) wavelet per frame
    :param frame: 1D array with frame samples
    :param wavelets: list of wavelets functions to test
    :param dl: decomposition level number
    :param rbl: range block level at which range blocks are
    :param bh: block height indicating how many levels are taking into IFS proces
    :param frame_ind: index of frame to be tested, for loggining
    :return: the best wavelet for provided frame
    """
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


def process_frame(frame: np.ndarray, frame_ind: int, metadata: dict, wavelets: list, dl: int, rbl: int,
                  bh: int) -> dict | None:
    """
    Processes frame for finding the best wavelet to calculated features - made for parallel processing
    :param metadata: signal metadata ("fs")
    :param frame: 1D array with frame samples
    :param wavelets: list of wavelets functions to test
    :param dl: decomposition level number
    :param rbl: range block level at which range blocks are
    :param bh: block height indicating how many levels are taking into IFS proces
    :param frame_ind: index of frame to be tested, for loggining
    :return: dictionary with extracted statistics with the best possible wavelet
    """
    # skip empty frames
    if np.all(frame == 0):
        print(f"Frame {frame_ind} skipped (all samples == 0).")
        return None

    features = feature_extractor(frame, metadata)

    # find best wavelet
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


def process_channel(channel: np.ndarray, channel_ind: int, metadata: dict, total_channels: int, wavelets: list, dl: int,
                    rbl: int, bh: int) -> list:
    """
    Processing single channel for finding the best wavelet to calculated features - parallel processing implementation
    :param channel: single channel from .wav file containing samples
    :param channel_ind: index of channel
    :param metadata: signal metadata ("fs")
    :param total_channels: how many channels in original .wav file
    :param wavelets: list of wavelets functions to test
    :param dl: decomposition level number
    :param rbl: range block level at which range blocks are
    :param bh: block height indicating how many levels are taking into IFS proces
    :return: list of extracted statistics with the best possible wavelet per frame extracted from original channel
    """
    print(f"Processing channel {channel_ind + 1}/{total_channels}...")
    channel = np.trim_zeros(channel, "fb")
    frames = split_into_frames(channel)

    # create a partial function with fixed parameters
    process_frame_partial = functools.partial(
        process_frame,
        metadata=metadata,
        wavelets=wavelets,
        dl=dl,
        rbl=rbl,
        bh=bh
    )

    # process frames in parallel
    results = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(process_frame_partial, frame, i) for i, frame in enumerate(frames)]
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)

    print(f"Finished processing channel {channel_ind + 1}/{total_channels}.")
    return results


def create_dataset(files: list):
    """
    Extract staitstical features from all .wav files (per each channel/frame of FRAME_SIZE)
    and calculate the wavelets which yields best PSNR while compressing
    :param files: list of .wav files to be processed
    """
    decomposition_level = 3
    block_height = 2
    range_blocks_level = decomposition_level - block_height
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

        for channel_ind, channel in enumerate(channels):
            file_results.extend(
                process_channel(channel, channel_ind, metadata, len(channels), wavelets, decomposition_level,
                                range_blocks_level, block_height))

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


folder_path = '../resources/'
folder_save_path = '../ml/datasets/best_wavelet'

files = [f for f in os.listdir(folder_path)]
create_dataset(files)
