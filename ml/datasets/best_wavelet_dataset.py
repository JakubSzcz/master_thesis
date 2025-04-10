import os
import numpy as np
import antropy as ant
import pandas as pd
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
    signal = np.array(signal)
    mean = np.mean(signal)
    variance = np.var(signal)
    std = np.std(signal)
    skewness = np.mean((signal - mean) ** 3) / (std ** 3 + 1e-8)  # Skewness
    energy = np.sum(signal ** 2)
    spectral_entropy = ant.spectral_entropy(signal, sf=metadata["fs"], method='fft', normalize=True)

    features = {
        "mean": mean,
        "variance": variance,
        "std": std,
        "skewness": skewness,
        "energy": energy,
        "spectral_entropy": spectral_entropy
    }

    return features


def feature_extractor_v2(signal: np.ndarray, metadata: dict) -> dict:
    """
    Extracts frequency domain from whole signal frame
    :param signal: 1d array with signal samples
    :param metadata: signal metadata ("fs")
    :return: dictionary with extracted statistics
    """
    # time based
    fs = metadata["fs"]

    signal = np.array(signal)

    # fft frequency beans
    fft_output = np.fft.fft(signal)  # symmetric
    frequency_base = np.fft.fftfreq(len(signal), d=1 / fs)

    idx = frequency_base >= 0
    fft_output_real = np.abs(fft_output[idx]) * 2 / len(signal)  # normalize magnitude, only positive frequencies

    return {"frequency_beans": fft_output_real.tolist()}


def split_into_frames(signal: np.ndarray, frame_size: int) -> list:
    """
    Splits long signal into frames, pad with zeros, if cannot be split equally
    :param frame_size: size of frame to be processed
    :param signal: 1d array with signal samples
    :return: list of separated frames from signal of size equals to FRAME_SIZE
    """
    if len(signal) < frame_size:
        pad_width = frame_size - len(signal)
        return [np.pad(signal, (0, pad_width), mode='constant')]
    else:
        return [signal[i: i + frame_size] for i in range(0, len(signal), frame_size)]


def find_best_wavelet_per_frame(frame: np.ndarray, wavelets: list, dl: int, rbl: int, bh: int, frame_ind: int) -> (
        str, np.float64):
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
    # print(f"Finding best wavelet for frame {frame_ind}...")
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

    # print(f"Best PSNR = {max_psnr} for wavelet: {best_wavelet} for frame {frame_ind}")
    return best_wavelet, max_psnr


def process_frame(frame: np.ndarray, frame_ind: int, metadata: dict, wavelets: list, dl: int, rbl: int,
                  bh: int, total_frames: int) -> dict | None:
    """
    Processes frame for finding the best wavelet to calculated features - made for parallel processing
    :param total_frames: how many frames are in channel
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
    # features = feature_extractor_v2(frame, metadata)

    # find best wavelet
    try:
        progress_indicator = int(0.05 * total_frames)
        if frame_ind % progress_indicator == 0:
            print(f"\tFrames processed: {frame_ind} / {total_frames}")
        best_wavelet, max_psnr = find_best_wavelet_per_frame(frame, wavelets, dl, rbl, bh, frame_ind)
        features["wavelet"] = best_wavelet
        features["psnr"] = max_psnr
        return features
    except ValueError:
        print(f"\tFrame {frame_ind} skipped (transformation failed).")
        return None
    except Exception:
        print(f"\tFrame {frame_ind} skipped (couldn't find best wavelet).")
        return None


def process_channel(channel: np.ndarray, channel_ind: int, metadata: dict, total_channels: int, wavelets: list, dl: int,
                    rbl: int, bh: int, frame_size: int) -> list:
    """
    Processing single channel for finding the best wavelet to calculated features - parallel processing implementation
    :param frame_size: size of frame to be processed
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
    frames = split_into_frames(channel, frame_size)
    total_frames = len(frames)
    print(f"Frames in channel: {total_frames}")

    # create a partial function with fixed parameters
    process_frame_partial = functools.partial(
        process_frame,
        metadata=metadata,
        wavelets=wavelets,
        dl=dl,
        rbl=rbl,
        bh=bh,
        total_frames=total_frames,
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


def create_dataset(files: list, folder_path, folder_save_path, wavelets_v2: bool = False):
    """
    Extract statistical features from all .wav files (per each channel/frame of FRAME_SIZE)
    and calculate the wavelets which yields best PSNR while compressing
    :param wavelets_v2: used trimmed list of wavelet
    :param files: list of .wav files to be processed
    """
    decomposition_level = 3
    block_height = 2
    FRAME_SIZE = 2 ** 11
    range_blocks_level = decomposition_level - block_height
    wavelets = ['db34', 'db36', 'db35', 'coif17', 'db38', 'db37', 'db32', 'coif16',
                'db33', 'db27', 'db18', 'db23', 'db31', 'db19', 'coif15', 'sym19']

    print(f"FILES: {files}")
    for file in files:
        file_results = []
        print(f"Processing {file}...")
        metadata, channels = read_wav_file(folder_path + file)

        for channel_ind, channel in enumerate(channels):
            file_results.extend(
                process_channel(channel, channel_ind, metadata, len(channels), wavelets, decomposition_level,
                                range_blocks_level, block_height, FRAME_SIZE))

        # Create and save DataFrame
        if file_results:
            if wavelets_v2:
                df = pd.DataFrame(file_results,
                                  columns=["frequency_beans", "wavelet"])
            else:
                df = pd.DataFrame(file_results,
                                  columns=["mean", "variance", "std", "skewness",
                                           "energy", "psnr", "wavelet", "spectral_entropy"])
            df.to_csv(folder_save_path + f'/best_wavelet_{file.split(".")[0]}.csv', index=False)
            print(f"Results saved for file {file}.")
        else:
            print("No valid frames were processed.")


if __name__ == "__main__":
    folder_path = './../../resources/'
    folder_save_path = 'best_wavelet'
    files = [f for f in os.listdir(folder_path)]
    create_dataset(files, folder_path, folder_save_path, False)
