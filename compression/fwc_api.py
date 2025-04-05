import time
import numpy as np
import compression.fractal_wavelet_compression_core as fwc
from util.wavFile import read_wav_file
from pympler import asizeof

# PARAMETERS
DECOMPOSITION_LEVEL = 6
BLOCK_HEIGHT = 3
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT
WAVELET_FAMILY = 'coif10'
STORES_ONLY_ALPHA = False
FRAME_SIZE = 2 ** 16


def split_into_frames(signal: np.ndarray) -> list:
    return [signal[i: i + FRAME_SIZE] for i in range(0, len(signal), FRAME_SIZE)]


def compress_frame(frame: np.ndarray) -> np.ndarray:
    wavelet_coefficients = fwc.wavelet_decomposition(frame, WAVELET_FAMILY, DECOMPOSITION_LEVEL, suppress_logs=True)

    # codded_data = (to_be_stored, coded_blocks)
    codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT,
                                      store_only_alpha=True, suppress_logs=True)

    return codded_data


def compress_file(file_path: str):
    start = time.time()
    file_name = file_path.split("/")[-1].split(".")[0]
    print("Compressing file: " + file_path.split("/")[-1])

    audio_meta_data, signal = read_wav_file(file_path)

    channels = [channel for channel in signal]

    codded_data = []
    for channel in channels:
        channel_codded_data = []
        frames = split_into_frames(channel)
        for frame in frames:
            channel_codded_data.append(compress_frame(frame))
        codded_data.append(channel_codded_data)

    print(f"Compression finished with {round(time.time() - start, 2)}s.")
    return [audio_meta_data["fs"], audio_meta_data["channels"], audio_meta_data["samples_n_per_channel"],
            audio_meta_data["byteDepth"], FRAME_SIZE, BLOCK_HEIGHT,
            RANGE_BLOCKS_LEVEL, WAVELET_FAMILY, file_name, codded_data]


# def decompress_file(encoded_file):
#

encoded_file = compress_file("../resources/confutatis.wav")
print(f"Total size: {asizeof.asizeof(encoded_file) / (1024 ** 2):.2f} MB")