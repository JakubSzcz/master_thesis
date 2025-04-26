import matplotlib.pyplot as plt
import numpy as np
import pywt

import util.math as mymath
from util.wavFile import read_wav_file


def print_attr_vs_orig(attractor: list | np.ndarray, original: list | np.ndarray, n_range: int = None,
                       n_domains: int = None, title: str = None, title_appendix: str = None):
    """
    Prints attractor vs original signals at the same plot.
    :param attractor: reconstructed signal/attractor to print
    :param original: original signal to print
    :param n_range: optional number of range blocks (for title purpose)
    :param n_domains: optional number of domain blocks (for title purpose)
    :param title: optional custom title
    :param title_appendix: optional custom title appendix for default title
    :return: plots attractor vs original signals and calculate Euclidian distance from both vectors
    """
    plt.figure()
    plt.grid()
    if title is None:
        if n_range is None or n_domains is None:
            title = "Attractor vs Original function"
        else:
            title = f"Attractor vs Original function; n_d = {n_domains}, n_r = {n_range}."

    if title_appendix is not None:
        title += " "
        title += title_appendix
    plt.title(title)
    plt.plot(attractor, label="Attractor")
    plt.plot(original, label="Original function", linestyle="--")
    plt.xlabel("Samples")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    print_mesures(attractor, original)


def print_mesures(reconstructed_signal: np.ndarray, original_signal: np.ndarray):
    """
    Prints mesures: L2 norm, Mean Square Error, Peak Signal to Noise Ratio
    :param reconstructed_signal: signal after reconstruction
    :param original_signal: original signal
    """
    mse = mymath.calculate_mse(reconstructed_signal, original_signal)
    euclidian_distance = mymath.distance(np.array(reconstructed_signal), original_signal)
    psnr = mymath.calculate_psnr(mse)
    print(f"Euclidian distance = {euclidian_distance}")
    print(f"MSE = {mse}")
    print(f"PSNR = {round(psnr, 3)} dB")
    return mse, psnr


def get_compression_rate(n_samples: int, block_height: int, wavelet_family: str, stores_only_alpha: bool = False,
                         bit_wise: bool = True, bit_depth: int = 16):
    """
    Prints compression rate
    :param n_samples: number of samples in original signal
    :param block_height: how many wavelet decomposition levels are to be encoded
    :param wavelet_family: family of wavelets e.g. db10
    :param stores_only_alpha: if only alpha parameter of affine transforms were saved
    :param bit_wise: flag indicating to consider compression rate in bytes, not samples numbers
    :param bit_depth: how many bits per single sample were used in original signal
    :return: calculated compression rate
    """

    filter_len = pywt.Wavelet(wavelet_family).dec_len
    n_ifs_parameters = n_samples
    for i in range(1, block_height + 1):
        n_ifs_parameters = (n_ifs_parameters + filter_len - 1) // 2

    coefficients_to_be_stored = (n_ifs_parameters + filter_len - 1) // 2 * 2

    n_ifs_parameters = n_ifs_parameters * 2 if stores_only_alpha else n_ifs_parameters * 3
    # TODO consider storing floats on 32 bits not 64
    # assuming float -> 64 bits
    float_bits_size = 64
    if bit_wise:
        n_samples *= bit_depth * 8
        coefficients_to_be_stored *= float_bits_size
        n_ifs_parameters *= float_bits_size
    compression_rate = n_samples / (n_ifs_parameters + coefficients_to_be_stored)
    to_print = f"Compression rate = {round(compression_rate, 3)}"
    if bit_wise:
        to_print += " (bit wise)"
    print(to_print)
    return compression_rate


def print_signal(signal: list | np.ndarray, title: str, plot_ranges_size: int = None, title_appendix: str = None):
    """
    Prints single signal.
    :param signal: signal to print
    :param title: custom title for plot
    :param plot_ranges_size: optional size of sub-blocks for printing vertical lines spacing range blocks
    :param title_appendix: optional custom title appendix for default title
    :return: plots signal
    """
    plt.figure()
    plt.grid()
    if title_appendix is not None:
        title += " "
        title += title_appendix
    plt.title(title)
    plt.plot(signal)
    if plot_ranges_size is not None:
        for p in range(0, len(signal), plot_ranges_size):
            plt.axvline(x=p, color='red', linestyle='--', alpha=0.7)
    plt.show()


def read_example_file(n_samples: int = 2 ** 12, samples_offset: int = 100000, file_type: str = "sound",
                      suppress_logs: bool = True):
    """
    Shortcut function for reading example audio files.
    :param suppress_logs: stop printing logs
    :param n_samples: how many samples to read (default 2 ** 12)
    :param samples_offset: how many samples to skip/shift (default 100000)
    :param file_type: type of audio file to read (default "sound")
    :return: Samples of single audio channel of audio file
    """
    if file_type == "sound":
        file = "../resources/sound.wav"
    elif file_type == "confutatis":
        file = "../resources/confutatis.wav"
    elif file_type == "rondo-alla-turca":
        file = "../resources/rondo-alla-turca.wav"
    elif file_type == "badinerie":
        file = "../resources/badinerie.wav"
    else:
        file = "../resources/en_speech.wav"
    audio_meta_data, signal = read_wav_file(file)
    audio_samplerate = audio_meta_data["fs"]
    bit_depth = audio_meta_data["byteDepth"]
    # reads only one channel with offset
    signal = signal[0][samples_offset:n_samples + samples_offset]

    if not suppress_logs:
        print(
            f"Audio parameters: fs = {audio_samplerate}Hz, samples = {n_samples}, bit depth = {8 * bit_depth}bits/sample, "
            f"duration = {round((1 / audio_samplerate) * n_samples, 2)}s.")
    return signal, audio_samplerate, bit_depth
