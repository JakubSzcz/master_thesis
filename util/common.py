import matplotlib.pyplot as plt
import numpy as np
import util.math as mymath
from util.wavFile import read_wav_file


def print_attr_vs_orig(attractor: list | np.ndarray, original: list | np.ndarray, n_range: int = None,
                       n_domains: int = None, title: str = None, title_appendix: str = None):
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

    print(f"Euclidian distance = {mymath.distance(attractor, original)}")
    # print(f"MSE = {mymath.calculate_mse(attractor, original)}")
    # print(f"RMS = {mymath.calculate_rms(attractor, original)}")


def print_signal(signal: list | np.ndarray, title: str, plot_ranges_size: int = None, title_appendix: str = None):
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


def read_example_file(n_samples: int = 2 ** 12, samples_offset: int = 100000, file_type: str = "sound"):
    if file_type == "sound":
        file = "../resources/sound.wav"
    else:
        file = "../resources/en_speech.wav"
    audio_meta_data, signal = read_wav_file(file)
    audio_samplerate = audio_meta_data["fs"]
    # reads only one channel with offset
    signal = signal[0][samples_offset:n_samples + samples_offset]

    print(f"Audio parameters: fs = {audio_samplerate}, samples = {n_samples}, "
          f"duration = {round((1 / audio_samplerate) * n_samples, 2)}s.")
    return signal, audio_samplerate
