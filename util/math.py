import numpy as np
import scipy.signal as signal
from numba import njit


@njit(cache=True)
def distance(x, y):
    """
    calculate the Euclidean distance between two vectors
    :param x: vector 1
    :param y: vector 2
    :return: distance
    """
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    return np.linalg.norm(x - y)


@njit(cache=True)
def calculate_alpha_beta(x, z, only_alpha: bool = False):
    """
    returns minimal value of alpha and beta for affine transformation
    :param only_alpha: flag whether only alpha is returned
    :param x: Vector x from which affine transformation is applied -> DOMAIN
    :param z: Vector z to which affine transformation will lead -> RANGE
    :return: tuple with alpha and beta parameters.
    """
    x = np.ascontiguousarray(x)
    z = np.ascontiguousarray(z)

    y = np.ones(len(x))

    # Denominator for both formulas
    denominator = np.dot(x, y) ** 2 - np.dot(x, x) * np.dot(y, y)

    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute alpha and beta.")

    # Calculate alpha
    alpha = (np.dot(y, z) * np.dot(x, y) - np.dot(y, y) * np.dot(x, z)) / denominator

    if only_alpha:
        return alpha, 0

    # Calculate beta
    beta = (np.dot(x, y) * np.dot(x, z) - np.dot(x, x) * np.dot(y, z)) / denominator

    return alpha, beta


@njit(cache=True)
def transform(alpha, beta, x):
    """
    performs affine transformation on the x with alpha and beta defined as: alpha * x + beta
    :param alpha: scaling transformation coefficient
    :param beta: translating transformation coefficient
    :param x: vector x on which affine transformation is applied
    :return: transformed vector
    """
    if beta == 0:
        return np.multiply(x, alpha)
    return np.multiply(x, alpha) + np.multiply(beta, np.ones(len(x)))


@njit(cache=True)
def compute_features(block: np.ndarray) -> list:
    """
    Computes statistical features for each block
    :param block: retrieved from signal/wavelet coefficients
    :return: list of statistical features of provided block: mean, variance, std, skewness, energy
    """
    block = np.ascontiguousarray(block)
    mean = np.mean(block)
    variance = np.var(block)
    std = np.std(block)
    skewness = np.mean((block - mean) ** 3) / (std ** 3 + 1e-8)  # Skewness
    energy = np.sum(block ** 2)  # Energy of the block

    return [mean, variance, std, skewness, energy]


def downsample(v):
    """
    downsample a vector by averaging 2 adjacent samples
    :param v: vector
    :return: downsampled vector v
    """
    return np.array(v).reshape(-1, 2).mean(axis=1)


def calculate_mse(x, y):
    return np.mean((x - y) ** 2)


def calculate_rms(x, y):
    return np.sqrt(calculate_mse(x, y))


def calculate_psnr(mse, max_sample_value=1.0):
    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10((max_sample_value ** 2) / mse)
    return psnr


def find_highest_frequency(signal_data, fs, threshold_ratio=0.1):
    """
    Returns highest frequency of signal data
    :param signal_data: array of samples of original signal
    :param fs: sampling frequency of signal
    :param threshold_ratio: significant frequencies threshold ratio
    :return: the highest frequency in signal
    """
    # Compute FFT
    fft_values = np.fft.rfft(signal_data)  # Compute FFT (only positive frequencies)
    freqs = np.fft.rfftfreq(len(signal_data), 1 / fs)  # Frequency bins
    magnitude = np.abs(fft_values)  # Get magnitude spectrum

    # Set a threshold to ignore low-amplitude noise
    threshold = max(magnitude) * threshold_ratio
    valid_freqs = freqs[magnitude > threshold]  # Filter significant frequencies

    # Get the highest existing frequency
    return max(valid_freqs) if len(valid_freqs) > 0 else 0


# Low-pass filter design
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Filters signal data using lowpass filter
    :param data: signal to be filtered
    :param cutoff: frequency cutoff for lowpass filter
    :param fs: sampling frequency
    :param order: order of lowpass filter
    :return: filtered signal
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)  # Butterworth filter
    filtered_signal = signal.filtfilt(b, a, data)  # Apply filter with zero-phase
    return filtered_signal
