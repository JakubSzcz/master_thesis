import numpy as np


def distance(x, y):
    """
    calculate the Euclidean distance between two vectors
    :param x: vector 1
    :param y: vector 2
    :return: distance
    """
    return np.linalg.norm(x - y)


def calculate_alpha_beta(x, z):
    """
    returns minimal value of alpha and beta for affine transformation
    :param x: Vector x from which affine transformation is applied -> DOMAIN
    :param z: Vector z to which affine transformation will lead -> RANGE
    :return:
    tuple: alpha and beta.
    """
    y = np.ones(len(x))

    # Denominator for both formulas
    denominator = np.dot(x, y) ** 2 - np.dot(x, x) * np.dot(y, y)

    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute alpha and beta.")

    # Calculate alpha
    alpha = (np.dot(y, z) * np.dot(x, y) - np.dot(y, y) * np.dot(x, z)) / denominator

    # Calculate beta
    beta = (np.dot(x, y) * np.dot(x, z) - np.dot(x, x) * np.dot(y, z)) / denominator

    return alpha, beta


def transform(alpha, beta, x):
    """
    performs affine transformation on the x with alpha and beta defined as: alpha * x + beta
    :param alpha: scaling transformation coefficient
    :param beta: translating transformation coefficient
    :param x: vector x on which affine transformation is applied
    :return: transformed vector
    """
    return np.multiply(x, alpha) + np.multiply(beta, np.ones(len(x)))


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
