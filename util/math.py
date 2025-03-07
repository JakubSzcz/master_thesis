import numpy as np


def distance(x, y):
    return np.linalg.norm(x - y)


def calculate_alpha_beta(x, z):
    """
    Parameters:
    x (numpy array): Vector x from which affine transformation is applied -> DOMAIN
    z (numpy array): Vector z to which affine transformation will lead -> RANGE

    Returns:
    tuple: alpha and beta.
    """
    y = np.ones(len(x))
    # Compute required dot products
    # xy = np.dot(x, y)  # <x, y>
    # yy = np.dot(y, y)  # <y, y>
    # xx = np.dot(x, x)  # <x, x>
    # xz = np.dot(x, z)  # <x, z>
    # yz = np.dot(y, z)  # <y, z>

    # Denominator for both formulas
    denominator = np.dot(x, y) ** 2 - np.dot(x, x) * np.dot(y, y)
    # denominator = xy ** 2 - xx * yy

    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute alpha and beta.")

    # Calculate alpha
    # alpha = (yz * xy - yy * xz) / denominator
    alpha = (np.dot(y, z) * np.dot(x, y) - np.dot(y, y) * np.dot(x, z)) / denominator

    # Calculate beta
    # beta = (xy * xz - xx * yz) / denominator
    beta = (np.dot(x, y) * np.dot(x, z) - np.dot(x, x) * np.dot(y, z)) / denominator

    return alpha, beta


def transform(alpha, beta, x):
    return np.multiply(x, alpha) + np.multiply(beta, np.ones(len(x)))


def downsample(v):
    return np.array(v).reshape(-1, 2).mean(axis=1)
