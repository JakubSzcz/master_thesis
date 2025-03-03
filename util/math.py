import numpy as np

def d_rms(x, y):
    # Calculate the difference vector
    #diff = x - y
    # Calculate the inner product <x-y, x-y>
    #inner_product = np.dot(diff, diff)
    # Take the square root to get the RMS distance
    #distance = np.sqrt(inner_product)

    return np.linalg.norm(x - y)

def calculate_alpha_beta(x, z):
    """
    Parameters:
    x (numpy array): Vector x from which affine transformation is applied -> DOMAIN
    z (numpy array): Vector z to which affine transformation will lead -> RANGE

    Returns:
    tuple: alpha and beta.
    """
    y = [1 for _ in range(len(x))]
    # Compute required dot products
    xy = np.dot(x, y)  # <x, y>
    yy = np.dot(y, y)  # <y, y>
    xx = np.dot(x, x)  # <x, x>
    xz = np.dot(x, z)  # <x, z>
    yz = np.dot(y, z)  # <y, z>

    # Denominator for both formulas
    denominator = xy ** 2 - xx * yy

    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute alpha and beta.")

    # Calculate alpha
    alpha = (yz * xy - yy * xz) / denominator

    # Calculate beta
    beta = (xy * xz - xx * yz) / denominator

    return alpha, beta

def transform(alpha, beta, x):
    y = [1 for _ in range(len(x))]
    return np.multiply(x, alpha) + np.multiply(beta, y)

def downsample(v):
    return [(v[i] + v[i + 1]) / 2 for i in range(0, len(v), 2)]