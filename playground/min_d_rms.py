import numpy as np
import random


def calculate_alpha_beta(x, y, z):
    """
    Calculates alpha and beta based on the provided formulas.

    Parameters:
    x (numpy array): Vector x.
    y (numpy array): Vector y.
    z (numpy array): Vector z.

    Returns:
    tuple: alpha and beta.
    """
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


y = np.array([1,1])
z = np.array([52, 104])

x1 = np.array([44, 112])
alpha1, beta1 = calculate_alpha_beta(x1,y,z)
print(f"alpha1 = {alpha1}, beta1 = {beta1}")

tolerance = 0.001
output = [random.randint(0, 1001), random.randint(0,1001)]
for i in range(1000):
    output = np.multiply(alpha1, output) + np.multiply(y, beta1)
    print(output)
