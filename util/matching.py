import numpy as np
from numba import njit

import util.math as mymath


@njit
def brute_force_r_to_d_matching(r: np.ndarray, d_matrix: np.ndarray, allow_threshold: bool = True) -> tuple:
    # threshold to stop searching if fulfilled
    d_threshold = 0.0001

    # parameters to encode
    distance_min = 1000000
    d_index = 0
    fit_alpha = 1
    fit_beta = 0

    # find the best base domain from domain pool to transform into range block with min d_rms
    for d_i, d in enumerate(d_matrix):
        alpha, beta = mymath.calculate_alpha_beta(d, r)
        transformed = mymath.transform(alpha, beta, d)
        distance_calc = mymath.distance(d, transformed)

        if distance_calc < distance_min:
            fit_alpha = alpha
            fit_beta = beta
            d_index = d_i
            distance_min = distance_calc

        # already found distance_min satisfies threshold, stop searching
        if distance_min < d_threshold and allow_threshold:
            break
    return d_index, fit_alpha, fit_beta
