import numpy as np
import util.math as mymath
import pywt
import time


# TODO CUSTOM OVERLAPPING - as for now only Cyclic buffer supported
# TODO disable logging parameter
# TODO IFS on lower layers
def get_sub_block(starting_ind: int, samples_to_add: int, org_block: np.ndarray) -> (int, np.ndarray):
    """
    Returns sub-block from array of coefficients starting from starting_ind,
    applying Cyclic Buffer if the starting_index + length of sub-block exceeds length of original array.
    :param starting_ind: index of sample in original array from which sub-block will be extracted
    :param samples_to_add: how many samples to add to create sub-block (length of sub-block)
    :param org_block: original array from which sub-block will be extracted
    :return: tuple of modified starting_index for next sub-block and sub-block
    """
    block = []
    # default addition, index not exceed
    if starting_ind + samples_to_add < len(org_block):
        block.extend(org_block[starting_ind:starting_ind + samples_to_add])
        starting_ind = starting_ind + samples_to_add
    # if starting index already in new cycle buffer, shift starting index and call get_sub_block with new values
    elif starting_ind > len(org_block):
        shift_counter = int(starting_ind / len(org_block))
        starting_ind = starting_ind - shift_counter * len(org_block)
        _, block = get_sub_block(starting_ind, samples_to_add, org_block)
    # go with buffer to the start, index not exceed
    elif starting_ind + samples_to_add == len(org_block):
        block.extend(org_block[starting_ind:])
        starting_ind = 0
    # go with buffer to the start, index is exceed, firstly add leftovers
    elif starting_ind + samples_to_add > len(org_block):
        samples_left = samples_to_add - (len(org_block) - starting_ind)
        block.extend(org_block[starting_ind:])
        starting_ind = 0
        block.extend(org_block[starting_ind:starting_ind + samples_left])
        starting_ind += samples_left

    return starting_ind, np.array(block)


def set_sub_block(starting_ind: int, samples_to_add: int, org_block: np.ndarray, new_block: np.ndarray) -> int:
    """
    Modifies original array with new sub-block according to starting_ind and samples_to_add applying Cyclic Buffer.
    :param starting_ind: index of sample in original array where values from sub-block will be placed
    :param samples_to_add: how many samples does sub-block have
    :param org_block: original array to be modified by new sub-block
    :param new_block: array of coefficients to modify org_block with
    :return: modified starting_index for next sub-block
    """
    # default, index not exceed
    if starting_ind + samples_to_add < len(org_block):
        org_block[starting_ind:starting_ind + samples_to_add] = new_block
        starting_ind = starting_ind + samples_to_add
    # go with buffer to the start, index not exceed
    elif starting_ind + samples_to_add == len(org_block):
        org_block[starting_ind:] = new_block
        starting_ind = 0
    # go with buffer to the start, index is exceed, firstly add leftovers
    elif starting_ind + samples_to_add > len(org_block):
        samples_left = samples_to_add - (len(org_block) - starting_ind)
        org_block[starting_ind:] = new_block[:(samples_to_add - samples_left)]
        starting_ind = 0
        org_block[starting_ind:starting_ind + samples_left] = (
                (new_block[(samples_to_add - samples_left):] + org_block[starting_ind:starting_ind + samples_left]) / 2)
        starting_ind += samples_left
    return starting_ind


def generate_blocks_matrix(blocks_level: int, block_height: int, coefficients: np.ndarray) -> np.ndarray:
    """
    Generates blocks matrix at some level from root up to block_height from coefficients array (subtrees)
    :param blocks_level: level at which blocks roots are
    :param block_height: how big the single block (tree) is
    :param coefficients: list of lists of wavelets coefficients at each levels
    :return: matrix of sub-blocks where each row corresponds to a sub-block
    """
    n_blocks = len(coefficients[blocks_level])
    blocks = [[] for _ in range(n_blocks)]
    # CYCLIC BUFFER
    for iter_n, i in enumerate(range(blocks_level, blocks_level + block_height)):
        samples_to_add = 2 ** iter_n
        starting_index = 0
        counter = 0
        while counter < n_blocks:
            # go with buffer to the start, index not exceed
            starting_index, temp_block = get_sub_block(starting_index, samples_to_add, coefficients[i])
            blocks[counter].extend(temp_block)
            counter += 1
    return np.array(blocks)


def generate_r_d(r_blocks_level: int, block_height: int, coefficients: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Based on the 'generate_blocks_matrix' function generates range blocks and domain blocks
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (tree) is
    :param coefficients: list of lists of wavelets coefficients at each levels
    :return: two matrix of sub-blocks (subtrees), ranges and domains blocks
    """
    start_time_blocks = time.time()
    print("starting generating blocks...")
    r = generate_blocks_matrix(r_blocks_level, block_height, coefficients)
    d = generate_blocks_matrix(r_blocks_level - 1, block_height, coefficients)
    print(f"blocks generation finished with {round(time.time() - start_time_blocks, 2)}s.")
    return r, d


def encode_wavelets(wavelets_coefficients: list, r_blocks_level: int, block_height: int) -> (np.ndarray, np.ndarray):
    """
    Performs fractal encoding of wavelets coefficients above some level of decomposition.
    :param wavelets_coefficients: list of lists of wavelets coefficients at each levels
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (tree) is
    :return: Returns tuple with wavelet coefficients below provided level to be stored directly and information
        for FWC decoding: (starting index of domain block, alpha parameter, beta parameter) for each range block
    """
    start_time_enc = time.time()
    print("starting encoding...")
    # prepare blocks
    a_coeffs = wavelets_coefficients[1:]
    coeffs_to_be_stored = wavelets_coefficients[:r_blocks_level + 1]
    r_matrix, d_matrix = generate_r_d(r_blocks_level, block_height, a_coeffs)
    n_range, _ = r_matrix.shape
    n_domain, _ = d_matrix.shape
    # encode
    uniq_d = set()
    progress_incrementor = 1 if int(0.05 * n_range) == 0 else int(0.05 * n_range)
    codded = []
    for r_i, r in enumerate(r_matrix):
        # progress logging
        if r_i % progress_incrementor == 0:
            print(f"\rProgress: {round(r_i * 100 / n_range, 2)}%.", end="", flush=True)
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

        # encoded parameters for each range block
        codded.append((d_index, fit_alpha, fit_beta))
        uniq_d.add(d_index)
    print("\rProgress: 100%.", flush=True)
    print(f"d used: {len(uniq_d)}/{n_domain}")
    print(f"encoding finished with {round(time.time() - start_time_enc, 2)}s.")
    return coeffs_to_be_stored, np.array(codded)


def decode(coded: tuple, wavelet_family: str, r_blocks_level: int, block_height: int, n_org_signal_samples: int,
           decoding_iter=10) -> np.ndarray:
    """
    Performs decoding proces of wavelets coefficients above some level of decomposition by using IFS.
    :param coded: tuple with wavelet coefficients below provided level stored directly and information
        for FWC decoding: (starting index of domain block, alpha parameter, beta parameter) for each range block
    :param wavelet_family: wavelet family used in decomposition process
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (tree) is
    :param n_org_signal_samples: number of samples in original signal
    :param decoding_iter: number of iterations for IFS decoding
    :return: reconstructed decoded signal
    """
    print("starting decoding...")
    start_time_dec = time.time()
    to_be_stored, coded_blocks = coded
    # get lengths of coeffs on each levels
    n_coeffs_level = []
    temp = n_org_signal_samples
    filter_len = pywt.Wavelet(wavelet_family).dec_len
    for i in range(block_height):
        temp = int((temp + filter_len - 1) / 2)
        n_coeffs_level.append(temp)
    n_coeffs_level = n_coeffs_level[::-1]

    # creating decoding base
    decoded = to_be_stored.copy()
    decoded.extend([np.random.uniform(0, 1, i) for i in n_coeffs_level])

    for _ in range(decoding_iter):
        for iter_n, i in enumerate(range(r_blocks_level, r_blocks_level + block_height)):
            samples_to_add = 2 ** iter_n
            starting_index = 0
            counter = 0
            while counter < len(coded_blocks):
                w = coded_blocks[counter]
                # get d block elements at level k-1
                _, d_to_decode_from = get_sub_block(int(w[0] * samples_to_add), samples_to_add, decoded[
                    iter_n + r_blocks_level])  # r is on K level, d is on K + 1 level but since decoded includes b_0 at 0 index, d level and r level are ++
                # perform transformation
                r_transformed = mymath.transform(w[1], w[2], d_to_decode_from)
                # set r_transformed
                starting_index = set_sub_block(starting_index, samples_to_add, decoded[iter_n + r_blocks_level + 1],
                                               r_transformed)
                counter += 1
    reconstructed_signal = pywt.waverec(decoded, wavelet_family)
    print(f"finished decoding with {round(time.time() - start_time_dec, 2)}s.")
    return np.array(reconstructed_signal)


def wavelet_decompostion(signal: np.ndarray, wavelet_family: str, decomposition_level: int) -> list:
    """
    Performs wavelet decomposition proces on original signal at provided level of decomposition
    :param signal: original signal to be decomposed
    :param wavelet_family: wavelet family used in decomposition process
    :param decomposition_level: how deep wavelet decomposition should be
    :return: signal decomposed into wavelet coefficients at different levels
    """
    max_decomp = pywt.dwt_max_level(len(signal), wavelet_family)
    assert max_decomp >= decomposition_level, \
        f"Desired wavelet decomposition level is too high. Maximum level is {max_decomp}."
    print("starting wavelet decomposition...")
    # DWT on X
    return pywt.wavedec(signal, wavelet_family, level=decomposition_level)
