import numpy as np
import pywt
import time
from enum import Enum
import faiss
from numba import njit

from util.matching import brute_force_r_to_d_matching
import util.math as mymath


# TODO CUSTOM OVERLAPPING - as for now only Cyclic buffer supported
# TODO IFS on lower layers
# TODO investigate other metric not only, L2 norm/dot product
# TODO asynchronous encoding
# TODO get/set sub_block_2d functions and test performance improvements
class MatchingType(Enum):
    BRUTE_FORCE = 1
    FAISS = 2


### CORE FUNCTIONS ###

def wavelet_decomposition(signal: np.ndarray, wavelet_family: str, decomposition_level: int,
                          suppress_logs: bool = False) -> list:
    """
    Performs wavelet decomposition proces on original signal at provided level of decomposition
    :param suppress_logs: stop printing logs
    :param signal: original signal to be decomposed
    :param wavelet_family: wavelet family used in decomposition process
    :param decomposition_level: how deep wavelet decomposition should be
    :return: signal decomposed into wavelet coefficients at different levels
    """
    max_decomp = pywt.dwt_max_level(len(signal), wavelet_family)
    assert max_decomp >= decomposition_level, \
        f"Desired wavelet decomposition level is too high. Maximum level is {max_decomp}."
    if not suppress_logs:
        print("Starting wavelet decomposition...")
    # DWT on X
    return pywt.wavedec(signal, wavelet_family, level=decomposition_level)


def encode_wavelets(wavelets_coefficients: list, r_blocks_level: int, block_height: int,
                    matching_type: MatchingType = MatchingType.FAISS, store_only_alpha: bool = False,
                    suppress_logs: bool = False) -> (
        np.ndarray, np.ndarray):
    """
    Performs fractal encoding of wavelets coefficients above some level of decomposition.
    :param suppress_logs: stop printing logs
    :param store_only_alpha: flag whether only alpha parameter is used while transforming blocks
    :param matching_type: what type of paring range to domain blocks to use [BRUTEFORCE, FAISS]
    :param wavelets_coefficients: list of lists of wavelets coefficients at each levels
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (tree) is
    :return: Returns tuple with wavelet coefficients below provided level to be stored directly and information
        for FWC decoding: (starting index of domain block, alpha parameter, optional beta parameter) for each range block
    """
    start_time_enc = time.time()
    if not suppress_logs:
        print("Starting encoding...")

    # prepare blocks
    a_coeffs = wavelets_coefficients[1:]
    coeffs_to_be_stored = wavelets_coefficients[:r_blocks_level + 1]
    r_matrix, d_matrix = generate_r_d(r_blocks_level, block_height, a_coeffs, suppress_logs=suppress_logs)

    n_range, dim = r_matrix.shape
    n_domain, _ = d_matrix.shape
    uniq_d = set()
    progress_incrementor = 1 if int(0.05 * n_range) == 0 else int(0.05 * n_range)
    codded = []

    # FAISS TYPE
    # matching preparation
    if matching_type == MatchingType.FAISS:
        index_faiss = faiss.IndexFlatIP(dim)
        index_faiss.add(d_matrix)
        _, best_matches_indices = index_faiss.search(r_matrix, 1)
        for r_i, best_matched in enumerate(best_matches_indices):
            d_index = best_matched[0]
            # progress logging
            if r_i % progress_incrementor == 0 and not suppress_logs:
                print(f"\rProgress: {round(r_i * 100 / n_range, 2)}%.", end="", flush=True)
            fit_alpha, fit_beta = mymath.calculate_alpha_beta(d_matrix[d_index], r_matrix[r_i], store_only_alpha)
            if store_only_alpha:
                codded.append((d_index, fit_alpha))
            else:
                codded.append((d_index, fit_alpha, fit_beta))
            uniq_d.add(d_index)
        if not suppress_logs:
            print("\rProgress: 100%.", flush=True)
            print(f"Unique d used: {len(uniq_d)}/{n_domain}")
            print(f"Encoding finished with {round(time.time() - start_time_enc, 2)}s.")

    # BRUTEFORCE TYPE
    # encoded parameters for each range block
    if matching_type == MatchingType.BRUTE_FORCE:
        for r_i, r in enumerate(r_matrix):
            # progress logging
            if r_i % progress_incrementor == 0 and not suppress_logs:
                print(f"\rProgress: {round(r_i * 100 / n_range, 2)}%.", end="", flush=True)

            # find best match
            d_index, fit_alpha, fit_beta = brute_force_r_to_d_matching(r, d_matrix)
            codded.append((d_index, fit_alpha, fit_beta))
            uniq_d.add(d_index)
        if not suppress_logs:
            print("\rProgress: 100%.", flush=True)
            print(f"Unique d used: {len(uniq_d)}/{n_domain}")
            print(f"Encoding finished with {round(time.time() - start_time_enc, 2)}s.")

    return coeffs_to_be_stored, np.array(codded)


def decode(coded: tuple, wavelet_family: str, r_blocks_level: int, block_height: int, n_org_signal_samples: int,
           decoding_iter: int = 10, suppress_logs: bool = False) -> np.ndarray:
    """
    Performs decoding proces of wavelets coefficients above some level of decomposition by using IFS.
    :param suppress_logs: stop printing logs
    :param coded: tuple with wavelet coefficients below provided level stored directly and information
        for FWC decoding: (starting index of domain block, alpha parameter, beta parameter) for each range block
    :param wavelet_family: wavelet family used in decomposition process
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (tree) is
    :param n_org_signal_samples: number of samples in original signal
    :param decoding_iter: number of iterations for IFS decoding
    :return: reconstructed decoded signal
    """
    if not suppress_logs:
        print("Starting decoding...")
    start_time_dec = time.time()
    to_be_stored, coded_blocks = coded
    if len(coded_blocks[0]) == 2:
        fit_beta = np.full((coded_blocks.shape[0], 1), 0)
        coded_blocks = np.hstack((coded_blocks, fit_beta))

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

    # wavelet coefficients IFS decoding
    decoded = wavelet_ifs_transform(decoded, coded_blocks, r_blocks_level, block_height, decoding_iter)

    reconstructed_signal = pywt.waverec(decoded, wavelet_family)
    if not suppress_logs:
        print(f"Decoding finished with {round(time.time() - start_time_dec, 2)}s.")
    return np.array(reconstructed_signal)


### SUB FUNCTIONS ###


@njit(cache=True)
def get_sub_block(starting_ind: int, samples_to_add: int, org_block: np.ndarray) -> (int, np.ndarray):
    """
    Returns sub-block from array of coefficients starting from starting_ind,
    applying Cyclic Buffer if the starting_index + length of sub-block exceeds length of original array.
    :param starting_ind: index of sample in original array from which sub-block will be extracted
    :param samples_to_add: how many samples to add to create sub-block (length of sub-block)
    :param org_block: 1D original array from which sub-block will be extracted
    :return: tuple of modified starting_index for next sub-block and 1D sub-block itself
    """
    n = len(org_block)

    # Cyclic Buffer index applied
    current_start = starting_ind % n

    # pre-allocate the output array
    sub_block = np.empty(samples_to_add, dtype=org_block.dtype)

    # fill the output array element-wise, modulo for cyclic access
    for i in range(samples_to_add):
        source_idx = (current_start + i) % n
        sub_block[i] = org_block[source_idx]

    # next starting index after samples_to_add
    next_starting_ind = (current_start + samples_to_add) % n

    return next_starting_ind, sub_block


@njit(cache=True)
def set_sub_block(starting_ind: int, samples_to_add: int, org_block: np.ndarray, new_block: np.ndarray) -> int:
    """
    Modifies original array with new sub-block according to starting_ind and samples_to_add applying Cyclic Buffer.
    :param starting_ind: index of sample in original array where values from sub-block will be placed
    :param samples_to_add: how many samples does sub-block have
    :param org_block: original array to be modified by new sub-block
    :param new_block: array of coefficients to modify org_block with
    :return: modified starting_index for next sub-block
    """
    n = len(org_block)
    current_start = starting_ind % n
    end_ind = current_start + samples_to_add

    # default, index not exceed
    if end_ind <= n:
        org_block[current_start:end_ind] = new_block
        next_starting_ind = end_ind

    # go with buffer to the start, index is exceed, firstly add leftovers
    else:
        # fit before the wrap
        end_part_len = n - current_start
        # to be placed at the beginning after wrapping
        start_part_len = samples_to_add - end_part_len

        # from current_start to the end
        if end_part_len > 0:  # in case if current_start is n-1
            org_block[current_start:] = new_block[:end_part_len]

        # from the beginning to leftovers
        if start_part_len > 0:
            target_slice_at_start = org_block[:start_part_len]
            source_slice_for_start = new_block[end_part_len: end_part_len + start_part_len]

            org_block[:start_part_len] = (source_slice_for_start + target_slice_at_start) / 2.0

        next_starting_ind = start_part_len
    return next_starting_ind


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


def generate_r_d(r_blocks_level: int, block_height: int, coefficients: np.ndarray, suppress_logs: bool = False) -> (
        np.ndarray, np.ndarray):
    """
    Based on the 'generate_blocks_matrix' function generates range blocks and domain blocks
    :param suppress_logs: stop printing logs
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (tree) is
    :param coefficients: list of lists of wavelets coefficients at each levels
    :return: two matrix of sub-blocks (subtrees), ranges and domains blocks
    """
    start_time_blocks = time.time()
    if not suppress_logs:
        print("Starting generating blocks...")
    r = generate_blocks_matrix(r_blocks_level, block_height, coefficients)
    d = generate_blocks_matrix(r_blocks_level - 1, block_height, coefficients)
    if not suppress_logs:
        print(f"Blocks generation finished with {round(time.time() - start_time_blocks, 2)}s.")
    return r, d


@njit
def wavelet_ifs_transform(coefficients_base: np.ndarray, coded_blocks: tuple, r_blocks_level: int, block_height: int,
                          decoding_iter: int) -> np.ndarray:
    """
    IFS decoding main functionalities extracted from decode function for numba compliance
    :param coefficients_base: 2D nd array with wavelet coefficients on first few levels and noise on the rest
    :param coded_blocks: information for FWC decoding: (starting index of domain block, alpha parameter,
        beta parameter) for each range block
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (tree) is
    :param decoding_iter: number of iterations for IFS decoding
    :return: reconstructed coefficient of wavelet decomposition of original signal
    """
    for _ in range(decoding_iter):
        for iter_n, i in enumerate(range(r_blocks_level, r_blocks_level + block_height)):
            samples_to_add = 2 ** iter_n
            starting_index = 0
            counter = 0
            while counter < len(coded_blocks):
                w = coded_blocks[counter]
                # get d block elements at level k-1
                _, d_to_decode_from = get_sub_block(int(w[0] * samples_to_add), samples_to_add, coefficients_base[
                    iter_n + r_blocks_level])  # r is on K level, d is on K + 1 level but since decoded includes b_0 at 0 index, d level and r level are ++
                # perform transformation
                r_transformed = mymath.transform(w[1], w[2], d_to_decode_from)
                # set r_transformed
                starting_index = set_sub_block(starting_index, samples_to_add,
                                               coefficients_base[iter_n + r_blocks_level + 1],
                                               r_transformed)
                counter += 1

    return coefficients_base
