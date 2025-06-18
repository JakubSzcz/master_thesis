import numpy as np
import pywt
import time
from enum import Enum
import faiss
from matplotlib import pyplot as plt
from numba import njit

from util.matching import brute_force_r_to_d_matching
import util.math as mymath


class MatchingType(Enum):
    BRUTE_FORCE = 1
    FAISS = 2


### CORE FUNCTIONS ###

def wavelet_decomposition(signal: np.ndarray, wavelet_family: str, decomposition_level: int,
                          suppress_logs: bool = False) -> list:
    """
    Performs wavelet decomposition proces on original signal at the provided level of decomposition
    :param suppress_logs: stop printing logs flag
    :param signal: original signal to be decomposed
    :param wavelet_family: wavelet family used in a decomposition process
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
                    suppress_logs: bool = False) -> (np.ndarray, np.ndarray):
    """
    Performs fractal encoding on DWT coefficients grid above r_blocks_level.
    :param suppress_logs: stop printing logs flag
    :param store_only_alpha: use only alpha parameter while transforming blocks flag
    :param matching_type: what type of paring range to domain blocks to use [BRUTE_FORCE, FAISS]
    :param wavelets_coefficients: nested lists of wavelet coefficients at each level
    :param r_blocks_level: at which range blocks roots are. Domain blocks level is at one level below
    :param block_height: how big the single block (tree) is
    :return: Returns tuple with wavelet coefficients below the provided level to be stored directly and information
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
        # get 5 best matches from FAISS
        _, best_matches_indices = index_faiss.search(r_matrix, 5)
        for r_i, best_matched in enumerate(best_matches_indices):

            # progress logging
            if r_i % progress_incrementor == 0 and not suppress_logs:
                print(f"\rProgress: {round(r_i * 100 / n_range, 2)}%.", end="", flush=True)

            best_dist = 1000000
            d_index = None
            fit_alpha = None
            fit_beta = None
            # find the best match based on the list of 5 returned from index
            for d_index_temp in best_matched:
                temp_fit_alpha, temp_fit_beta = mymath.calculate_alpha_beta(d_matrix[d_index_temp], r_matrix[r_i],
                                                                            store_only_alpha)
                distance = mymath.distance(d_matrix[d_index_temp], r_matrix[r_i])
                if distance < best_dist:
                    best_dist = distance
                    d_index = d_index_temp
                    fit_alpha = temp_fit_alpha
                    fit_beta = temp_fit_beta

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

            # find the best match
            d_index, fit_alpha, fit_beta = brute_force_r_to_d_matching(r, d_matrix)
            codded.append((d_index, fit_alpha, fit_beta))
            uniq_d.add(d_index)

        if not suppress_logs:
            print("\rProgress: 100%.", flush=True)
            print(f"Unique d used: {len(uniq_d)}/{n_domain}")
            print(f"Encoding finished with {round(time.time() - start_time_enc, 2)}s.")

    return coeffs_to_be_stored, np.array(codded)


def decode(coded: tuple, wavelet_family: str, r_blocks_level: int, block_height: int, n_org_signal_samples: int,
           decoding_iter: int = 10, suppress_logs: bool = False, original_signal: np.ndarray = None) -> np.ndarray:
    """
    Performs decoding proces on DWT grid above some level using IFS.
    :param original_signal: numpy array with original signal, for comparison purposes,
    :param suppress_logs: stop printing logs flag
    :param coded: tuple with directly stored DWT grid below r_blocks_level and information for FWC decoding:
        (starting index of domain block, alpha parameter, beta parameter) for each range block above r_blocks_level
    :param wavelet_family: wavelet family used in decomposition process
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (tree) is
    :param n_org_signal_samples: number of samples in the original signal
    :param decoding_iter: number of iterations for IFS decoding
    :return: reconstructed signal
    """
    if not suppress_logs:
        print("Starting decoding...")
    start_time_dec = time.time()
    to_be_stored_full_precision, coded_blocks = coded

    # use only an alpha parameter case
    if len(coded_blocks[0]) == 2:
        fit_beta = np.full((coded_blocks.shape[0], 1), 0)
        coded_blocks = np.hstack((coded_blocks, fit_beta))

    to_be_stored = []
    # simulate 16bits precision
    for level_org in to_be_stored_full_precision:
        level_f16 = level_org.astype(np.float16)
        level_f64_with_loss = level_f16.astype(np.float64)
        to_be_stored.append(level_f64_with_loss)

    coded_blocks = np.array(coded_blocks, dtype=np.float16)
    coded_blocks = np.array(coded_blocks, dtype=np.float64)

    # get lengths of coefficients at each level
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

    # decoding process
    decoded = wavelet_ifs_transform(decoded, coded_blocks, r_blocks_level, block_height, decoding_iter)

    # reconstruction comparison at each level of decomposition
    if original_signal is not None and not suppress_logs:
        to_be_stored_len = len(to_be_stored)
        wavelet_decomp_org = pywt.wavedec(original_signal, wavelet_family, level=(len(decoded) - 1))
        for i in range(to_be_stored_len, (len(decoded))):
            plt.plot(wavelet_decomp_org[i], label=f"original {i}")
            plt.plot(decoded[i], label=f"decoded {i}")
            plt.legend()
            mre_per_level = round(mymath.calculate_mre(wavelet_decomp_org[i], decoded[i]), 5)
            print(f"{i}th level: d={mymath.distance(wavelet_decomp_org[i], decoded[i])}, mre={mre_per_level}")
            plt.title(f"mre {mre_per_level} for {i}th level")
            plt.show()

    reconstructed_signal = pywt.waverec(decoded, wavelet_family)

    if not suppress_logs:
        print(f"Decoding finished with {round(time.time() - start_time_dec, 2)}s.")

    return np.array(reconstructed_signal)


### SUB FUNCTIONS ###


@njit(cache=True)
def get_sub_block(starting_ind: int, samples_to_add: int, org_block: np.ndarray) -> (int, np.ndarray):
    """
    Returns sub-block from single level of DWT grid starting from starting_ind,
    applying Cyclic Buffer if the starting_index + length of sub-block exceeds length of original array.
    :param starting_ind: index of starting sample in original array, from which sub-block will be extracted
    :param samples_to_add: length of sub-block - how many samples to take
    :param org_block: 1D array of single level from DWT grid
    :return: tuple: starting_index for next sub-block and extracted sub-block itself
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
def set_sub_block(starting_ind: int, samples_to_add: int, org_block: np.ndarray, new_block: np.ndarray | None) -> int:
    """
    Modifies original array of single level DWT grid with new sub-block values according to starting_ind and
        samples_to_add applying Cyclic Buffer.
    :param starting_ind: index of starting sample in original array, at which values from sub-block will be placed
    :param samples_to_add: length of sub-block - how many samples to replace
    :param org_block: original array of single level DWT grid
    :param new_block: sub-block with coefficients values to modify org_block with
    :return: starting_index for next sub-block
    """
    n = len(org_block)
    current_start = starting_ind % n
    end_ind = current_start + samples_to_add

    # default, array length not exceeded
    if end_ind <= n:
        org_block[current_start:end_ind] = new_block
        next_starting_ind = end_ind

    # array length is exceeded - add leftovers and go with buffer to the beginning with the rest of values
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
    Generates block matrix from DWT grid by splitting it into subtrees with roots at blocks_level up to block_height.
        Each next level of subtree has 2 times more elements than its predecessor
    :param blocks_level: level at which blocks roots are
    :param block_height: how big the single block (tree) is - how many levels of DWT grid builds up the subtree
    :param coefficients: nested list of DWT grid
    :return: matrix of sub-blocks (subtrees) where each row corresponds to a sub-block
    """
    n_blocks = len(coefficients[blocks_level])
    blocks = [[] for _ in range(n_blocks)]
    # CYCLIC BUFFER
    for iter_n, i in enumerate(range(blocks_level, blocks_level + block_height)):
        samples_to_add = 2 ** iter_n
        starting_index = 0
        counter = 0
        while counter < n_blocks:
            # go with buffer to the start, index not exceeds
            starting_index, temp_block = get_sub_block(starting_index, samples_to_add, coefficients[i])
            blocks[counter].extend(temp_block)
            counter += 1
    return np.array(blocks)


def generate_r_d(r_blocks_level: int, block_height: int, coefficients: np.ndarray, suppress_logs: bool = False) -> (
        np.ndarray, np.ndarray):
    """
    Based on the 'generate_blocks_matrix' function generates range blocks and domain blocks
    :param suppress_logs: stop printing logs flag
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (subtree) is
    :param coefficients: nested list of DWT grid
    :return: two matrix of sub-blocks (subtrees) - range and domain blocks
    """
    start_time_blocks = time.time()
    if not suppress_logs:
        print("Starting generating blocks...")
    r = generate_blocks_matrix(r_blocks_level, block_height, coefficients)
    d = generate_blocks_matrix(r_blocks_level - 1, block_height, coefficients)
    if not suppress_logs:
        print(f"Blocks generation finished with {round(time.time() - start_time_blocks, 2)}s.")
    return r, d


@njit(cache=True)
def wavelet_ifs_transform(coefficients_base: list, coded_blocks: tuple | np.ndarray, r_blocks_level: int,
                          block_height: int, decoding_iter: int) -> list:
    """
    IFS decoding main functionalities extracted from decode function for numba compliance
    :param coefficients_base: list of np.ndarray with wavelet coefficients on the first few levels and noise on the rest
    :param coded_blocks: information for FWC decoding: (starting index of domain block, alpha parameter,
        beta parameter) for each range block
    :param r_blocks_level: level at which range blocks roots are. Domain blocks is at one level below
    :param block_height: how big the single block (subtree) is
    :param decoding_iter: number of iterations for IFS decoding
    :return: reconstructed DWT grid of original signal
    """
    for _ in range(decoding_iter):
        for iter_n, i in enumerate(range(r_blocks_level, r_blocks_level + block_height)):
            samples_to_add = 2 ** iter_n
            starting_index = 0
            counter = 0
            while counter < len(coded_blocks):
                w = coded_blocks[counter]
                # get d block elements at level k-1
                # NOTE! r is on K level, d is on K - 1 level but since decoded variable includes b_0 at 0 index,
                # d level and r level are incremented by 1
                _, d_to_decode_from = get_sub_block(int(w[0] * samples_to_add), samples_to_add, coefficients_base[
                    iter_n + r_blocks_level])
                # perform transformation
                r_transformed = mymath.transform(w[1], w[2], d_to_decode_from)
                # set r_transformed
                starting_index = set_sub_block(starting_index, samples_to_add,
                                               coefficients_base[iter_n + r_blocks_level + 1],
                                               r_transformed)
                counter += 1

    return coefficients_base

# TO BE DONE
# def predict_wavelet(signal: np.ndarray, fs: int) -> str:
#     print("Starting wavelet prediction...")
#     # PARAMETERS
#     model_path = "./../ml/models/best_wavelet/mlp_100_4_200_01.joblib"
#     # based on the empirical results
#     wavelet_dict = {"db": "db34", "coif": "coif16", "sym": "sym19"}
#
#     # performs fft and process output
#     fft_output = mymath.extract_fft(signal, fs)
#     fft_output = mymath.log_transform(fft_output)
#     fft_100_bins = mymath.get_100_mean_bins(fft_output).reshape(1, -1)
#
#     # loading model and prediction
#     model_b_w = joblib.load(model_path)
#     predicted_wavelet_family = model_b_w.predict(fft_100_bins)
#     return wavelet_dict[predicted_wavelet_family[0]]
