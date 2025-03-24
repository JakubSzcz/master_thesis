import pprint

from util.wavFile import read_wav_file
import numpy as np
import sounddevice as sd
import util.math as mymath
import util.common as common
import pywt
import random

def create_block_from_pyramid(root_level: int, block_height: int, starting_index: int,
                              a: np.ndarray) -> list:
    """
    Create a single pyramid block from a wavelet decomposition.
    :param root_level: wavelet decomposition level where the pyramid starting node is located
    :param block_height: how many decomposed levels constitute the block
    :param starting_index: index of the root coefficient used for generating the block
    :param a: list of all coefficients from decomposition TO BE ACCESSED AS CLASS PARAMETER
    :return: list of range blocks composed of coefficients under chosen root coefficient. e.g. [a_k_j, [a_k+1_j*2, a_k+1_(j*2)+1...]...]
    """
    # TODO make a documentation
    # TODO check if recursive version is faster
    # TODO all pyramid in loop
    # pyramid root init
    pyramid = [a[root_level][starting_index]]
    n_elements_in_pyramid_level = 2
    for i in range(root_level + 1, root_level + block_height):
        starting_index *= 2
        pyramid.append(a[i][starting_index: starting_index + n_elements_in_pyramid_level])
        n_elements_in_pyramid_level *= 2
    return pyramid


def generate_blocks(signal: np.ndarray, decomposition_level: int, k: int, i: int,
                    wavelet_family: str = 'db1', return_coeffs: bool = False) -> (list, list, list):
    """
    Performs wavelet decomposition and then generates range and domain blocks and wavelet decomposition
     coefficients that have been left to be stored
    :param i: at which level domain blocks are
    :param return_coeffs: flag to return wavelet decomposition coefficients
    :param signal: original signal to be decomposed
    :param decomposition_level: wavelet decomposition level
    :param k: at which level range blocks are
    :param wavelet_family: base wavelet family
    :return: tuple of three list: range blocks, domain blocks and the remaining coefficients below chosen
     level k to be stored directly
    """
    # TODO the k value for range and for domain should be passed as parameter, now assuming that range blocks are always one level below domain // to be verified if even possible
    # TODO validation if proper wavelet family member was provided
    # TODO signal must be of size n ** 2 - validation
    # TODO documentation
    # TODO verify that returning coefficients to be stored works properly
    # TODO returns list of [float, np.ndarray(arrays)] -> list[float, list ....]
    # assert that desired decomposition can be achieved
    assert decomposition_level <= pywt.dwt_max_level(len(signal),
                                                     wavelet_family), "Desired decomposition level is too high."
    # assert that range block height is valid
    assert 0 < k < decomposition_level, "Desired range block height is not valid."

    # perform DWT on signal
    wave_coeff_pyramid = pywt.wavedec(X, wavelet_family,
                                      level=DECOMP_LEVEL)  # [low freq -> b_00, high_1 (deepest) --> a_0, high_2, ... high_n (shallowest) a_n]

    # separate detail coefficients cD (a) -> high freq and approximation coefficients cA (b) - low freq
    # b = wave_coeff_pyramid[0]  # low freq b_0
    a = wave_coeff_pyramid[1:]  # high, from lowest to highest [a_0, a_1 ... a_n]

    # create blocks
    block_height = decomposition_level - k  # level for root subtree for generating blocks from pyramid

    range_blocks = []  # range blocks
    # generate range blocks
    # TODO a should be access as a class parameter
    # TODO check if can be done in one loop
    for j, a_k in enumerate(a[k]):
        range_blocks.append(create_block_from_pyramid(k, block_height, j, a))

    domain_blocks = []  # domain blocks
    # generate domain blocks
    for j, a_k in enumerate(a[i]):  # root
        domain_blocks.append(create_block_from_pyramid(i, block_height, j, a))

    if return_coeffs:
        return range_blocks, domain_blocks, wave_coeff_pyramid[0:k], wave_coeff_pyramid
    else:
        return range_blocks, domain_blocks, wave_coeff_pyramid[0:k]

def compute_S(alpha: int, beta: int, gamma: int, delta: int, c: list, max_k: int = 10):
    """
    Computes the sum S_{α,β,γ,δ} based on the given mathematical formula.

    Parameters:
    alpha (int): Index α
    beta (int): Index β
    gamma (int): Index γ
    delta (int): Index δ
    c (list of lists): A 2D list or NumPy array where c[i][j] stores coefficient values
    max_k (int): Upper limit for k' (controls depth of summation)

    Returns:
    float: Computed sum S_{α,β,γ,δ}
    """

    S = 0  # Initialize sum

    # Outer sum over k'
    for k_prime in range(max_k):
        two_k_prime = 2 ** k_prime  # Compute 2^k'

        # Inner sum over l'
        for l_prime in range(two_k_prime):
            # Compute indices
            row1, col1 = alpha + k_prime, two_k_prime * beta + l_prime
            row2, col2 = gamma + k_prime, two_k_prime * delta + l_prime

            # Check if indices are within bounds
            if row1 < len(c) and col1 < len(c[row1]) and row2 < len(c) and col2 < len(c[row2]):
                c1 = c[row1][col1]
                c2 = c[row2][col2]
            else:
                c1, c2 = 0, 0  # If out of bounds, treat as zero

            # Accumulate the sum
            S += c1 * c2

    return S


# PARAMETERS
n = 4
n_samples = 2 ** n  # samples in base signal
wave_offset = 10000
DECOMP_LEVEL = 4
K = 2  # decomposition level at which range blocks roots are
I = 1  # # decomposition level at which domain blocks roots are
# RANGE_BLOCK_HEIGHT = 2  # DECOMP_LEVEL - K


# LOADING AUDIO
file = "../resources/sound.wav"
# file = "../resources/en_speech.wav"
audio_meta_data, X = read_wav_file(file)
audio_samplerate = audio_meta_data["fs"]
X = X[0][wave_offset:n_samples + wave_offset]


print(f"Audio parameters: fs = {audio_samplerate}, samples = {n_samples}, "
      f"duration = {round((1 / audio_samplerate) * n_samples, 2)}s.")

# WAVELET DECOMPOSITION
# coefficients = wave_coeff_pyramid = pywt.wavedec(X, 'db1', level=DECOMP_LEVEL)
# a = coefficients[1:] # high freq coeffs

R, D, to_store, coefficients = generate_blocks(X, DECOMP_LEVEL, K, I)
a = coefficients[1:] # high freq coeffs

# ENCODING
codded = []
range_roots = a[K]
n_range = len(range_roots)
scale_factor = 2 ** (K - I)
progress_incrementor = int(0.05 * n_range)

# for each range block root
for l, a_k in enumerate(a[K]):
    if l % progress_incrementor == 0:
        print(f"\rProgress: {round(l * 100 / n_range, 2)}%.", end="", flush=True)
    min_distance = 1000000000
    fit_alpha = 1
    domain_indx = 0
    for j, b_k in enumerate(a[K - 1]):
        S_range_domain = compute_S(K, l, K - 1, j, a)
        S_domain = compute_S(K - 1, j, K - 1, j, a)
        S_range = compute_S(K, l, K, l, a)
        calc_alpha = scale_factor * S_range_domain / S_domain
        calc_distance = np.sqrt(S_range - calc_alpha * scale_factor * S_range_domain)
        if calc_distance < min_distance:
            min_distance = calc_distance
            fit_alpha = calc_alpha
            domain_indx = j
    codded.append((fit_alpha, domain_indx))

# # DECODING
print("starting decoding...")
# parameters
n_range = len(codded)
decoded = []
for i in range(0, DECOMP_LEVEL - K):
    decoded.append(np.random.uniform(0, 1, n_range * 2 ** i))
for iterat in range(100):
    print(f"iter: {iterat}/10")

    for i, a_rec in enumerate(decoded):
        domain_root_index = codded[i][1]

        for k_prime in range(DECOMP_LEVEL):
            two_k_prime = 2 ** k_prime  # Compute 2^k'

            # Inner sum over l'
            for l_prime in range(two_k_prime):
                # Compute indices
                row_r, col_r = i + k_prime, two_k_prime * i + l_prime
                row_d, col_d = domain_root_index + k_prime, two_k_prime * domain_root_index + l_prime

                # Check if indices are within bounds
                if row_r < len(decoded) and col_r < len(decoded[row_r]) and row_d < len(decoded) and col_d < len(
                        decoded[row_d]):
                    decoded[row_r][col_r] = codded[i][0] * decoded[row_d][col_d]
print(len(decoded))
print(len(decoded) == len(a[K:]))

pprint.pprint(decoded[1][0:10])
pprint.pprint(a[1][0:10])
