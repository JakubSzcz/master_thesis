from util.wavFile import read_wav_file
import numpy as np
import pywt


# TODO make a class of it

def create_block_from_pyramid(decomp_level: int, r_block_height: int, coeffs, starting_index: int,
                              a: np.ndarray) -> list:
    """
    Create a single pyramid block from a wavelet decomposition.
    :param decomp_level: wavelet decomposition level
    :param r_block_height: how many decomposed levels constitute the block
    :param coeffs: pyramid root coefficient TO BE DELETED check todos
    :param starting_index: index of the root coefficient used for generating the block
    :param a: list of all coefficients from decomposition TO BE ACCESSED AS CLASS PARAMETER
    :return: list of range blocks composed of coefficients under chosen root coefficient. e.g. [a_k_j, [a_k+1_j*2, a_k+1_(j*2)+1...]...]
    """
    # TODO make a documentation
    # TODO check if recursive version is faster
    # TODO coeffs parametr can be ommited insted use a[root_level]
    # pyramid root init
    root_level = decomp_level - r_block_height
    pyramid = [coeffs]
    n_elements_in_pyramid_level = 2
    for i in range(root_level + 1, decomp_level):
        pyramid.append(a[i][starting_index: starting_index + n_elements_in_pyramid_level])
        starting_index *= 2
        n_elements_in_pyramid_level *= 2
    return pyramid


def generate_blocks(signal: np.ndarray, decomposition_level: int, range_block_height: int,
                    wavelet_family: str = 'db1') -> (list, list, list):
    """
    Performs wavelet decomposition and then generates range and domain blocks and wavelet decomposition
     coefficients that have been left to be stored
    :param signal: original signal to be decomposed
    :param decomposition_level: wavelet decomposition level
    :param range_block_height: how many decomposed levels constitute the block
    :param wavelet_family: base wavelet family
    :return: tuple of three list: range blocks, domain blocks and the remaining coefficients below chosen
     level k to be stored directly
    """
    # TODO the k value for range and for domain should be passed as parameter, now assuming that range blocks are always one level below domain
    # TODO validation if proper wavelet family member was provided
    # TODO signal must be of size n ** 2 - validation
    # TODO documentation
    # TODO verify that returning coefficients to be stored works properly
    # TODO returns list of [float, np.ndarray(arrays)] -> list[float, list ....]
    # assert that desired decomposition can be achieved
    assert decomposition_level <= pywt.dwt_max_level(len(signal),
                                                     wavelet_family), "Desired decomposition level is too high."
    # assert that range block height is valid
    assert 0 < range_block_height < decomposition_level, "Desired range block height is not valid."

    # perform DWT on signal
    wave_coeff_pyramid = pywt.wavedec(X, wavelet_family,
                                      level=DECOMP_LEVEL)  # [low freq -> b_00, high_1 (deepest) --> a_0, high_2, ... high_n (shallowest) a_n]

    # separate detail coefficients cD (a) -> high freq and approximation coefficients cA (b) - low freq
    b = wave_coeff_pyramid[0]  # low freq b_0
    a = wave_coeff_pyramid[1:]  # high, from lowest to highest [a_0, a_1 ... a_n]

    # create blocks
    domain_block_height = range_block_height + 1  # from the pyramid top, without b as fundament
    k = decomposition_level - range_block_height  # level for root subtree for generating blocks from pyramid

    range_blocks = []  # range blocks
    # generate range blocks
    # TODO a should be access as a class parameter
    for i, a_k in enumerate(a[k]):
        range_blocks.append(create_block_from_pyramid(decomposition_level, range_block_height, a_k, i, a))

    domain_blocks = []  # domain blocks
    # generate domain blocks
    for i, a_k in enumerate(a[k - 1]):  # root
        domain_blocks.append(create_block_from_pyramid(decomposition_level, domain_block_height, a_k, i, a))

    return range_blocks, domain_blocks, [b, a[0:k]]


# parameters
n = 16
n_samples = 2 ** n  # samples in base signal
wave_offset = 100000

# generating base image
file = "../resources/sound.wav"
# file = "../resources/en_speech.wav"
audio_meta_data, X = read_wav_file(file)
audio_samplerate = audio_meta_data["fs"]
X = X[0][wave_offset:n_samples + wave_offset]

print(f"Audio parameters: fs = {audio_samplerate}, samples = {n_samples}, "
      f"duration = {round((1 / audio_samplerate) * n_samples, 2)}s.")

# generating wavelets coefficients pyramid
DECOMP_LEVEL = 8
# create blocks
RANGE_BLOCK_HEIGHT = 6  # from the pyramid top, without b, on the bottom a_0, on the top a_(DECOMP_LEVEL - 1)
DOMAIN_BLOCK_HEIGHT = RANGE_BLOCK_HEIGHT + 1  # from the pyramid top, without b

# generating range and domains blocks
R, D, coff_to_be_stored = generate_blocks(X, DECOMP_LEVEL, RANGE_BLOCK_HEIGHT)
