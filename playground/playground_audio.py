from util.wavFile import read_wav_file
import numpy as np
import pywt


# TODO make a class of it

def create_block_from_pyramid(a, current_level: int, starting_index: int, coeffs: np.floating | np.ndarray,
                              max_level: int) -> list | np.floating:
    # TODO documentation
    # TODO a should be access as a class parameter
    coeffs_retrieved = []

    # final statement if top of pyramid reached, end recursion
    if current_level == max_level - 1:
        return coeffs
    else:
        # save coefficients from current level
        coeffs_retrieved.append(coeffs)

        # prepare parameters for the next level
        # every next level there are 2 times more coefficients, and starting_index is 2 times greater
        starting_index *= 2
        if type(coeffs) is np.ndarray:
            next_level_n_coeffs = len(coeffs) * 2
        else:
            # only for (starting) level = k
            next_level_n_coeffs = 2
        current_level += 1
        # recurrently save coefficients from upper levels
        coeffs_retrieved.append(
            create_block_from_pyramid(a, current_level, starting_index,
                                      a[current_level][starting_index:starting_index + next_level_n_coeffs],
                                      max_level))
    return coeffs_retrieved


def generate_blocks(signal: np.ndarray, decomposition_level: int, range_block_height: int,
                    wavelet_family: str = 'db1') -> (list, list, list):
    # TODO the k value for range and for domain should be passed as parameter, now assuming that range blocks are always one level below domain
    # TODO validation if proper wavelet family member was provided
    # TODO signal must be of size n ** 2
    # TODO documentation
    # TODO verify that returning coefficients to be stored works properly
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

    R = []  # range blocks
    # generate range blocks
    # TODO a should be access as a class parameter
    for i, a_k in enumerate(a[k]):
        R.append(create_block_from_pyramid(a, k, i, a_k, decomposition_level))

    D = []  # domain blocks
    # generate domain blocks
    for i, a_k in enumerate(a[k - 1]):  # root
        D.append(create_block_from_pyramid(a, k - 1, i, a_k, decomposition_level - 1))

    # validation
    assert len(R) == 2 ** (decomposition_level - range_block_height), "Error occurred when generating range blocks."
    assert len(D) == 2 ** (
            decomposition_level - range_block_height - 1), "Error occurred when generating domain blocks."  # assuming that domain starting level is one level bellow range

    return R, D, b.tolist() + a[0:k]


# parameters
n = 6
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
DECOMP_LEVEL = 6
# create blocks
RANGE_BLOCK_HEIGHT = 3  # from the pyramid top, without b, on the bottom a_0, on the top a_(DECOMP_LEVEL - 1)
DOMAIN_BLOCK_HEIGHT = RANGE_BLOCK_HEIGHT + 1  # from the pyramid top, without b

R, D, _ = generate_blocks(X, DECOMP_LEVEL, RANGE_BLOCK_HEIGHT)
