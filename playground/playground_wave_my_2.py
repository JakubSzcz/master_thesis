import pprint

from matplotlib import pyplot as plt

from util.wavFile import read_wav_file
import numpy as np
import sounddevice as sd
import util.math as mymath
import util.common as common
import pywt
import time
import random


# TODO make a class of it

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


def generate_blocks(signal: np.ndarray, decomposition_level: int, block_height: int,
                    wavelet_family: str = 'db1', return_coeffs: bool = False) -> (list, list, list):
    """
    Performs wavelet decomposition and then generates range and domain blocks and wavelet decomposition
     coefficients that have been left to be stored
    :param return_coeffs: flag to return wavelet decomposition coefficients
    :param signal: original signal to be decomposed
    :param decomposition_level: wavelet decomposition level
    :param block_height: how many decomposed levels constitute to the block
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
    assert 0 < block_height < decomposition_level, "Desired range block height is not valid."

    # perform DWT on signal
    wave_coeff_pyramid = pywt.wavedec(X, wavelet_family,
                                      level=DECOMP_LEVEL)  # [low freq -> b_00, high_1 (deepest) --> a_0, high_2, ... high_n (shallowest) a_n]

    # separate detail coefficients cD (a) -> high freq and approximation coefficients cA (b) - low freq
    b = wave_coeff_pyramid[0]  # low freq b_0
    a = wave_coeff_pyramid[1:]  # high, from lowest to highest [a_0, a_1 ... a_n]

    # create blocks
    k = decomposition_level - block_height  # level for root subtree for generating blocks from pyramid

    range_blocks = []  # range blocks
    # generate range blocks
    # TODO a should be access as a class parameter
    # TODO check if can be done in one loop
    for i, a_k in enumerate(a[k]):
        range_blocks.append(create_block_from_pyramid(k, block_height, i, a))

    domain_blocks = []  # domain blocks
    # generate domain blocks
    for i, a_k in enumerate(a[k - 1]):  # root
        domain_blocks.append(create_block_from_pyramid(k - 1, block_height, i, a))

    if return_coeffs:
        return range_blocks, domain_blocks, wave_coeff_pyramid[
                                            0:k + 1], wave_coeff_pyramid  # [0:k+1] +1 since 0's element is b
    else:
        return range_blocks, domain_blocks, wave_coeff_pyramid[0:k + 1]


# parameters
n = 9
n_samples = 2 ** n  # samples in base signal
wave_offset = 100000

# generating base image
#file = "../resources/sound.wav"
file = "../resources/en_speech.wav"
audio_meta_data, X = read_wav_file(file)
audio_samplerate = audio_meta_data["fs"]
X = X[0][wave_offset:n_samples + wave_offset]
# print(pywt.wavelist(family=None, kind='discrete'))
wavelet = 'haar'

print(f"Audio parameters: fs = {audio_samplerate}, samples = {n_samples}, "
      f"duration = {round((1 / audio_samplerate) * n_samples, 2)}s.")

# generating wavelets coefficients pyramid
DECOMP_LEVEL = 4
# create blocks
RANGE_BLOCK_HEIGHT = 3  # from the pyramid top, without b, on the bottom a_0, on the top a_(DECOMP_LEVEL - 1)
K = DECOMP_LEVEL - RANGE_BLOCK_HEIGHT

# generating range and domains blocks
start_time_enc = time.time()
print("starting encoding...")
R, D, coff_to_be_stored, all_coeffs = generate_blocks(X, DECOMP_LEVEL, RANGE_BLOCK_HEIGHT, wavelet_family=wavelet,
                                                      return_coeffs=True)
# flatten R and D
R_flatten = []
D_flatten = []

for r in R:
    R_flatten.append(np.hstack(r))

for d in D:
    D_flatten.append(np.hstack(d))

# D_flatten = random.sample(D_flatten, len(D_flatten) // 2)

# ENCODING
n_range = len(R_flatten)
uniq_d = set()
progress_incrementor = 1 if int(0.05 * n_range) == 0 else int(0.05 * n_range)
codded = []
for r_i, r in enumerate(R_flatten):
    # progress logging
    if r_i % progress_incrementor == 0:
        print(f"\rProgress: {round(r_i * 100 / n_range, 2)}%.", end="", flush=True)
    # parameters to encode
    distance_min = 1000000
    d_index = 0
    fit_alpha = 1
    fit_beta = 0

    # find the best base domain from domain pool to transform into range block with min d_rms
    for d_i, d in enumerate(D_flatten):
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
print(f"encoding finished with {round(time.time() - start_time_enc, 2)}s.")
print(f"d used: {len(uniq_d)}/{len(all_coeffs[K-1])}")

# DECODING
print("starting decoding...")
start_time_dec = time.time()
# parameters
range_block_size = len(R_flatten[0])

# prepare base random vectors pyramid for reconstruction for level above K
decoded = coff_to_be_stored.copy()
# decoded_coeffs_base = [np.random.uniform(0, 1, 2 ** (n - i + 2)) for i in range(DECOMP_LEVEL, K, -1)]
decoded_coeffs_base = [np.random.uniform(0, 1, 2 ** (n - i)) for i in range(RANGE_BLOCK_HEIGHT, 0, -1)]
decoded.extend(decoded_coeffs_base)  # CONTAINS b AT 0 INDEX, AT K + 1 INDEX ARE RANGE BLOCKS

# iteratively perform transformation for each range blocks
for _ in range(10):
    for ind, w in enumerate(codded):
        for k_prim in range(0, DECOMP_LEVEL - K):
            k_prim_pow = 2 ** k_prim
            decoded[K + 1 + k_prim][ind * k_prim_pow: ind * k_prim_pow + k_prim_pow] = (
                mymath.transform(w[1], w[2], decoded[K + k_prim][w[0] * k_prim_pow: w[0] * k_prim_pow + k_prim_pow]))

# TODO try filtering coeffs before
pprint.pprint(decoded)
reconstructed_signal = pywt.waverec(decoded, wavelet)
print(f"finished decoding with {round(time.time() - start_time_dec, 2)}s.")

# LOW PASS FILTERING
highest_freq = mymath.find_highest_frequency(X, audio_samplerate)
filtered = mymath.butter_lowpass_filter(reconstructed_signal, highest_freq + 0.0001, audio_samplerate)

# PLOTTING
# for ind, lev in enumerate(decoded):
#     plt.plot(lev)
#     if ind == 0:
#         plt.title(f"{wavelet} my_2 wavelet decomposition reconstructed level b_{ind}")
#     else:
#         plt.title(f"{wavelet} my_2 wavelet decomposition reconstructed level a_{ind - 1}")
#     plt.show()

common.print_signal(X, "original signal")
common.print_signal(reconstructed_signal, "reconstructed signal")
common.print_signal(filtered, "filtered signal")
common.print_attr_vs_orig(reconstructed_signal, X)

# PLAYING
sd.play(X, samplerate=audio_samplerate, blocking=True)
sd.play(np.array(reconstructed_signal), samplerate=audio_samplerate, blocking=True)
sd.play(np.array(filtered) * 5, samplerate=audio_samplerate, blocking=True)
