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


# TODO CYCLIC BUFFER and CUSTOM OVERLAPPING

def get_sub_block(starting_ind, samps_to_add, org_block):
    # go with buffer to the start, index not exceed
    block = []
    if starting_ind + samps_to_add == len(org_block):
        block.extend(org_block[starting_ind:])
        starting_ind = 0
    # go with buffer to the start, index is exceed, firstly add leftovers
    elif starting_ind + samps_to_add > len(org_block):
        samples_left = samps_to_add - (len(org_block) - starting_ind)
        block.extend(org_block[starting_ind:])
        starting_ind = 0
        block.extend(org_block[starting_ind:starting_ind + samples_left])
        starting_ind += samples_left
    else:
        block.extend(org_block[starting_ind:starting_ind + samps_to_add])
        starting_ind = starting_ind + samps_to_add
    return starting_ind, block


def set_sub_block(starting_ind, samps_to_add, org_block, new_block):
    # go with buffer to the start, index not exceed
    if starting_ind + samps_to_add == len(org_block):
        print("dobija do konca")
        org_block[starting_ind:] = new_block
        starting_ind = 0
    # go with buffer to the start, index is exceed, firstly add leftovers
    elif starting_ind + samps_to_add > len(org_block):
        print("zeruje index")
        samples_left = samps_to_add - (len(org_block) - starting_ind)
        org_block[starting_ind:] = new_block[:(samps_to_add - samples_left)]
        starting_ind = 0
        org_block[starting_ind:starting_ind + samples_left] = new_block[(samps_to_add - samples_left):]
        starting_ind += samples_left
    else:
        print("miesci sie")
        org_block[starting_ind:starting_ind + samps_to_add] = new_block
        starting_ind = starting_ind + samps_to_add
    return starting_ind


def generate_blocks_matrix(K, BH, coefs):
    n_blocks = len(coefs[K])
    blocks = [[] for _ in range(n_blocks)]
    # CYCLIC BUFFER
    for iter_n, i in enumerate(range(K, K + BH)):
        samples_to_add = 2 ** iter_n
        starting_index = 0
        counter = 0
        while counter < n_blocks:
            # go with buffer to the start, index not exceed
            starting_index, temp_block = get_sub_block(starting_index, samples_to_add, coefs[i])
            blocks[counter].extend(temp_block)
            counter += 1
    return np.array(blocks)


def generate_r_d(K, BH, coefs):
    start_time_blocks = time.time()
    print("starting generating blocks...")
    r = generate_blocks_matrix(K, BH, coefs)
    d = generate_blocks_matrix(K - 1, BLOCK_HEIGHT, coefs)
    print(f"blocks generation finished with {round(time.time() - start_time_blocks, 2)}s.")
    return r, d


def encode_wavelets(r_matrix, d_matrix):
    start_time_enc = time.time()
    print("starting encoding...")
    n_range, _ = r_matrix.shape
    n_domain, _ = d_matrix.shape
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
    return np.array(codded)


# PARAMETERS
n = 9
n_samples = 2 ** n  # samples in base signal
wave_offset = 100000
DECOMP_LEVEL = 4
BLOCK_HEIGHT = 3
K = DECOMP_LEVEL - BLOCK_HEIGHT

# reading base image
sound = "../resources/sound.wav"
speech = "../resources/en_speech.wav"
file = speech
audio_meta_data, X = read_wav_file(file)
audio_samplerate = audio_meta_data["fs"]
X = X[0][wave_offset:n_samples + wave_offset]
wavelet_family = 'haar'
print(f"Audio parameters: fs = {audio_samplerate}, samples = {n_samples}, "
      f"duration = {round((1 / audio_samplerate) * n_samples, 2)}s.")

# DWT on X
wave_coeffes = pywt.wavedec(X, wavelet_family, level=DECOMP_LEVEL)
coeffs = wave_coeffes[1:]
to_be_stored = wave_coeffes[:K + 1]

# PREPARE BLOCKS
range_blocks_matrix, domain_blocks_matrix = generate_r_d(K, BLOCK_HEIGHT, coeffs)

# ENCODING
coded_blocks = encode_wavelets(range_blocks_matrix, domain_blocks_matrix)

# DECODING
# get lengths of coeffs on each levels
n_coeffs_level = []
temp = n_samples
filter_len = pywt.Wavelet(wavelet_family).dec_len
for i in range(BLOCK_HEIGHT):
    temp = int((temp + filter_len - 1) / 2)
    n_coeffs_level.append(temp)
n_coeffs_level = n_coeffs_level[::-1]

# creating decoding base
decoded = to_be_stored.copy()
decoded.extend([np.random.uniform(0, 1, i) for i in n_coeffs_level])

# TODO check first generate blocks then decoding
# decoding
print("starting decoding...")
start_time_dec = time.time()
for _ in range(10):
    for iter_n, i in enumerate(range(K, K + BLOCK_HEIGHT)):
        samples_to_add = 2 ** iter_n
        starting_index = 0
        counter = 0
        while counter < len(coded_blocks):
            w = coded_blocks[counter]
            # get d block elements at level k-1
            _, d_to_decode_from = get_sub_block(int(w[0] * samples_to_add), samples_to_add, decoded[
                iter_n + K])  # r is on K level, d is on K + 1 level but since decoded includes b_0 at 0 index, d level and r level are ++
            # perform transformation
            r_transformed = mymath.transform(w[1], w[2], d_to_decode_from)
            # set r_transformed
            starting_index = set_sub_block(starting_index, samples_to_add, decoded[iter_n + K + 1], r_transformed)
            counter += 1
pprint.pprint(decoded)
reconstructed_signal = pywt.waverec(decoded, wavelet_family)
print(f"finished decoding with {round(time.time() - start_time_dec, 2)}s.")

# for ind, lev in enumerate(decoded):
#     plt.plot(lev)
#     if ind == 0:
#         plt.title(f"{wavelet_family} my_1 wavelet decomposition reconstructed level b_{ind}")
#     else:
#         plt.title(f"{wavelet_family} my_1 wavelet decomposition reconstructed level a_{ind - 1}")
#     plt.show()

# PRINTING
common.print_signal(X, "original signal")
common.print_signal(reconstructed_signal, "reconstructed signal")
common.print_attr_vs_orig(reconstructed_signal, X)

# PLAYING
sd.play(X, samplerate=audio_samplerate, blocking=True)
sd.play(np.array(reconstructed_signal), samplerate=audio_samplerate, blocking=True)