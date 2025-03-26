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

def generate_blocks_matrix(K, BH, coefs, ):
    n_blocks = len(coefs[K])
    blocks = [[] for _ in range(n_blocks)]
    # CYCLIC BUFFER
    for iter_n, i in enumerate(range(K, K + BH)):
        samples_to_add = 2 ** iter_n
        starting_index = 0
        counter = 0
        while counter < len(coefs[K]):
            # go with buffer to the start, index not exceed
            if starting_index + samples_to_add == len(coefs[i]):
                blocks[counter].extend(coefs[i][starting_index:])
                starting_index = 0
            # go with buffer to the start, index is exceed, firstly add leftovers
            elif starting_index + samples_to_add > len(coefs[i]):
                samples_left = samples_to_add - (len(coefs[i]) - starting_index)
                blocks[counter].extend(coefs[i][starting_index:])
                starting_index = 0
                blocks[counter].extend(coefs[i][starting_index:starting_index + samples_left])
                starting_index += samples_left
            else:
                blocks[counter].extend(coefs[i][starting_index:starting_index + samples_to_add])
                starting_index = starting_index + samples_to_add
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
        print(distance_min)
        uniq_d.add(d_index)
    print("\rProgress: 100%.", flush=True)
    print(f"d used: {len(uniq_d)}/{n_domain}")
    print(f"encoding finished with {round(time.time() - start_time_enc, 2)}s.")
    return np.array(codded)


# PARAMETERS
n = 10
n_samples = 2 ** n  # samples in base signal
wave_offset = 100000
DECOMP_LEVEL = 4
BLOCK_HEIGHT = 3
K = DECOMP_LEVEL - BLOCK_HEIGHT

# reading base image
sound = "../resources/sound.wav"
speech = "../resources/en_speech.wav"
file = sound
audio_meta_data, X = read_wav_file(file)
audio_samplerate = audio_meta_data["fs"]
X = X[0][wave_offset:n_samples + wave_offset]
wavelet_family = 'haar'
print(f"Audio parameters: fs = {audio_samplerate}, samples = {n_samples}, "
      f"duration = {round((1 / audio_samplerate) * n_samples, 2)}s.")

# DWT on X
wave_coeffes = pywt.wavedec(X, wavelet_family, level=DECOMP_LEVEL)
coeffs = wave_coeffes[1:]

# PREPARE BLOCKS
range_blocks_matrix, domain_blocks_matrix = generate_r_d(K, BLOCK_HEIGHT, coeffs)

# ENCODING
coded_blocks = encode_wavelets(range_blocks_matrix, domain_blocks_matrix)

# DECODING
