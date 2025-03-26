import pywt
import cv2
import random
import numpy as np

from util.wavFile import read_wav_file

n = 6
n_samples = 2 ** n  # samples in base signal
wave_offset = 10000
#n_samples = 100

# generating base image
# file = "../resources/sound.wav"
file = "../resources/en_speech.wav"
audio_meta_data, X = read_wav_file(file)
audio_samplerate = audio_meta_data["fs"]
X = X[0][wave_offset:n_samples + wave_offset]
#print(pywt.wavelist())
wavelet_family = "db2"
wave = pywt.Wavelet(wavelet_family)

DECOMP_LEVEL = 4

wave_coeff_pyramid = pywt.wavedec(X, wavelet_family,
                                  level=DECOMP_LEVEL, mode="symmetric")

for level in wave_coeff_pyramid:
    print(len(level))


