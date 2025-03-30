import pywt

from util.wavFile import read_wav_file

n = 10
n_samples = 2 ** n  # samples in base signal
wave_offset = 10000
# n_samples = 100

# generating base image
# file = "../resources/sound.wav"
file = "../resources/en_speech.wav"
audio_meta_data, X = read_wav_file(file)
audio_samplerate = audio_meta_data["fs"]
X = X[0][wave_offset:n_samples + wave_offset]
# print(pywt.wavelist())
wavelet_family = "db2"
wave = pywt.Wavelet(wavelet_family).dec_len

DECOMP_LEVEL = 4

wave_coeff_pyramid = pywt.wavedec(X, wavelet_family,
                                  level=DECOMP_LEVEL, mode="symmetric")

test = [i for i in range(1, 11)]


def modify(test_array, start_ind):
    for i in range(start_ind, len(test_array)):
        test_array[i] = test_array[i] ** 2
    return test_array


print(f"przed {test}")
modify(test, 5)
print(f"po {test}")
# for level in wave_coeff_pyramid:
#     print(len(level))
