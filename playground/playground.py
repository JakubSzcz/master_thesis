import pywt

all_wavelets = pywt.wavelist(kind="discrete")
test = []
for wavelet in all_wavelets:
    print(f"{wavelet} - L = {pywt.Wavelet(wavelet).dec_len} ")
