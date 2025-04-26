import compression.fractal_wavelet_compression_core as fwc
import util.common as common
import util.math as mymath

# PARAMETERS
DECOMPOSITION_LEVEL = 3
BLOCK_HEIGHT = 2
RANGE_BLOCKS_LEVEL = DECOMPOSITION_LEVEL - BLOCK_HEIGHT
n_samples = 2 ** 11
file_types = ("sound", "speech", "confutatis", "badinerie", "rondo-alla-turca")
file_type = file_types[2]
wavelet_families = ["db34", "coif16", "sym19"]
STORES_ONLY_ALPHA = False

for offset_step in range(1000, 2 ** 15, n_samples):

    # READING ORIGINAL SIGNAL
    original_signal, audio_samplerate, bit_depth = common.read_example_file(n_samples=n_samples, file_type=file_type,
                                                                            suppress_logs=True, samples_offset=offset_step)

    # PREDICT BEST WAVELET
    best_wavelet_predicted = fwc.predict_wavelet(original_signal, audio_samplerate)
    best_wavelet_calculated = ""
    best_psnr = -100000
    for wavelet_family in wavelet_families:
        # DWT on signal (wavelet decomposition)
        wavelet_coefficients = fwc.wavelet_decomposition(original_signal, wavelet_family, DECOMPOSITION_LEVEL,
                                                         suppress_logs=True)

        # ENCODING
        # codded_data = (to_be_stored, coded_blocks)
        codded_data = fwc.encode_wavelets(wavelet_coefficients, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT,
                                          store_only_alpha=STORES_ONLY_ALPHA, suppress_logs=True)

        # DECODING
        decoded_signal = fwc.decode(codded_data, wavelet_family, RANGE_BLOCKS_LEVEL, BLOCK_HEIGHT, n_samples,
                                    suppress_logs=True)

        # PRINTING
        mse = mymath.calculate_mse(decoded_signal, original_signal)
        psnr = mymath.calculate_psnr(mse)
        if psnr > best_psnr:
            best_psnr = psnr
            best_wavelet_calculated = wavelet_family
    print(f"wavelet_calculated={best_wavelet_calculated}, wavelet_predicted={best_wavelet_predicted}, {best_wavelet_calculated == best_wavelet_predicted}")
