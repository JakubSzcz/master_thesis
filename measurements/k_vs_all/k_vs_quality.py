import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import compression.fractal_wavelet_compression_core as fwc
import pandas as pd
import util.common as common
import util.math as mymath

csv_path = "k_vs_quality.csv"
BHs = list(range(2, 7))
Ns = list(range(10, 21))
wavelets = ['db1', 'db7', 'db12', 'coif8', 'db32', 'db37', 'coif14', 'coif16']
wavelets_lengths = [2, 14, 24, 48, 64, 74, 84, 96]
Ns_skip_per_wavelet = [0, 0, 1, 2, 2, 3, 3, 3]
# # coif 16 min 2 ** 13
# # coif 14 min 2 ** 13
# # db 37 min 2 ** 13
# # db 32 min 2 ** 12
# # coif 8 min 2 ** 12
# # db 12 min 2 ** 11
# # db 7 min 2 ** 10
# # db 1 min 2 ** 6

# Align N values per wavelet
Ns_aligned = [Ns[skip:] for skip in Ns_skip_per_wavelet]


# Worker function to be run in parallel
def process_task(task):
    file_id, BH, wavelet, wavelet_index, N = task
    try:
        original_signal, _, _ = common.read_example_file(n_samples=2 ** N, file_type="sound", suppress_logs=True)
        wavelet_coefficients = fwc.wavelet_decomposition(original_signal, wavelet, BH + 1, suppress_logs=True)
        codded_data = fwc.encode_wavelets(wavelet_coefficients, 1, BH, store_only_alpha=True, suppress_logs=True)
        decoded_signal = fwc.decode(codded_data, wavelet, 1, BH, 2 ** N, suppress_logs=True)

        mse = mymath.calculate_mse(original_signal, decoded_signal)
        mre = mymath.calculate_mre(original_signal, decoded_signal)
        psnr = mymath.calculate_psnr(mse)
        k = common.get_compression_rate(2 ** N, BH, wavelet)

        return {
            'file_id': file_id,
            'psnr': psnr,
            'mre': mre,
            'wavelet': wavelet,
            'L': wavelets_lengths[wavelet_index],
            'n': N,
            'BH': BH,
            'k': k
        }
    except Exception as e:
        print(f"Error processing task (file_id={file_id}): {e}")
        return None


if __name__ == "__main__":
    # warmup
    common.testing_warmup()
    # Create list of tasks
    tasks = []
    file_id = 0
    for BH in BHs:
        for wavelet_index, wavelet in enumerate(wavelets):
            for N in Ns_aligned[wavelet_index]:
                tasks.append((file_id, BH, wavelet, wavelet_index, N))
                file_id += 1

    # Run in parallel
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(f"[{completed}/{total}] Task completed")
            if result is not None:
                results.append(result)

    # Save results to CSV
    df = pd.DataFrame(results)
    print(df)
    file_exists = os.path.isfile(csv_path)
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
