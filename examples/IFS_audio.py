from IFS.IFS_1d import IFS
import numpy as np
import sounddevice as sd
import util.common as common
import util.math as mymath

# parameters
n_domains = 10  # number of Domain blocks
range_block_size = 10  # number of Range blocks

# reading audio files
original_signal, audio_samplerate = common.read_example_file(n_samples=1000)

ifs = IFS()
ifs.setup_values(range_block_size, 10, 0.0001)

# PARTITIONING
# range blocks, covering all the signal, no overlapping allowed
# domain blocks, overlapping allowed
R, D, D_start_sample = ifs.generate_range_domain_blocks(n_domains, original_signal)

# CODING
codded_data = ifs.encode(R, D, D_start_sample)

# DECODING
decoded_signal = ifs.decode(codded_data)

# LOW PASS FILTERING
highest_freq = mymath.find_highest_frequency(original_signal, audio_samplerate) + 0.0001
filtered_signal = mymath.butter_lowpass_filter(decoded_signal, highest_freq, audio_samplerate)

# PLOTTING
common.print_attr_vs_orig(decoded_signal, original_signal, n_range=range_block_size, n_domains=n_domains)
common.print_signal(original_signal, "Original signal")
common.print_signal(decoded_signal, "Decoded signal")
common.print_signal(filtered_signal, "Filtered signal")
common.print_attr_vs_orig(filtered_signal, original_signal, title="Original vs filtered attractor")

# PLAYING
sd.play(np.array(R).flatten(), samplerate=audio_samplerate, blocking=True)
sd.play(np.array(decoded_signal), samplerate=audio_samplerate, blocking=True)
sd.play(np.array(filtered_signal), samplerate=audio_samplerate, blocking=True)
