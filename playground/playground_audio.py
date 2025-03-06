from IFS.IFS_1d import IFS
from util.wavFile import read_wav_file
import numpy as np
import sounddevice as sd
import util.common as common

# parameters
n_samples = 50000  # samples in base signal
n_domains = 500  # number of Domains blocks
wave_offset = 100000

# generating base image
t = np.linspace(0, 1, n_samples)
audio_meta_data, X = read_wav_file("../resources/example.wav")
audio_samplerate = audio_meta_data["fs"]
X = X[0][wave_offset:n_samples + wave_offset]

ifs = IFS()
ifs.setup_values(10,10,0.0001)

# PARTITIONING
# range blocks, covering all the signal, no overlapping allowed
# domain blocks, overlapping allowed
R, D, D_start_sample = ifs.generate_range_domain_blocks(n_domains, X)

# CODING
codded = ifs.encode(R, D, D_start_sample)

# DECODING
decoded = ifs.decode(codded)

# PLOTTING
common.print_attr_vs_orig(decoded, X)
common.print_signal(X, "Original signal")
common.print_signal(decoded, "Decoded signal")

# PLAYING
sd.play(np.array(R).flatten(), samplerate=audio_samplerate, blocking=True)
sd.play(np.array(decoded), samplerate=audio_samplerate, blocking=True)
