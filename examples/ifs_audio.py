from IFS.IFS_1d import IFS
from util.wavFile import read_wav_file
import numpy as np
import sounddevice as sd
import util.common as common

# parameters
n_samples = 50000  # samples in base signal
n_domains = 1000 # number of Domain blocks
n_range = 10 # number of Range blocks
wave_offset = 100000

# reading audio files
sound = "../resources/sound.wav"
speech = "../resources/en_speech.wav"
file = sound
audio_meta_data, X = read_wav_file(file)
audio_samplerate = audio_meta_data["fs"]
# reads only one channel with offset
X = X[0][wave_offset:n_samples + wave_offset]

print(f"Audio parameters: fs = {audio_samplerate}, samples = {n_samples}, "
      f"duration = {round((1 / audio_samplerate) * n_samples, 2)}s.")

ifs = IFS()
ifs.setup_values(n_range, 10, 0.0001)

# PARTITIONING
# range blocks, covering all the signal, no overlapping allowed
# domain blocks, overlapping allowed
R, D, D_start_sample = ifs.generate_range_domain_blocks(n_domains, X)

# CODING
codded = ifs.encode(R, D, D_start_sample)

# DECODING
decoded = ifs.decode(codded)

# PLOTTING
common.print_attr_vs_orig(decoded, X, n_range=n_range, n_domains=n_domains)
common.print_signal(X, "Original signal")
common.print_signal(decoded, "Decoded signal")

# PLAYING
sd.play(np.array(R).flatten(), samplerate=audio_samplerate, blocking=True)
sd.play(np.array(decoded), samplerate=audio_samplerate, blocking=True)
