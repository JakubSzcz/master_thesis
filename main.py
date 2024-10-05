from util.waveFile import read_wav_file
import numpy as np
import matplotlib.pyplot as plt

meta, data = read_wav_file("resources/example.wav", mono=True)

time_seq = np.arange(0, meta["time_duration"], 1/float(meta["fs"]))

plt.plot(time_seq, data)