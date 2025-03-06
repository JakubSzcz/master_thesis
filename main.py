from util.wavFile import read_wav_file
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pywt

meta, data = read_wav_file("resources/sound.wav")
data = data[0]
time_seq = np.arange(0, meta["time_duration"], 1/float(meta["fs"]))

# #data = data[0] # only one channel
# # in Hz
# f = 10
# fs = 10000
# # in sec
# t_start = 0
# t_end = 2
lev = 10
results = pywt.wavedec(data, 'db1', level=lev)
count = len(results)

# plt.figure(1,(16,16))
# for i in range(count):
#     plt.subplot(count,1,i+1)
#     plt.plot(results[i])
#     plt.grid()
#     if i == 0:
#         plt.title("CA")
#         print(f"CA len = {len(results[i])}")
#         print()
#     else:
#         plt.title(f"CD_{count - i}")
#         print(f"CD len = {len(results[i])}")
#
#
plt.figure(2, (16,16))
# plt.plot(time_seq, data)
# plt.grid()
# plt.title("Original signal")
# plt.show()
plt.subplot(2,1,1)
plt.plot(results[1])
plt.grid()
plt.title(f"CD_{count}")
plt.subplot(2,1,2)
plt.plot(results[-1])
plt.grid()
plt.title(f"CD_{1}")
plt.show()







