import wave
import numpy as np

"""
For PCM_S coding
"""


def read_wav_file(filename: str, amp_normalized: bool = True) -> (dict, np.ndarray):
    # assertions
    assert filename.endswith('.wav'), "Invalid file extension provided."
    assert filename != "", "Empty filename provided."

    # open file
    wav_file = wave.open(filename, 'r')

    bd = wav_file.getsampwidth()
    assert bd >= 1, "Invalid byte depth retrieved."

    # read all samples from frames depending on how many frames they were saved
    if bd == 1:
        samples = np.frombuffer(wav_file.readframes(-1), np.int8)  # one sample stored on 1 byte = 8 bit
    elif bd == 2:
        samples = np.frombuffer(wav_file.readframes(-1), np.int16)  # one sample stored on 2 bytes = 16 bit
    elif bd == 3:
        samples = np.frombuffer(wav_file.readframes(-1), np.int32)
    elif bd == 4:
        samples = np.frombuffer(wav_file.readframes(-1), np.int64)
    else:
        raise ValueError("Invalid byte depth retrieved.")

    nchannels = wav_file.getnchannels()
    assert nchannels >= 1, "Invalid number of channels retrieved."

    # normalize with the highest possible value for specified byte depth
    if amp_normalized:
        samples = samples / 2 ** (bd * 8 - 1)  # 1 bit for sign

    # separate channels
    samples.shape = -1, nchannels
    samples = samples.T

    metadata = {"fs": wav_file.getframerate(), "channels": nchannels, "byteDepth": bd,
                "samples_n_per_channel": wav_file.getnframes(),
                "time_duration": wav_file.getnframes() / float(wav_file.getframerate())}

    return metadata, samples
