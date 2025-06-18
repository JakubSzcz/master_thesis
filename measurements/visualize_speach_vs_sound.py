import util.common as common
import matplotlib.pyplot as plt

sound_t, _, _ = common.read_example_file(n_samples=2 ** 13, file_type="sound",
                                         suppress_logs=True)
speach_t, _, _ = common.read_example_file(n_samples=2 ** 13, file_type="speach",
                                          suppress_logs=True)
plt.figure(figsize=(12, 7))
plt.suptitle("Porównanie przebiegów czasowych sygnału typu muzyka oraz mowa", fontsize=16)

plt.subplot(1, 2, 1)
plt.title("Muzyka")
plt.plot(sound_t, label="Muzyka")
plt.xlabel("Numer próbki")
plt.ylabel("Amplituda")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Mowa")
plt.plot(speach_t, label="Mowa", color="orange")
plt.xlabel("Numer próbki")
plt.ylabel("Amplituda")
plt.grid(True)

plt.tight_layout()
plt.show()
