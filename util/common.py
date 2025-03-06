import matplotlib.pyplot as plt
import util.math as mymath


def print_attr_vs_orig(attractor: list, original: list):
    plt.figure()
    plt.grid()
    plt.title("Attractor vs Original function")
    plt.plot(attractor, label="Attractor")
    plt.plot(original, label="Original function", linestyle="--")
    plt.legend()
    plt.show()

    print(f"RMS = {mymath.d_rms(attractor, original)}")


def print_signal(signal: list, title: str):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.plot(signal)
    plt.show()
