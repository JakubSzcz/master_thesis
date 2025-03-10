import matplotlib.pyplot as plt
import util.math as mymath


def print_attr_vs_orig(attractor: list, original: list, n_range: int = None, n_domains: int = None):
    plt.figure()
    plt.grid()
    if n_range is None or n_domains is None:
        title = "Attractor vs Original function"
    else:
        title = f"Attractor vs Original function; n_d = {n_domains}, n_r = {n_range}."
    plt.title(title)
    plt.plot(attractor, label="Attractor")
    plt.plot(original, label="Original function", linestyle="--")
    plt.xlabel("Samples")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    print(f"Euclidian distance = {mymath.distance(attractor, original)}")
    #print(f"MSE = {mymath.calculate_mse(attractor, original)}")
    #print(f"RMS = {mymath.calculate_rms(attractor, original)}")


def print_signal(signal: list, title: str):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.plot(signal)
    plt.show()
