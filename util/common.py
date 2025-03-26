import matplotlib.pyplot as plt
import util.math as mymath


def print_attr_vs_orig(attractor: list, original: list, n_range: int = None, n_domains: int = None, title: str = None):
    plt.figure()
    plt.grid()
    if title is None:
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


def print_signal(signal: list, title: str, plot_ranges_size: int = None):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.plot(signal)
    if plot_ranges_size is not None:
        for p in range(0, len(signal), plot_ranges_size):
            plt.axvline(x=p, color='red', linestyle='--', alpha=0.7)
    plt.show()
