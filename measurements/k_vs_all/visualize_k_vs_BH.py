import matplotlib.pyplot as plt


def get_n_coefficients(N, L, BH):
    crs = []
    T = N
    for _ in range(BH):
        T = (T + L - 1) // 2
    M = ((T + L - 1) // 2) * 2
    d = (N * 16) / (T * 48 + M * 16)
    crs.append(d)
    return crs


BHs = [i for i in range(2, 8)]
Ns = [i for i in range(8, 21, 3)]
Ls = [4, 44, 74, 104]

plt.figure(figsize=(12, 8))
plt.suptitle(r"Zmiana poziomu kompresji $k$ dla różnych wartości parametrów $L$, $N$ oraz $\mathit{BH}$",
             fontsize=17)
crs_all = []
for j, L in enumerate(Ls):
    crs_all = []
    for N in Ns:
        temp_n = []
        for BH in BHs:
            temp_n.append(get_n_coefficients(2 ** N, L, BH))
        crs_all.append(temp_n)

    plt.subplot(2, 2, j + 1)
    max_k = crs_all[-1]
    crs_all = crs_all[:len(crs_all) - 1]
    plt.title(fr"$L={{{L}}}$", fontsize=13)

    for i, crs in enumerate(crs_all):
        plt.plot(BHs, crs, label=fr"$N=2^{{{Ns[i]}}}$")
    plt.plot(BHs, max_k, label="Maksymalny poziom kompresji", linestyle="--", linewidth=2)
    plt.xlabel(r"$\mathit{BH}$", fontsize=13)
    plt.ylabel(r"$k$", fontsize=13)
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
