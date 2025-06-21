import matplotlib.pyplot as plt


def get_n_coefficients(N, L, BH):
    crs = []
    for l in L:
        T = N
        for _ in range(BH):
            T = (T + l - 1) // 2
        M = ((T + l - 1) // 2) * 2
        d = (N * 16) / (T * 48 + M * 16)
        crs.append(d)
    return crs


Ns = [2 ** 10, 2 ** 19]
L = [i for i in range(2, 110, 6)]
BH = 5

crs_all = []
for N in Ns:
    crs_all.append(get_n_coefficients(N, L, BH))

plt.figure(figsize=(8, 5))
title_prefix = r"Zmiana poziomu kompresji $k$" + " dla \nróżnych wartości długości filtra " + r"$L$"
title_sufix = r" oraz parametru $\mathit{BH=5}$"
plt.suptitle(title_prefix + title_sufix, fontsize=16)

plt.subplot(1, 2, 1)
plt.title("N=$2^{12}$", fontsize=13)
plt.plot(L, crs_all[0], label="Poziom kompresji k", color="orange")
plt.xlabel(r"$L$", fontsize=13)
plt.ylabel(r"$k$", fontsize=13)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("N=$2^{18}$", fontsize=14)
plt.plot(L, crs_all[1], label="Poziom kompresji k")
plt.xlabel(r"$L$", fontsize=13)
plt.ylabel(r"$k$", fontsize=13)
plt.grid(True)

plt.tight_layout()
plt.show()

Ns = [i for i in range(10, 21)]
crs_all_2 = []
for N in Ns:
    crs_all_2.append(get_n_coefficients(2 ** N, L, BH))
delta_crs = [c[0] - c[-1] for c in crs_all_2]

plt.figure(figsize=(7, 5))
plt.title(
    "Maksymalna różnica poziomu kompresji " + r"$\Delta_{max} \ k$ przy $L_{\text{min}}=2$ oraz $L_{\text{max}}=104$ " + "\ndla sygnałów o różnych długościach oraz dla parametru " + r"$BH=5$")
plt.plot(Ns, delta_crs, label="Różnica poziomu kompresji k")
plt.xlabel(r"N=$2^x$", fontsize=13)
plt.ylabel(r"$\Delta_{max} \ k$", fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()
