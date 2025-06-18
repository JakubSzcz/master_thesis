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


L = 64
BHs = [i for i in range(2, 6)]
print(BHs)
Ns = [i for i in range(10, 41)]
crs_all = []
for BH in BHs:
    bhs_temp = []
    for N in Ns:
        bhs_temp.append(get_n_coefficients(2 ** N, L, BH))
    crs_all.append(bhs_temp)

plt.figure(figsize=(10, 6))
plt.suptitle("Zmiana poziomu kompresji k dla różnych wartości\ndługości sygnału N oraz wysokości bloku BH, przy L=64",
             fontsize=16)

for i, BH in enumerate(BHs):
    plt.subplot(2, 2, i + 1)

    plt.title(f"BH={BH}")
    plt.plot(Ns, crs_all[i], label="Poziomu kompresji k")
    plt.xlabel(r"N=$2^x$")
    plt.ylabel("k")
    plt.grid(True)
plt.tight_layout()
plt.show()

Ls = [4, 24, 44, 64, 84, 104]
BH = 5

crs_all_v2 = []
for L in Ls:
    ls_temp = []
    for N in Ns:
        ls_temp.append(get_n_coefficients(2 ** N, L, BH))
    crs_all_v2.append(ls_temp)

plt.figure(figsize=(6, 5))
plt.title("Zmiana poziomu kompresji k dla różnych wartości\ndługości sygnału N oraz długości filtra L, przy BH=5")
for i, L in enumerate(Ls):
    plt.plot(Ns, crs_all_v2[i], label=f"L={L}")
plt.xlabel(r"N=$2^x$")
plt.ylabel("k")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
