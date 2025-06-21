import pandas as pd
import matplotlib.pyplot as plt

# n vs bh
path = "n_vs_bh.csv"
df = pd.read_csv(path)

# calc mean and std for each iteration
result = df.groupby(['n', 'bh']).agg(
    psnr_mean=("psnr", "mean"),
    psnr_std=("psnr", "std"),
    mre_mean=("mre", "mean"),
    mre_std=("mre", "std")
).reset_index()

# grouping
# drop n == 19
result = result[result['n'] != 19]

bh_unique = result['bh'].unique()
n_uniques = result['n'].unique()
data = [result[result['bh'] == 3], result[result['n'] == 15]]

for i, d in enumerate(data):
    parameter_name = ("N", "BH", 3, r"N=$2^x$") if i == 0 else ("BH", "N", 15, "BH")
    suffix  = fr"${{{parameter_name[1]}}}={{{parameter_name[2]}}}$" if i == 0 else fr"${{{parameter_name[1]}}}=2^{{{parameter_name[2]}}}$"
    plt.figure(figsize=(8, 6))
    plt.suptitle(
        fr"Porównanie metryk $\mathrm{{PSNR}}$ oraz $\mathrm{{MRE}}$" + "\n" + fr"zrekonstruowanego sygnału w zależności od ${{{parameter_name[0]}}}$ oraz dla "+ suffix,
        fontsize=16)

    x = d[parameter_name[0].lower()]
    y_p = d["psnr_mean"]
    y_m = d["mre_mean"]
    y_p_std = d["psnr_std"]
    y_m_std = d["mre_std"]

    plt.subplot(1, 2, 1)
    plt.title(r"$\mathrm{PSNR}$")
    plt.plot(x, y_p, label=r"$\mathrm{PSNR}$")
    plt.fill_between(x, y_p - y_p_std, y_p + y_p_std, alpha=0.2)
    plt.xlabel(parameter_name[3], fontsize=13)
    plt.ylabel(r"$dB$", fontsize=13)
    plt.xticks(x)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title(r"$\mathrm{MRE}$")
    plt.plot(x, y_m, label="MRE", color="orange")
    plt.fill_between(x, y_m - y_m_std, y_m + y_m_std, alpha=0.2, color="orange")
    plt.xlabel(parameter_name[3], fontsize=13)
    plt.xticks(x)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# plt.figure(figsize=(12, 9))
# plt.suptitle(
#     r"Porównanie metryk jakości $\mathrm{PSNR}$ oraz $\mathrm{MRE}$ zrekonstruowanego" + "\nsygnału za pomocą FWC w zależności od wartości parametrów " + r"$N$ oraz $BH$",
#     fontsize=16)
#
# for i, d in enumerate(data):
#     parameter_name = ("N", "BH", 3, r"N=$2^x$") if i == 0 else ("BH", "N", 15, "BH")
#     x = d[parameter_name[0].lower()]
#     y_p = d["psnr_mean"]
#     y_m = d["mre_mean"]
#     y_p_std = d["psnr_std"]
#     y_m_std = d["mre_std"]
#
#     plt.subplot(2, 2, (i * 2 + 1))
#     plt.title(
#         fr"$\mathrm{{PSNR}}$ dla ${{{parameter_name[1]}}}={{{parameter_name[2]}}}$")
#     plt.plot(x, y_p, label=r"$\mathrm{PSNR}$")
#     plt.fill_between(x, y_p - y_p_std, y_p + y_p_std, alpha=0.2)
#     plt.xlabel(parameter_name[3], fontsize=13)
#     plt.ylabel(r"$dB$", fontsize=13)
#     plt.xticks(x)
#     plt.legend()
#     plt.grid(True)
#
#     plt.subplot(2, 2, (i * 2 + 2))
#     plt.title(
#         rf"$\mathrm{{MRE}}$ dla ${{{parameter_name[1]}}}={{{parameter_name[2]}}}$")
#     plt.plot(x, y_m, label="MRE", color="orange")
#     plt.fill_between(x, y_m - y_m_std, y_m + y_m_std, alpha=0.2, color="orange")
#     plt.xlabel(parameter_name[3], fontsize=13)
#     plt.xticks(x)
#     plt.legend()
#     plt.grid(True)
#
# plt.tight_layout()
# plt.show()

# summary
# n vs bh
path_v2 = "n_vs_bh_v2.csv"
df2 = pd.read_csv(path_v2)

# grouping
result2 = df2.groupby(['n']).agg(
    psnr_max=("psnr", "max"),
    psnr_min=("psnr", "min"),
    mre_max=("mre", "max"),
    mre_min=("mre", "min"),
).reset_index()
result2["psnr_delta"] = result2["psnr_max"] - result2["psnr_min"]
result2["mre_delta"] = result2["mre_max"] - result2["mre_min"]

# print(result2)
# # Assume you already have `result` DataFrame from the groupby
# result['exec_total_time'] = result['exec_time_comp_mean'] + result['exec_time_decomp_mean']
#
# # Pivot the data to form a 2D grid suitable for surface plotting
# pivot = result.pivot(index='n', columns='bh', values='exec_total_time')
#
# # Get X, Y, Z data
# X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
# Z = pivot.values
#
# # Plot
# fig = plt.figure(figsize=(9, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
#
# # Labels
# ax.set_xlabel("Wysokość bloku")
# ax.set_xticks(bh_unique)
# ax.set_ylabel(r"Liczba próbek [$2^i$]")
# ax.set_yticks(n_uniques)
# ax.set_zlabel("Czas wykonywania [s]")
#
# fig.colorbar(surf, shrink=0.4, aspect=8)
# plt.suptitle("Wykres całkowitego czasu kompresji oraz dekomprwsji\nalgorytmu FWC w zależności parametrów N oraz BH",
#              fontsize=16)
# plt.tight_layout()
# plt.show()
