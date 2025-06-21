import pandas as pd
import matplotlib.pyplot as plt
import util.common as common
import seaborn as sns

# Load CSV
df = pd.read_csv("k_vs_quality.csv")
df["k_new"] = df.apply(lambda row: common.get_compression_rate(2 ** row["n"], row["BH"], row["wavelet"]), axis=1)

df["BH"] = df["BH"].astype(int)

# Unique wavelets
unique_pairs_list = list(df[["L", "wavelet"]].drop_duplicates().itertuples(index=False, name=None))

for L, wavelet in unique_pairs_list:
    subset_psnr = df[df["wavelet"] == wavelet].pivot(index="n", columns="BH", values="psnr")
    subset_k = df[df["wavelet"] == wavelet].pivot(index="n", columns="BH", values="k_new")

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(rf"Mapy cieplne $\mathrm{{PSNR}}$ oraz $k$ dla falki {wavelet} – $L={{{L}}}$", fontsize=19)

    # PSNR heatmap
    sns.heatmap(subset_psnr, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title(r"$\mathrm{PSNR}$", fontsize=16)
    axes[0].set_xlabel(r"$\mathit{BH}$", fontsize=13)
    axes[0].set_ylabel(r"$N=2^n$", fontsize=13)

    # k heatmap
    sns.heatmap(subset_k, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title(r"$k$", fontsize=16)
    axes[1].set_xlabel(r"$\mathit{BH}$", fontsize=13)
    axes[1].set_ylabel(r"$N=2^n$", fontsize=13)

    # Layout
    plt.tight_layout()
    plt.show()

# # Plot for each wavelet
# for wavelet in unique_wavelets:
#     wavelet_df = df[df["wavelet"] == wavelet]
#     bh_values = sorted(wavelet_df["BH"].unique())
#
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
#     fig.suptitle(f"Wavelet: {wavelet}", fontsize=16)
#
#     # PSNR subplot
#     for bh in bh_values:
#         subset = wavelet_df[wavelet_df["BH"] == bh]
#         axes[0].plot(subset["n"], subset["psnr"], label=f"BH={bh}", marker='o')
#     axes[0].set_title("PSNR vs n")
#     axes[0].set_xlabel("n")
#     axes[0].set_ylabel("PSNR")
#     axes[0].legend()
#     axes[0].grid(True)
#
#     # k subplot
#     for bh in bh_values:
#         subset = wavelet_df[wavelet_df["BH"] == bh]
#         axes[1].plot(subset["n"], subset["k"], label=f"BH={bh}", marker='o')
#     axes[1].set_title("Compression Rate (k) vs n")
#     axes[1].set_xlabel("n")
#     axes[1].set_ylabel("k")
#     axes[1].legend()
#     axes[1].grid(True)
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()
