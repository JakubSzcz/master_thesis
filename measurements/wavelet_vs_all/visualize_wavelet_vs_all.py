import pandas as pd
import pywt
import seaborn as sns
import matplotlib.pyplot as plt

from util.math import remove_outliers_df

# wavelet vs all
path = "wavelet_vs_all.csv"
df = pd.read_csv(path)
df = remove_outliers_df(df, "psnr")
df = remove_outliers_df(df, "mre")

category_order = ['classic', 'pop', 'pop_sing', 'noise', 'speach']
custom_labels = ['Muzyka klasyczna', 'Muzyka pop', 'Muzyka pop (ze śpiewem)', 'Tło i hałas', 'Mowa']

wavelets =  ['coif16', 'coif17', 'coif14', 'db37', 'db34', 'coif15', 'db32', 'db38', 'coif13', 'coif12']
x_ticks = []
for wavelet in wavelets:
   x_ticks.append(f"{wavelet}\n"+fr"$L={{{pywt.Wavelet(wavelet).dec_len}}}$")

palette = ['#AEC6CF', '#FFB347', '#B39EB5', '#77DD77', '#F49AC2', '#FFD1DC', '#BFD8B8', '#CBAACB', '#FFDAC1',
           '#E0BBE4']

for i, label in enumerate(category_order):
    data_df = df[df['file_cat'] == label]

    plt.figure(figsize=(11, 5))
    plt.suptitle(
        f"Jakości kompresji FWC z wykorzystaniem\n różnych falek dla sygnału typu {custom_labels[i].lower()}",
        fontsize=16)

    # PSNR
    plt.subplot(1, 2, 1)
    sns.boxplot(x='wavelet', y='psnr', hue='wavelet', data=data_df, palette=palette)

    plt.xlabel('Falka')
    plt.ylabel('PSNR [dB]')
    plt.grid()
    plt.title('PSNR')
    plt.tight_layout()

    # MRE
    plt.subplot(1, 2, 2)
    sns.boxplot(x='wavelet', y='mre', hue='wavelet', data=data_df, palette=palette)

    plt.xlabel('Falka')
    plt.ylabel('MRE')
    plt.grid()
    plt.title('MRE')
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(11, 5))
df["total_time"] = df["exec_time_comp"] + df["exec_time_decomp"]
df = remove_outliers_df(df, "total_time")

sns.boxplot(x='wavelet', y='total_time', hue='wavelet', data=df, palette=palette)

plt.xlabel('Falka\ndługość falki')
plt.xticks(ticks=range(len(wavelets)), labels=x_ticks)
plt.ylabel('Czas wykonywania [s]')
plt.grid()
plt.title('Całkowity czas kompresji w zależności od użytej falki')
plt.tight_layout()
plt.show()
