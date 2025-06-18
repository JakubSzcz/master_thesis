import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from util.math import remove_outliers_df

# type vs quality
path = "type_vs_quality.csv"
df = pd.read_csv(path)

category_order = ['classic', 'pop', 'pop_sing', 'noise', 'speach']
custom_labels = ['Muzyka klasyczna', 'Muzyka pop', 'Muzyka pop\n(ze śpiewem)', 'Tło i hałas', 'Mowa']
palette = ['#AEC6CF', '#FFB347', '#B39EB5', '#77DD77', '#F49AC2']

# PSNR
plt.figure(figsize=(8, 6))

sns.boxplot(x='file_cat', y='psnr', hue='file_cat', data=remove_outliers_df(df, "psnr"), order=category_order,
            palette=palette, legend=False)

plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels)
plt.xlabel('Kategoria sygnału dźwiękowego')
plt.ylabel('PSNR [dB]')
plt.grid()
plt.title('Wykres pudełkowy jakości kompresji PSNR\nw zależności od typu sygnału dźwiękowego', fontsize=16)
plt.show()

# MRE
plt.figure(figsize=(8, 6))

sns.boxplot(x='file_cat', y='mre', hue='file_cat', data=remove_outliers_df(df, "mre"), order=category_order,
            palette=palette, legend=False)

plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels)
plt.xlabel('Kategoria sygnału dźwiękowego')
plt.ylabel('MRE')
plt.grid()
plt.title('Wykres pudełkowy jakości kompresji MRE\nw zależności od typu sygnału dźwiękowego', fontsize=16)
plt.show()

# MRE
plt.figure(figsize=(8, 6))

sns.boxplot(x='file_cat', y='exec_time', hue='file_cat', data=remove_outliers_df(df, "exec_time"), order=category_order,
            palette=palette, legend=False)

plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels)
plt.xlabel('Kategoria sygnału dźwiękowego')
plt.ylabel('Czas wykonywania [s]')
plt.grid()
plt.title('Wykres pudełkowy czasu kompresji \nw zależności od typu sygnału dźwiękowego', fontsize=16)
plt.show()
