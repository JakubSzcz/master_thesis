import pandas as pd
import matplotlib.pyplot as plt

# numba vs plain
path_numba = "numba_vs_plain_numba_v2.csv"
path_plain = "numba_vs_plain_plain_v2.csv"

df_n = pd.read_csv(path_numba)
df_p = pd.read_csv(path_plain)

# calc mean and std for each iteration
result_n = df_n.groupby(['n_sample', 'block_height'])['exec_time'].agg(
    exec_time_mean='mean',
    exec_time_std='std'
).reset_index()

result_p = df_p.groupby(['n_sample', 'block_height'])['exec_time'].agg(
    exec_time_mean='mean',
    exec_time_std='std'
).reset_index()



# printing
block_heights = result_n['block_height'].unique()
plt.figure(figsize=(10, 8))

block_height_grouped_n = result_n[result_n['block_height'] == 3]
block_height_grouped_p = result_p[result_p['block_height'] == 3]

mean_block_grouped_n = block_height_grouped_n["exec_time_mean"].agg("mean")
print(mean_block_grouped_n)
mean_block_grouped_p = block_height_grouped_p["exec_time_mean"].agg("mean")
print(mean_block_grouped_p)

x_n = block_height_grouped_n["n_sample"]
y_n = block_height_grouped_n["exec_time_mean"]
x_p = block_height_grouped_p["n_sample"]
y_p = block_height_grouped_p["exec_time_mean"]
std_n = block_height_grouped_n["exec_time_std"]
std_p = block_height_grouped_p["exec_time_std"]

plt.plot(x_n, y_n, label="Numba", color='blue')
plt.fill_between(x_n, y_n - std_n, y_n + std_n, color='blue', alpha=0.2)
plt.plot(x_p, y_p, label="Brak Numba", color='orange')
plt.fill_between(x_p, y_p - std_p, y_p + std_p, color='orange', alpha=0.2)

plt.title("Porównanie czasu wykonywania kompresji i dekompresje FWC\nz wykorzystaniem biblioteki Numba oraz bez niej")
plt.xlabel(r"Liczba próbek [$2^i$]")
plt.ylabel("Czas wykonywania [s]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# for i, block_height in enumerate(block_heights):
#     block_height_grouped_n = result_n[result_n['block_height'] == block_height]
#     block_height_grouped_p = result_p[result_p['block_height'] == block_height]
#
#     mean_block_grouped_n = block_height_grouped_n["exec_time_mean"].agg("mean")
#     print(mean_block_grouped_n)
#     mean_block_grouped_p = block_height_grouped_p["exec_time_mean"].agg("mean")
#     print(mean_block_grouped_p)
#
#     x_n = block_height_grouped_n["n_sample"]
#     y_n = block_height_grouped_n["exec_time_mean"]
#     x_p = block_height_grouped_p["n_sample"]
#     y_p = block_height_grouped_p["exec_time_mean"]
#     std_n = block_height_grouped_n["exec_time_std"]
#     std_p = block_height_grouped_p["exec_time_std"]
#
#     plt.subplot(3, 1, i + 1)
#
#     plt.plot(x_n, y_n, label="Numba", color='blue')
#     plt.fill_between(x_n, y_n - std_n, y_n + std_n, color='blue', alpha=0.2)
#     plt.plot(x_p, y_p, label="Brak Numby", color='orange')
#     plt.fill_between(x_p, y_p - std_p, y_p + std_p, color='orange', alpha=0.2)
#
#     plt.title(f"Czas wykonywania dla bloków o wysokości równej {block_height}")
#     plt.xlabel(r"Liczba próbek [$2^i$]")
#     plt.ylabel("Czas wykonywania [s]")
#     plt.legend()
#     plt.grid()
# plt.tight_layout()
# plt.show()
