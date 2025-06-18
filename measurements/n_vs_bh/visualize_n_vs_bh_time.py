import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# n vs bh
path = "n_vs_bh.csv"
df = pd.read_csv(path)

# calc mean and std for each iteration
result = df.groupby(['n', 'bh']).agg(
    exec_time_comp_mean=('exec_time_comp', 'mean'),
    exec_time_comp_std=('exec_time_comp', 'std'),
    exec_time_decomp_mean=('exec_time_decomp', 'mean'),
    exec_time_decomp_std=('exec_time_decomp', 'std')
).reset_index()

# grouping
bh_unique = result['bh'].unique()
n_uniques = result['n'].unique()
data = [result[result['bh'] == 3], result[result['n'] == 15]]

plt.figure(figsize=(12, 9))
plt.suptitle("Porównanie czasów kompresji i dekompresji FWC\nw zależności od wartości parametrów N oraz BH", fontsize=16)

for i, d in enumerate(data):
    parameter_name = ("N", "BH", 3, r"N=$2^x$") if i == 0 else ("BH", "N", r"$2^{15}$", "BH")
    x = d[parameter_name[0].lower()]
    y_c = d["exec_time_comp_mean"]
    y_d = d["exec_time_decomp_mean"]
    y_c_std = d["exec_time_comp_std"]
    y_d_std = d["exec_time_decomp_std"]

    plt.subplot(2, 2, (i * 2 + 1))
    plt.title(
        f"Czas kompresji {parameter_name[1]}={parameter_name[2]}")
    plt.plot(x, y_c, label="Kompresja")
    plt.fill_between(x, y_c - y_c_std, y_c + y_c_std, alpha=0.2)
    plt.xlabel(parameter_name[3])
    plt.ylabel("Czas wykonywania [s]")
    plt.xticks(x)
    plt.grid(True)

    plt.subplot(2, 2, (i * 2 + 2))
    plt.title(
        f"Czas dekompresji {parameter_name[1]}={parameter_name[2]}")
    plt.plot(x, y_d, label="Dekompresja", color="orange")
    plt.fill_between(x, y_d - y_c_std, y_d + y_c_std, alpha=0.2, color="orange")
    plt.xlabel(parameter_name[3])
    plt.ylabel("Czas wykonywania [s]")
    plt.xticks(x)
    plt.grid(True)

plt.tight_layout()
plt.show()

# Assume you already have `result` DataFrame from the groupby
result['exec_total_time'] = result['exec_time_comp_mean'] + result['exec_time_decomp_mean']

# Pivot the data to form a 2D grid suitable for surface plotting
pivot = result.pivot(index='n', columns='bh', values='exec_total_time')

# Get X, Y, Z data
X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
Z = pivot.values

# Plot
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Labels
ax.set_xlabel("Wysokość bloku")
ax.set_xticks(bh_unique)
ax.set_ylabel(r"Liczba próbek [$2^i$]")
ax.set_yticks(n_uniques)
ax.set_zlabel("Czas wykonywania [s]")

fig.colorbar(surf, shrink=0.4, aspect=8)
plt.suptitle("Wykres całkowitego czasu kompresji oraz dekompresji\nalgorytmu FWC w zależności parametrów N oraz BH",
             fontsize=16)
plt.tight_layout()
plt.show()
