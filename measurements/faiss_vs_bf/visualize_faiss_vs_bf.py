import pandas as pd
import matplotlib.pyplot as plt

# fais vs plain
path_bf = "faiss_vs_bf_bf_v2.csv"
path_faiss = "faiss_vs_bf_faiss_v2.csv"

df_b = pd.read_csv(path_bf)
df_f = pd.read_csv(path_faiss)

# calc mean and std for each iteration
result_b = df_b.groupby(['n_sample'])['exec_time'].agg(
    exec_time_mean='mean',
    exec_time_std='std'
).reset_index()

result_f = df_f.groupby(['n_sample'])['exec_time'].agg(
    exec_time_mean='mean',
    exec_time_std='std'
).reset_index()

# printing
plt.figure(figsize=(10, 8))
plt.title("Porównanie czasów kompresji FWC z wykorzystaniem\nindeksu biblioteki FAISS oraz metody siłowej",
          fontsize=16)

x_b = result_b["n_sample"]
y_b = result_b["exec_time_mean"]
x_f = result_f["n_sample"]
y_f = result_f["exec_time_mean"]
std_b = result_b["exec_time_std"]
std_f = result_f["exec_time_std"]

plt.plot(x_b, y_b, label="Metoda siłowa + Numba", color='orange')
plt.fill_between(x_b, y_b - std_b, y_b + std_b, color='orange', alpha=0.2)
plt.plot(x_f, y_f, label="FAISS + Numba", color='blue')
plt.fill_between(x_f, y_f - std_f, y_f + std_f, color='blue', alpha=0.2)

plt.xlabel(r"Liczba próbek [$2^i$]")
plt.ylabel("Czas kodowania [s]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
