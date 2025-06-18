import pandas as pd
import pywt

# type vs quality
path = "wavelet_occurrences.csv"
df = pd.read_csv(path)

df_t10 = df['best_wavelet_psnr'].value_counts().head(10).reset_index().rename(
    columns={'index': 'count', 'best_wavelet_psnr': 'wavelet'})

new_data = []

for _, row in df_t10.iterrows():
    wavelet_name = row['wavelet']
    wavelet_family = "Coiflet" if "coif" in wavelet_name else "Daubechies"
    temp = {
        "Rodzina falki": wavelet_family,
        "Identyfikator PyWavelet": wavelet_name,
        "N falki": row['count'],
        "Długość filtra QMF": pywt.Wavelet(wavelet_name).dec_len
    }
    new_data.append(temp)

df_new = pd.DataFrame(new_data)
print(df_new)
print(df_new["Identyfikator PyWavelet"].tolist())
