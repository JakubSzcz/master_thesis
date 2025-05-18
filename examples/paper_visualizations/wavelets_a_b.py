import pywt
import numpy as np
import matplotlib.pyplot as plt


wavelet_name = 'db4'
wavelet = pywt.Wavelet(wavelet_name)

level = 5  # Level of refinement for visualization

phi, psi, x = wavelet.wavefun(level=level)

# Adjust x to be centered around 0 for better visualization
if len(x) > 0:
    x = x - x[len(x) // 2]
else:
    print("Error: x grid for wavelet function is empty. Cannot proceed.")
    exit()

# --- Visualization of Scaling ---
plt.figure(figsize=(12, 10))
plt.suptitle(f"Skalowanie oraz translacja falki matki '{wavelet_name}'", fontsize=16)

plt.subplot(2, 1, 1)
plt.title("Skalowanie")

# Original mother wavelet
plt.plot(x, psi, label='Falka matka (a=1, b=0)', lw=2)

# Scaled versions
scales = [0.5, 2.0]  # s > 1: wider (stretched), 0 < s < 1: narrower (compressed)

for s_val in scales:

    # Create the transformed x-coordinates for interpolation
    x_transformed_for_interp = x / s_val

    scaled_psi_values = (1 / np.sqrt(s_val)) * np.interp(x_transformed_for_interp, x, psi, left=0, right=0)

    # Plotting against the original x-axis to show the effect of scaling
    plt.plot(x, scaled_psi_values, label=f'Falka przeskalowana (a={s_val})', linestyle='--')

plt.xlabel("t [s]")
plt.ylabel("Amplituda")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)

# --- Visualization of Translation (Shift) ---
plt.subplot(2, 1, 2)
plt.title("Translacja")

# Original mother wavelet (at scale s=1)
plt.plot(x, psi, label='Falka matka (a=1, b=0)', lw=2)

# Translated versions
translations = [-2.0, 3.0]  # b values (time shifts)

for t_shift in translations:

    x_shifted_for_interp = x - t_shift

    translated_psi_values = np.interp(x_shifted_for_interp, x, psi, left=0, right=0)

    plt.plot(x, translated_psi_values, label=f'Falka przesunięta (b={t_shift})', linestyle='-.')

plt.xlabel("t [s]")
plt.ylabel("Amplituda")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
plt.show()
