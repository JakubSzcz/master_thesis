import numpy as np
import matplotlib.pyplot as plt
import util.math as mymath

n_samples = 10000

t = np.linspace(0, 1, n_samples)
X = np.sin(t*np.pi)
w_c = 0.5

I = [X[0:int(n_samples/2)], X[int(n_samples/2):]]
J = []
for j in range(4):
    J.append(X[j*int(n_samples/4):(j+1)*int(n_samples/4)])

# encoding
codded = []
for r_i, r in enumerate(J):
    min_rms = 10000
    fit_d_i = 0
    fit_alpha = 1
    fit_beta = 0
    for d_i, d in enumerate(I):
        d_down = mymath.downsample(d)
        alpha, beta = mymath.calculate_alpha_beta(d_down, r)
        d_rms_cal = mymath.d_rms(d_down, mymath.transform(alpha, beta, d_down))
        if d_rms_cal < min_rms:
            fit_d_i = d_i
            fit_alpha = alpha
            fit_beta = beta
            min_rms = d_rms_cal

    codded.append((fit_d_i, fit_alpha, fit_beta))

#decoding
decoded = J
to_decode_from = I
# alpha, beta
w = [(0.5, 0), (0.5, 0.25), (0.5, 0.25), (0.5, 0.5)]
for _ in range(20):
    for ind, phi in enumerate(codded):
        decoded[ind] = mymath.transform(phi[1], phi[2], mymath.downsample(to_decode_from[phi[0]]))
    temp = np.array(decoded).flatten().tolist()
    to_decode_from = [temp[0:int(n_samples/2)], temp[int(n_samples/2):]]

result = np.array(decoded).flatten()
plt.figure()
plt.grid()
plt.title("Attractor vs Original function")
plt.plot(t, result, label="Attractor")
plt.plot(t, X, label="Original function")
plt.legend()
plt.show()

print(f"RMS = {mymath.d_rms(result, X)}")
