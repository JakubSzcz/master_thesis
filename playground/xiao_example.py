import util.math as utmat

R = [32, 38]
D = [ [32,38,55,72], [40,48,50,58]]

D_down = [[(d[i] + d[i+1])/2 for i in range(0, len(d), 2)] for d in D]
print(f"Down sampled D = {D_down}")

min_rms = 1000000
alpha, beta = 0, 0
for d in D_down:
    print(f"For d = {d}")
    alpha, beta = utmat.calculate_alpha_beta(d, R)
    r_recon = utmat.transform(alpha, beta, d)
    rms = utmat.d_rms(r_recon, R)
    print(f"alpha = {alpha}, beta = {beta}, rms = {rms}")
    if rms < min_rms:
        min_rms = rms
print(min_rms)

temp = D_down[1]
for i in range(10):
    temp = utmat.transform(alpha, beta, temp)
    print(temp)