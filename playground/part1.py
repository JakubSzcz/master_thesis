import numpy as np
import matplotlib.pyplot as plt
import pywt

# generating signal
f_s = 1000
t_start = 0
t_stop = 5
n = (t_stop-t_start)*f_s

f_1 = 1
f_2 = 20
f_3 = 32

t = np.linspace(t_start, t_stop, n)
x_org = np.sin(2*np.pi*f_1*t) + np.sin(2*np.pi*f_2*t) + np.sin(2*np.pi*f_3*t)

# plt.figure()
# plt.plot(t, x_org)
# plt.grid()
# plt.title("Original signal")
# plt.show()


(ca, cd) = pywt.wavedec(x_org, 'haar', level=1)

plt.figure(1,(13,13))
plt.subplot(3,1,1)
plt.plot(ca)
plt.grid()
plt.title("CA")
plt.subplot(3,1,2)
plt.plot(cd)
plt.grid()
plt.title("CD")
plt.subplot(3,1,3)
plt.plot(t, x_org)
plt.grid()
plt.title("Original signal")
plt.show()
