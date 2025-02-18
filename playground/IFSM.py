import numpy as np
import matplotlib.pyplot as plt

# Define the base space X = [0, 1]
X = np.linspace(0, 1, 1000)

# Define the function u(x) = 4x(1-x)
def u(x):
    return 4 * x * (1 - x)

# Define the contraction maps w_1 and w_2
def w_1(x):
    return 0.6 * x

def w_2(x):
    return 0.6 * x + 0.4

# Define the grey level maps phi_1 and phi_2
def phi_1(t):
    return 0.5 * t + 0.2

def phi_2(t):
    return 0.5 * t + 0.5

# Define the fractal components f_i(x)
def f_1(x):
    # Compute the preimage w_1^{-1}(x)
    preimage = x / 0.6
    if 0 <= preimage <= 1:  # Check if preimage is in X
        return phi_1(u(preimage))
    else:
        return 0

def f_2(x):
    # Compute the preimage w_2^{-1}(x)
    preimage = (x - 0.4) / 0.6
    if 0 <= preimage <= 1:  # Check if preimage is in X
        return phi_2(u(preimage))
    else:
        return 0

# Define the fractal transform Tu(x)
def Tu(x):
    return f_1(x) + f_2(x)

# Compute the original function u(x) and the fractal transform Tu(x)
u_values = u(X)
Tu_values = np.array([Tu(x) for x in X])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X, u_values, label="Original Function $u(x)$", color="blue")
plt.plot(X, Tu_values, label="Fractal Transform $Tu(x)$", color="red", linestyle="--")
plt.title("Fractal Transform using IFSM")
plt.xlabel("$x$")
plt.ylabel("Function Value")
plt.legend()
plt.grid(True)
plt.show()