import random

import numpy as np
from matplotlib import pyplot as plt


def get_cords(init_points):
    x_cord = []
    y_cord = []
    for point in init_points:
        x_cord.append(point[0])
        y_cord.append(point[1])

    return x_cord, y_cord



#points = [(random.randint(0,10), random.randint(0,10))for _ in range(3)]
points = [(0,0), (10,0), (5,5*np.sqrt(3))]
x_cord, y_cord = get_cords(points)
plt.figure()
plt.scatter(x_cord, y_cord)
plt.title("Random points - initial setup")
plt.show()

# IFS
ITER = 4

def w1(points: list):
    return [(point[0]*0.5, point[1] * 0.5)for point in points]

def w2(points: list):
    return [(point[0]*0.5 + 0.25, point[1] * 0.5 + 0.5)for point in points]

def w3(points: list):
    return [(point[0]*0.5 + 0.5, point[1] * 0.5)for point in points]

to_plot = []
for i in range(ITER):

    w1_points = w1(points)
    w2_points = w2(points)
    w3_points = w3(points)

    points += w1_points + w2_points + w3_points
    to_plot += points
x_cord, y_cord = get_cords(to_plot)
plt.figure()
plt.scatter(x_cord, y_cord)
plt.title("Sierpinski Random Generation - iteration:")
plt.show()
