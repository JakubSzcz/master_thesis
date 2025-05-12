import matplotlib.pyplot as plt
import numpy as np


def sierpinski_triangle(iteration, vertices):
    if iteration == 0:
        # Base case: return the vertices of the triangle
        return [vertices]
    else:
        # Recursive case: divide the triangle into 3 smaller triangles
        v1, v2, v3 = vertices
        mid1 = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)
        mid2 = ((v2[0] + v3[0]) / 2, (v2[1] + v3[1]) / 2)
        mid3 = ((v3[0] + v1[0]) / 2, (v3[1] + v1[1]) / 2)

        # Recursively generate the smaller triangles
        triangles = []
        triangles.extend(sierpinski_triangle(iteration - 1, (v1, mid1, mid3)))
        triangles.extend(sierpinski_triangle(iteration - 1, (mid1, v2, mid2)))
        triangles.extend(sierpinski_triangle(iteration - 1, (mid3, mid2, v3)))

        return triangles


def plot_sierpinski_triangle(iteration, vertices, save=False):

    for i in range(iteration):
        triangles = sierpinski_triangle(i, vertices)

        plt.figure(figsize=(8, 8))
        for triangle in triangles:
            x = [triangle[0][0], triangle[1][0], triangle[2][0], triangle[0][0]]
            y = [triangle[0][1], triangle[1][1], triangle[2][1], triangle[0][1]]
            plt.plot(x, y, 'k-', linewidth=2)

        plt.axis('off')
        plt.tight_layout()
        if save:
            plt.savefig(f'sierpinski_{i + 1}.jpg', dpi=300, bbox_inches='tight')
        plt.show()
        plt.show()


# Define the initial vertices of the triangle
vertices = ((0, 0), (1, 0), (0.5, np.sqrt(3) / 2))

# Plot the 1st iteration
plot_sierpinski_triangle(4, vertices, save=False)
