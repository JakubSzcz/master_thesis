import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi


def koch_iteration(points):
    """Generate the next iteration of the Koch curve."""
    new_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p4 = points[i + 1]

        # Calculate the vector from p0 to p4
        v = p4 - p0

        # Points that divide the segment in 3 equal parts
        p1 = p0 + v / 3
        p3 = p0 + 2 * v / 3

        # Middle point (the peak of the equilateral triangle)
        # Rotate the vector p1->p3 by 60 degrees around p1
        angle = pi / 3  # 60 degrees in radians
        x = (p3[0] - p1[0]) * cos(angle) - (p3[1] - p1[1]) * sin(angle)
        y = (p3[0] - p1[0]) * sin(angle) + (p3[1] - p1[1]) * cos(angle)
        p2 = p1 + np.array([x, y])

        # Add the new points to the list
        new_points.extend([p0, p1, p2, p3])

    new_points.append(points[-1])
    return np.array(new_points)


def create_koch_curve(iterations=4, save=False):
    """Create a Koch curve with a given number of iterations."""
    # Start with a horizontal line segment
    points = np.array([[0, 0], [1, 0]])

    # Show the initial line segment in its own figure
    plt.figure(figsize=(10, 3))
    plt.plot(points[:, 0], points[:, 1], color="black", linewidth=2)

    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.2, 0.5)
    plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(f'koch_{0}.jpg', dpi=300, bbox_inches='tight')
    plt.show()

    # Perform the specified number of iterations
    for i in range(iterations):
        points = koch_iteration(points)

        # Plot the current iteration in a new figure
        plt.figure(figsize=(10, 3))
        plt.plot(points[:, 0], points[:, 1], color="black", linewidth=2)
        plt.axis('equal')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.2, 0.5)
        plt.axis('off')
        plt.tight_layout()
        if save:
            plt.savefig(f'koch_{i + 1}.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    return points


# Create and display the Koch curve with 4 iterations
final_points = create_koch_curve(4, save=False)