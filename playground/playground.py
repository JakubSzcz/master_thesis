import matplotlib.pyplot as plt


def cantor_set(ax, x, y, length, depth):
    if depth == 0:
        ax.plot([x, x + length], [y, y], color="black", linewidth=2)
    else:
        # Draw the line for the current segment
        ax.plot([x, x + length], [y, y], color="black", linewidth=2)

        # Recursive calls for left and right segments after removing the middle third
        cantor_set(ax, x, y - 1, length / 3, depth - 1)
        cantor_set(ax, x + 2 * length / 3, y - 1, length / 3, depth - 1)


#Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1)
ax.set_ylim(-6, 1)
ax.axis("off")

# Generate and plot the Cantor set
cantor_set(ax, 0, 0, 1, 5)

plt.show()
