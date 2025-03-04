import cv2
import numpy as np

# Read the image
img = cv2.imread("../resources/Lenna.png", cv2.IMREAD_UNCHANGED)

# Convert BGR to RGB (if needed)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)  # (height, width, channels)