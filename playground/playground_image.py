import cv2
import random
import numpy as np

# Read the image
img = cv2.imread("../resources/lena_gray.png", cv2.IMREAD_GRAYSCALE)

n_samples = img.shape[0]  # samples in base signal
block_size = 16  # size of range block (domain block = *2)
n_domains = 300  # number of Domains blocks
n_range = int(n_samples / block_size) ** 2  # number of range block
decoding_iterations = 10  # iterations number while decoding

R = []
for i in range(0, n_samples, block_size):
    for j in range(0, n_samples, block_size):
        R.append(img[j:j + block_size, i:i + block_size])

D = []
for _ in range(n_domains):
    i = random.randint(0, n_samples - (block_size * 2))
    j = random.randint(0, n_samples - (block_size * 2))
    D.append(img[j:j + (block_size * 2), i:i + (block_size * 2)])
