import cv2
import skimage.feature as sk
import numpy as np

# Read an image from the disk and convert it into grayscale
image = cv2.imread("images/nature.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate GLCM of the grayscale image
glcm = sk.greycomatrix(image, [2], [0, np.pi / 2])
print(glcm)
