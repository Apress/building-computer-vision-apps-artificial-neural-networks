import cv2
import numpy as np

# Load an image
image = cv2.imread("images/boat.jpg")
# convert the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original Grayscale Image", image)

# Binarization using adaptive thresholding and simple mean
binarized = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
cv2.imshow("Binarized Image with Simple Mean", binarized)

# Binarization using adaptive thresholding and Gaussian Mean
binarized = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
cv2.imshow("Binarized Image with Gaussian Mean", binarized)

cv2.waitKey(0)