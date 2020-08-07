import cv2
import numpy as np

# Load the park image
parkImage = cv2.imread("images/park.jpg")
cv2.imshow("Original Image", parkImage)

# Gaussian blurring with 3x3 kernel height and 0 for standard deviation to calculate from the kernel
GaussianFiltered = cv2.GaussianBlur(parkImage, (5,5), 0)
cv2.imshow("Gaussian Blurred Image", GaussianFiltered)

cv2.waitKey(0)