import cv2
import numpy as np

# Load an image
image = cv2.imread("images/scanned_doc.png")
# convert the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Grayscale Receipt", image)

# Binarize the image using thresholding
(T, binarizedImage) = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)
cv2.imshow("Binarized Receipt", binarizedImage)

# Binarization with inverse thresholding
(Ti, inverseBinarizedImage) = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Inverse Binarized Receipt", inverseBinarizedImage)
cv2.waitKey(0)
