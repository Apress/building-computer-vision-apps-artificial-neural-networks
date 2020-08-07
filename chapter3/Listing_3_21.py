import cv2
import numpy as np

# Load an image
image = cv2.imread("images/sudoku.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Blurred image", image)

# Canny function for edge detection
canny = cv2.Canny(image, 50, 170)
cv2.imshow("Canny Edges", canny)

cv2.waitKey(0)
