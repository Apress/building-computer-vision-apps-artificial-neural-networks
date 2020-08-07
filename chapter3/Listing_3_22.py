import cv2
import numpy as np

# Load an image
image = cv2.imread("images/sudoku.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Blurred image", image)

# Binarize the image
(T,binarized) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("Binarized image", binarized)

# Canny function for edge detection
canny = cv2.Canny(binarized, 0, 255)
cv2.imshow("Canny Edges", canny)

(contours, hierarchy) = cv2.findContours(canny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours determined are ", format(len(contours)))

copiedImage = image.copy()
cv2.drawContours(copiedImage, contours, -1, (0,255,0), 2)
cv2.imshow("Contours", copiedImage)
cv2.waitKey(0)
