import cv2
import numpy as np

# Load the image
park = cv2.imread("images/nature.jpg")
cv2.imshow("Original Park Image", park)

#Define the kernal
kernal = (3,3)
blurred3x3 = cv2.blur(park,kernal)
cv2.imshow("3x3 Blurred Image", blurred3x3)

blurred5x5 = cv2.blur(park,(5,5))
cv2.imshow("5x5 Blurred Image", blurred5x5)

blurred7x7 = cv2.blur(park, (7,7))
cv2.imshow("7x7 Blurred Image", blurred7x7)
cv2.waitKey(0)
