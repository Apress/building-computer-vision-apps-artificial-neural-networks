from __future__ import print_function
import cv2
import numpy as np

# Load image
imagePath = "images/zebrasmall.png"
image = cv2.imread(imagePath)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# Crop the image to get only the face of the zebra
croppedImage = image[0:150, 0:250]
cv2.imshow("Cropped Image", croppedImage)
cv2.waitKey(0)
