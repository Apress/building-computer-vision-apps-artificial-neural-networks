from __future__ import print_function
import cv2
import numpy as np

# Load image
imagePath = "images/zebrasmall.png"
image = cv2.imread(imagePath)
(h,w) = image.shape[:2]

#Define translation matrix
center = (h//2, w//2)
angle = -45
scale = 1.0

rotationMatrix = cv2.getRotationMatrix2D(center, angle, scale)

# Rotate the image
rotatedImage = cv2.warpAffine(image, rotationMatrix, (image.shape[1], image.shape[0]))

cv2.imshow("Rotated image", rotatedImage)
cv2.waitKey(0)