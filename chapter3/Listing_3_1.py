from __future__ import print_function
import cv2
import numpy as np

# Load image
imagePath = "images/zebra.png"
image = cv2.imread(imagePath)

# Get image shape which returns height, width, and channels as a tuple. Calculate the aspect ratio
(h, w) = image.shape[:2]
aspect = w / h

# lets resize the image to  decrease height by half of the original image.
# Remember, pixel values must be integers.
height = int(0.5 * h)
width =  int(height * aspect)

# New image dimension as a tuple
dimension = (height, width)
resizedImage = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized Image", resizedImage)

# Resize using x and y factors
resizedWithFactors = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LANCZOS4)
cv2.imshow("Resized with factors", resizedWithFactors)
cv2.waitKey(0)
