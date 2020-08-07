from __future__ import print_function
import cv2
import numpy as np

image1Path = "images/zebra.png"
image2Path = "images/nature.jpg"

image1 = cv2.imread(image1Path)
image2 = cv2.imread(image2Path)

# resize the two images to make them of the same dimension. This is a must to add two images
resizedImage1 = cv2.resize(image1,(300,300),interpolation=cv2.INTER_AREA)
resizedImage2 = cv2.resize(image2,(300,300),interpolation=cv2.INTER_AREA)

# This is a simple addition of two images
resultant = cv2.add(resizedImage1, resizedImage2)

# Display these images to see the difference
cv2.imshow("Resized 1", resizedImage1)
cv2.waitKey(0)

cv2.imshow("Resized 2", resizedImage2)
cv2.waitKey(0)

cv2.imshow("Resultant Image", resultant)
cv2.waitKey(0)

# This is weighted addition of the two images
weightedImage = cv2.addWeighted(resizedImage1,0.7, resizedImage2, 0.3, 0)
cv2.imshow("Weighted Image", weightedImage)
cv2.waitKey(0)

imageEnhanced = 255*resizedImage1
cv2.imshow("Enhanced Image", imageEnhanced)
cv2.waitKey(0)

arrayImage = resizedImage1+resizedImage2
cv2.imshow("Array Image", arrayImage)
cv2.waitKey(0)