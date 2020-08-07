from __future__ import print_function
import cv2
import numpy as np

image1Path = "images/zebra.png"
image2Path = "images/nature.jpg"
image3Path = "images/receipt.jpg"
image4Path = "images/cat1.png"
image5Path = "images/cat2.png"

image1 = cv2.imread(image1Path)
image2 = cv2.imread(image2Path)
image3 = cv2.imread(image3Path)
image4 = cv2.imread(image4Path)
image5 = cv2.imread(image5Path)

# resize the two images to make them of the same dimension. This is a must to add two images
resizedImage1 = cv2.resize(image1,(300,300),interpolation=cv2.INTER_AREA)
resizedImage2 = cv2.resize(image2,(300,300),interpolation=cv2.INTER_AREA)
resizedImage3 = cv2.resize(image3,(300,500),interpolation=cv2.INTER_AREA)
resizedImage4 = cv2.resize(image4,(int(500*image4.shape[1]/image4.shape[0]), 500),interpolation=cv2.INTER_AREA)
resizedImage5 = cv2.resize(image5,(int(500*image4.shape[1]/image4.shape[0]), 500),interpolation=cv2.INTER_AREA)

cv2.imshow("Screen 1", resizedImage4)
cv2.imshow("Screen 2", resizedImage5)

cv2.imshow("Diff",cv2.subtract(resizedImage4, resizedImage5))
cv2.waitKey(-1)


# subtract images
subtractedImage = cv2.subtract(resizedImage1, resizedImage2)
cv2.imshow("Subtracted Image", subtractedImage)
cv2.waitKey(-1)

subtractedImage2 = resizedImage1 - resizedImage2
cv2.imshow("Numpy Sub Image", subtractedImage2)
cv2.waitKey(-1)

subtractedImage3 = resizedImage1 - 50
cv2.imshow("Constant Sub Image", subtractedImage3)
cv2.waitKey(-1)

