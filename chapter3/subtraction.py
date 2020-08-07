import cv2
import numpy as np


image1Path = "images/cat1.png"
image2Path = "images/cat2.png"

image1 = cv2.imread(image1Path)
image2 = cv2.imread(image2Path)

# resize the two images to make them of the same dimensions. This is a must to subtract two images
resizedImage1 = cv2.resize(image1,(int(500*image1.shape[1]/image1.shape[0]), 500),interpolation=cv2.INTER_AREA)
resizedImage2 = cv2.resize(image2,(int(500*image2.shape[1]/image2.shape[0]), 500),interpolation=cv2.INTER_AREA)

cv2.imshow("Cat 1", resizedImage1)
cv2.imshow("Cat 2", resizedImage2)

# Subtract image 1 from 2
cv2.imshow("Diff Cat1 and Cat2",cv2.subtract(resizedImage2, resizedImage1))
cv2.waitKey(-1)


# subtract images 2 from 1
subtractedImage = cv2.subtract(resizedImage1, resizedImage2)
cv2.imshow("Cat2 subtracted from Cat1", subtractedImage)
cv2.waitKey(-1)

# Numpy Subtraction Cat2 from Cat1
subtractedImage2 = resizedImage2 - resizedImage1
cv2.imshow("Numpy Subracts Images", subtractedImage2)
cv2.waitKey(-1)

# A constant subtraction
subtractedImage3 = resizedImage1 - 50
cv2.imshow("Constant Subracted from the image", subtractedImage3)
cv2.waitKey(-1)

