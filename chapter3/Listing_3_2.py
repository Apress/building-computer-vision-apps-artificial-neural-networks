from __future__ import print_function
import cv2
import numpy as np

#Load image
imagePath = "images/soccer-in-green.jpg"
image = cv2.imread(imagePath)

#Define translation matrix
translationMatrix = np.float32([[1,0,50],[0,1,20]])

#Move the image
movedImage = cv2.warpAffine(image, translationMatrix, (image.shape[1], image.shape[0]))

cv2.imshow("Moved image", movedImage)
cv2.waitKey(0)