import cv2
import numpy as np
from skimage import feature as sk

#Load an image from the disk
image = cv2.imread("images/obama.jpg")
#Resize the image.
image = cv2.resize(image,(int(image.shape[0]/5),int(image.shape[1]/5)))

# HOG calculation
(HOG, hogImage) = sk.hog(image, orientations=9, pixels_per_cell=(8, 8),
	cells_per_block=(2, 2), visualize=True, transform_sqrt=True, block_norm="L2-Hys", feature_vector=True)

print("Image Dimension",image.shape)
print("Feature Vector Dimension:", HOG.shape)

#showing the original and HOG images
cv2.imshow("Original image", image)
cv2.imshow("HOG Image", hogImage)
cv2.waitKey(0)
