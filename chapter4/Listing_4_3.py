import cv2
import numpy as np
from matplotlib import pyplot as plot

# Read an image and convert it into grayscale
image = cv2.imread("images/nature.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Image", image)

# calculate histogram of the original image
hist = cv2.calcHist([image], [0], None, [256], [0,255])

# Plot histogram graph
#plot.figure()
plot.title("Grayscale Histogram of Original Image")
plot.xlabel("Bins")
plot.ylabel("Number of Pixels")
plot.plot(hist)
plot.show()

equalizedImage = cv2.equalizeHist(image)
cv2.imshow("Equalized Image", equalizedImage)

# calculate histogram of the original image
histEqualized = cv2.calcHist([equalizedImage], [0], None, [256], [0,255])

# Plot histogram graph
#plot.figure()
plot.title("Grayscale Histogram of Equalized Image")
plot.xlabel("Bins")
plot.ylabel("Number of Pixels")
plot.plot(histEqualized)
plot.show()
cv2.waitKey(0)


