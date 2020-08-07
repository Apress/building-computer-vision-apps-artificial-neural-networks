import cv2
import numpy as np
from skimage import feature as sk
from matplotlib import pyplot as plt

#Load an image from the disk, resize and convert to grayscale
image = cv2.imread("images/obama.jpg")
image = cv2.resize(image, (int(image.shape[0]/5), int(image.shape[1]/5)))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# calculate Histogram of original image and plot it
originalHist = cv2.calcHist(image, [0], None, [256], [0,256])

plt.figure()
plt.title("Histogram of Original Image")
plt.plot(originalHist, color='r')

# Calculate LBP image and histogram over the LBP, then plot the histogram
radius = 3
points = 3*8
# LBP calculation
lbp = sk.local_binary_pattern(image, points, radius, method='default')
lbpHist, _ = np.histogram(lbp, density=True, bins=256, range=(0, 256))

plt.figure()
plt.title("Histogram of LBP Image")
plt.plot(lbpHist, color='g')
plt.show()

#showing the original and LBP images
cv2.imshow("Original image", image)
cv2.imshow("LBP Image", lbp)
cv2.waitKey(0)
