import cv2
import numpy as np
from matplotlib import pyplot as plot

# Read a color image
image = cv2.imread("images/nature.jpg")

cv2.imshow("Original Color Image", image)
#Remember OpenCV stores color in BGR sequence instead of RBG.
colors = ("blue", "green", "red")
# calculate histogram
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [32], [0,256])
    # Plot histogram graph
    plot.plot(hist, color=color)

plot.title("RGB Color Histogram")
plot.xlabel("Bins")
plot.ylabel("Number of Pixels")
plot.show()
cv2.waitKey(0)


