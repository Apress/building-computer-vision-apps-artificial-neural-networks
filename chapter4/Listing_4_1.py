import cv2
from matplotlib import pyplot as plot

# Read an image and convert it to grayscale
image = cv2.imread("images/nature.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Image", image)

# calculate histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 255])

# Plot histogram graph
plot.figure()
plot.title("Grayscale Histogram")
plot.xlabel("Bins")
plot.ylabel("Number of Pixels")
plot.plot(hist)
plot.show()
cv2.waitKey(0)
