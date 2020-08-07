import cv2
import numpy as np

# Load the image
natureImage = cv2.imread("images/nature.jpg")

# Split the image into component colors
(b,g,r) = cv2.split(natureImage)

# show the blue image
cv2.imshow("Blue Image", b)

# Show the green image
cv2.imshow("Green image", g)

# Show the red image
cv2.imshow("Red image", r)

cv2.waitKey(0)
