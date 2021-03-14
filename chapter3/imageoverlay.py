import cv2

# Read the two images

image1 = cv2.imread("images/nature.jpg")
image2 = cv2.imread("images/zebrasmall.png")

cv2.imshow("Nature", image1)
cv2.imshow("Zebra", image2)
cv2.waitKey(0)


