import cv2

# Load image
imagePath = "images/zebrasmall.png"
image = cv2.imread(imagePath)

# Flip horizontally
flippedHorizontally = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flippedHorizontally)
cv2.waitKey(-1)

# Flip vertically
flippedVertically = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flippedVertically)
cv2.waitKey(-1)

# Flip horizontally and then vertically
flippedHV = cv2.flip(image, -1)
cv2.imshow("Flipped H and V", flippedHV)
cv2.waitKey(0)
