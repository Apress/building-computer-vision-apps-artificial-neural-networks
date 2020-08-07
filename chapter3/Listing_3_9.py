import cv2
import numpy as np

# Load an image
natureImage = cv2.imread("images/nature.jpg")
cv2.imshow("Original Nature Image", natureImage)

# Create a rectangular mask
maskImage = cv2.rectangle(np.zeros(natureImage.shape[:2], dtype="uint8"),
                     (50, 50), (int(natureImage.shape[1])-50, int(natureImage.shape[0] / 2)-50), (255, 255, 255), -1)
cv2.imshow("Mask Image", maskImage)
cv2.waitKey(0)

# Using bitwise_and operation perform masking. Notice the mask=maskImage argument
masked = cv2.bitwise_and(natureImage, natureImage, mask=maskImage)
cv2.imshow("Masked image", masked)
cv2.waitKey(0)
