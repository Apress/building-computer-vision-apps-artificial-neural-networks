import cv2

# Load a noisy image
saltpepperImage = cv2.imread("images/salt-pepper.jpg")
cv2.imshow("Original noisy image", saltpepperImage)

# Median filtering for noise reduction
blurredImage3 = cv2.medianBlur(saltpepperImage, 3)
cv2.imshow("Blurred image 3", blurredImage3)

# Median filtering for noise reduction
blurredImage5 = cv2.medianBlur(saltpepperImage, 5)
cv2.imshow("Blurred image 5", blurredImage5)


cv2.waitKey(0) 