import cv2

# Load a noisy image
noisyImage = cv2.imread("images/nature.jpg")
cv2.imshow("Original image", noisyImage)

# Bilateral Filter with
filteredImage5 = cv2.bilateralFilter(noisyImage, 5, 150, 50)
cv2.imshow("Blurred image 5", filteredImage5)

# Bilateral blurring with kernel 7
filteredImage7 = cv2.bilateralFilter(noisyImage, 7, 160, 60)
cv2.imshow("Blurred image 7", filteredImage7)

cv2.waitKey(0)
