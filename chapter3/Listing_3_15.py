import cv2

# Load a noisy image
noisyImage = cv2.imread("images/nature.jpg")
cv2.imshow("Original image", noisyImage)

# Bilateral Filter with
fileteredImag5 = cv2.bilateralFilter(noisyImage, 5, 150,50)
cv2.imshow("Blurred image 5", fileteredImag5)

# Bilateral blurring with kernal 7
fileteredImag7 = cv2.bilateralFilter(noisyImage, 7, 160,60)
cv2.imshow("Blurred image 7", fileteredImag7)

cv2.waitKey(0)
