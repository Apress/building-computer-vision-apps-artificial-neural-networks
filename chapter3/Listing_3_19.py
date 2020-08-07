import cv2
import numpy as np
# Load an image
image = cv2.imread("images/sudoku.jpg")
cv2.imshow("Original Image", image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bilateralFilter(image, 5, 50, 50)
cv2.imshow("Blurred image", image)

# Sobel gradient detection
sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
sobelx = np.uint8(np.absolute(sobelx))
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
sobely = np.uint8(np.absolute(sobely))

cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)

# Schar gradient detection by passing ksize = -1 to Sobel function
scharx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=-1)
scharx = np.uint8(np.absolute(scharx))
schary = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=-1)
schary = np.uint8(np.absolute(schary))
cv2.imshow("Schar X", scharx)
cv2.imshow("Schar Y", schary)

cv2.waitKey(0)
