import cv2

# image path
image_path = "images/marsrover.png"
# Read or load image from its path
image = cv2.imread(image_path)
# image is a NumPy array
print("Dimension of the image: ", image.ndim)
print("Image height: ", format(image.shape[0]))
print("Image width: ", format(image.shape[1]))
print("Image channels: ", format(image.shape[2]))
print("Size of the image array: ", image.size)
# Display the image and wait until a key is pressed
cv2.imshow("My Image", image)
cv2.waitKey(0)
