from __future__ import print_function
import cv2

# image path
image_path = "images/marsrover.png"
# Read or load image from its path
image = cv2.imread(image_path)
# set the start and end coordinates
# of the top-left and bottom-right corners of the rectangle
start = (100,70)
end = (350,380)
# Set the color and thickness of the outline
color = (0,255,0)
thickness = 5
# Draw the rectangle
cv2.rectangle(image, start, end, color, thickness)
# Save the modified image with the rectangle drawn to it.
cv2.imwrite("rectangle.jpg", image)
# Display the modified image
cv2.imshow("Rectangle", image)
cv2.waitKey(0)