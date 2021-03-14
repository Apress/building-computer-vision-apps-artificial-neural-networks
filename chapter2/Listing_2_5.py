import cv2
import numpy as np

# create a new canvas
canvas = np.zeros((200, 200, 3), dtype = "uint8")
center = (100,100)
radius = 50
color = (0,0,255)
thickness = 5
cv2.circle(canvas, center, radius, color, thickness)
cv2.imwrite("circle.jpg", canvas)
cv2.imshow("My Circle", canvas)
cv2.waitKey(0)
