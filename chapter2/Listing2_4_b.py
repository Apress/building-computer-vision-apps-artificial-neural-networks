from __future__ import print_function
import cv2
import numpy as np

# create a new canvas
canvas = np.zeros((200, 200, 3), dtype = "uint8")
start = (10,10)
end = (100,100)
color = (0,0,255)
thickness = 5
cv2.rectangle(canvas, start, end, color, thickness)
cv2.imwrite("rectangle.jpg", canvas)
cv2.imshow("Rectangle", canvas)
cv2.waitKey(0)
