import cv2
import numpy as np

# create a circle
circle = cv2.circle(np.zeros((200, 200, 3), dtype = "uint8"), (100,100), 90, (255,255,255), -1)
cv2.imshow("A white circle", circle)
cv2.waitKey(0)

# create a square
square = cv2.rectangle(np.zeros((200,200,3), dtype= "uint8"), (30,30), (170,170),(255,255,255), -1)
cv2.imshow("A white square", square)
cv2.waitKey(0)

#bitwise AND
bitwiseAnd = cv2.bitwise_and(square, circle)
cv2.imshow("AND Operation", bitwiseAnd)
cv2.waitKey(0)

#bitwise OR
bitwiseOr = cv2.bitwise_or(square, circle)
cv2.imshow("OR Operation", bitwiseOr)
cv2.waitKey(0)

#bitwise XOR
bitwiseXor = cv2.bitwise_xor(square, circle)
cv2.imshow("XOR Operation", bitwiseXor)
cv2.waitKey(0)

#bitwise NOT
bitwiseNot = cv2.bitwise_not(square)
cv2.imshow("NOT Operation", bitwiseNot)
cv2.waitKey(0)
