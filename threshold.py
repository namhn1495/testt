import cv2

import numpy as np

image = cv2.imread("topview/frame20.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("threshold",thresh1)
cv2.waitKey(0)