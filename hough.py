import cv2
import numpy as np

img = cv2.imread('video01/frame0.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,120)

for i in  range(0,lines.shape[0]):
    rho,theta = lines[i,0]
    degree = np.degrees(theta)
    # if ( (degree>=10 and degree <= 85) or degree>=95):
    if True:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)


cv2.imshow("hough",  img)
cv2.waitKey(0)