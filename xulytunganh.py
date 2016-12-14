import test
import numpy as np
import cv2
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged



frame = cv2.imread('topview/frame159.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# topview
# height, width = gray.shape
# tl = [height/4, height - height/4]
# tr = [width-height/4,height - height/4]
# bl = [0,height]
# br = [width,height]
# pts = np.array([tl,tr,bl,br], dtype="float32")
# warped = test.four_point_transform(gray, pts)

# detect line
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("thresh", thresh)
blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
# auto = auto_canny(blurred)
# auto = cv2.Canny(blurred, 225, 250)
auto = cv2.Canny(blurred, 225, 250)
lines = cv2.HoughLines(auto, 1, np.pi / 180, 30)
colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
if(lines is not None):
    for i in range(0, lines.shape[0]):
        rho, theta = lines[i, 0]
        degree = np.degrees(theta)
        # if((degree>=0.0 and degree <= 40)or (degree>=150)):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(colored, (x1, y1), (x2, y2), (0, 127, 255), 2)
        cv2.imshow("Detectline", colored)

    # raw_input("press")

cv2.waitKey(0)
cv2.destroyAllWindows()