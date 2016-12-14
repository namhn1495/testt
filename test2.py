import test
import numpy as np
import cv2
from scipy import ndimage
cap = cv2.VideoCapture("04.avi")
while(cap.isOpened()):
    ret, frame = cap.read()

    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # denoise = cv2.fastNlMeansDenoising(gray,None,10,7,21)
        # med_denoised = ndimage.median_filter(gray, 10)
        med_denoised = cv2.GaussianBlur(gray,(21,21),0)
        ret, thresh = cv2.threshold(med_denoised,170,255,cv2.THRESH_BINARY)
        cv2.imshow("gray",gray)
        cv2.imshow("denoise",med_denoised)
        
        cv2.imshow("thresh",thresh)
    else:
        break
    # raw_input("press")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
