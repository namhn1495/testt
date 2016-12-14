import test
import numpy as np
import cv2
from scipy.spatial import distance

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def distance(p0, p1, p2): # p3 is the point
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    nom = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = float(float(((y2 - y1)**2 + (x2 - x1) ** 2)) ** 0.5)
    result = nom / denom
    return result

cap = cv2.VideoCapture("04.avi")
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    if(ret):
        # filename = "video04/frame" + str(count) +".jpg"
        # count+=1
        # gray frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoise = cv2.fastNlMeansDenoising(gray,None,10,7,21)
        cv2.imshow("gray",denoise)
        cv2.imshow("denoise",denoise);
        # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        # cv2.imwrite(filename, thresh)

        # topview
        # height, width = gray.shape
        # tl = [height/4, height - height/4]
        # tr = [width-height/4,height - height/4]
        # bl = [0,height]
        # br = [width,height]
        # pts = np.array([tl,tr,bl,br], dtype="float32")
        # M,warped = test.four_point_transform(gray, pts)
        # height, width = warped.shape

        # filename = "gg/frame" + str(count) + ".jpg"
        # count += 1
        # cv2.imwrite(filename, warped)
        # cv2.imshow('original',gray)
        # cv2.imshow('frame',warped)

        # detect line
        # ret, thresh = cv2.threshold(warped, 160, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh',thresh)
        # cv2.imwrite(filename, thresh)
        # denoise = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)

        # cv2.imshow('denoise',denoise)
        # ret, thresh = cv2.threshold(denoise, 160, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh',thresh)

        # blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
        # ret, thresh = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh',thresh)
        # auto = auto_canny(blurred)
        # auto = cv2.Canny(thresh, 225, 250)
        # auto = cv2.Canny(blurred, 225, 250)

        # cv2.imshow('bien',auto)
        # dmin = 10000;
        # d2min = 10000;
        # tl = (0,0)
        # bl = (0,0)
        # tr = (width/2,0)
        # br = (width/2,0)
        # tl2 = (0,0)
        # bl2 = (0,0)
        # tr2 = (width/2,0)
        # br2 = (width/2,0)
        #
        # lines = cv2.HoughLines(auto, 1, np.pi / 180, 60)
        # colored = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
        # if(lines is not None):
        #     for i in range(0, lines.shape[0]):
        #         rho, theta = lines[i, 0]
        #         degree = np.degrees(theta)
        #         if((degree>=0.0 and degree <= 60) or (degree>=140)):
        #             a = np.cos(theta)
        #             b = np.sin(theta)
        #             x1 = int(rho/a)
        #             y1 = 0
        #             x2 = int((rho - height*b)/a)
        #             y2 = height
        #
        #             # cv2.line(colored, (x1, y1), (x2, y2), (0, 127, 255), 2)
        #             p1 =(x1,y1)
        #             p2 = (x2,y2)
        #             p3 = (0,height/2);
        #             p4 = (width,height/2)
        #             d = distance(p3,p1,p2)
        #             if(d<dmin):
        #                 dmin = d
        #
        #                 tl = (x1,y1)
        #                 bl = (x2,y2)
        #                 # tr = (x1+25,y1)
        #                 # br = (x2+25,y2)
        #             d = distance(p4, p1, p2)
        #             if (d < d2min):
        #                 d2min = d
        #                 tr2 = (x1, y1)
        #                 br2 = (x2, y2)
        #                 # tr2 = (x1 - 25, y1)
        #                 # br2 = (x2 - 25, y2)
        #
        #     print str(tl)+","+str(bl)
        #     cv2.line(colored, tl, bl, (0, 127, 255), 2)
        #     # cv2.line(colored, tr, br, (0, 127, 255), 2)
        #     cv2.line(colored, tr2, br2, (0, 127, 255), 2)
        #     # cv2.line(colored, tr2, br2, (0, 127, 255), 2)
        #     tbx1 = (tl[0]+bl[0])/2
        #     tby1 = (tl[1]+bl[1])/2
        #     tbx2 = (tr2[0]+br2[0])/2
        #     tby2 = (tr2[1]+br2[1])/2
        #     print str((tbx1+tbx2)/2) +","+ str( (tby1+tby2)/2)
        #     cv2.circle(colored, ((tbx1+tbx2)/2, (tby1+tby2)/2), 2, (0,127,255), -1)
        #     # cv2.imshow("Detectline", colored)
        #     after = cv2.warpPerspective(colored, M, (gray.shape[1], gray.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        #     # print  after
        #     frame[frame.shape[0] - frame.shape[0]/4:frame.shape[0], 0:frame.shape[1] ] = after[frame.shape[0] - frame.shape[0]/4:frame.shape[0], 0:frame.shape[1]]

            # result = cv2.addWeighted(after, 0.8, frame, 0.2, 0)
            # result = cv2.bitwise_and(after, after, mask=frame)

            # frame[frame.shape[0] - frame.shape[0]/4:height, 0:width] = after
            # cv2.imshow("ReverseDetectline", frame)

            # filename = "topview03/frame" + str(count) + ".jpg"
            # count += 1
            # cv2.imwrite(filename, colored)
        # print tl

    else:
        break
    # raw_input("press")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


