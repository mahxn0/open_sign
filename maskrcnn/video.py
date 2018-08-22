# -*- coding:utf-8-*-

import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        #show_info(cam)
        #detect(img,net)
        cv2.imshow("capture", img)
    if 0xFF == ord('q') & cv2.waitKey(5) == 27:
        break
cap.release()
cv2.destroyAllWindows()
