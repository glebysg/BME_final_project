import numpy as np
import cv2

cap = cv2.VideoCapture('./data/10secs/3_1.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    print(ret)
    if ret:
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
