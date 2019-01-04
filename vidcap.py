import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def nothing(x):
    return x

while True:
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame1', frame)
    thl = 0
    thu = 255

    frameg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame2 = cv2.threshold(frameg, thresh=127, maxval=255, type=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
   
    cv2.imshow('frame2', frame2)

    cv2.imshow('grayscale', frameg)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

