import numpy as np
import cv2
 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
while True:
    # frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # lav grid for nemmere debug af colors
    pixelFrequency = 50
    x = pixelFrequency
    y = pixelFrequency
    while x < gray.shape[1]:
        cv2.line(gray, (x, 0), (x, gray.shape[0]), (0, 0, 255), 1, 1)
        x += pixelFrequency

    while y < gray.shape[0]:
        cv2.line(gray, (0, y), (gray.shape[1], y), (0, 255, 0), 1, 1)
        y += pixelFrequency


    # vis webcam stream i grayscale
    #cv2.imshow('Normal', frame)
    cv2.imshow('Grayscale conversion', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()