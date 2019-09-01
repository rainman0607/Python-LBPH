"""
This is for image creation for training purpose.
"""
import os
import time
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

print("What should the label be named?")
label = input()
path = "images/" + str(label.replace(" ", "-")) + "/"
os.mkdir(path)
print("Billeder bliver taget om 3 sekunder, roter dit hovede med uret. 3..2..1")
time.sleep(3)
i = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        i += 1
        print(x, y, w, h)
        facial_region_g = gray[y:y+h, x:x+w] # Grayscale image for LBPH
        facial_region_c = frame[y:y+h, x:x+w] # Colour image
        if i < 51:
            cv2.imwrite(path + str(i) + '.png', gray) # For training purpose later on
            print("Billede nr " + str(i) + " er blevet taget")
        # Calculate coordinates
        axis_end_x = x + w
        axis_end_y = y + h

        cv2.rectangle(frame, (x, y), (axis_end_x, axis_end_y), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if i >= 51:
        print("DONZO!")
        break
cap.release()
cv2.destroyAllWindows()
