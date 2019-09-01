import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels = {"name": 1}
with open("labels.pickle", "rb") as f:
    orig_labels = pickle.load(f)
    labels = {v: k for k, v in orig_labels.items()}
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x, y, w, h) # X, Y and WIDTH, HEIGHT in PX
        facial_region_g = gray[y:y + h, x:x + w]  # Grayscale image used for Local Binary Pattern Histogram calculations
        facial_region_c = frame[y:y + h, x:x + w]  # Colour image
        """
        id_: ID from pickle when we trained the algorithm
        conf: Confidence, shall only recognize face, if the confidence is over or equal to 45
        """
        id_, conf = recognizer.predict(facial_region_g)
        if conf >= 45:
            name = labels[id_]
            # Calculate coordinates
            axis_end_x = x + w
            axis_end_y = y + h
            print(name + " - " + str(round(conf)) + "%")
            cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()
