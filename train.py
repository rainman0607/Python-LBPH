import os
import cv2
import pickle
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

# load haarcascade and start lbph recognizer
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

curr_id = 0
label_ids = {}

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(os.path.dirname(path))
        if label in label_ids:
            pass
        else:
            label_ids[label] = curr_id
            curr_id += 1

        id_ = label_ids[label]
        pil_img = Image.open(path)
        img_arr = np.array(pil_img, "uint8")
        faces = face_cascade.detectMultiScale(img_arr, scaleFactor=1.5, minNeighbors=5)

        for(x, y, w, h) in faces:
            roi = img_arr[y:y+h, x:x+w]
            x_train.append(roi)
            y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
