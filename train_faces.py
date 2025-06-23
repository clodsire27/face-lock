import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

labels = {"태영": 0, "은아": 1}
faces = []
ids = []

for name, label in labels.items():
    folder = f"{name}"
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in detected:
            face = gray[y:y+h, x:x+w]
            faces.append(face)
            ids.append(label)

recognizer.train(faces, np.array(ids))
recognizer.save("face_model.yml")
print("학습 완료")
