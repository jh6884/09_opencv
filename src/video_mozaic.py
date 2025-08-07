import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
rate = 15

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        mozaic_area = img[y:y+h, x:x+w]
        mozaic_area = cv2.resize(mozaic_area, (w//rate, h//rate))
        mozaic_area = cv2.resize(mozaic_area, (w, h), interpolation=cv2.INTER_AREA)
        img[y:y+h, x:x+w] = mozaic_area

cv2.imshow('camera', img)
cv2.waitKey(0)
cv2.destroyAllWindows()