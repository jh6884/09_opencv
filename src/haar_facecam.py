import cv2
import dlib
import numpy as np

# 얼굴과  검출을 위한 케스케이드 분류기 생성 
# face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye.xml')
cnn_face_detector = dlib.cnn_face_detection_model_v1('../data/mmod_human_face_detector.dat')

# 카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)
while cap.isOpened():    
    ret, img = cap.read()  # 프레임 읽기
    if ret:
        img_resized = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        face_detections = cnn_face_detector(gray, 1)
        for idx, face_detection in enumerate(face_detections):
            left, top, right, bottom, confidence = face_detection.rect.left(), face_detection.rect.top(), face_detection.rect.right(), face_detection.rect.bottom(), face_detection.confidence
            print(f'confidence{idx+1}: {confidence}')  # print confidence of the detection
            cv2.rectangle(img_resized, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow('camera',img_resized)
    else:
        break
    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()