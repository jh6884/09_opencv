import cv2, time
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성 --- ①
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def detect_faces(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for rect in faces:
        # 얼굴 영역을 좌표로 변환 후 사각형 표시 --- ③
        x,y = rect.left(), rect.top()
        w,h = rect.right()-x, rect.bottom()-y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
        # 얼굴 랜드마크 검출 --- ④
        shape = predictor(gray, rect)
        for i in range(68):
            # 부위별 좌표 추출 및 표시 --- ⑤
            part = shape.part(i)
            cv2.circle(frame, (part.x, part.y), 2, (0, 0, 255), -1)
            #cv2.putText(frame, str(i), (part.x, part.y), cv2.FONT_HERSHEY_PLAIN, 0.5,(255,255,255), 1, cv2.LINE_AA)

def frame_per_second(frame, start_time, frame_count):
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        str_fps = f"FPS: {fps:.2f}"
        cv2.putText(frame, str_fps, (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # 변수 초기화 후 반환
        return time.time(), 0
    else:
        # 1초가 지나지 않았을 경우 현재 상태 유지
        return start_time, frame_count

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    detect_faces(img, detector, predictor)
    time_for_fps = time.time()
    frame_count = 0
    frame_per_second(img, time_for_fps, frame_count)

    cv2.imshow('camera', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()