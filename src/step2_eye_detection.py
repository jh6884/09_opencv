import cv2, time
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성 --- ①
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 시간 변수 초기화
prev_time = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print('no frame.');break
    
    # fps 표기를 위해 시간을 측정
    curr_time = time.time()
    sec = curr_time - prev_time
    fps = 1 / sec
    str_fps = "FPS : %0.1f" % fps

    # fps를 영상 위에 표시
    cv2.putText(img, str_fps, (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
    prev_time = curr_time

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 얼굴 영역 검출 --- ②
    faces = detector(gray)
    for rect in faces:
        # 얼굴 영역을 좌표로 변환 후 사각형 표시 --- ③
        x,y = rect.left(), rect.top()
        w,h = rect.right()-x, rect.bottom()-y
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
        # 얼굴 랜드마크 검출 --- ④
        shape = predictor(gray, rect)
        areax = []
        areay = []
        for i in range(36, 48):
            # 부위별 좌표 추출 및 표시 --- ⑤
            part = shape.part(i)
            areax.append(part.x)
            areay.append(part.y)
            print(min(areax), max(areax))
            cv2.rectangle(img, (min(areax), min(areay)), (max(areax), max(areay)), (0, 255, 0), 1)
#            cv2.circle(img, (part.x, part.y), 2, (0, 0, 255), -1)
#            cv2.putText(img, str(i), (part.x, part.y), cv2.FONT_HERSHEY_PLAIN, 0.5,(255,255,255), 1, cv2.LINE_AA)
    
    cv2.imshow("face landmark", img)
    if cv2.waitKey(1)== 27:
        break
cap.release()