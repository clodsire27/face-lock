import cv2 as cv
import numpy as np
import time
import sys

#카메라 연결(cv.CAP_DSHOW는 맥이나 일부 윈도우에서는 안돌아가서 뺌)
cap = cv.VideoCapture(0)    
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

# 얼굴 검출기 로드
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    sys.exit('얼굴 검출기 로드 실패')

#얼굴인식 알고리즘과 미리 학습된 우리 얼굴인식 파일
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")
#인식된 라벨번호 이름으로 매핑하고, 우리 암호 손가락 갯수
label_map = {0: "Taeyoung", 1: "Euna"}
gesture_sequence = [0, 5, 2]
frame_count_for_gesture = 3
gesture_delay = 0.5

#제스쳐 상황 저장
#잠금 해제된 사람 목록
gesture_states = {}
unlocked_faces = set()

#손 인식 잘하기 위한 손 색상 범위 지정
lower = np.array([0, 20, 70])
upper = np.array([20, 255, 255])

#얼굴 모자이크 함수 -> ratio 0.1은 먼가 약해서 0.05로 더 강하게 설정함
def mosaic_face(frame, x, y, w, h, ratio=0.05):
    h_frame, w_frame = frame.shape[:2]  #프레임 앞에 두개->높이,너비 가져오는거
    x, y = max(0, x), max(0, y)
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)
    face_roi = frame[y:y+h, x:x+w]
    small = cv.resize(face_roi, (0, 0), fx=ratio, fy=ratio)
    mosaic = cv.resize(small, (w, h), interpolation=cv.INTER_NEAREST)
    frame[y:y+h, x:x+w] = mosaic

#손가락 외곽선 분석해서 손가락 개수 추정-> 손가락 사이 오목한 부분 계산해서 추정
def count_fingers(contour):
    hull = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull)
    count = 0

    if defects is not None and len(defects) > 0:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start, end, far = contour[s][0], contour[e][0], contour[f][0]
            a = np.linalg.norm(start - end)
            b = np.linalg.norm(start - far)
            c = np.linalg.norm(end - far)
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c + 1e-5))
            if angle < np.pi / 2 and d > 10000:
                count += 1

    if count == 0:
        M = cv.moments(contour)
        if M['m00'] == 0:
            return 0
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = np.array([cx, cy])

        max_dist = max(np.linalg.norm(pt[0] - center) for pt in contour)
        area = cv.contourArea(contour)

        # 조건 조정: 주먹과 손가락 1개 구분
        if area > 5000 and max_dist > 200:
            return 1
        else:
            return 0
    else:
        return count + 1

#손가락 수 보면서 우리가 설정한 값에 맞게 진행중인지 판단하는거 -> 일정한 프레임동안 같은 동작하고 순서가 일치하면 언락되게
def process_gesture(fingers, identity):  # identity는 고유 이름(태영, 은아, 게스트 누구 이런거)
    current_time = time.time()
    if identity not in gesture_states:
        gesture_states[identity] = {
            "same_finger_count": 0,
            "prev_fingers": -1,
            "gesture_index": 0,
            "last_gesture_time": 0
        }
    state = gesture_states[identity]
    if fingers == state["prev_fingers"]:
        state["same_finger_count"] += 1
    else:
        state["same_finger_count"] = 1
        state["prev_fingers"] = fingers
    if state["same_finger_count"] >= frame_count_for_gesture:
        if current_time - state["last_gesture_time"] < gesture_delay:
            return False
        expected = gesture_sequence[state["gesture_index"]]
        if fingers == expected:
            state["gesture_index"] += 1
            state["last_gesture_time"] = current_time
            state["same_finger_count"] = 0
            if state["gesture_index"] == len(gesture_sequence):
                gesture_states[identity] = {
                    "same_finger_count": 0,
                    "prev_fingers": -1,
                    "gesture_index": 0,
                    "last_gesture_time": 0
                }
                return True
        else:
            state["gesture_index"] = 0
    gesture_states[identity] = state
    return False


#메인 시작-> 프레임 실시간으로 읽음
while True:
    ret, frame = cap.read()
    if not ret:
        break
#프레임의 높이랑 너비 정함
    h_frame, w_frame = frame.shape[:2]

    #얼굴 검출이랑 인식 하려고 흑백 이미지로 변환하고 여러 크기의 얼굴 찾음
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #1.3은 이미지 피라미드, 5는 얼굴로 판단하는 최소 조건

    #검출된 얼굴들 다 루프돌면서 라벨번호랑 인식 신뢰도 분석
    for (x, y, w, h) in faces:
        label, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # 신뢰도 값 높으면 label map으로 이름 찾음(태영이나 은아), 라벨에 해당되는 이름 없으면 게스트_번호 부여
        if confidence < 50:
            name = label_map.get(label)
            if name is None:
                label_map[label] = f"Guest_{label}"
                name = label_map[label]
        else:
            # 인식이 불확실할 경우에도 고유하게 name 설정
            name = f"Unknown_{label}_{int(confidence)}"

        # 이름 기반으로 잠금 상태 관리
        unlocked = name in unlocked_faces

        # 손 인식 영역 추출 -> 얼굴 왼쪽에만 손이 있다는 뜻
        roi_x_end = max(0, x)
        roi_x_start = max(0, x - 500)
        roi_y_start = y
        roi_y_end = min(y + h + 200, h_frame)

        roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        #색 공간 hsv로 바꿔서 피부톤 범위는 마스크처리해서 손 잘 인식하도록
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)

        #커널 설정해서  잡음 제거
        kernel = np.ones((3, 3), np.uint8)  #3*3크기 커널
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)   #노이즈 제거
        mask = cv.dilate(mask, kernel, iterations=1)    #손 팽창시켜서 손 더 두껍고 잘보이게 설정
        blur = cv.GaussianBlur(mask, (3, 3), 0)     #손 가우시안 블러처리 해서 윤곽선 자연스럽게 잡히게 함

        #손 윤곽선 찾기, 기본값은 손가락 0개
        contours, _ = cv.findContours(blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        fingers = 0
        hand_detected = False
        
        #손 감지되면 가장 큰 외곽선(손) 찾고, 면적 중심좌표 계산한 뒤, 아까 만든 함수로 손가락 개수 추정
        if contours:
            cnt = max(contours, key=cv.contourArea)
            area = cv.contourArea(cnt)
            if area > 1000:
                M = cv.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00']) + roi_x_start
                    cy = int(M['m01'] / M['m00']) + roi_y_start
                    fingers = count_fingers(cnt)
                    hand_detected = True
                    cv.drawContours(frame, [cnt + [roi_x_start, roi_y_start]], -1, (0, 255, 0), 2)
                    cv.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                    cv.putText(frame, f"Fingers: {fingers}", (cx + 10, cy),
                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #잠금 안풀렸을 때 손 인식되면 process_gesture(fingers, name) 이거 손 제스쳐 인식하는거 호출, 올바른 손가락 순서로 동작하면 true 반환
        #이름 기준으로 잠금 해제(태영이 풀었는데 은아가 해제되면 안되니까)
        if not unlocked and hand_detected:
            if process_gesture(fingers, name):
                unlocked_faces.add(name)
                unlocked = True
        #언락되면 초록테두리+이름+언락 표시
        if unlocked:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f"{name} (Unlock)", (x, y - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        #아니면 모자이크+빨간글씨로 락
        else:
            mosaic_face(frame, x, y, w, h)
            cv.putText(frame, f"{name} (Lock)", (x, y - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
    #프레임 어떤 상황인지 실시간으로 표시, esc 눌리면 루프 끝
    cv.imshow("Face Control", frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
