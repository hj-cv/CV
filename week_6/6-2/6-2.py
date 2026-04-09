import cv2  # 영상 처리를 위한 OpenCV 라이브러리를 임포트합니다.
import mediapipe as mp  # 얼굴 랜드마크 검출을 위해 Mediapipe 라이브러리를 임포트합니다.

# FaceMesh 모델 및 인식 옵션을 초기화합니다.
mp_face_mesh = mp.solutions.face_mesh  # Mediapipe의 FaceMesh 솔루션 모듈을 불러옵니다.
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)  # 실시간 검출을 위한 모델 인스턴스를 생성합니다.

# 비디오 파일을 로드하고 출력 창의 속성을 설정합니다.
cap = cv2.VideoCapture('6-2/human.mp4')  # 분석 대상인 비디오 파일을 읽어옵니다.
cv2.namedWindow('Face Mesh Analysis', cv2.WINDOW_NORMAL)  # 창 크기 조절이 가능하도록 윈도우를 생성합니다.

# 원본 해상도를 유지하며 프레임을 읽고 분석하는 메인 루프입니다.
while True:  # 프로그램이 종료될 때까지 무한 반복합니다.
    ret, frame = cap.read()  # 비디오 파일에서 한 프레임을 읽어옵니다.
    
    # 영상 재생이 끝나면 처음으로 되돌려 무한 재생을 수행합니다.
    if not ret:  # 더 이상 읽을 프레임이 없는지 확인합니다.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 영상의 재생 위치를 0번 프레임(처음)으로 설정합니다.
        continue  # 다음 루프로 이동하여 처음부터 다시 재생합니다.

    # 모델 분석을 위해 색상 공간을 변환하고 추론을 수행합니다.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 형식을 모델 입력용 RGB 형식으로 변환합니다.
    results = face_mesh.process(image_rgb)  # 변환된 이미지에서 얼굴 랜드마크 좌표를 추출합니다.

    # 검출된 랜드마크 좌표를 원본 영상 크기에 맞춰 시각화합니다.
    if results.multi_face_landmarks:  # 하나 이상의 얼굴이 검출되었는지 확인합니다.
        for face_landmarks in results.multi_face_landmarks:  # 검출된 각 얼굴 데이터에 접근합니다.
            for lm in face_landmarks.landmark:  # 얼굴을 구성하는 468개의 개별 점들을 순회합니다.
                ih, iw, _ = frame.shape  # 좌표 계산을 위해 이미지의 높이와 너비를 가져옵니다.
                cv2.circle(frame, (int(lm.x * iw), int(lm.y * ih)), 2, (0, 255, 0), -1)  # 변환된 좌표 위치에 초록색 점을 그립니다.

    # 처리 결과를 화면에 출력하고 프로그램 종료를 제어합니다.
    cv2.imshow('Face Mesh Analysis', frame)  # 랜드마크가 시각화된 프레임을 윈도우에 표시합니다.
    if cv2.waitKey(30) & 0xFF == 27:  # 30ms 동안 입력을 대기하며 ESC 키(27)가 눌리면 루프를 탈출합니다.
        break  # 종료 조건 만족 시 반복문을 종료합니다.

# 사용한 자원을 반환하고 프로그램을 종료합니다.
cap.release()  # 비디오 캡처 객체의 메모리 점유를 해제합니다.
cv2.destroyAllWindows()  # 생성된 모든 OpenCV 윈도우 창을 닫습니다.