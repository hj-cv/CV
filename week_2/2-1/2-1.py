import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # 수치 연산을 위한 넘파이 임포트
import glob  # 파일 경로 패턴 매칭을 위한 glob 임포트
import os  # 운영체제 인터페이스를 위한 os 임포트

base_path = os.path.dirname(os.path.abspath(__file__)) # 현재 실행 중인 파일의 절대 경로 추출
images_path = os.path.join(base_path, '..', 'images', 'calibration_images', 'left*.jpg') # 이미지 폴더 경로 생성
images = glob.glob(images_path) # 해당 경로의 모든 이미지 파일 리스트 확보

CHECKERBOARD = (9, 6) # 체크보드 내부 코너 개수 설정
square_size = 25.0 # 격자 한 칸의 실제 크기(25mm) 설정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 코너 정밀화 정지 조건 설정
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32) # 실제 좌표를 담을 0행렬 생성
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) # x, y 격자 좌표 대입
objp *= square_size # 격자 크기를 곱해 실제 좌표값(mm)으로 변환

objpoints, imgpoints = [], [] # 3D 실제 좌표와 2D 이미지 좌표를 담을 리스트 초기화
img_size = None # 이미지 해상도를 저장할 변수 초기화

for fname in images: # 모든 이미지에 대해 루프 수행
    img = cv2.imread(fname) # 이미지 로드
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 그레이스케일로 변환
    img_size = gray.shape[::-1] # 이미지 크기 정보 저장
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None) # 체크보드 코너 검출
    if ret: # 코너 검출 성공 시
        objpoints.append(objp) # 3D 실제 좌표 데이터 추가
        imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)) # 정밀화된 2D 코너 좌표 추가

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None) # 카메라 캘리브레이션 수행
print(f"K:\n{K}\ndist:\n{dist}") # 계산된 내부 행렬 K와 왜곡 계수 출력

img = cv2.imread(images[0]) # 보정 테스트를 위해 첫 번째 이미지 로드
dst = cv2.undistort(img, K, dist) # 카메라 파라미터를 이용해 왜곡 보정 수행
cv2.imshow('Calibration Result', dst) # 결과 영상 표시
cv2.waitKey(0) # 키 입력 대기
cv2.destroyAllWindows() # 모든 창 닫기