import cv2  # OpenCV 라이브러리를 임포트합니다.
import numpy as np  # 수치 연산을 위한 넘파이 라이브러리를 임포트합니다.
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 matplotlib의 pyplot 모듈을 임포트합니다.
import os  # 운영체제와 상호작용하여 파일 경로를 다루기 위해 os 모듈을 임포트합니다.

base_path = os.path.dirname(os.path.abspath(__file__))  # 실행 중인 스크립트 파일의 절대 경로를 가져옵니다.
image_path = os.path.join(base_path, '..', 'image', 'mot_color70.jpg')  # 상위 디렉토리의 image 폴더 내 mot_color70.jpg 파일 경로를 설정합니다.
img = cv2.imread(image_path)  # cv2.imread 함수를 사용하여 설정된 경로의 이미지를 읽어옵니다.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 색상 공간을 matplotlib 표시를 위해 RGB로 변환합니다.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # SIFT 연산을 위해 입력 이미지를 흑백(그레이스케일)으로 변환합니다.

sift = cv2.SIFT_create(nfeatures=500)  # 검출할 특징점의 최대 개수를 500개로 제한하여 SIFT 객체를 생성합니다.
kp, des = sift.detectAndCompute(gray, None)  # 변환된 흑백 이미지에서 특징점(kp)과 기술자(des)를 동시에 검출하고 계산합니다.

img_kp = cv2.drawKeypoints(img_rgb, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 이미지 위에 특징점의 크기와 방향을 포함하여 시각화합니다.

plt.subplot(1, 2, 1)  # 1행 2열로 구성된 화면의 첫 번째 칸을 선택합니다.
plt.imshow(img_rgb)  # 선택된 칸에 원본 RGB 이미지를 표시합니다.
plt.title('Original')  # 해당 이미지의 제목을 'Original'로 설정합니다.
plt.axis('off')  # 이미지의 가로세로 축 수치 표시를 제거합니다.
plt.subplot(1, 2, 2)  # 1행 2열로 구성된 화면의 두 번째 칸을 선택합니다.
plt.imshow(img_kp)  # 선택된 칸에 SIFT 특징점이 그려진 이미지를 표시합니다.
plt.title('SIFT Keypoints')  # 해당 이미지의 제목을 'SIFT Keypoints'로 설정합니다.
plt.axis('off')  # 이미지의 가로세로 축 수치 표시를 제거합니다.
plt.show()  # 생성된 모든 플롯을 화면에 출력합니다.