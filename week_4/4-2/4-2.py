import cv2  # 이미지 처리를 위한 OpenCV 라이브러리를 임포트합니다.
import numpy as np  # 행렬 연산 및 수치 처리를 위해 넘파이를 임포트합니다.
import matplotlib.pyplot as plt  # 결과 시각화를 위한 matplotlib를 임포트합니다.
import os  # 파일 시스템 경로 관리를 위해 os 모듈을 임포트합니다.

base_path = os.path.dirname(os.path.abspath(__file__))  # 현재 실행 중인 파이썬 파일의 절대 경로를 변수에 저장합니다.
path1 = os.path.join(base_path, '..', 'image', 'mot_color70.jpg')  # 첫 번째 이미지 파일인 mot_color70.jpg의 경로를 생성합니다.
path2 = os.path.join(base_path, '..', 'image', 'mot_color83.jpg')  # 두 번째 이미지 파일인 mot_color83.jpg의 경로를 생성합니다.

img1 = cv2.imread(path1)  # 경로를 통해 첫 번째 이미지를 불러옵니다.
img2 = cv2.imread(path2)  # 경로를 통해 두 번째 이미지를 불러옵니다.
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # 첫 번째 이미지를 시각화를 위해 RGB로 변환합니다.
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # 두 번째 이미지를 시각화를 위해 RGB로 변환합니다.

sift = cv2.SIFT_create()  # SIFT 특징 추출기 객체를 생성합니다.
kp1, des1 = sift.detectAndCompute(img1, None)  # 첫 번째 이미지에서 특징점과 특징 기술자를 추출합니다.
kp2, des2 = sift.detectAndCompute(img2, None)  # 두 번째 이미지에서 특징점과 특징 기술자를 추출합니다.

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # L2 거리를 사용하고 교차 검사를 수행하는 BFMatcher 객체를 생성합니다.
matches = bf.match(des1, des2)  # 추출된 두 기술자 간의 최적 매칭 쌍을 찾습니다.
matches = sorted(matches, key=lambda x: x.distance)  # 매칭된 결과들을 거리(유사도) 기준으로 오름차순 정렬합니다.

img_match = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # 상위 50개의 매칭점들을 이미지 사이에 선으로 연결하여 그립니다.

plt.figure(figsize=(15, 7))  # 출력을 위한 캔버스의 크기를 설정합니다.
plt.imshow(img_match)  # 매칭 결과가 그려진 이미지를 표시합니다.
plt.title('SIFT Feature Matching')  # 전체 플롯의 제목을 설정합니다.
plt.axis('off')  # 가로세로 축 정보를 숨깁니다.
plt.show()  # 결과를 최종적으로 화면에 표시합니다.