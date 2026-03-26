import cv2  # 컴퓨터 비전 라이브러리인 OpenCV를 임포트합니다.
import numpy as np  # 행렬 연산 및 좌표 변환을 위해 넘파이를 임포트합니다.
import matplotlib.pyplot as plt  # 이미지 시각화를 위해 matplotlib를 임포트합니다.
import os  # 파일 경로 설정을 위해 os 라이브러리를 임포트합니다.

base_path = os.path.dirname(os.path.abspath(__file__))  # 실행 중인 파일의 절대 경로를 기준으로 기본 경로를 설정합니다.
img1 = cv2.imread(os.path.join(base_path, '..', 'image', 'img2.jpg'))  # 정합의 대상이 될 첫 번째 이미지를 읽어옵니다.
img2 = cv2.imread(os.path.join(base_path, '..', 'image', 'img1.jpg'))  # 정합의 기준이 될 두 번째 이미지를 읽어옵니다.

sift = cv2.SIFT_create()  # SIFT 알고리즘 객체를 생성합니다.
kp1, des1 = sift.detectAndCompute(img1, None)  # 첫 번째 이미지에서 특징점과 기술자를 추출합니다.
kp2, des2 = sift.detectAndCompute(img2, None)  # 두 번째 이미지에서 특징점과 기술자를 추출합니다.

bf = cv2.BFMatcher()  # 특징점 사이의 거리를 계산하기 위한 BFMatcher 객체를 생성합니다.
matches = bf.knnMatch(des1, des2, k=2)  # 각 특징점마다 가장 유사한 2개의 매칭점을 찾습니다.
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]  # Lowe's Ratio Test를 적용해 거리 비율이 0.7 미만인 신뢰할 수 있는 점들만 선별합니다.

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 선별된 특징점들 중 첫 번째 이미지의 좌표를 추출합니다.
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 선별된 특징점들 중 두 번째 이미지의 좌표를 추출합니다.

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # RANSAC 알고리즘을 사용하여 두 이미지 사이의 호모그래피 행렬 H를 계산합니다.

h1, w1 = img1.shape[:2]  # 첫 번째 이미지의 높이와 너비를 가져옵니다.
h2, w2 = img2.shape[:2]  # 두 번째 이미지의 높이와 너비를 가져옵니다.
result_img = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))  # 계산된 행렬 H를 바탕으로 첫 번째 이미지를 원근 변환하여 넓은 캔버스에 배치합니다.
result_img[0:h2, 0:w2] = img2  # 변환된 이미지 영역의 왼쪽 상단에 기준 이미지인 두 번째 이미지를 덮어씌웁니다.
result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # 결과물을 시각화하기 위해 BGR에서 RGB로 색상을 변환합니다.

plt.figure(figsize=(15, 10))  # 출력될 창의 크기를 가로 15, 세로 10으로 설정합니다.
plt.imshow(result_rgb)  # 최종적으로 정합된 이미지를 화면에 표시합니다.
plt.title('Image Alignment using Homography')  # 이미지의 제목을 설정합니다.
plt.axis('off')  # 좌표 축 표시를 제거합니다.
plt.show()  # 전체 시각화 결과를 출력합니다.