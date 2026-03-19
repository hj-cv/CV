import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # 수치 연산을 위한 넘파이 임포트
import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib 임포트
import os  # 경로 설정을 위한 os 임포트

base_path = os.path.dirname(os.path.abspath(__file__))  # 현재 실행 파일의 절대 경로 추출
image_path = os.path.join(base_path, '..', 'images', 'edgeDetectionImage.jpg')  # 이미지 상대 경로 생성
img = cv2.imread(image_path)  # cv.imread()를 사용하여 이미지를 불러옴
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # matplotlib 출력을 위해 BGR을 RGB로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cv.cvtColor()를 사용하여 그레이스케일로 변환

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # x축 방향의 에지 검출 (ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # y축 방향의 에지 검출 (ksize=3)
magnitude = cv2.magnitude(sobel_x, sobel_y)  # cv.magnitude()를 사용하여 에지 강도 계산
magnitude_uint8 = cv2.convertScaleAbs(magnitude)  # cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 uint8로 변환

plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 서브플롯 생성
plt.imshow(img_rgb)  # 원본 이미지 시각화
plt.title('Original')  # 서브플롯 제목 설정
plt.axis('off')  # 축 눈금 숨김
plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 서브플롯 생성
plt.imshow(magnitude_uint8, cmap='gray')  # cmap='gray'를 사용하여 흑백으로 에지 강도 이미지 시각화
plt.title('Sobel Edge Magnitude')  # 서브플롯 제목 설정
plt.axis('off')  # 축 눈금 숨김
plt.show()  # 화면에 출력