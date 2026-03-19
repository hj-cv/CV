import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # 수치 연산을 위한 넘파이 임포트
import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib 임포트
import os  # 경로 설정을 위한 os 임포트

base_path = os.path.dirname(os.path.abspath(__file__))  # 현재 실행 파일의 절대 경로 추출
image_path = os.path.join(base_path, '..', 'images', 'dabo.jpg')  # 이미지 상대 경로 생성
img = cv2.imread(image_path)  # 이미지 불러오기
result_img = img.copy()  # 직선을 그릴 원본 이미지 복사본 생성

edges = cv2.Canny(img, 100, 200)  # cv.Canny()를 사용하여 에지 맵 생성 (threshold1=100, threshold2=200)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)  # 허프 변환을 사용하여 직선 검출

if lines is not None:  # 직선이 하나라도 검출되었다면
    for line in lines:  # 검출된 모든 직선에 대해 반복
        x1, y1, x2, y2 = line[0]  # 직선의 시작점과 끝점 좌표 추출
        cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # cv.line()을 사용하여 빨간색(0,0,255), 두께 2로 직선 그림

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # matplotlib 출력을 위해 원본 BGR을 RGB로 변환
result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # matplotlib 출력을 위해 결과 BGR을 RGB로 변환

plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 서브플롯 생성
plt.imshow(img_rgb)  # 원본 이미지 시각화
plt.title('Original')  # 서브플롯 제목 설정
plt.axis('off')  # 축 눈금 숨김
plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 서브플롯 생성
plt.imshow(result_rgb)  # 직선이 그려진 결과 이미지 시각화
plt.title('Hough Lines')  # 서브플롯 제목 설정
plt.axis('off')  # 축 눈금 숨김
plt.show()  # 화면에 출력