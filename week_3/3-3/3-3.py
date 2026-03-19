import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # 수치 연산을 위한 넘파이 임포트
import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib 임포트
import os  # 경로 설정을 위한 os 임포트

base_path = os.path.dirname(os.path.abspath(__file__))  # 현재 실행 파일의 절대 경로 추출
image_path = os.path.join(base_path, '..', 'images', 'coffee cup.JPG')  # 이미지 상대 경로 생성
img = cv2.imread(image_path)  # 이미지 불러오기

mask = np.zeros(img.shape[:2], np.uint8)  # 이미지 크기와 동일한 빈 마스크 생성
bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델 초기화
fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델 초기화

rect = (50, 50, img.shape[1]-100, img.shape[0]-100)  # 초기 사각형 영역 설정 (x, y, width, height)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)  # cv.grabCut()를 사용하여 대화식 분할 수행

mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')  # 확실한 배경과 예상 배경을 0, 나머지를 1로 변경
result_img = img * mask2[:, :, np.newaxis]  # 마스크를 원본 이미지에 곱하여 배경 제거

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # matplotlib 출력을 위해 원본 BGR을 RGB로 변환
result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # matplotlib 출력을 위해 결과 BGR을 RGB로 변환

plt.subplot(1, 3, 1)  # 1행 3열의 첫 번째 서브플롯 생성
plt.imshow(img_rgb)  # 원본 이미지 시각화
plt.title('Original')  # 서브플롯 제목 설정
plt.axis('off')  # 축 눈금 숨김
plt.subplot(1, 3, 2)  # 1행 3열의 두 번째 서브플롯 생성
plt.imshow(mask2, cmap='gray')  # 마스크 이미지 시각화
plt.title('Mask')  # 서브플롯 제목 설정
plt.axis('off')  # 축 눈금 숨김
plt.subplot(1, 3, 3)  # 1행 3열의 세 번째 서브플롯 생성
plt.imshow(result_rgb)  # 배경 제거된 결과 이미지 시각화
plt.title('Result')  # 서브플롯 제목 설정
plt.axis('off')  # 축 눈금 숨김
plt.show()  # 화면에 출력