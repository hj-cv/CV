import cv2
import numpy as np
import os

base_path = os.path.dirname(os.path.abspath(__file__)) # 현재 실행 중인 파일의 절대 경로 추출
img_path = os.path.join(base_path, '..', 'images', 'rose.png') # 한 단계 상위 폴더의 images 내 rose.png 경로 생성

img = cv2.imread(img_path) # 이미지 파일 로드
rows, cols = img.shape[:2] # 이미지의 높이(rows)와 너비(cols) 정보 획득

# 이미지 중심 기준 +30도 회전 및 0.8배 축소를 위한 2x3 변환 행렬 생성 [cite: 86, 87, 89]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 0.8)

# 변환 행렬의 마지막 열을 수정하여 x축 +80px, y축 -40px 평행이동 정보 반영 [cite: 88, 91]
M[0, 2] += 80
M[1, 2] -= 40

# 어파인 변환 행렬을 적용하여 회전, 크기 조절, 이동이 통합된 결과 영상 생성 [cite: 90]
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('Transformation', dst) # 최종 변환 결과를 'Transformation' 창에 표시
cv2.waitKey(0) # 사용자로부터 키 입력이 있을 때까지 무한 대기
cv2.destroyAllWindows() # 실행 종료 후 생성된 모든 영상 창을 닫음