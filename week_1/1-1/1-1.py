import cv2 as cv
import numpy as np
import sys

img = cv.imread('../cat.png') # 이미지 로드 

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 그레이스케일 변환 

gray_3channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 그레이스케일 이미지를 np.hstack 연결을 위해 3채널(BGR)로 변환
result = np.hstack((img, gray_3channel)) # 원본 이미지와 변환된 이미지를 가로로 결합

cv.imshow('Original and Gray', result) # 결과 표시 
cv.waitKey() # 사용자로부터 키 입력이 있을 때까지 대기
cv.destroyAllWindows() # 생성된 모든 영상 창을 닫음