import cv2 as cv
import numpy as np
import sys

img = cv.imread('cat.png') # 이미지 로드 

if img is None:
    sys.exit('파일을 찾을 수 없습니다.') # 파일 확인 [cite: 2190, 2456]

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 그레이스케일 변환 

# hstack을 위해 그레이스케일 이미지를 3채널로 변경하거나 원본을 1채널로 표시할 수 없으므로 
# 과제 힌트에 따라 np.hstack 사용을 위해 gray 영상을 3채널 형식으로 복사하여 연결
gray_3channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
result = np.hstack((img, gray_3channel)) # 가로로 연결 

cv.imshow('Original and Gray', result) # 결과 표시 
cv.waitKey()
cv.destroyAllWindows()