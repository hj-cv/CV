import cv2 as cv
import sys
import numpy as np
import os

base_path = os.path.dirname(os.path.abspath(__file__)) # 현재 실행 중인 파일(1-3.py)의 절대 경로를 기준으로 설정
img_path = os.path.join(base_path, '..', 'cat.png') # 한 단계 상위 폴더의 cat.png 경로를 생성

img_orig = cv.imread(img_path) # 원본 이미지 로드
if img_orig is None:
    sys.exit() # 파일이 없을 경우 종료

img = img_orig.copy() # 화면 표시용 이미지 복사본 생성
ix, iy = -1, -1 # 마우스 시작 좌표 초기화
is_drawing = False # 드래그 상태 확인용 플래그
roi = None # 추출된 ROI를 저장할 변수

def select_roi(event, x, y, flags, param): # ROI 선택을 위한 마우스 콜백 함수
    global ix, iy, is_drawing, img, roi
    
    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭 시
        is_drawing = True # 드래그 시작
        ix, iy = x, y # 시작 좌표 저장
    elif event == cv.EVENT_MOUSEMOVE: # 마우스 이동 시
        if is_drawing: # 드래그 중인 경우
            img = img_orig.copy() # 이전 사각형 잔상 제거를 위해 원본 복사
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2) # 현재 드래그 영역 시각화
    elif event == cv.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼을 뗐을 때
        is_drawing = False # 드래그 종료
        cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2) # 최종 영역 확정 시각화
        roi = img_orig[min(iy, y):max(iy, y), min(ix, x):max(ix, x)] # Numpy 슬라이싱으로 ROI 추출
        if roi.size > 0: # 유효한 영역인 경우
            cv.imshow('ROI', roi) # 추출된 ROI를 별도 창에 표시

cv.namedWindow('Select ROI') # 메인 윈도우 생성
cv.setMouseCallback('Select ROI', select_roi) # 마우스 콜백 함수 등록

while True: # 이벤트 루프
    cv.imshow('Select ROI', img) # 현재 이미지 상태 표시
    key = cv.waitKey(1) & 0xFF # 키 입력 대기
    
    if key == ord('r'): # 'r' 키 입력 시
        img = img_orig.copy() # 이미지 리셋
        cv.destroyWindow('ROI') # ROI 창 닫기
    elif key == ord('s'): # 's' 키 입력 시
        if roi is not None: # ROI가 존재할 경우
            cv.imwrite('roi_result.jpg', roi) # 이미지 파일로 저장
    elif key == ord('q'): # 'q' 키 입력 시
        break # 루프 종료

cv.destroyAllWindows() # 모든 창 닫기