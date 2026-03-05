import cv2
import numpy as np
import sys
import os


base_path = os.path.dirname(os.path.abspath(__file__)) # 현재 실행 중인 파일(1-2.py)의 절대 경로를 기준으로 설정
img_path = os.path.join(base_path, '..', 'cat.png') # 한 단계 상위 폴더의 cat.png 경로를 생성

drawing = False # 마우스 클릭 상태를 확인하기 위한 플래그 변수
color = (255, 0, 0) # 기본 붓 색상을 파란색으로 설정
brush_size = 5 # 초기 붓 크기를 5로 설정

def paint(event, x, y, flags, param): # 마우스 이벤트 처리를 위한 콜백 함수 정의
    global drawing, color # 전역 변수 drawing과 color를 함수 내에서 사용

    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼을 눌렀을 때
        drawing = True # 그리기 상태를 True로 변경
        color = (255, 0, 0) # 색상을 파란색으로 설정
        cv2.circle(img, (x, y), brush_size, color, -1) # 클릭한 위치에 현재 붓 크기로 원을 그림
    elif event == cv2.EVENT_RBUTTONDOWN: # 마우스 오른쪽 버튼을 눌렀을 때
        drawing = True # 그리기 상태를 True로 변경
        color = (0, 0, 255) # 색상을 빨간색으로 설정
        cv2.circle(img, (x, y), brush_size, color, -1) # 클릭한 위치에 현재 붓 크기로 원을 그림
    elif event == cv2.EVENT_MOUSEMOVE: # 마우스를 움직일 때
        if drawing: # 그리기 상태가 True인 경우에만 실행
            cv2.circle(img, (x, y), brush_size, color, -1) # 마우스 경로를 따라 원을 그림
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP: # 마우스 버튼을 떼었을 때
        drawing = False # 그리기 상태를 False로 변경

img = cv2.imread(img_path) # cat.png 이미지를 로드

if img is None:
    sys.exit() # 이미지 파일이 없으면 종료

cv2.namedWindow('Paint') # 'Paint'라는 이름의 윈도우 창 생성
cv2.setMouseCallback('Paint', paint) # 'Paint' 창에 마우스 콜백 함수 등록

while True: # 무한 루프 시작
    cv2.imshow('Paint', img) # 이미지를 화면에 표시
    key = cv2.waitKey(1) & 0xFF # 1ms 동안 키 입력을 대기

    if key == ord('q'): # 'q' 키를 누르면 루프 종료
        break
    elif key == ord('+') or key == ord('='): # '+' 또는 '=' 키를 누르면 붓 크기 증가
        brush_size = min(15, brush_size + 1) # 최대 크기를 15로 제한
    elif key == ord('-'): # '-' 키를 누르면 붓 크기 감소
        brush_size = max(1, brush_size - 1) # 최소 크기를 1로 제한

cv2.destroyAllWindows() # 생성된 모든 윈도우 창을 닫음