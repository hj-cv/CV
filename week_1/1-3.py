import cv2 as cv
import sys

img_orig = cv.imread('cat.png') # 이미지 로드
if img_orig is None:
    sys.exit('파일을 찾을 수 없습니다.')

img = img_orig.copy()
ix, iy = -1, -1
is_drawing = False
roi = None

def select_roi(event, x, y, flags, param):
    global ix, iy, is_drawing, img, roi
    
    if event == cv.EVENT_LBUTTONDOWN:
        is_drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if is_drawing:
            img = img_orig.copy() # 드래그 중 사각형 잔상 제거
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2) # 영역 시각화 
    elif event == cv.EVENT_LBUTTONUP:
        is_drawing = False
        cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        # ROI 추출 (numpy 슬라이싱) 
        roi = img_orig[min(iy, y):max(iy, y), min(ix, x):max(ix, x)]
        if roi.size > 0:
            cv.imshow('ROI', roi) # 별도 창에 출력 

cv.namedWindow('Select ROI')
cv.setMouseCallback('Select ROI', select_roi)

while True:
    cv.imshow('Select ROI', img)
    key = cv.waitKey(1)
    
    if key == ord('r'): # 리셋 
        img = img_orig.copy()
        cv.destroyWindow('ROI')
    elif key == ord('s'): # 저장 
        if roi is not None:
            cv.imwrite('roi_result.jpg', roi)
            print('ROI saved as roi_result.jpg')
    elif key == ord('q'):
        break

cv.destroyAllWindows()