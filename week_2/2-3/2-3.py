import cv2
import numpy as np
import os
from pathlib import Path

base_path = os.path.dirname(os.path.abspath(__file__)) # 스크립트 파일의 절대 경로 추출
left_path = os.path.join(base_path, '..', 'images', 'left.png') # 왼쪽 이미지 경로 설정
right_path = os.path.join(base_path, '..', 'images', 'right.png') # 오른쪽 이미지 경로 설정
output_dir = Path(base_path) / "outputs" # 결과물 저장 폴더 경로 설정
output_dir.mkdir(parents=True, exist_ok=True) # 폴더가 없으면 생성

left_color = cv2.imread(left_path) # 왼쪽 컬러 이미지 로드
right_color = cv2.imread(right_path) # 오른쪽 컬러 이미지 로드
if left_color is None or right_color is None: exit() # 로드 실패 시 프로그램 종료

f, B = 700.0, 0.12 # 카메라 초점 거리 및 베이스라인 설정 [cite: 126, 127]
rois = {"Painting": (55, 50, 130, 110), "Frog": (90, 265, 230, 95), "Teddy": (310, 35, 115, 90)} # 분석 대상 ROI 영역 정의 [cite: 111]

imgL = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY) # 왼쪽 이미지를 그레이스케일로 변환 [cite: 110]
imgR = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY) # 오른쪽 이미지를 그레이스케일로 변환 [cite: 110]

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15) # StereoBM 객체 생성 [cite: 110, 132]
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0 # 시차 계산 및 16배 스케일링 보정 [cite: 134, 135]

valid_mask = disparity > 0 # 유효한 시차 값(0보다 큰 값) 추출을 위한 마스크 생성 [cite: 110]
depth_map = np.zeros_like(disparity) # 깊이 지도를 담을 0행렬 초기화
depth_map[valid_mask] = (f * B) / disparity[valid_mask] # $Z = \frac{fB}{d}$ 공식을 이용해 거리 계산 [cite: 115, 125]

results = {} # 분석 결과 저장을 위한 딕셔너리
for name, (x, y, w, h) in rois.items(): # 각 ROI 영역 순회 [cite: 111]
    roi_disp = disparity[y:y+h, x:x+w] # 시차 맵에서 해당 영역 슬라이싱
    roi_depth = depth_map[y:y+h, x:x+w] # 깊이 맵에서 해당 영역 슬라이싱
    mask = roi_disp > 0 # 해당 영역 내 유효 픽셀 선별
    results[name] = { # 평균 시차 및 깊이 계산 결과 저장 [cite: 111]
        "avg_disp": np.mean(roi_disp[mask]) if np.any(mask) else 0,
        "avg_depth": np.mean(roi_depth[mask]) if np.any(mask) else 0
    }

disp_tmp = disparity.copy() # 시각화용 시차 맵 복사
disp_tmp[disp_tmp <= 0] = np.nan # 유효하지 않은 값 제외
d_min, d_max = np.nanpercentile(disp_tmp, 5), np.nanpercentile(disp_tmp, 95) # 정규화를 위한 백분위수 계산 [cite: 116]
disp_scaled = np.clip((disp_tmp - d_min) / (d_max - d_min + 1e-6), 0, 1) # 0~1 사이로 정규화
disp_vis = np.zeros_like(disparity, dtype=np.uint8) # 시각화 배경 이미지 생성
disp_vis[~np.isnan(disp_tmp)] = (disp_scaled[~np.isnan(disp_tmp)] * 255).astype(np.uint8) # 유효 픽셀 8비트 변환
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET) # 컬러맵 적용 (가까울수록 빨강)

for name, (x, y, w, h) in rois.items(): # 시각화 결과에 ROI 박스 및 이름 표기
    cv2.rectangle(left_color, (x, y), (x + w, y + h), (0, 255, 0), 2) # 녹색 사각형 그리기
    cv2.putText(left_color, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # 이름 작성

cv2.imwrite(str(output_dir / "disparity_color.jpg"), disparity_color) # 결과 시차 맵 저장
for name, data in results.items(): # 분석 데이터 터미널 출력 [cite: 111]
    print(f"{name}: Avg Disparity = {data['avg_disp']:.2f}, Avg Depth = {data['avg_depth']:.2f}")

cv2.imshow('Disparity Map', disparity_color) # 결과 화면 표시
cv2.waitKey(0) # 키 입력 대기
cv2.destroyAllWindows() # 모든 창 닫기