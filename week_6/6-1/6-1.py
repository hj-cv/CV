import cv2  # OpenCV 라이브러리를 임포트합니다.
import numpy as np  # 수치 연산을 위한 넘파이 라이브러리를 임포트합니다.
import os  # 파일 경로 처리를 위해 os 모듈을 임포트합니다.
from filterpy.kalman import KalmanFilter  # 객체 위치 예측을 위한 칼만 필터를 임포트합니다.
from scipy.optimize import linear_sum_assignment  # 최적 매칭을 위한 헝가리안 알고리즘을 임포트합니다.

base_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 절대 경로를 가져옵니다.
weights_path = os.path.join(base_path, '..', 'L06', 'yolov3.weights')  # 가중치 파일의 상대 경로를 설정합니다.
cfg_path = os.path.join(base_path, '..', 'L06', 'yolov3.cfg')  # 설정 파일의 상대 경로를 설정합니다.
video_path = os.path.join(base_path, '..', 'L06', 'slow_traffic_small.mp4')  # 비디오 파일의 상대 경로를 설정합니다.

class KalmanBoxTracker:  # 개별 객체를 추적하는 클래스를 정의합니다.
    count = 0  # 객체 ID 부여를 위한 카운터를 초기화합니다.
    def __init__(self, bbox):  # 객체 생성 시 초기화를 수행합니다.
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 7차원 상태와 4차원 관측값을 가진 필터를 생성합니다.
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0], [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])  # 상태 전이 행렬을 설정합니다.
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])  # 관측 행렬을 설정합니다.
        self.kf.R[2:,2:] *= 10.  # 측정 노이즈를 설정합니다.
        self.kf.P[4:,4:] *= 1000.  # 초기 오차 공분산을 설정합니다.
        self.kf.P *= 10.  # 전체 공분산을 조절합니다.
        self.kf.Q[-1,-1] *= 0.01  # 프로세스 노이즈를 설정합니다.
        self.kf.Q[4:,4:] *= 0.01  # 속도 관련 노이즈를 설정합니다.
        self.kf.x[:4] = np.array([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2]*bbox[3], bbox[2]/bbox[3]]).reshape((4, 1))  # 초기 상태를 설정합니다.
        self.id = KalmanBoxTracker.count  # 고유 ID를 할당합니다.
        KalmanBoxTracker.count += 1  # 다음 ID를 위해 카운터를 증가시킵니다.

    def update(self, bbox):  # 새로운 관측값으로 상태를 갱신합니다.
        self.kf.update(np.array([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2]*bbox[3], bbox[2]/bbox[3]]).reshape((4, 1)))  # 칼만 갱신을 수행합니다.

    def predict(self):  # 다음 시점의 위치를 예측합니다.
        if((self.kf.x[6]+self.kf.x[2]) <= 0): self.kf.x[6] *= 0.0  # 수치 안정성을 체크합니다.
        self.kf.predict()  # 칼만 예측을 수행합니다.
        return self.kf.x[:4].reshape((4,))  # 예측된 [cx, cy, s, r] 값을 반환합니다.

class Sort:  # 다중 객체 추적을 관리하는 클래스를 정의합니다.
    def __init__(self):  # 클래스 초기화를 수행합니다.
        self.trackers = []  # 추적기 리스트를 생성합니다.

    def update(self, dets):  # 검출 결과를 바탕으로 추적기를 업데이트합니다.
        trks = np.zeros((len(self.trackers), 5))  # 예측값을 담을 배열을 생성합니다.
        for t, trk in enumerate(trks):  # 기존 추적기들을 순회합니다.
            pos = self.trackers[t].predict()  # 각 추적기의 다음 위치를 예측합니다.
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]  # 예측 위치를 배열에 저장합니다.
        
        iou_matrix = np.zeros((len(dets), len(self.trackers)))  # IoU 행렬을 생성합니다.
        for d, det in enumerate(dets):  # 검출된 객체들을 순회합니다.
            for t, trk in enumerate(trks):  # 예측된 객체들을 순회합니다.
                w_p, h_p = np.sqrt(trk[2] * trk[3]), trk[2] / np.sqrt(trk[2] * trk[3])  # 예측 박스의 너비와 높이를 계산합니다.
                tx1, ty1, tx2, ty2 = trk[0]-w_p/2, trk[1]-h_p/2, trk[0]+w_p/2, trk[1]+h_p/2  # 예측 박스의 좌표를 계산합니다.
                dx1, dy1, dx2, dy2 = det[0], det[1], det[0]+det[2], det[1]+det[3]  # 검출 박스의 좌표를 계산합니다.
                xx1, yy1, xx2, yy2 = max(dx1, tx1), max(dy1, ty1), min(dx2, tx2), min(dy2, ty2)  # 교차 영역을 계산합니다.
                w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)  # 교차 영역의 크기를 계산합니다.
                iou_matrix[d, t] = (w*h) / (det[2]*det[3] + trk[2] - (w*h))  # IoU 값을 계산하여 행렬에 넣습니다.

        matched_indices = linear_sum_assignment(-iou_matrix)  # 헝가리안 알고리즘으로 최적 매칭을 수행합니다.
        matched_indices = np.array(matched_indices).T  # 결과를 전치하여 쌍으로 만듭니다.

        unmatched_dets = [d for d in range(len(dets)) if d not in matched_indices[:,0]]  # 매칭 안 된 검출물을 찾습니다.
        unmatched_trks = [t for t in range(len(self.trackers)) if t not in matched_indices[:,1]]  # 매칭 안 된 추적기를 찾습니다.

        for m in matched_indices:  # 매칭된 쌍을 업데이트합니다.
            if iou_matrix[m[0], m[1]] < 0.3:  # IoU가 낮으면 매칭을 취소합니다.
                unmatched_dets.append(m[0])  # 검출물을 미매칭으로 분류합니다.
                unmatched_trks.append(m[1])  # 추적기를 미매칭으로 분류합니다.
            else: self.trackers[m[1]].update(dets[m[0]])  # 성공적인 매칭은 추적기를 업데이트합니다.

        for i in sorted(unmatched_trks, reverse=True): self.trackers.pop(i)  # 사라진 객체의 추적기를 제거합니다.
        for i in unmatched_dets: self.trackers.append(KalmanBoxTracker(dets[i]))  # 새로운 객체에 추적기를 생성합니다.

        return [[t.kf.x[0].item(), t.kf.x[1].item(), t.kf.x[2].item(), t.kf.x[3].item(), t.id] for t in self.trackers]  # 결과를 반환합니다.

net = cv2.dnn.readNet(weights_path, cfg_path)  # YOLOv3 모델을 로드합니다.
layer_names = net.getLayerNames()  # 레이어 이름을 가져옵니다.
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # 출력 레이어를 설정합니다.
tracker = Sort()  # SORT 추적기를 초기화합니다.
cap = cv2.VideoCapture(video_path)  # 비디오 파일을 엽니다.

while cap.isOpened():  # 비디오가 열려 있는 동안 반복합니다.
    ret, frame = cap.read()  # 프레임을 읽어옵니다.
    if not ret: break  # 프레임이 없으면 종료합니다.
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # 이미지를 전처리합니다.
    net.setInput(blob)  # 네트워크에 입력합니다.
    outs = net.forward(output_layers)  # 추론을 수행합니다.
    
    boxes, confidences = [], []  # NMS 적용을 위해 박스와 신뢰도를 담을 리스트를 생성합니다.
    for out in outs:  # 각 레이어의 출력을 확인합니다.
        for detection in out:  # 각 검출 항목을 확인합니다.
            scores = detection[5:]  # 클래스 점수를 가져옵니다.
            class_id = np.argmax(scores)  # 가장 높은 점수의 ID를 찾습니다.
            confidence = scores[class_id]  # 해당 클래스의 확률값을 가져옵니다.
            if confidence > 0.5:  # 신뢰도가 0.5보다 높을 때만 처리합니다.
                w, h = int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])  # 너비와 높이를 계산합니다.
                x, y = int(detection[0] * frame.shape[1] - w / 2), int(detection[1] * frame.shape[0] - h / 2)  # 좌표를 계산합니다.
                boxes.append([x, y, w, h])  # 박스 리스트에 추가합니다.
                confidences.append(float(confidence))  # 신뢰도 리스트에 추가합니다.

    # NMS(Non-Maximum Suppression)를 사용하여 중복 검출된 박스들을 제거합니다.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    dets = [boxes[i] for i in indices] if len(indices) > 0 else []  # 최종 남은 박스들만 dets에 저장합니다.

    tracks = tracker.update(np.array(dets))  # 필터링된 검출물로 추적기를 갱신합니다.

    for trk in tracks:  # 추적 결과를 순회합니다.
        cx, cy, s, r, tid = trk[0], trk[1], trk[2], trk[3], int(trk[4])  # 상태 변수를 가져옵니다.
        w = np.sqrt(s * r)  # 면적과 종횡비로 너비를 복원합니다.
        h = s / w  # 면적과 너비로 높이를 복원합니다.
        x, y = int(cx - w / 2), int(cy - h / 2)  # 좌상단 좌표로 변환합니다.
        cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 255, 0), 2)  # 박스를 그립니다.
        cv2.putText(frame, f"ID: {tid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # ID를 씁니다.

    cv2.imshow("Multi-Object Tracking (SORT)", frame)  # 화면에 출력합니다.
    if cv2.waitKey(1) & 0xFF == 27: break  # ESC 키로 종료합니다.

cap.release()  # 캡처를 종료합니다.
cv2.destroyAllWindows()  # 창을 모두 닫습니다.