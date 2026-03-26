# 과제 요약

## 4-1. SIFT를 이용한 특징점 검출 및 시각화
- [cite_start]**기능**: 주어진 이미지에서 크기와 회전 변화에 강인한 특징점(Keypoint)을 추출하고 시각적으로 표시함[cite: 12, 17].
- **해결 방법**:
  - [cite_start]`cv.imread()`로 이미지(`mot_color70.jpg`)를 로드한 뒤 `cv.cvtColor()`를 사용해 그레이스케일로 변환함[cite: 34].
  - [cite_start]`cv.SIFT_create(nfeatures=500)`를 사용하여 특징점 개수가 제한된 SIFT 객체를 생성함[cite: 20, 25].
  - [cite_start]`detectAndCompute()`를 호출하여 이미지 내 특징점과 해당 지점의 기술자(Descriptor)를 동시에 검출함[cite: 21].
  - [cite_start]`cv.drawKeypoints()` 함수에 `flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` 설정을 적용하여 특징점의 방향과 크기를 시각화함[cite: 22, 26].
  - [cite_start]Matplotlib을 사용하여 원본 이미지와 특징점이 표시된 이미지를 나란히 배치하여 결과를 출력함[cite: 23].
- **결과 이미지**:
  ![4-1](./4-1/4-1.png)

## 4-2. SIFT를 이용한 두 영상 간 특징점 매칭
- [cite_start]**기능**: 두 이미지 사이의 대응되는 특징점을 찾아 연결함으로써 영상 간의 상관관계를 시각화함[cite: 29, 32].
- **해결 방법**:
  - [cite_start]`mot_color70.jpg`와 `mot_color83.jpg` 두 장의 이미지를 입력받아 각각 SIFT 특징점을 추출함[cite: 32, 38].
  - [cite_start]`cv.BFMatcher()` 객체를 생성할 때 `cv.NORM_L2` 거리 측정 방식과 `crossCheck=True`를 설정하여 매칭의 정확도를 높임[cite: 39, 42].
  - `match()` 함수를 통해 두 기술자 집합 간의 최적 매칭 쌍을 찾고, 유사도가 높은(거리가 짧은) 순으로 정렬함.
  - [cite_start]`cv.drawMatches()`를 사용하여 상위 50개의 매칭 결과를 두 영상 사이에 직선으로 연결하여 시각화함[cite: 40].
- **결과 이미지**:
  ![4-2](./4-2/4-2.png)

## 4-3. 호모그래피를 이용한 이미지 정합 (Image Alignment)
- [cite_start]**기능**: 두 이미지 간의 기하학적 변환 관계인 호모그래피를 계산하여 하나의 좌표계로 통합하는 정합(Alignment)을 수행함[cite: 47, 50].
- **해결 방법**:
  - [cite_start]`cv.SIFT_create()`를 통해 특징점을 추출하고, `cv.BFMatcher().knnMatch()`를 사용하여 각 점당 최근접 이웃 2개를 검출함[cite: 53, 54].
  - [cite_start]최근접 이웃 거리 비율(Lowe's ratio test)을 0.7 미만으로 적용하여 신뢰할 수 있는 매칭점만 선별함[cite: 44, 62].
  - [cite_start]선별된 대응점 좌표들을 `cv.findHomography()`에 입력하고 `cv.RANSAC` 알고리즘을 적용하여 이상점(Outlier)을 제거한 정확한 변환 행렬 $H$를 계산함[cite: 55, 60].
  - [cite_start]`cv.warpPerspective()`를 사용하여 기준 이미지를 변환 행렬에 맞춰 뒤틀고, 두 이미지를 합친 크기의 캔버스`(w1+w2, max(h1,h2))`에 배치하여 정합을 완료함[cite: 56, 61].
- **결과 이미지**:
  ![4-3](./4-3/4-3(32).png)