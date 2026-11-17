## Day 03 - Joint Angle, Normalization & Frame Feature

### 오늘 배운 내용

- COCO 17 Keypoint와 Confidence 확인
- 세 Keypoint를 이용한 관절각도 계산
- Hip Center 기준 몸 중심 이동
- Shoulder Width 기준 크기 정규화
- Near / Far 정규화 전·후 비교
- Person A / Person B 비교
- COCO Left / Right 기준 이해
- Keypoint 좌우 반전
- 34개 Normalized 좌표 + 4개 관절각도
- 38-D Frame Feature 생성
- 동작을 설명하는 Feature 선택
- 영상의 각 Frame에서 38-D Feature 생성
- Feature가 시간에 따라 변하는 과정 확인

### Day 03 핵심 흐름

```text
YOLO Pose
        ↓
COCO 17 Keypoint
+
Confidence
        ↓
관절각도
        ↓
Hip Center
        ↓
Shoulder Width Normalization
        ↓
Normalized Keypoint 34개
+
Joint Angle 4개
        ↓
38-D Frame Feature
        ↓
영상의 각 Frame에 반복
        ↓
시간에 따른 Feature 변화
        ↓
Sequence의 재료
```

### 오늘 추가한 주요 코드

```text
src/
└── feature_utils.py

scripts/
├── 11_check_day03_data.py
├── 12_joint_angle.py
├── 13_normalize_pose.py
├── 14_compare_normalization.py
├── 15_flip_keypoints.py
├── 16_make_frame_feature.py
├── 17_arms_up_rule.py
└── 18_video_frame_features.py
```

### 오늘 이해한 핵심

한 장의 Pose에서 얻은 Keypoint 좌표를 그대로 사용하는 것이 아니라,
몸의 위치와 크기 차이를 줄인 Normalized 좌표와 관절각도를 이용하여
한 Frame의 자세를 38개의 숫자로 표현했습니다.

또한 동작을 구분할 때는 38개의 숫자를 무조건 모두 보는 것이 아니라,
손목과 어깨의 상대 높이처럼 해당 동작을 설명하는 Feature를 선택하는 것이 중요하다는 것을 확인했습니다.

영상에서는 같은 38-D Feature를 Frame마다 반복해서 만들고,
Feature가 시간에 따라 변하는 모습을 확인했습니다.

### Mini Challenge

팀별로 직접 준비한 자세와 영상을 이용하여

- 관절각도
- Confidence
- Near / Far 정규화
- 38-D Frame Feature
- 의미 있는 Feature 선택
- 영상에서의 Feature 변화

과정을 다시 수행했습니다.

### 다음 수업

4일차에는 여러 Frame의 Feature를 시간 순서로 묶어
하나의 동작을 표현하는 **Sequence**로 확장합니다.