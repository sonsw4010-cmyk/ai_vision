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

## Day 04 - Action Clip & Sequence

### 목표

3일차의 38-D Frame Feature를
영상의 시간 순서대로 연결하여
Action Sequence를 만든다.

### 사용한 Action

- bend_return
- leg_raise_lower
- walk_turn_walk
- drink_return
- sit_stand
- wave

### Pipeline

```text
Action Clip
→ Frame
→ YOLO Pose
→ COCO 17 Keypoint
→ Normalization
→ Joint Angle
→ 38-D Frame Feature
→ 시간 순서 유지
→ Sequence
→ Action Label
```

### 주요 코드

```text
scripts/
├── 19_check_day04_data.py
├── 20_check_action_classes.py
├── 21_extract_clip_keyframes.py
├── 22_build_action_sequences.py
├── 23_summarize_action_sequences.py
└── 24_inspect_sequence_stages.py
```

### 주요 결과

- 6개의 Action Sequence CSV 생성
- 총 824 Frame 처리
- Pose 성공 824 / 824
- Feature 생성 성공 824 / 824
- Action별 Feature의 시간 변화 확인
- 직접 촬영한 Action에도 같은 Sequence Pipeline 적용

### 오늘 확인한 핵심

- Pose는 한 순간의 자세이다.
- Sequence는 Frame Feature의 시간 흐름이다.
- Action은 한 Frame이 아니라 시간에 따른 자세 변화로 이해해야 한다.
- Action마다 중요하게 변하는 Feature가 다를 수 있다.
- 대표 Sample만으로 동작이 잘 보이지 않으면 더 많은 시점을 확인해야 한다.

### 다음 수업

Day 05에서는 여러 Subject와 반복 Take를 이용하여
Sequence 데이터를 학습용 Dataset 구조로 확장하고
품질검수와 데이터 분리를 진행한다.

## Day 05 - Keypoint Dataset QA & Subject-wise Split

### Dataset

```text
3 Subjects
×
6 Actions
×
1 Take
=
18 MP4
```

```text
A001 bend_return
A002 leg_raise_lower
A003 walk_turn_walk
A004 drink_return
A005 sit_stand
A006 wave
```

### Pipeline

```text
18 MP4
        ↓
COCO 17 Keypoint Dataset
        ↓
자동 QA
        ↓
AUTO_PASS 13
CHECK 5
        ↓
CHECK Human QA
        ↓
Human PASS 5
        ↓
Approved 18 Clips
        ↓
Subject-wise Split
        ↓
S01 → Train
S02 → Validation
S03 → Test
        ↓
Leakage check: OK
```

### 주요 결과

- Keypoint CSV: 18개
- AUTO_PASS: 13개
- CHECK: 5개
- CHECK Human PASS: 5개
- Approved Clips: 18개
- Train: S01 / 6 Clips
- Validation: S02 / 6 Clips
- Test: S03 / 6 Clips
- Subject overlap: 없음
- Clip overlap: 없음

### Random Frame Split 비교

```text
Train Frames: 1946
Test Frames : 487

Subject overlap:
S01, S02, S03

Clip overlap:
18
```

따라서 최종 평가에는 Random Frame Split을 사용하지 않고
Subject-wise Split을 사용합니다.

### 주요 실행 파일

```text
scripts/
├── 25_check_day05_data.py
├── 26_extract_keypoint_dataset.py
├── 27_keypoint_qa.py
├── 28_validate_human_review.py
├── 29_build_approved_manifest.py
├── 30_build_subject_split.py
├── 31_check_split_leakage.py
└── 32_demo_frame_split_leakage.py
```

### Next

Day 06에서는 Day 05의 Keypoint Dataset과 Subject-wise Split을 그대로 사용하여
Body Normalization과 Joint Angle Feature를 만들고
Action Recognition Baseline 학습으로 연결합니다.

---

## Day 06 - Action Recognition Baseline

### Input

Day 05 Approved Keypoint Dataset

- S01 → Train (6 Clips)
- S02 → Validation (6 Clips)
- S03 → Test (6 Clips)
- 6 Actions: `bend_return`, `drink_return`, `leg_raise_lower`, `sit_stand`, `walk_turn_walk`, `wave`
- Total: 18 Clips

### Pipeline

```text
Keypoint CSV (18개)
  ↓
38-D Frame Feature (좌표 34개 + 관절 각도 4개)
  ↓
Sequence Resampling (30 Step)
  ↓
Sequence Feature (1,140-D)
  ↓
Random Forest Baseline 학습 (S01, 300 Trees)
  ↓
Validation 평가 (S02) & 오분류 분석
```

### Baseline 설정

- Sequence Steps: 30
- Feature Count: 1,140
- n_estimators: 300
- max_depth: null
- random_state: 42

### Baseline Validation 결과 (S02)

- Accuracy: 0.5000 (3 / 6)
- Macro F1: 0.4444

Confusion Matrix & Misclassified:
- `walk_turn_walk` → `leg_raise_lower` (conf: 0.213)
- `drink_return` → `wave` (conf: 0.227)
- `wave` → `drink_return` (conf: 0.220)

### 주요 실행 파일

```text
scripts/
├── 33_check_day06_inputs.py
├── 34_build_sequence_dataset.py
├── 35_train_rf_baseline.py
├── 36_evaluate_validation.py
└── 37_analyze_misclassified.py
```

