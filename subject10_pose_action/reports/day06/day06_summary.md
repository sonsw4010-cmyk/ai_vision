# Day 06 Action Recognition Baseline 요약 보고서

## 1. 입력 데이터

- 기반 데이터: Day 05 승인 Keypoint Dataset (18 Clips)
- S01: Train (6 Clips)
- S02: Validation (6 Clips)
- S03: Test (6 Clips)

## 2. Sequence Feature 구성

- 1 Frame = 관절 정규화 좌표 34개 + 관절 각도 4개 = 38-D Feature
- 1 Clip = 30 Step 리샘플링 × 38-D = 1,140-D Sequence Feature
- 산출 파일:
  - `data/day06/sequences/train.csv` (S01, 6행)
  - `data/day06/sequences/val.csv` (S02, 6행)
  - `data/day06/sequences/test.csv` (S03, 6행)

## 3. Baseline 모델 학습

- 모델 알고리즘: Random Forest Classifier
- Tree 수: 300개
- Random Seed: 42
- 학습 데이터: S01 (6 Clips)
- 산출 모델: `artifacts/day06/rf_action_baseline.joblib`
- Feature 컬럼: `artifacts/day06/rf_feature_columns.json`

## 4. Validation 평가 결과 (S02)

- Accuracy: 0.5000 (3 / 6 Correct)
- Macro F1: 0.4444
- 정답 동작:
  - `bend_return`: PASS (1.00)
  - `leg_raise_lower`: PASS (1.00)
  - `sit_stand`: PASS (1.00)
- 오답 동작 (3건):
  - `walk_turn_walk` → `leg_raise_lower` (conf: 0.213)
  - `drink_return` → `wave` (conf: 0.227)
  - `wave` → `drink_return` (conf: 0.220)

## 5. 산출물 및 주요 파일

- `scripts/33_check_day06_inputs.py`: Day06 입력 데이터 정합성 검사
- `scripts/34_build_sequence_dataset.py`: 1,140-D Sequence Feature 변환
- `scripts/35_train_rf_baseline.py`: Random Forest Baseline 학습
- `scripts/36_evaluate_validation.py`: S02 검증 및 혼동행렬 생성
- `scripts/37_analyze_misclassified.py`: 오분류 클립 추적 분석
