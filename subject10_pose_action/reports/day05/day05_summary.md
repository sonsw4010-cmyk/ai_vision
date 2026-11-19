# Day 05 Summary

## 1. Dataset

- Subjects: S01, S02, S03
- Actions: 6
- Take: Action당 1
- Total Clips: 18

## 2. Keypoint Dataset

- 생성된 Keypoint CSV: 18개
- COCO Keypoint: 17개
- 저장 값:
  - Pixel 좌표
  - Image Normalized 좌표
  - Confidence

## 3. 자동 QA

- AUTO_PASS: 13
- CHECK: 5
- Pose Ratio: 18개 모두 1.0
- CHECK 주요 원인:
  - S02 A001~A005의 single_person_ratio (배경/소품에 의한 다중 인물 검출)

## 4. Human QA

- 필수 Human QA 대상: CHECK 5개
- Human PASS: 5개
- RESHOOT: 0개
- EXCLUDE: 0개
- AUTO_PASS 13개:
  - 필수 Human QA 생략
  - 동작 정상 확인 완료

## 5. Approved Dataset

- AUTO_PASS: 13개
- CHECK 후 Human PASS: 5개
- Approved Clips: 18개

## 6. Subject-wise Split

- Train: S01 / 6 Clips
- Validation: S02 / 6 Clips
- Test: S03 / 6 Clips
- 각 Split에 A001~A006 모두 존재

## 7. Leakage 검사

- Subject overlap: 없음
- Clip overlap: 없음
- Action Coverage: 정상
- Result: Leakage check: OK

## 8. Random Frame Split 비교

- Train Frames: 1946
- Test Frames: 487
- Subject overlap: S01, S02, S03
- Clip overlap: 18

Random Frame Split에서는 같은 사람과 같은 Clip의 Frame이
Train과 Test에 함께 들어갈 수 있음을 확인했습니다.

## 9. 현재 Dataset의 한계

현재 데이터는:

3 Subjects × 6 Actions × 1 Take = 18 Clips

의 최소 교육용 Dataset입니다.

Dataset 구축·QA·Subject-wise Split과
Action Recognition Pipeline을 배우기에는 적절하지만
현업 배포 성능이나 충분한 일반화 성능을 판단하기에는 작습니다.

## 10. 오늘의 핵심

- Pose Dataset은 학습 전에 품질을 검사해야 한다.
- CHECK 데이터만 필요한 Human QA를 수행할 수 있다.
- 평가용 데이터는 사람 기준으로 분리하여 Leakage를 막아야 한다.
