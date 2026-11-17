# Day 04 Action Clip & Sequence Summary

## 1. 오늘 한 일

6개의 Action MP4를 Frame 단위로 분석하여
3일차의 38-D Frame Feature를 시간 순서대로 연결한
Sequence CSV를 만들었다.

사용한 Action:

- bend_return
- leg_raise_lower
- walk_turn_walk
- drink_return
- sit_stand
- wave

## 2. 오늘 만든 Pipeline

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

## 3. 본 실습 Sequence 결과

6개 영상에서 총 824 Frame을 처리했다.

- Pose 성공: 824 / 824
- 38-D Feature 성공: 824 / 824
- 6개 Action 모두 pose_ok_ratio = 1.0
- 6개 Action 모두 feature_ok_ratio = 1.0

## 4. Sequence 품질에서 확인한 점

walk_turn_walk은 다른 Action보다:

- 평균 Keypoint Confidence가 낮았다.
- LOW Keypoint 수가 많았다.
- 인접 Frame의 정규화 Feature 변화량이 매우 크게 나타났다.

하지만 Pose와 Feature 생성 자체는 모든 Frame에서 성공했다.

mean_adjacent_feature_change는
Action Recognition Accuracy가 아니라
Frame 사이 자세 변화 정도를 보기 위한 참고값이다.

## 5. Action별 Feature 변화

- wave:
  오른쪽 Wrist와 오른쪽 Elbow Angle 변화가 잘 나타났다.

- drink_return:
  오른쪽 Wrist가 올라가고 Elbow Angle이 크게 작아졌다가 복귀했다.

- bend_return:
  중간 구간에서 양쪽 Wrist의 상대 위치가 변하고 다시 시작 상태에 가까워졌다.

- sit_stand:
  Knee Angle과 Hip Center 높이 변화에서 일어서는 과정이 나타났다.

- walk_turn_walk:
  Hip Center x에서 화면 이동이 뚜렷하게 나타났다.

- leg_raise_lower:
  대표 5개 시점만으로 핵심 순간이 명확하지 않아
  Sample 수를 늘려 볼 필요가 있었다.

## 6. 오늘 이해한 핵심

Pose는 한 순간의 자세이고,
Action은 여러 Pose가 시간에 따라 변하는 과정이다.

한 Frame의 38-D Feature만으로는
Action 전체를 설명하기 어렵다.

여러 Frame의 Feature를 원래 시간 순서대로 연결한
Sequence가 필요하다.

## 7. Mini Challenge

- Action 1:
- Action 2:
- Pose / Feature 품질:
- 시간 변화가 가장 잘 보였던 Feature:
- 두 Action에서 가장 크게 달랐던 점:

## 8. 현재 데이터의 한계

현재 6개의 Clip은 Sequence 구조를 이해하기 위한
소규모 실습 데이터이다.

실제 Action Recognition 학습에는
여러 Subject와 Action별 여러 Take가 추가로 필요하다.