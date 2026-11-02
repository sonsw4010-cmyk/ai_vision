# Day04 Mini Challenge

검사 이미지:
rotate.jpg

주요 변화:
회전

선택한 방법:
SIFT + Homography

선택 이유:
회전된 로고의 특징을 비교하고
실제 위치까지 확인하기 위해 선택했다.

결과:
로고 위치를 사각형으로 찾았다.

관찰:
회전된 이미지에서도 특징점 Matching이 유지되었다.

한계:
반사나 가림이 심해지면
특징점이 줄어들 수 있다.

Threshold:
Day05 Calibration 후 결정