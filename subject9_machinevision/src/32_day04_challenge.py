from pathlib import Path

import cv2


reference_path = Path(
    "samples/day04/reference/reference_label.jpg"
)

challenge_dir = Path(
    "samples/day04/challenge"
)

output_dir = Path(
    "outputs/day04/challenge"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


challenge_files = [
    challenge_dir / "shift.jpg",
    challenge_dir / "rotate.jpg",
    challenge_dir / "distance.jpg",
    challenge_dir / "tilt_glare.jpg",
]


# --------------------------------------------------
# 1. 사용할 방법을 선택합니다.
#
# TEMPLATE
# ORB
# SIFT
# SIFT_HOMOGRAPHY
# --------------------------------------------------

selected_method = "CHOOSE_METHOD"


reference = cv2.imread(
    str(reference_path)
)

if reference is None:
    raise FileNotFoundError(
        reference_path
    )


# --------------------------------------------------
# 2. Challenge 이미지를 하나씩 읽습니다.
# --------------------------------------------------

for target_path in challenge_files:

    target = cv2.imread(
        str(target_path)
    )

    if target is None:
        print(
            "읽기 실패:",
            target_path
        )
        continue


    # --------------------------------------------------
    # TODO
    #
    # 자신이 선택한 방법에 맞게
    # 27~30번 코드에서 필요한 부분을 가져옵니다.
    #
    # TEMPLATE
    # → Score + Location
    #
    # ORB
    # → Good Match
    #
    # SIFT
    # → Good Match
    #
    # SIFT_HOMOGRAPHY
    # → Good Match + Inlier + 로고 위치
    # --------------------------------------------------


    print(
        target_path.name,
        "method=",
        selected_method,
    )