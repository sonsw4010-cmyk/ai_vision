from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
import argparse
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 Human QA 기본 파일 경로 설정
# ==================================================
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REVIEW_PATH = (
    ROOT
    / "data"
    / "day05"
    / "qa"
    / "keypoint_clip_review.csv"
)


# ==================================================
# ③ 허용되는 Human QA 최종 판정값 정의
# ==================================================
ALLOWED = {
    "PASS",
    "RESHOOT",
    "EXCLUDE",
}


# ==================================================
# ④ 터미널 실행 옵션 설정
# ==================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--review-path",
        default=str(
            DEFAULT_REVIEW_PATH
        ),
    )

    parser.add_argument(
        "--expected-clips",
        type=int,
        default=18,
    )

    return parser.parse_args()


# ==================================================
# ⑤ main() 실행 옵션 읽기 및 Review CSV 경로 준비
# ==================================================
def main() -> None:
    args = parse_args()

    review_path = Path(
        args.review_path
    )

    if not review_path.exists():
        print(
            "[ERROR] Human QA 파일을 찾을 수 없습니다."
        )
        print(review_path)
        raise SystemExit(1)

    df = pd.read_csv(
        review_path
    )

    if len(df) != args.expected_clips:
        print(
            "[ERROR] Human QA Clip 수가 "
            "예상과 다릅니다."
        )
        print(
            f"expected={args.expected_clips}, "
            f"actual={len(df)}"
        )
        raise SystemExit(1)

    decisions = (
        df["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    unreviewed_mask = (
        (decisions == "")
        | (decisions == "UNREVIEWED")
    )

    if unreviewed_mask.any():
        unreviewed = df.loc[
            unreviewed_mask,
            [
                "clip_id",
                "qa_status",
                "human_decision",
            ],
        ]

        print()
        print(
            "[HUMAN QA NOT COMPLETE]"
        )
        print()
        print(
            unreviewed.to_string(
                index=False
            )
        )

        print()
        print(
            f"남은 Clip: "
            f"{int(unreviewed_mask.sum())}"
        )

        print()
        print(
            "원본 MP4를 확인한 뒤 "
            "human_decision을"
        )
        print(
            "PASS / RESHOOT / EXCLUDE 중 "
            "하나로 입력하세요."
        )

        raise SystemExit(1)

    invalid_mask = (
        ~decisions.isin(ALLOWED)
    )

    if invalid_mask.any():
        invalid = df.loc[
            invalid_mask,
            [
                "clip_id",
                "qa_status",
                "human_decision",
            ],
        ]

        print()
        print(
            "[INVALID HUMAN DECISION]"
        )
        print()
        print(
            invalid.to_string(
                index=False
            )
        )

        print()
        print(
            "human_decision은 "
            "PASS / RESHOOT / EXCLUDE만 "
            "사용할 수 있습니다."
        )

        raise SystemExit(1)

    print()
    print("[Human QA Result]")
    print(
        decisions.value_counts()
    )

    print()
    print(
        f"Reviewed clips: {len(df)}"
    )
    print(
        "Human review: OK"
    )


if __name__ == "__main__":
    main()
