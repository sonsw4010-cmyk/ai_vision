from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
import argparse
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 기본 입력·출력 경로 설정
# ==================================================
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REVIEW_PATH = (
    ROOT
    / "data"
    / "day05"
    / "qa"
    / "keypoint_clip_review.csv"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "approved_clip_manifest.csv"
)


# ==================================================
# ③ 터미널 실행 옵션 설정
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
        "--output-path",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


# ==================================================
# ④ main() 실행 옵션과 파일 경로 준비
# ==================================================
def main() -> None:
    args = parse_args()

    review_path = Path(
        args.review_path
    )

    output_path = Path(
        args.output_path
    )

    if not review_path.exists():
        raise FileNotFoundError(
            review_path
        )

    df = pd.read_csv(
        review_path
    )

    decisions = (
        df["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    approved = df[
        decisions == "PASS"
    ].copy()

    if approved.empty:
        raise RuntimeError(
            "PASS 데이터가 없습니다."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    approved.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Approved clips: "
        f"{len(approved)}"
    )
    print()
    print("[Subject distribution]")
    print(
        approved["subject"]
        .value_counts()
        .sort_index()
    )
    print()
    print("[Action distribution]")
    print(
        approved["action_label"]
        .value_counts()
        .sort_index()
    )
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
