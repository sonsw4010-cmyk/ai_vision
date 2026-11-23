from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    load_config,
    resolve_device,
)

# 파일명이 숫자로 시작하므로 importlib로 26번 모듈을 불러옵니다.
_spec = importlib.util.spec_from_file_location(
    "extract_keypoint_dataset",
    ROOT / "scripts" / "26_extract_keypoint_dataset.py",
)
extract26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract26)


ACTION_FILE = ROOT / "configs" / "motion_capture_action_classes.csv"


def load_action_map() -> dict[str, str]:
    with ACTION_FILE.open("r", encoding="utf-8-sig") as file:
        return {
            row["action_id"].strip().upper(): row["action_label"].strip()
            for row in csv.DictReader(file)
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        default="S01",
    )
    parser.add_argument(
        "--video-root",
        default=str(ROOT / "data" / "motion_capture"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "motion_capture_keypoints"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject = args.subject.upper()
    video_dir = Path(args.video_root) / subject
    output_root = Path(args.output_root)

    if not video_dir.exists():
        raise FileNotFoundError(video_dir)

    clips = sorted(video_dir.glob("*.mp4"))
    if not clips:
        raise RuntimeError(f"MP4가 없습니다: {video_dir}")

    # 파일명 규칙 검사 및 Action ID 수집
    action_ids = set()
    for clip in clips:
        match = extract26.FILENAME_PATTERN.match(clip.name)
        if match is None:
            raise ValueError(f"파일명 규칙 오류: {clip.name}")
        action_ids.add(match.group(2).upper())

    # 동작 이름은 motion_capture_action_classes.csv에서 읽습니다.
    action_map = load_action_map()
    missing = sorted(action_ids - set(action_map))
    if missing:
        raise RuntimeError(
            f"{ACTION_FILE.name}에 없는 Action ID: {missing}"
        )

    # 기존 manifest 읽기 (캐시 재사용)
    manifest_path = output_root / f"manifest_{subject}.csv"
    existing_manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig") as file:
            for r in csv.DictReader(file):
                existing_manifest[r["clip_id"]] = r

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(str(ROOT / config["pose_model"]))

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / subject).mkdir(parents=True, exist_ok=True)

    print(f"Subject: {subject}")
    print(f"Clips: {len(clips)} / Actions: {len(action_map)}")
    print(f"Device: {device}")
    print(f"Existing in manifest: {len(existing_manifest)}")
    print()

    manifest_rows = []
    for index, clip in enumerate(clips, start=1):
        clip_id = clip.stem
        flat_csv = output_root / subject / f"{clip_id}.csv"
        
        # 이미 존재하고 manifest에도 있으면 재사용
        if clip_id in existing_manifest and flat_csv.exists() and flat_csv.stat().st_size > 0:
            print(f"[{index}/{len(clips)}] [CACHED] {clip.name}")
            manifest_rows.append(existing_manifest[clip_id])
            continue

        print(f"[{index}/{len(clips)}] [EXTRACTING] {clip.name}")
        item = extract26.process_clip(
            model=model,
            clip_path=clip,
            action_map=action_map,
            config=config,
            device=device,
            output_root=output_root,
        )
        # 26번 코드는 S01/<label>/ 하위에 저장하므로
        # 영상 폴더와 같은 S01/ 바로 아래로 옮깁니다.
        nested_csv = ROOT / item["keypoint_csv"]
        nested_csv.replace(flat_csv)
        if not any(nested_csv.parent.iterdir()):
            nested_csv.parent.rmdir()
        item["keypoint_csv"] = str(flat_csv.relative_to(ROOT))
        item["orientation"] = (
            "portrait"
            if item["height"] >= item["width"]
            else "landscape"
        )
        manifest_rows.append(item)
        print(
            f"  pose_ratio={item['pose_ratio']} "
            f"mean_conf={item['mean_keypoint_conf']} "
            f"{item['orientation']}"
        )

    # clip_id 정렬
    manifest_rows.sort(key=lambda x: x["clip_id"])

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(manifest_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"Subject {subject} keypoint extraction: COMPLETE ({len(manifest_rows)} clips)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
