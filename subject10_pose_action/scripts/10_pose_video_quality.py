from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ② 프로젝트 경로 설정 및 공통 함수 불러오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    load_config,
    resolve_device,
)
from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ③ 터미널에서 영상 경로 및 실행 옵션 입력받기
def parse_args():
    parser = argparse.ArgumentParser()

    # 분석할 MP4 영상 또는 비디오 폴더 (기본값: videos 폴더 전체)
    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day02"
            / "videos"
        ),
        help="분석할 MP4 영상 파일 경로 또는 비디오들이 들어있는 폴더 경로",
    )

    # LOW 관절을 판단할 Confidence 기준
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    # 최대 처리 Frame 수 (0이면 영상 전체)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="처리할 최대 Frame 수. 0이면 영상 끝까지 처리",
    )

    return parser.parse_args()


# ④ 단일 동영상 분석 및 리포트 저장 함수
def process_single_video(
    source: Path,
    model: YOLO,
    config: dict,
    device: str,
    max_frames: int,
    kp_conf: float,
) -> None:
    print(f"\n========================================")
    print(f"영상 분석 시작: {source.name}")
    print(f"========================================")

    capture = cv2.VideoCapture(str(source))

    # 영상 연결 여부 확인
    if not capture.isOpened():
        print(f"영상을 열 수 없습니다: {source}")
        return

    # 영상 FPS·해상도 및 첫 Frame 확인
    fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = fps if fps > 0 else 30.0

    success, frame = capture.read()
    if not success:
        capture.release()
        print(f"영상의 첫 Frame을 읽지 못했습니다: {source}")
        return

    height, width = frame.shape[:2]

    # Pose 결과 MP4 저장 준비
    output_video = (
        ROOT
        / "reports"
        / "day02"
        / "videos"
        / f"{source.stem}_pose.mp4"
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        print(f"결과 MP4 VideoWriter 생성 실패: {output_video}")
        return

    # Frame 품질 CSV와 Keypoint CSV 저장 준비
    frame_table = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / f"{source.stem}_frame_quality.csv"
    )
    keypoint_table = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / f"{source.stem}_keypoints.csv"
    )
    frame_table.parent.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    keypoint_rows = []
    frame_id = 0

    # Frame별 YOLO Pose 추론 및 결과 영상 저장
    while success:
        frame_id += 1

        results = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )

        result = results[0]
        writer.write(result.plot())

        people = len(result.boxes) if result.boxes is not None else 0
        mean_kp_conf = 0.0
        low_keypoints = 17
        lowest_name = "NO_PERSON"
        lowest_conf = 0.0

        if (
            result.boxes is not None
            and result.keypoints is not None
            and result.keypoints.xy is not None
            and len(result.boxes) > 0
            and len(result.keypoints.xy) > 0
        ):
            box_areas = (
                result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0]
            ) * (
                result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1]
            )
            main_idx = int(box_areas.argmax().item())

            main_xy = result.keypoints.xy[main_idx].cpu().numpy()
            main_conf = result.keypoints.conf[main_idx].cpu().numpy()

            mean_kp_conf = float(main_conf.mean())
            low_keypoints = int((main_conf < kp_conf).sum())

            lowest_id = int(main_conf.argmin())
            lowest_name = COCO_KEYPOINT_NAMES[lowest_id]
            lowest_conf = float(main_conf[lowest_id])

            for keypoint_id, name in enumerate(COCO_KEYPOINT_NAMES):
                x, y = main_xy[keypoint_id]
                confidence = main_conf[keypoint_id]

                keypoint_rows.append(
                    {
                        "frame_id": frame_id,
                        "time_sec": round((frame_id - 1) / output_fps, 3),
                        "keypoint_id": keypoint_id,
                        "keypoint_name": name,
                        "pixel_x": round(float(x), 2),
                        "pixel_y": round(float(y), 2),
                        "confidence": round(float(confidence), 4),
                    }
                )

        frame_rows.append(
            {
                "frame_id": frame_id,
                "time_sec": round((frame_id - 1) / output_fps, 3),
                "people": people,
                "mean_keypoint_conf": round(mean_kp_conf, 4),
                "low_keypoint_count": low_keypoints,
                "lowest_keypoint": lowest_name,
                "lowest_keypoint_conf": round(lowest_conf, 4),
                "keypoint_threshold": kp_conf,
            }
        )

        if frame_id % 30 == 0:
            print(f"[{source.name}] Processed frame: {frame_id}")

        if max_frames > 0 and frame_id >= max_frames:
            break

        success, frame = capture.read()

    capture.release()
    writer.release()

    # Frame별 품질 요약 CSV 작성
    with frame_table.open("w", newline="", encoding="utf-8-sig") as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "time_sec",
                "people",
                "mean_keypoint_conf",
                "low_keypoint_count",
                "lowest_keypoint",
                "lowest_keypoint_conf",
                "keypoint_threshold",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)

    # Keypoint 상세 CSV 저장
    with keypoint_table.open("w", newline="", encoding="utf-8-sig") as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "time_sec",
                "keypoint_id",
                "keypoint_name",
                "pixel_x",
                "pixel_y",
                "confidence",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(keypoint_rows)

    print(f"완료: {source.name} (총 {frame_id}프레임)")
    print(f"  - Pose 영상: {output_video}")
    print(f"  - 품질 요약: {frame_table}")
    print(f"  - 관절 좌표: {keypoint_table}")


def main() -> None:
    args = parse_args()
    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(f"경로가 존재하지 않습니다: {source}")

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    # 1. 대상이 폴더인 경우: 폴더 안의 모든 동영상(.mp4 등)을 찾아 순차 분석
    if source.is_dir():
        video_files = sorted(
            [f for f in source.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
        )
        if not video_files:
            print(f"폴더 내에 분석할 동영상 파일이 없습니다: {source}")
            return

        print(f"\n[일괄 분석 모드] 총 {len(video_files)}개의 영상을 분석합니다.")
        for idx, video in enumerate(video_files, start=1):
            print(f"\n>>> [{idx}/{len(video_files)}] 작업 중...")
            process_single_video(
                source=video,
                model=model,
                config=config,
                device=device,
                max_frames=args.max_frames,
                kp_conf=args.kp_conf,
            )
        print(f"\n전체 {len(video_files)}개 영상의 분석과 저장이 모두 완료되었습니다!")

    # 2. 대상이 단일 동영상 파일인 경우: 해당 파일 1개만 분석
    elif source.is_file():
        process_single_video(
            source=source,
            model=model,
            config=config,
            device=device,
            max_frames=args.max_frames,
            kp_conf=args.kp_conf,
        )
    else:
        raise ValueError(f"유효하지 않은 경로입니다: {source}")


if __name__ == "__main__":
    main()
