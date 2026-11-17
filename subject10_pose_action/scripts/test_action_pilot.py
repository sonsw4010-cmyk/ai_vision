"""
========================================================================================
3조 무인 상점 이상행동 감지 - [범용 동작 구간(Start-Peak-End) 자동 탐지] 스크립트

[⭐ 핵심 신규 기능 ⭐]
1. 전신 운동 에너지(Total Motion Energy) 기반 시작/끝 탐지:
   - 특정 관절(손목)에 국한되지 않고 17개 모든 관절의 순간 이동 속도 합산!
   - 발차기, 때리기, 쓰러지기 등 '모든 이상행동'에 공통 적용 가능한 범용 알고리즘!
2. 동작 구간(Action Boundary) 자동 트리밍:
   - 대기 상태(Idle) -> 동작 시작(Start) -> 절정(Peak) -> 동작 종료(End) 구간 자동 분할
3. 영상 화면에 현재 구간 [IDLE / ACTION / RETURN] 실시간 자막 표시!
========================================================================================
"""

from pathlib import Path
import sys
import math
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# 1. ROOT 설정 및 정규화 모듈 연결
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_utils import normalize_by_shoulder_width

# 분석할 동영상 (동연 님 키오스크 내려치기 정면 뷰)
VIDEO_PATH = Path("/mnt/c/Users/user1/Desktop/모션/동연/20261116_142236.mp4")

OUTPUT_DIR = Path("/home/user1/ai_vision/subject10_pose_action/reports/pilot_team3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "kiosk_strike_segmented.mp4"

print("=" * 70)
print(f"🎬 [STEP 1] 범용 구간 탐지 영상 로드: {VIDEO_PATH.name}")
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise FileNotFoundError(f"영상을 열 수 없습니다: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps if fps > 0 else 0

print(f"   - 해상도: {width}x{height}, FPS: {fps:.1f}, 전체: {total_frames}프레임 ({duration_sec:.2f}초)")

# 1배속 재생 비디오 녹화기
output_fps = fps / 2.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, output_fps, (width, height))

model = YOLO("yolo11m-pose.pt")

print("⚡ [STEP 2] 1차 전수 분석: 17개 전신 관절 운동 에너지 추적 중...")

# --------------------------------------------------------------------------------------
# 1차 루프: 모든 프레임에서 전신 관절 좌표 및 순간 운동 에너지 수집
# --------------------------------------------------------------------------------------
frame_data = []
frame_idx = 0
prev_kpts = None
prev_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % 2 == 0:
        time_sec = frame_idx / fps
        pred = model(frame, imgsz=1280, conf=0.50, max_det=1, verbose=False)[0]

        if pred.boxes is not None and len(pred.boxes) > 0 and pred.keypoints is not None:
            kpts = pred.keypoints.xy[0].cpu().numpy()  # (17, 2)

            # ⭐ [범용 전신 운동 에너지 계산]
            # 17개 관절 각각이 이전 프레임 대비 이동한 유클리드 거리(픽셀)의 합
            if prev_kpts is not None and prev_time is not None:
                dt = time_sec - prev_time
                if dt > 0:
                    displacements = np.linalg.norm(kpts - prev_kpts, axis=1) # 17개 관절별 이동거리
                    motion_energy = np.sum(displacements) / dt              # 전신 운동 에너지 (px/sec)
                else:
                    motion_energy = 0
            else:
                motion_energy = 0

            prev_kpts = kpts.copy()
            prev_time = time_sec

            # 정규화 및 손목 속도 계산
            norm_kpts, _, shoulder_w = normalize_by_shoulder_width(kpts)
            rw_y = float(kpts[10][1])

            frame_data.append({
                "frame": frame_idx,
                "time_sec": time_sec,
                "motion_energy": motion_energy,
                "raw_frame": frame,
                "pred": pred,
                "rw_y": rw_y,
                "shoulder_w": shoulder_w
            })

    frame_idx += 1

cap.release()

# --------------------------------------------------------------------------------------
# [STEP 3] 범용 알고리즘으로 시작(Start) - 절정(Peak) - 종료(End) 자동 판별
# --------------------------------------------------------------------------------------
print("\n🧠 [STEP 3] 범용 시작/절정/종료 구간 자동 탐색 중...")
df_energy = pd.DataFrame([{"frame": d["frame"], "time_sec": d["time_sec"], "energy": d["motion_energy"]} for d in frame_data])

# 3프레임 스무딩으로 떨림 제거
df_energy["smooth_energy"] = df_energy["energy"].rolling(window=5, min_periods=1, center=True).mean()

# 평상시 대기 상태의 기준 에너지 (하위 30% 수준)
baseline_energy = df_energy["smooth_energy"].quantile(0.30)
# 동작 시작으로 판단할 임계값 (평상시의 3배 이상 또는 최소 400 px/s)
start_threshold = max(baseline_energy * 3.0, 400.0)

# 1. 절정(Peak): 전신 에너지가 가장 거대하게 폭발한 프레임!
peak_row_idx = df_energy["smooth_energy"].idxmax()
peak_time = df_energy.loc[peak_row_idx, "time_sec"]
peak_frame = df_energy.loc[peak_row_idx, "frame"]
max_energy = df_energy.loc[peak_row_idx, "smooth_energy"]

# 2. 시작(Start): 절정 직전에서 에너지가 임계값을 뚫고 올라오기 시작한 지점
before_peak = df_energy.loc[:peak_row_idx]
start_candidates = before_peak[before_peak["smooth_energy"] < start_threshold]
if len(start_candidates) > 0:
    start_row_idx = start_candidates.index[-1]
else:
    start_row_idx = 0
start_time = df_energy.loc[start_row_idx, "time_sec"]
start_frame = df_energy.loc[start_row_idx, "frame"]

# 3. 종료(End): 절정 이후에서 에너지가 다시 임계값 밑으로 가라앉은 지점
after_peak = df_energy.loc[peak_row_idx:]
end_candidates = after_peak[after_peak["smooth_energy"] < start_threshold]
if len(end_candidates) > 0:
    end_row_idx = end_candidates.index[0]
else:
    end_row_idx = len(df_energy) - 1
end_time = df_energy.loc[end_row_idx, "time_sec"]
end_frame = df_energy.loc[end_row_idx, "frame"]

action_duration = end_time - start_time

print("=" * 70)
print("🎯 [전신 운동 에너지 기반 자동 탐지 결과]")
print(f"  * 대기 상태 평균 에너지 : {baseline_energy:.1f} px/s")
print(f"  * 시작 임계값(Threshold): {start_threshold:.1f} px/s")
print(f"  -------------------------------------------------------------")
print(f"  🏁 동작 시작 (START) : {start_time:.2f}초 (프레임 {int(start_frame)})")
print(f"  🔥 동작 절정 (PEAK)  : {peak_time:.2f}초 (프레임 {int(peak_frame)}) [에너지: {max_energy:.0f} px/s]")
print(f"  🛑 동작 종료 (END)   : {end_time:.2f}초 (프레임 {int(end_frame)})")
print(f"  ⏱️ 순수 동작 구간    : {action_duration:.2f}초 (전체 {duration_sec:.2f}초 중 알맹이만 도려냄!)")
print("=" * 70)

# --------------------------------------------------------------------------------------
# [STEP 4] 구간 상태(IDLE / ACTION IN PROGRESS / FINISHED) 자막 비디오 렌더링
# --------------------------------------------------------------------------------------
print("\n📼 [STEP 4] 구간 라벨이 오버레이된 최종 동영상 렌더링 중...")

for item in frame_data:
    t = item["time_sec"]
    pred = item["pred"]
    annotated = pred.plot(labels=False)

    # 구간 판별
    if t < start_time:
        stage_text = "STAGE: [1/3] IDLE / WAITING"
        stage_color = (200, 200, 200) # 회색
    elif start_time <= t <= end_time:
        if abs(t - peak_time) < 0.15:
            stage_text = "STAGE: [2/3] PEAK IMPACT! (STRIKE!)"
            stage_color = (0, 0, 255) # 빨간색 (절정 타격)
        else:
            stage_text = "STAGE: [2/3] ACTION IN PROGRESS"
            stage_color = (0, 165, 255) # 주황색 (동작 진행 중)
    else:
        stage_text = "STAGE: [3/3] ACTION FINISHED / RETURN"
        stage_color = (0, 255, 0) # 초록색 (종료/복귀)

    # 반투명 오버레이 박스
    overlay = annotated.copy()
    cv2.rectangle(overlay, (30, 40), (880, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    # 대시보드 텍스트 출력
    cv2.putText(annotated, "[TEAM 3] UNIVERSAL ACTION SEGMENTATION", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(annotated, f"Time: {t:.2f}s | Energy: {item['motion_energy']:.0f} px/s", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(annotated, f"Boundary: [{start_time:.2f}s -> {end_time:.2f}s] (Peak: {peak_time:.2f}s)", (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated, stage_text, (50, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, stage_color, 3)

    writer.write(annotated)

writer.release()

# 구간 정보를 포함한 CSV 저장
df_summary = pd.DataFrame([{
    "action_name": "kiosk_strike",
    "total_duration": duration_sec,
    "start_time": start_time,
    "peak_time": peak_time,
    "end_time": end_time,
    "action_duration": action_duration,
    "peak_energy": max_energy
}])
summary_csv = OUTPUT_DIR / "action_segmentation_summary.csv"
df_summary.to_csv(summary_csv, index=False)

print(f"\n🎉 렌더링 완료!")
print(f"📼 구간 탐지 비디오: {OUTPUT_VIDEO_PATH}")
print(f"📊 구간 요약 리포트: {summary_csv}")
