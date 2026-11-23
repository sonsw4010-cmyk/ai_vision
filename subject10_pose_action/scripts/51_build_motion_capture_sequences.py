import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/user1/ai_vision/subject10_pose_action")
sys.path.insert(0, str(ROOT))

from src.sequence_utils import (
    build_frame_feature_matrix,
    flatten_sequence,
    resample_sequence,
    sequence_feature_names,
)

CONFIG_PATH = ROOT / "configs" / "motion_capture.json"
SPLIT_DIR = ROOT / "data" / "motion_capture" / "splits"
OUTPUT_DIR = ROOT / "data" / "motion_capture_sequences"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

target_steps = int(config.get("sequence_steps", 30))
min_valid_frames = int(config.get("min_valid_frames", 10))
feature_names = sequence_feature_names(target_steps)

print(f"Target steps: {target_steps}, Feature count: {len(feature_names)}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for split_name in ["train", "val", "test"]:
    split_file = SPLIT_DIR / f"{split_name}.csv"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    
    manifest = pd.read_csv(split_file)
    print(f"\nProcessing {split_name} ({len(manifest)} clips)...")
    
    rows = []
    skipped = []
    
    for _, item in manifest.iterrows():
        clip_id = item["clip_id"]
        keypoint_path = ROOT / str(item["keypoint_csv"])
        
        if not keypoint_path.exists():
            skipped.append({"clip_id": clip_id, "reason": "KEYPOINT_CSV_MISSING"})
            continue
            
        frame_df = pd.read_csv(keypoint_path)
        frame_matrix, frame_ids = build_frame_feature_matrix(frame_df)
        
        if len(frame_matrix) < min_valid_frames:
            skipped.append({"clip_id": clip_id, "reason": f"TOO_FEW_VALID_FRAMES ({len(frame_matrix)} < {min_valid_frames})"})
            continue
            
        sequence = resample_sequence(frame_matrix, frame_ids, target_steps=target_steps)
        flattened = flatten_sequence(sequence)
        
        if len(flattened) != target_steps * 38:
            raise RuntimeError(f"Expected {target_steps * 38} features, got {len(flattened)}")
            
        row = {
            "clip_id": clip_id,
            "subject": item["subject"],
            "action_id": item["action_id"],
            "action_label": item["action_label"],
            "take": item["take"],
            "valid_frames": len(frame_matrix),
            "first_valid_frame": int(frame_ids[0]),
            "last_valid_frame": int(frame_ids[-1]),
        }
        
        for name, value in zip(feature_names, flattened):
            row[name] = float(value)
            
        rows.append(row)
        
    print(f"[{split_name.upper()}] Successfully converted: {len(rows)} clips, Skipped: {len(skipped)}")
    
    if skipped:
        skip_path = OUTPUT_DIR / f"{split_name}_skipped.csv"
        pd.DataFrame(skipped).to_csv(skip_path, index=False, encoding="utf-8-sig")
        print(f"Skipped info saved: {skip_path}")
        
    result_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / f"{split_name}.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path} (shape: {result_df.shape})")
    
    # Check classes
    classes = sorted(result_df["action_id"].unique())
    print(f"Classes in {split_name}: {len(classes)} classes")
    assert len(classes) == 17, f"Expected 17 classes in {split_name}, but found {len(classes)}"

print("\n[SUCCESS] Motion Capture Sequence Datasets built successfully!")
