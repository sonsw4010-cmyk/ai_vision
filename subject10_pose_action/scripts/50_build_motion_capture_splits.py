import json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/user1/ai_vision/subject10_pose_action")

# 1. Write configs/motion_capture.json
config = {
    "sequence_steps": 30,
    "min_valid_frames": 10,
    "random_state": 42,
    "n_estimators": 300,
    "max_depth": None,
    "train_subjects": ["S01", "S02", "S03"],
    "val_subjects": ["S04"],
    "test_subjects": ["S05"],
    "action_classes_csv": "configs/motion_capture_action_classes.csv"
}

config_path = ROOT / "configs" / "motion_capture.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"[OK] Saved config: {config_path}")

# 2. Read manifests
manifests = []
keypoint_dir = ROOT / "data" / "motion_capture_keypoints"
for s in ["S01", "S02", "S03", "S04", "S05"]:
    m_file = keypoint_dir / f"manifest_{s}.csv"
    if not m_file.exists():
        raise FileNotFoundError(f"Missing {m_file}")
    df_s = pd.read_csv(m_file)
    print(f"Loaded {s}: {len(df_s)} clips")
    manifests.append(df_s)

all_df = pd.concat(manifests, ignore_index=True)
print(f"Total clips combined: {len(all_df)}")

# Check keypoint files existence
missing_files = []
for _, row in all_df.iterrows():
    kp_path = ROOT / str(row["keypoint_csv"])
    if not kp_path.exists():
        missing_files.append(str(kp_path))

if missing_files:
    raise FileNotFoundError(f"Missing keypoint files: {missing_files[:5]} (total {len(missing_files)})")
print("[OK] All 170 keypoint CSV files verified on disk.")

# 3. Assign splits
split_map = {
    "S01": "train",
    "S02": "train",
    "S03": "train",
    "S04": "val",
    "S05": "test",
}
all_df["split"] = all_df["subject"].map(split_map)

train_df = all_df[all_df["split"] == "train"].copy()
val_df = all_df[all_df["split"] == "val"].copy()
test_df = all_df[all_df["split"] == "test"].copy()

print(f"Train count: {len(train_df)}")
print(f"Val count:   {len(val_df)}")
print(f"Test count:  {len(test_df)}")

# Leakage check
train_sub = set(train_df["subject"].unique())
val_sub = set(val_df["subject"].unique())
test_sub = set(test_df["subject"].unique())
assert len(train_sub.intersection(val_sub)) == 0, "Train and Val subjects overlap!"
assert len(train_sub.intersection(test_sub)) == 0, "Train and Test subjects overlap!"
assert len(val_sub.intersection(test_sub)) == 0, "Val and Test subjects overlap!"
print(f"[OK] No subject leakage. Train: {train_sub}, Val: {val_sub}, Test: {test_sub}")

# Class distribution check
expected_classes = sorted(all_df["action_id"].unique())
print(f"Unique classes count: {len(expected_classes)}")
assert len(expected_classes) == 17, f"Expected 17 classes, got {len(expected_classes)}"

assert set(train_df["action_id"].unique()) == set(expected_classes), "Train missing some classes!"
assert set(val_df["action_id"].unique()) == set(expected_classes), "Val missing some classes!"
assert set(test_df["action_id"].unique()) == set(expected_classes), "Test missing some classes!"
print("[OK] All 17 classes present in all splits (Train, Val, Test).")

# 4. Save splits
split_dir = ROOT / "data" / "motion_capture" / "splits"
split_dir.mkdir(parents=True, exist_ok=True)

all_df.to_csv(split_dir / "all_manifest.csv", index=False, encoding="utf-8-sig")
train_df.to_csv(split_dir / "train.csv", index=False, encoding="utf-8-sig")
val_df.to_csv(split_dir / "val.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(split_dir / "test.csv", index=False, encoding="utf-8-sig")

print(f"[SUCCESS] Split files saved to {split_dir}")
