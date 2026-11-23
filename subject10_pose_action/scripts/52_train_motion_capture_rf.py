import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path("/home/user1/ai_vision/subject10_pose_action")

CONFIG_PATH = ROOT / "configs" / "motion_capture.json"
SEQ_DIR = ROOT / "data" / "motion_capture_sequences"
ARTIFACT_DIR = ROOT / "artifacts" / "motion_capture"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# Load sequences
train_df = pd.read_csv(SEQ_DIR / "train.csv")
val_df = pd.read_csv(SEQ_DIR / "val.csv")
test_df = pd.read_csv(SEQ_DIR / "test.csv")

META_COLUMNS = {
    "clip_id",
    "subject",
    "action_id",
    "action_label",
    "take",
    "valid_frames",
    "first_valid_frame",
    "last_valid_frame",
}

feature_cols = [c for c in train_df.columns if c not in META_COLUMNS]
print(f"Feature columns count: {len(feature_cols)}")

X_train = train_df[feature_cols].to_numpy()
y_train = train_df["action_label"].to_numpy()

X_val = val_df[feature_cols].to_numpy()
y_val = val_df["action_label"].to_numpy()

X_test = test_df[feature_cols].to_numpy()
y_test = test_df["action_label"].to_numpy()

classes = sorted(list(set(y_train)))
print(f"Training on {len(X_train)} clips across {len(classes)} action classes.")
print(f"Subjects in train: {list(train_df['subject'].unique())}")
print(f"Subjects in val:   {list(val_df['subject'].unique())}")
print(f"Subjects in test:  {list(test_df['subject'].unique())}")

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=int(config.get("n_estimators", 300)),
    max_depth=config.get("max_depth", None),
    random_state=int(config.get("random_state", 42)),
    class_weight="balanced",
    n_jobs=-1,
)

print("\nFitting RandomForestClassifier...")
model.fit(X_train, y_train)
print("[OK] Model training complete.")

# Evaluate Train
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

# Evaluate Val
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred, average="macro")
val_report = classification_report(y_val, val_pred, output_dict=True)

# Evaluate Test
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average="macro")
test_report = classification_report(y_test, test_pred, output_dict=True)

print(f"\n================ PERFORMANCE METRICS ================")
print(f"Train Accuracy:      {train_acc * 100:.2f}% ({sum(train_pred == y_train)}/{len(y_train)})")
print(f"Validation Accuracy: {val_acc * 100:.2f}% ({sum(val_pred == y_val)}/{len(y_val)}) | Macro F1: {val_f1:.4f}")
print(f"Test Accuracy:       {test_acc * 100:.2f}% ({sum(test_pred == y_test)}/{len(y_test)}) | Macro F1: {test_f1:.4f}")
print(f"=====================================================\n")

# Detailed Test Report
print("--- TEST CLASSIFICATION REPORT ---")
print(classification_report(y_test, test_pred))

# Detailed Val Report
print("--- VALIDATION CLASSIFICATION REPORT ---")
print(classification_report(y_val, val_pred))

# Save artifacts
model_path = ARTIFACT_DIR / "rf_action_motion_capture.joblib"
joblib.dump(model, model_path)
print(f"[SAVED] Model: {model_path}")

feature_list_path = ARTIFACT_DIR / "rf_feature_columns.json"
with open(feature_list_path, "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2, ensure_ascii=False)
print(f"[SAVED] Feature Columns: {feature_list_path}")

train_info = {
    "train_clips": len(X_train),
    "val_clips": len(X_val),
    "test_clips": len(X_test),
    "subjects": {
        "train": list(train_df["subject"].unique()),
        "val": list(val_df["subject"].unique()),
        "test": list(test_df["subject"].unique())
    },
    "classes": classes,
    "feature_count": len(feature_cols),
    "sequence_steps": int(config.get("sequence_steps", 30)),
    "min_valid_frames": int(config.get("min_valid_frames", 10)),
    "n_estimators": int(config.get("n_estimators", 300)),
    "max_depth": config.get("max_depth", None),
    "random_state": int(config.get("random_state", 42)),
    "metrics": {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_f1),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1)
    }
}

train_info_path = ARTIFACT_DIR / "rf_training_info.json"
with open(train_info_path, "w", encoding="utf-8") as f:
    json.dump(train_info, f, indent=2, ensure_ascii=False)
print(f"[SAVED] Training Info: {train_info_path}")

summary = {
    "train_accuracy": float(train_acc),
    "val_accuracy": float(val_acc),
    "val_macro_f1": float(val_f1),
    "test_accuracy": float(test_acc),
    "test_macro_f1": float(test_f1),
    "val_report": val_report,
    "test_report": test_report
}
summary_path = ARTIFACT_DIR / "evaluation_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[SAVED] Evaluation Summary: {summary_path}")

print("\n[ALL COMPLETE] Motion Capture Model Trained and Evaluated!")
