import os
import sys
import time
import json
import random
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import (
  f1_score, precision_score, recall_score,
  confusion_matrix, classification_report
)
try:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import seaborn as sns
  _MATPLOTLIB_AVAILABLE = True
except ImportError:
  _MATPLOTLIB_AVAILABLE = False

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")

from ml.training.config import TrainingConfig
from ml.training.model import build_lstm_model, save_model_summary
from ml.preprocessing.encode_labels import encode_dataset_labels
from ml.preprocessing.augment import augment_dataset

def set_all_seeds(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"🌱 Fixed random seed set to: {seed}")

def train_model(
    config_path: str = None,
    npz_path: str = None,
    split_path: str = None,
    cli_overrides: dict = None
) -> str:
  
  startTime = time.time()
  
  # Load configuration
  if config_path and os.path.exists(config_path):
    config = TrainingConfig.load_json(config_path)
  else:
    config = TrainingConfig()

  # Apply CLI overrides (§6)
  if cli_overrides:
    for k, v in cli_overrides.items():
      if v is not None and hasattr(config, k):
        setattr(config, k, v)

  set_all_seeds(config.random_seed)

  # Find dataset if not specified
  processed_dir = os.path.join("ml", "data", "processed")
  if not npz_path:
    npz_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith(".npz")]
    if not npz_files:
      print("❌ ERROR: No processed .npz dataset found in ml/data/processed/. Run validate.py first.", file=sys.stderr)
      sys.exit(1)
    npz_path = sorted(npz_files)[-1]

  if not split_path:
    split_path = npz_path.replace(".npz", "_split.json")
    if not os.path.exists(split_path):
      print(f"❌ ERROR: Split manifest not found: {split_path}. Run split.py first.", file=sys.stderr)
      sys.exit(1)

  # Load raw dataset & split indices
  data = np.load(npz_path)
  X = data["X"] # (N, 30, 63)
  raw_labels = data["labels"].tolist()
  ids = data["ids"].tolist()

  with open(split_path, "r", encoding="utf-8") as f:
    split_data = json.load(f)

  train_id_set = set(split_data["train_ids"])
  val_id_set = set(split_data["val_ids"])
  test_id_set = set(split_data["test_ids"])

  # Shape check against config (§6)
  _, seq_len, feat_count = X.shape
  if seq_len != config.sequence_length or feat_count != config.feature_count:
    print(
      f"❌ ERROR: Shape mismatch! Dataset shape is ({seq_len}, {feat_count}), but config specifies ({config.sequence_length}, {config.feature_count}).",
      file=sys.stderr,
    )
    sys.exit(1)

  # Encode labels and save label_map
  y, label_to_idx, _ = encode_dataset_labels(npz_path)
  num_classes = len(label_to_idx)

  # Separate train, val, test splits
  train_mask = [ids[i] in train_id_set for i in range(len(ids))]
  val_mask = [ids[i] in val_id_set for i in range(len(ids))]
  test_mask = [ids[i] in test_id_set for i in range(len(ids))]

  X_train, y_train = X[train_mask], y[train_mask]
  X_val, y_val = X[val_mask], y[val_mask]
  X_test, y_test = X[test_mask], y[test_mask]

  # Apply augmentation ONLY to training set if enabled (§10)
  if config.augmentation_enabled:
    X_train, y_train = augment_dataset(
      X_train, y_train, augmentation_factor=config.augmentation_factor, random_seed=config.random_seed
    )

  print(f"\n🚀 Starting Training Run:")
  print(f" - Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
  print(f" - Classes ({num_classes}): {list(label_to_idx.keys())}")

  # Build Keras LSTM model (§5)
  model = build_lstm_model(num_classes, config)

  # Prepare model version folder (§8)
  timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
  short_hash = os.path.basename(npz_path).replace(".npz", "")[:8]
  model_version = f"v1_{timestamp_str}_{short_hash}"
  
  model_dir = os.path.join("ml", "models", model_version)
  os.makedirs(model_dir, exist_ok=True)

  # Save model summary
  save_model_summary(model, os.path.join(model_dir, "model_summary.txt"))
  model.summary()

  # Checkpoint path — save best val_accuracy model
  ckpt_path = os.path.join(model_dir, "best_checkpoint.keras")

  # Callbacks (§13 spec)
  callbacks = [
    tf.keras.callbacks.EarlyStopping(
      monitor=config.early_stopping_monitor,
      patience=config.early_stopping_patience,
      restore_best_weights=config.restore_best_weights,
      verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
      filepath=ckpt_path,
      monitor="val_accuracy",
      save_best_only=True,
      verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
      monitor="val_loss",
      factor=0.5,
      patience=max(5, config.early_stopping_patience // 3),
      min_lr=1e-6,
      verbose=1,
    ),
  ]

  # Train Keras Model
  history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=config.epochs,
    batch_size=config.batch_size,
    callbacks=callbacks,
    verbose=1,
  )

  wall_clock_sec = round(time.time() - startTime, 2)

  # Save Keras Model Weights
  model_save_path = os.path.join(model_dir, "model.keras")
  model.save(model_save_path)
  print(f"💾 Model saved to: {model_save_path}")

  # Save Artifacts Bundle (§8)
  # 1. label_map.json
  with open(os.path.join(model_dir, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump({"label_to_index": label_to_idx, "index_to_label": {v: k for k, v in label_to_idx.items()}, "num_classes": num_classes}, f, indent=2)

  # 2. config.json
  config.save_json(os.path.join(model_dir, "config.json"))

  # 3. preprocessing_config.json
  preproc_config = {
    "featureCount": config.feature_count,
    "sequenceLength": config.sequence_length,
    "normalization": "wrist_relative_middle_mcp_scale",
    "degenerate_epsilon": 1e-6,
  }
  with open(os.path.join(model_dir, "preprocessing_config.json"), "w", encoding="utf-8") as f:
    json.dump(preproc_config, f, indent=2)

  # 4. training_history.json
  hist_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
  with open(os.path.join(model_dir, "training_history.json"), "w", encoding="utf-8") as f:
    json.dump(hist_dict, f, indent=2)

  # Evaluate on all splits
  train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
  val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
  test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

  # Compute F1, precision, recall on test set (§13)
  test_preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
  test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
  test_macro_precision = float(precision_score(y_test, test_preds, average="macro", zero_division=0))
  test_macro_recall = float(recall_score(y_test, test_preds, average="macro", zero_division=0))
  idx_to_label = {v: k for k, v in label_to_idx.items()}
  test_report = classification_report(
    y_test, test_preds,
    target_names=[idx_to_label.get(i, str(i)) for i in range(num_classes)],
    output_dict=True, zero_division=0
  )

  # Compute val F1
  val_preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
  val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))

  # 90% target check
  target_reached = val_acc >= 0.90 and test_acc >= 0.90
  target_status = "TARGET REACHED ✅" if target_reached else "TARGET NOT REACHED ❌"

  print(f"\n📊 Training Complete in {wall_clock_sec}s:")
  print(f" - Train Acc : {train_acc:.4f}")
  print(f" - Val Acc   : {val_acc:.4f}  | Val Macro F1 : {val_macro_f1:.4f}")
  print(f" - Test Acc  : {test_acc:.4f}  | Test Macro F1: {test_macro_f1:.4f}")
  print(f" - Status    : {target_status}")

  # Save confusion matrix image
  cm = confusion_matrix(y_test, test_preds)
  cm_path = os.path.join(model_dir, "confusion_matrix.png")
  if _MATPLOTLIB_AVAILABLE:
    fig, ax = plt.subplots(figsize=(max(8, num_classes), max(6, num_classes - 2)))
    sns.heatmap(
      cm, annot=(num_classes <= 20), fmt="d", cmap="Blues",
      xticklabels=[idx_to_label.get(i, str(i)) for i in range(num_classes)],
      yticklabels=[idx_to_label.get(i, str(i)) for i in range(num_classes)],
      ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Test Set (Acc={test_acc*100:.1f}%)")
    plt.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"📊 Confusion matrix saved → {cm_path}")

  # Save per-class classification report
  report_path = os.path.join(model_dir, "classification_report.json")
  with open(report_path, "w", encoding="utf-8") as f:
    json.dump(test_report, f, indent=2)
  print(f"📋 Classification report → {report_path}")

  # Append to runs.jsonl experiment tracking log (§9)
  run_record = {
    "run_id": f"run_{model_version}",
    "model_version": model_version,
    "date": datetime.now().isoformat(),
    "dataset_source_files": [npz_path],
    "dataset_hash": short_hash,
    "num_samples_total": len(X),
    "num_samples_per_class": {lbl: raw_labels.count(lbl) for lbl in set(raw_labels)},
    "classes": sorted(list(label_to_idx.keys())),
    "architecture_summary": f"LSTM({config.lstm_units_1})->Dropout({config.dropout_1})->LSTM({config.lstm_units_2})->Dropout({config.dropout_2})->Dense({config.dense_units})->Dense({num_classes})",
    "config": config.to_dict(),
    "epochs_run": len(history.history["loss"]),
    "early_stopped": len(history.history["loss"]) < config.epochs,
    "target_90_reached": target_reached,
    "train_accuracy_final": float(train_acc),
    "val_accuracy_final": float(val_acc),
    "val_macro_f1": val_macro_f1,
    "test_accuracy": float(test_acc),
    "test_macro_f1": test_macro_f1,
    "test_macro_precision": test_macro_precision,
    "test_macro_recall": test_macro_recall,
    "confusion_matrix_path": cm_path if _MATPLOTLIB_AVAILABLE else None,
    "classification_report_path": report_path,
    "best_checkpoint_path": ckpt_path,
    "leakage_guard_applied": split_data.get("leakage_guard_applied", False),
    "grouping_metadata_available": split_data.get("leakage_guard_applied", False),
    "random_seed": config.random_seed,
    "wall_clock_seconds": wall_clock_sec,
    "notes": f"Augmentation {'enabled' if config.augmentation_enabled else 'disabled'}.",
  }

  exp_dir = os.path.join("ml", "experiments")
  os.makedirs(exp_dir, exist_ok=True)
  runs_jsonl_path = os.path.join(exp_dir, "runs.jsonl")

  with open(runs_jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(run_record) + "\n")

  print(f"📝 Appended experiment record to: {runs_jsonl_path}")
  return model_version

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train Sign Bridge Keras LSTM Model")
  parser.add_argument("--config", help="Path to config JSON file")
  parser.add_argument("--npz", help="Path to processed .npz dataset")
  parser.add_argument("--split", help="Path to _split.json file")
  parser.add_argument("--epochs", type=int)
  parser.add_argument("--batch-size", type=int)
  parser.add_argument("--lr", type=float)
  parser.add_argument("--seed", type=int)
  args = parser.parse_args()

  cli_overrides = {
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.lr,
    "random_seed": args.seed,
  }

  train_model(args.config, args.npz, args.split, cli_overrides)
