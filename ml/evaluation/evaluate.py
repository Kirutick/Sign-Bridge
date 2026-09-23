import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
  accuracy_score,
  precision_recall_fscore_support,
  confusion_matrix,
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")

from ml.evaluation.error_analysis import analyze_confusion_pairs
from ml.preprocessing.encode_labels import encode_dataset_labels

def render_confusion_matrix_png(
    cm: np.ndarray,
    class_names: list,
    output_path: str
):
  """
  Renders raw counts and row-normalized percentages confusion matrix heatmap (§7).
  """
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  cm_norm = np.nan_to_num(cm_norm)

  fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
  
  # Format cell annotations with both count and percentage
  annot = np.empty_like(cm, dtype=object)
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

  sns.heatmap(
    cm_norm,
    annot=annot,
    fmt="",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
    cbar=True,
  )
  
  ax.set_title("Sign Bridge LSTM Confusion Matrix (Test Set)", fontsize=12, fontweight="bold", pad=12)
  ax.set_xlabel("Predicted Sign Class", fontsize=10, fontweight="bold")
  ax.set_ylabel("True Sign Class", fontsize=10, fontweight="bold")
  plt.tight_layout()
  
  plt.savefig(output_path)
  plt.close()
  print(f"📊 Confusion matrix heatmap saved to: {output_path}")

def generate_model_card(
    model_dir: str,
    model_version: str,
    eval_metrics: dict,
    class_names: list,
    top_confused: list,
    leakage_guard_applied: bool,
    dataset_info: dict
):
  """
  Generates comprehensive, honest MODEL_CARD.md Markdown document (§8, §12).
  """
  card_path = os.path.join(model_dir, "MODEL_CARD.md")
  
  per_class_table = "| Class Name | Test Samples (n) | Precision | Recall | F1-Score |\n|---|---|---|---|---|\n"
  for cls_name, metrics in eval_metrics["per_class"].items():
    per_class_table += f"| `{cls_name}` | {metrics['sample_count']} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} |\n"

  confused_str = ""
  if top_confused:
    for c in top_confused:
      confused_str += f"- {c['summary']}\n"
  else:
    confused_str = "- No misclassifications observed on test set.\n"

  train_acc = eval_metrics.get("train_accuracy", 0.0)
  val_acc = eval_metrics.get("val_accuracy", 0.0)
  test_acc = eval_metrics["accuracy"]
  acc_gap = train_acc - val_acc

  overfitting_status = "⚠️ Moderate Overfitting Detected (Gap > 10%)" if acc_gap > 0.10 else "✅ Healthy (No Significant Overfitting)"

  card_content = f"""# Model Card: Sign Bridge LSTM Baseline (`{model_version}`)

## 1. Model Overview
- **Model Identifier:** `{model_version}`
- **Architecture:** Keras Sequential 2-Layer LSTM Baseline
- **Input Tensor Shape:** `(sequence_length=30, feature_count=63)`
- **Output Classes ({len(class_names)}):** `{", ".join(class_names)}`
- **Primary Framework:** TensorFlow 2.17+ / Keras

---

## 2. Intended Use & Deployment Compatibility
- **Intended Use:** Internal baseline benchmark for real-time temporal sign language sequence classification.
- **Export Compatibility:** Standardized feature vector shape (`30x63`, wrist-relative normalized). Ready for TF.js / TFLite deployment handoff.
- **Explicit Non-Use:** Not intended for production safety-critical medical/legal translation without multi-performer validation.

---

## 3. Training & Validation Performance
- **Train Accuracy:** `{train_acc:.4f}`
- **Validation Accuracy:** `{val_acc:.4f}`
- **Held-out Test Accuracy:** `{test_acc:.4f}`
- **Test Macro F1:** `{eval_metrics['macro_f1']:.4f}`
- **Overfitting Audit:** {overfitting_status} (Train-Val Gap: `{acc_gap*100:.1f}%`)

---

## 4. Per-Class Performance Breakdown (§7)

{per_class_table}

### Most Confused Label Pairs:
{confused_str}

---

## 5. Honest Limitations & Data Leakage Assessment (§12)

- **Leakage Prevention Status:** {"✅ Group-Aware Performer Split Applied" if leakage_guard_applied else "⚠️ Leakage Unverified — Random Sample Fallback Used"}
- **Performer Generalization Disclaimer:** {"Performer group split enforced. Model evaluated on unseen performers." if leakage_guard_applied else "Dataset lacked performer IDs. Test accuracy may be optimistic due to near-duplicate sequences across splits."}
- **Sample Coverage Notice:** All classes evaluated with n sample counts noted above. Small sample classes (n < 10) should be considered low-confidence benchmarks.
"""

  with open(card_path, "w", encoding="utf-8") as f:
    f.write(card_content)

  print(f"📄 MODEL_CARD.md generated at: {card_path}")

def evaluate_model(model_version: str, npz_path: str = None, split_path: str = None):
  model_dir = os.path.join("ml", "models", model_version)
  
  if not os.path.exists(model_dir):
    print(f"❌ ERROR: Model folder not found: {model_dir}", file=sys.stderr)
    sys.exit(1)

  # Load model & artifacts
  model_path = os.path.join(model_dir, "model.keras")
  model = tf.keras.models.load_model(model_path)

  with open(os.path.join(model_dir, "label_map.json"), "r", encoding="utf-8") as f:
    label_map_data = json.load(f)
    label_to_idx = label_map_data["label_to_index"]
    idx_to_label = {int(k): v for k, v in label_map_data["index_to_label"].items()}

  with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
    config_data = json.load(f)

  with open(os.path.join(model_dir, "training_history.json"), "r", encoding="utf-8") as f:
    history_data = json.load(f)

  # Find dataset and split
  processed_dir = os.path.join("ml", "data", "processed")
  if not npz_path:
    npz_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith(".npz")]
    npz_path = sorted(npz_files)[-1]

  if not split_path:
    split_path = npz_path.replace(".npz", "_split.json")

  data = np.load(npz_path)
  X = data["X"]
  raw_labels = data["labels"].tolist()
  ids = data["ids"].tolist()

  with open(split_path, "r", encoding="utf-8") as f:
    split_data = json.load(f)

  test_id_set = set(split_data["test_ids"])
  test_mask = [ids[i] in test_id_set for i in range(len(ids))]

  y_all, _, _ = encode_dataset_labels(npz_path, existing_map_path=os.path.join(model_dir, "label_map.json"))

  X_test, y_test = X[test_mask], y_all[test_mask]

  print(f"\n🧪 Evaluating Model '{model_version}' on Held-Out Test Set ({len(X_test)} samples)...")

  # Predict probabilities
  y_pred_probs = model.predict(X_test, verbose=0)
  y_pred = np.argmax(y_pred_probs, axis=1)

  # Overall Accuracy
  test_acc = accuracy_score(y_test, y_pred)

  # Per-Class Precision, Recall, F1
  class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
  precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(class_names)), zero_division=0
  )

  macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
  weight_p, weight_r, weight_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

  # Confusion Matrix
  cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
  top_confused = analyze_confusion_pairs(cm, idx_to_label)

  # Format per-class dictionary
  per_class_metrics = {}
  for idx, cls_name in enumerate(class_names):
    per_class_metrics[cls_name] = {
      "sample_count": int(support[idx]),
      "precision": round(float(precision[idx]), 4),
      "recall": round(float(recall[idx]), 4),
      "f1_score": round(float(f1[idx]), 4),
    }

  eval_report = {
    "model_version": model_version,
    "dataset_hash": split_data.get("npz_path", ""),
    "test_sample_count": len(X_test),
    "accuracy": round(float(test_acc), 4),
    "macro_precision": round(float(macro_p), 4),
    "macro_recall": round(float(macro_r), 4),
    "macro_f1": round(float(macro_f1), 4),
    "weighted_f1": round(float(weight_f1), 4),
    "train_accuracy": round(history_data["accuracy"][-1], 4),
    "val_accuracy": round(history_data["val_accuracy"][-1], 4),
    "per_class": per_class_metrics,
    "top_confused_pairs": top_confused,
    "confusion_matrix_raw": cm.tolist(),
  }

  # Save evaluation_report.json (§7)
  eval_json_path = os.path.join(model_dir, "evaluation_report.json")
  with open(eval_json_path, "w", encoding="utf-8") as f:
    json.dump(eval_report, f, indent=2)

  # Render confusion matrix heatmap PNG (§7)
  cm_png_path = os.path.join(model_dir, "confusion_matrix.png")
  render_confusion_matrix_png(cm, class_names, cm_png_path)

  # Generate MODEL_CARD.md (§8, §12)
  generate_model_card(
    model_dir,
    model_version,
    eval_report,
    class_names,
    top_confused,
    split_data.get("leakage_guard_applied", False),
    split_data,
  )

  print("\n================ EVALUATION SUMMARY ================")
  print(f"Model Version: {model_version}")
  print(f"Test Accuracy: {test_acc*100:.2f}% | Test Macro F1: {macro_f1:.4f}")
  print("Per-Class Breakdown:")
  for cls_name, m in per_class_metrics.items():
    print(f" - {cls_name:15s}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f} (n={m['sample_count']})")
  print("===================================================\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate Sign Bridge Trained Model")
  parser.add_argument("--model-version", required=True, help="Model version folder name in ml/models/")
  parser.add_argument("--npz", help="Path to processed .npz dataset")
  parser.add_argument("--split", help="Path to _split.json file")
  args = parser.parse_args()

  evaluate_model(args.model_version, args.npz, args.split)
