import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")

def create_deterministic_label_map(
    labels: List[str],
    existing_map_path: str = None
) -> Tuple[Dict[str, int], Dict[int, str]]:
  
  unique_labels = sorted(list(set(labels)))

  if existing_map_path and os.path.exists(existing_map_path):
    print(f"📖 Loading existing label map from: {existing_map_path}")
    with open(existing_map_path, "r", encoding="utf-8") as f:
      existing_data = json.load(f)
      existing_map: Dict[str, int] = existing_data.get("label_to_index", {})

    # Append new labels at the end of existing mapping (§3 policy b)
    label_to_idx = dict(existing_map)
    next_idx = max(existing_map.values()) + 1 if existing_map else 0

    new_labels_added = []
    for lbl in unique_labels:
      if lbl not in label_to_idx:
        label_to_idx[lbl] = next_idx
        next_idx += 1
        new_labels_added.append(lbl)

    if new_labels_added:
      print(f"➕ Appended new labels at end of existing mapping: {new_labels_added}")
  else:
    # Deterministic alphabetical ordering (§3)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}

  idx_to_label = {idx: lbl for lbl, idx in label_to_idx.items()}
  return label_to_idx, idx_to_label

def check_class_imbalance(labels: List[str]):
  counts = {}
  for lbl in labels:
    counts[lbl] = counts.get(lbl, 0) + 1

  sorted_counts = sorted(counts.values())
  median_count = np.median(sorted_counts)

  print("\n⚖️ Class Imbalance Audit:")
  print(f" - Min count: {min(sorted_counts)} | Median count: {median_count} | Max count: {max(sorted_counts)}")

  imbalanced_classes = [lbl for lbl, count in counts.items() if count < (median_count / 2.0)]
  if imbalanced_classes:
    print(f"⚠️ WARNING: Severe class imbalance detected! The following classes have < 50% of median samples ({median_count/2.0:.1f}):")
    for cls in imbalanced_classes:
      print(f"   - Class '{cls}': {counts[cls]} samples")
  else:
    print(" ✅ Class distribution is balanced.")

def encode_dataset_labels(
    npz_path: str,
    output_dir: str = None,
    existing_map_path: str = None
) -> Tuple[np.ndarray, Dict[str, int], str]:
  
  if not os.path.exists(npz_path):
    print(f"❌ ERROR: Processed dataset .npz not found: {npz_path}", file=sys.stderr)
    sys.exit(1)

  data = np.load(npz_path)
  raw_labels = data["labels"].tolist()

  label_to_idx, idx_to_label = create_deterministic_label_map(raw_labels, existing_map_path)
  check_class_imbalance(raw_labels)

  y = np.array([label_to_idx[lbl] for lbl in raw_labels], dtype=np.int32)

  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    map_json_path = os.path.join(output_dir, "label_map.json")
    map_data = {
      "label_to_index": label_to_idx,
      "index_to_label": idx_to_label,
      "num_classes": len(label_to_idx)
    }
    with open(map_json_path, "w", encoding="utf-8") as f:
      json.dump(map_data, f, indent=2)
    print(f"✅ Saved label map to: {map_json_path}")
  else:
    map_json_path = ""

  return y, label_to_idx, map_json_path

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Encode dataset labels deterministically")
  parser.add_argument("--npz", required=True, help="Path to processed .npz dataset file")
  parser.add_argument("--output-dir", help="Output directory to save label_map.json")
  parser.add_argument("--existing-map", help="Path to existing label_map.json")
  args = parser.parse_args()

  encode_dataset_labels(args.npz, args.output_dir, args.existing_map)
