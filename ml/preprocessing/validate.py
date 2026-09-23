import os
import sys
import json
import hashlib
import argparse
import math
import numpy as np
from typing import Dict, List, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")

SUPPORTED_SCHEMA_VERSIONS = [1]
DEFAULT_FEATURE_COUNT = 63
DEFAULT_SEQUENCE_LENGTH = 30
MIN_SAMPLES_PER_LABEL = 5

def compute_dataset_hash(raw_bytes: bytes, feature_count: int, sequence_length: int) -> str:
  hasher = hashlib.sha256()
  hasher.update(raw_bytes)
  hasher.update(f"{feature_count}_{sequence_length}".encode("utf-8"))
  return hasher.hexdigest()[:16]

def validate_dataset_file(
    file_path: str,
    expected_feature_count: int = DEFAULT_FEATURE_COUNT,
    expected_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    min_samples_per_label: int = MIN_SAMPLES_PER_LABEL,
) -> Tuple[np.ndarray, List[str], List[str], List[str], Dict[str, Any], str]:
  
  if not os.path.exists(file_path):
    print(f"❌ ERROR: Raw dataset file does not exist: {file_path}", file=sys.stderr)
    sys.exit(1)

  with open(file_path, "rb") as f:
    raw_bytes = f.read()

  try:
    data = json.loads(raw_bytes.decode("utf-8"))
  except Exception as e:
    print(f"❌ ERROR: Failed to parse JSON file {file_path}: {e}", file=sys.stderr)
    sys.exit(1)

  # Check top-level schema fields (§2a, §2b, §2c)
  version = data.get("version")
  if version not in SUPPORTED_SCHEMA_VERSIONS:
    print(
      f"❌ ERROR: Unsupported schema version '{version}'. Importer supports version(s): {SUPPORTED_SCHEMA_VERSIONS}",
      file=sys.stderr,
    )
    sys.exit(1)

  feature_count = data.get("featureCount", DEFAULT_FEATURE_COUNT)
  sequence_length = data.get("sequenceLength", DEFAULT_SEQUENCE_LENGTH)

  if feature_count != expected_feature_count:
    print(
      f"❌ ERROR: Feature count mismatch! Dataset has {feature_count}, expected {expected_feature_count}.",
      file=sys.stderr,
    )
    sys.exit(1)

  if sequence_length != expected_sequence_length:
    print(
      f"❌ ERROR: Sequence length mismatch! Dataset has {sequence_length}, expected {expected_sequence_length}.",
      file=sys.stderr,
    )
    sys.exit(1)

  raw_samples = data.get("samples", [])
  if not isinstance(raw_samples, list):
    print("❌ ERROR: 'samples' field is missing or not a JSON array.", file=sys.stderr)
    sys.exit(1)

  total_read = len(raw_samples)
  rejected_list: List[Dict[str, Any]] = []

  valid_sequences: List[List[List[float]]] = []
  valid_labels: List[str] = []
  valid_ids: List[str] = []
  valid_performers: List[str] = []
  seen_ids = set()

  # Per-sample validation (§2d)
  for idx, sample in enumerate(raw_samples):
    sample_id = sample.get("id", f"sample_{idx}")
    sample_label = sample.get("label", "")
    metadata = sample.get("metadata", {})
    performer_id = metadata.get("performerId", "default_performer")

    if not sample_label or not isinstance(sample_label, str) or not sample_label.strip():
      rejected_list.append({"id": sample_id, "reason": "Missing or non-string label", "label": sample_label})
      continue

    clean_label = sample_label.trim() if hasattr(sample_label, "trim") else sample_label.strip()

    if sample_id in seen_ids:
      # Deduplicate ID collision
      sample_id = f"{sample_id}_dedup_{idx}"
    seen_ids.add(sample_id)

    seq = sample.get("sequence")
    if not isinstance(seq, list) or len(seq) != sequence_length:
      rejected_list.append({
        "id": sample_id,
        "reason": f"Sequence row count ({len(seq) if isinstance(seq, list) else 0}) != {sequence_length}",
        "label": clean_label,
      })
      continue

    is_malformed = False
    for r_idx, row in enumerate(seq):
      if not isinstance(row, list) or len(row) != feature_count:
        rejected_list.append({
          "id": sample_id,
          "reason": f"Row {r_idx} feature count ({len(row) if isinstance(row, list) else 0}) != {feature_count}",
          "label": clean_label,
        })
        is_malformed = True
        break

      for c_idx, val in enumerate(row):
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
          rejected_list.append({
            "id": sample_id,
            "reason": f"Row {r_idx} col {c_idx} contains non-finite value: {val}",
            "label": clean_label,
          })
          is_malformed = True
          break
      if is_malformed:
        break

    if not is_malformed:
      valid_sequences.append(seq)
      valid_labels.append(clean_label)
      valid_ids.append(sample_id)
      valid_performers.append(performer_id)

  # Summarize per-label counts
  label_counts: Dict[str, int] = {}
  for lbl in valid_labels:
    label_counts[lbl] = label_counts.get(lbl, 0) + 1

  print("\n================ DATASET VALIDATION REPORT ================")
  print(f"File Path: {file_path}")
  print(f"Total Samples Read: {total_read}")
  print(f"Valid Samples Retained: {len(valid_sequences)}")
  print(f"Rejected Samples Count: {len(rejected_list)}")

  if rejected_list:
    print("\nRejected Samples Summary:")
    for rej in rejected_list[:10]: # Print first 10
      print(f" - ID: {rej['id']} | Label: {rej['label']} | Reason: {rej['reason']}")
    if len(rejected_list) > 10:
      print(f" ... and {len(rejected_list) - 10} more.")

  print("\nPer-Label Sample Counts:")
  for lbl, count in sorted(label_counts.items()):
    status_str = "✅ PASS" if count >= min_samples_per_label else "❌ FAIL (< min threshold)"
    print(f" - {lbl:20s}: {count:4d} samples | {status_str}")
  print("===========================================================\n")

  # Hard failure checks (§2f)
  if len(valid_sequences) == 0:
    print("❌ ERROR: Zero valid samples remain after dataset validation!", file=sys.stderr)
    sys.exit(1)

  for lbl, count in label_counts.items():
    if count < min_samples_per_label:
      print(
        f"❌ ERROR: Label '{lbl}' has only {count} valid sample(s), which is below the minimum threshold ({min_samples_per_label}).",
        file=sys.stderr,
      )
      sys.exit(1)

  # Convert to numpy array of shape (N, 30, 63)
  X = np.array(valid_sequences, dtype=np.float32)
  dataset_hash = compute_dataset_hash(raw_bytes, feature_count, sequence_length)

  # Save cached .npz file (§2g)
  processed_dir = os.path.join("ml", "data", "processed")
  os.makedirs(processed_dir, exist_ok=True)
  output_npz_path = os.path.join(processed_dir, f"{dataset_hash}.npz")

  meta_info = {
    "file_path": file_path,
    "total_read": total_read,
    "valid_count": len(valid_sequences),
    "rejected_count": len(rejected_list),
    "feature_count": feature_count,
    "sequence_length": sequence_length,
    "label_counts": label_counts,
  }

  np.savez_compressed(
    output_npz_path,
    X=X,
    labels=np.array(valid_labels),
    ids=np.array(valid_ids),
    performers=np.array(valid_performers),
    meta=json.dumps(meta_info),
  )

  print(f"✅ Validation successful. Processed dataset saved to: {output_npz_path}")

  # Update DATASET_MANIFEST.md
  manifest_path = os.path.join("ml", "data", "DATASET_MANIFEST.md")
  manifest_entry = f"| {dataset_hash} | {os.path.basename(file_path)} | {len(valid_sequences)} | {len(label_counts)} | {feature_count} | {sequence_length} |\n"
  
  if not os.path.exists(manifest_path):
    with open(manifest_path, "w", encoding="utf-8") as f:
      f.write("# Dataset Processing Manifest\n\n| Dataset Hash | Source File | Valid Samples | Classes Count | Feature Count | Sequence Length |\n|---|---|---|---|---|---|\n")
  
  with open(manifest_path, "a", encoding="utf-8") as f:
    f.write(manifest_entry)

  return X, valid_labels, valid_ids, valid_performers, meta_info, output_npz_path

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Validate Phase 3 JSON Export Dataset")
  parser.add_argument("--input", required=True, help="Path to raw exported JSON file")
  parser.add_argument("--feature-count", type=int, default=DEFAULT_FEATURE_COUNT)
  parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
  parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_LABEL)
  args = parser.parse_args()

  validate_dataset_file(
    args.input,
    expected_feature_count=args.feature_count,
    expected_sequence_length=args.sequence_length,
    min_samples_per_label=args.min_samples,
  )
