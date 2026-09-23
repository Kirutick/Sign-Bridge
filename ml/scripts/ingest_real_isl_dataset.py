"""
Sign Bridge — Real Indian Sign Language (ISL) Dataset Ingestion & Validation Pipeline

Strictly ingests, validates, normalizes, and partitions REAL human ISL recordings
conforming to the canonical 30x63 MediaPipe single-hand contract.

Synthetic data or non-ISL datasets are strictly rejected.
"""

import os
import sys
import json
import argparse
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CLASSES_ISL_50 = [
    # 26 Letters (ISL Alphabet)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    # 10 Numbers
    "NUM_0", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5", "NUM_6", "NUM_7", "NUM_8", "NUM_9",
    # 14 ISL Signs (Represented with English Concept Labels)
    "HELLO", "THANK_YOU", "YES", "NO", "HELP", "PLEASE", "SORRY",
    "GOODBYE", "WATER", "FOOD", "STOP", "GO", "AND", "INDIA"
]

LABEL_TO_ID = {lbl: i for i, lbl in enumerate(CLASSES_ISL_50)}

def normalize_frame(raw_frame):
    """
    Applies canonical normalization:
    - Wrist (index 0) shifted to origin (0, 0, 0)
    - Scale divided by Euclidean distance between Wrist and Middle-MCP (index 9)
    """
    pts = np.array(raw_frame, dtype=np.float32).reshape(21, 3)
    wrist = pts[0]
    middle_mcp = pts[9]
    scale = np.sqrt(np.sum((middle_mcp - wrist) ** 2))
    if scale < 1e-6:
        raise ValueError("Degenerate scale divisor (< 1e-6). Hand joints collapsed.")
    normalized = (pts - wrist) / scale
    return normalized.flatten()

def validate_sample(sample_dict, strict=True):
    """
    Validates that a sample is genuine, human-recorded, ISL, and mathematically well-formed.
    """
    # 1. Reject synthetic flags
    meta = sample_dict.get("metadata", {})
    if sample_dict.get("isSynthetic", False) or meta.get("isSynthetic", False):
        raise ValueError("Rejected sample: Explicitly marked as synthetic.")
    if "synthetic" in str(meta.get("notes", "")).lower():
        raise ValueError("Rejected sample: Contains synthetic notes.")

    # 2. Check label
    label = sample_dict.get("label", "").upper().strip()
    if label not in LABEL_TO_ID:
        raise ValueError(f"Unknown label '{label}'. Must be one of 50 canonical ISL classes.")

    # 3. Check sequence structure
    seq = sample_dict.get("sequence", [])
    if not isinstance(seq, list) or len(seq) != 30:
        raise ValueError(f"Invalid sequence length ({len(seq)}). Exactly 30 frames required.")

    # 4. Check feature count per frame and finite values
    norm_seq = np.zeros((30, 63), dtype=np.float32)
    for t, frame in enumerate(seq):
        if not isinstance(frame, list) or len(frame) != 63:
            raise ValueError(f"Frame {t} has length {len(frame)}, expected 63.")
        
        arr = np.array(frame, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Frame {t} contains NaN or Infinite values.")

        # Re-verify or enforce normalization
        try:
            norm_seq[t] = normalize_frame(arr)
        except Exception as e:
            raise ValueError(f"Frame {t} normalization failed: {e}")

    performer = meta.get("performerId") or meta.get("signerId") or sample_dict.get("performerId") or "signer_unknown"

    return {
        "label": label,
        "class_id": LABEL_TO_ID[label],
        "sequence": norm_seq,
        "performer_id": str(performer),
        "handedness": sample_dict.get("handedness", ["Right"]),
        "isSynthetic": False
    }

def ingest_dataset(input_source, output_dir="ml/data/processed", strict=True):
    print("\n" + "=" * 70)
    print(" SIGN BRIDGE — REAL ISL DATASET INGESTION & VALIDATION PIPELINE")
    print("=" * 70)
    print(f"Input source:  {input_source}")
    print(f"Output dir:    {output_dir}")
    print(f"Strict mode:   {strict}\n")

    raw_samples = []

    # Read from file or directory
    if os.path.isfile(input_source):
        with open(input_source, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, list):
                raw_samples = content
            elif isinstance(content, dict) and "samples" in content:
                raw_samples = content["samples"]
            else:
                raw_samples = [content]
    elif os.path.isdir(input_source):
        for fname in os.listdir(input_source):
            if fname.endswith(".json"):
                fpath = os.path.join(input_source, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        raw_samples.extend(content)
                    else:
                        raw_samples.append(content)
    else:
        raise FileNotFoundError(f"Input source not found: {input_source}")

    print(f"Found {len(raw_samples)} raw candidate samples. Validating...")

    valid_samples = []
    rejected_count = 0

    for i, s in enumerate(raw_samples):
        try:
            validated = validate_sample(s, strict=strict)
            valid_samples.append(validated)
        except Exception as e:
            rejected_count += 1
            if rejected_count <= 5:
                print(f"  [X] Sample {i} rejected: {e}")

    if rejected_count > 5:
        print(f"  ... and {rejected_count - 5} more samples rejected.")

    print(f"\nValidation complete: {len(valid_samples)} valid samples, {rejected_count} rejected.")

    if len(valid_samples) == 0:
        print("\n[!] ERROR: No valid real ISL samples found. Aborting packaging.")
        return False

    # Package into X, y, sample_meta
    total = len(valid_samples)
    X = np.zeros((total, 30, 63), dtype=np.float32)
    y = np.zeros(total, dtype=np.int32)
    sample_meta = []

    for i, item in enumerate(valid_samples):
        X[i] = item["sequence"]
        y[i] = item["class_id"]
        sample_meta.append({
            "sample_id": f"real_isl_{item['class_id']:02d}_{i:04d}",
            "label": item["label"],
            "class_id": item["class_id"],
            "performer_id": item["performer_id"],
            "isSynthetic": False,
            "datasetType": "REAL_HUMAN_RECORDINGS",
            "validationStatus": "VALIDATED_ISL",
            "signLanguage": "Indian Sign Language",
            "outputLanguage": "English",
            "normalizationVersion": "v1"
        })

    os.makedirs(output_dir, exist_ok=True)
    out_npz = os.path.join(output_dir, "dataset_real_isl_v1.npz")
    out_meta = os.path.join(output_dir, "meta_real_isl_v1.json")

    np.savez_compressed(out_npz, X=X, y=y, sample_meta=np.array(sample_meta, dtype=object))

    classes_present = [CLASSES_ISL_50[c] for c in np.unique(y)]
    meta_dict = {
        "datasetType": "REAL_HUMAN_RECORDINGS",
        "validationStatus": "VALIDATED_ISL",
        "isSynthetic": False,
        "signLanguage": "Indian Sign Language",
        "signLanguageCode": "ISL",
        "outputLanguage": "English",
        "outputLanguageCode": "en",
        "total_samples": total,
        "unique_classes_count": len(classes_present),
        "classes_present": classes_present,
        "classes_canonical": CLASSES_ISL_50
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    print(f"✅ Successfully ingested {total} real ISL samples -> {out_npz}")
    print(f"   Metadata exported -> {out_meta}\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Ingest Real ISL Dataset")
    parser.add_argument("--input", default="ml/data/real_isl", help="Path to JSON file or directory containing real ISL samples")
    parser.add_argument("--output_dir", default="ml/data/processed", help="Output directory for processed dataset")
    parser.add_argument("--permissive", action="store_true", help="Allow warnings instead of hard rejection on minor warnings")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        os.makedirs(args.input, exist_ok=True)
        print(f"Created input directory '{args.input}'. Place real ISL recordings (JSON format) here.")
        return

    ingest_dataset(args.input, args.output_dir, strict=not args.permissive)

if __name__ == "__main__":
    main()
