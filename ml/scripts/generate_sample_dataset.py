import os
import sys
import json
import uuid
import argparse
import numpy as np
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")

CLASSES = ["HELLO", "THANK_YOU", "YES", "NO"]
PERFORMERS = ["performer_1", "performer_2", "performer_3"]
SAMPLES_PER_CLASS = 15
SEQUENCE_LENGTH = 30
FEATURE_COUNT = 63

def generate_mock_landmark_sequence(class_idx: int, frame_idx: int) -> list:
  landmarks = []
  base_freq = (class_idx + 1) * 0.1
  
  for lm in range(21):
    if lm == 0:
      x, y, z = 0.0, 0.0, 0.0
    elif lm == 9:
      x, y, z = 0.0, 0.2 + np.sin(frame_idx * base_freq) * 0.01, 0.0
    else:
      x = (lm % 5) * 0.05 + np.cos(frame_idx * base_freq + lm) * 0.02
      y = (lm // 5) * 0.08 + np.sin(frame_idx * base_freq + lm) * 0.02
      z = (lm * 0.01) + np.cos(frame_idx * 0.05) * 0.01

    landmarks.extend([float(x), float(y), float(z)])

  return landmarks

def generate_dataset_export(output_path: str):
  samples = []

  for class_idx, label in enumerate(CLASSES):
    for s_idx in range(SAMPLES_PER_CLASS):
      performer = PERFORMERS[s_idx % len(PERFORMERS)]
      sequence = []

      for f_idx in range(SEQUENCE_LENGTH):
        seq_frame = generate_mock_landmark_sequence(class_idx, f_idx)
        sequence.append(seq_frame)

      sample = {
        "id": str(uuid.uuid4()),
        "label": label,
        "sequence": sequence,
        "sequenceLength": SEQUENCE_LENGTH,
        "featureCount": FEATURE_COUNT,
        "createdAt": datetime.now().isoformat(),
        "handedness": ["Right"],
        "metadata": {
          "performerId": performer,
          "notes": "Synthetic mock test sequence",
          "fps": 30,
          "appVersion": "0.3.0"
        }
      }
      samples.append(sample)

  export_data = {
    "version": 1,
    "featureCount": FEATURE_COUNT,
    "sequenceLength": SEQUENCE_LENGTH,
    "exportedAt": datetime.now().isoformat(),
    "samples": samples
  }

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(export_data, f, indent=2)

  print(f"[OK] Synthetic Phase 3 dataset export created: {output_path}")
  print(f" - Total Samples: {len(samples)} ({len(CLASSES)} classes x {SAMPLES_PER_CLASS} samples)")
  print(f" - Performers: {PERFORMERS}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate synthetic Phase 3 dataset export for testing")
  parser.add_argument("--output", default="ml/data/raw/dataset_mock.json", help="Output JSON file path")
  args = parser.parse_args()

  generate_dataset_export(args.output)
