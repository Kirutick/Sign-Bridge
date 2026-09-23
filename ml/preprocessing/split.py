import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import GroupShuffleSplit

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

def perform_group_aware_split(
    npz_path: str,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = 42
) -> Dict[str, Any]:
  
  if not os.path.exists(npz_path):
    print(f"❌ ERROR: Processed dataset file not found: {npz_path}", file=sys.stderr)
    sys.exit(1)

  data = np.load(npz_path, allow_pickle=True)
  ids = data["ids"].tolist()
  labels = data["labels"].tolist()
  performers = data["performers"].tolist() if "performers" in data else ["default_performer"] * len(ids)

  # Also load source_video_ids if available (added by extract_landmarks_from_media.py)
  source_video_ids = data["source_video_ids"].tolist() if "source_video_ids" in data else [None] * len(ids)

  total_samples = len(ids)
  unique_performers = sorted(list(set(performers)))
  unique_source_videos = sorted(list(set(v for v in source_video_ids if v)))

  has_group_metadata = len(unique_performers) > 1 and not all(p == "default_performer" for p in performers)
  has_video_metadata = len(unique_source_videos) > 1

  print("\n✂️ DATASET SPLITTING ANALYSIS:")
  print(f" - Total Samples           : {total_samples}")
  print(f" - Unique Performers       : {len(unique_performers)} ({unique_performers[:5]})")
  print(f" - Unique Source Videos    : {len(unique_source_videos)}");

  train_ids: List[str] = []
  val_ids: List[str] = []
  test_ids: List[str] = []
  leakage_guard_applied = False
  grouping_method = "none"

  def _split_by_groups(groups: List[str], all_group_labels: List[str], num_groups: int, label: str):
    """Generic group-based split returning (train_g, val_g, test_g)."""
    rng = np.random.RandomState(random_seed)
    shuffled = list(set(g for g in groups if g))
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n >= 3:
      return shuffled[2:], [shuffled[0]], [shuffled[1]]
    elif n == 2:
      return [shuffled[1]], [shuffled[0]], [shuffled[0]]
    else:
      return shuffled, shuffled, shuffled

  if has_video_metadata:
    print("🔒 Source-Video Leakage Guard: ACTIVE (splitting strictly by source video)")
    leakage_guard_applied = True
    grouping_method = "source_video"
    train_g, val_g, test_g = _split_by_groups(unique_source_videos, source_video_ids, len(unique_source_videos), "source video")
    train_g_set = set(train_g)
    val_g_set = set(val_g)
    test_g_set = set(test_g)
    print(f" - Video Group Allocations: Train={train_g}, Val={val_g}, Test={test_g}")
    for i, s_id in enumerate(ids):
      sv = source_video_ids[i]
      if sv in train_g_set:
        train_ids.append(s_id)
      elif sv in val_g_set:
        val_ids.append(s_id)
      elif sv in test_g_set:
        test_ids.append(s_id)
      else:
        train_ids.append(s_id)  # Unclassified → train

  elif has_group_metadata:
    print("🔒 Group-Aware Leakage Guard: ACTIVE (Splitting strictly by performer groups)")
    leakage_guard_applied = True
    grouping_method = "performer"

    rng = np.random.RandomState(random_seed)
    shuffled_performers = list(unique_performers)
    rng.shuffle(shuffled_performers)

    num_groups = len(shuffled_performers)
    if num_groups >= 3:
      # Allocate at least 1 performer group to val and test
      val_groups = [shuffled_performers[0]]
      test_groups = [shuffled_performers[1]]
      train_groups = shuffled_performers[2:]
    elif num_groups == 2:
      val_groups = [shuffled_performers[0]]
      test_groups = [shuffled_performers[0]] # Shared val/test group if only 2
      train_groups = [shuffled_performers[1]]
    else:
      train_groups = shuffled_performers
      val_groups = shuffled_performers
      test_groups = shuffled_performers

    print(f" - Group Allocations: Train={train_groups}, Val={val_groups}, Test={test_groups}")

    for i in range(total_samples):
      p = performers[i]
      s_id = ids[i]
      if p in train_groups:
        train_ids.append(s_id)
      elif p in val_groups:
        val_ids.append(s_id)
      elif p in test_groups:
        test_ids.append(s_id)

  else:
    print("⚠️ WARNING: No multi-performer metadata detected! Falling back to random sample split.")
    print("⚠️ NOTE: Test metrics MUST be flagged as potentially optimistic due to near-duplicate sequence leakage risks across splits.")
    
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(total_samples)

    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_ids = [ids[i] for i in shuffled_indices[:train_end]]
    val_ids = [ids[i] for i in shuffled_indices[train_end:val_end]]
    test_ids = [ids[i] for i in shuffled_indices[val_end:]]

  # Fallback if any split ends up empty
  if not val_ids:
    val_ids = train_ids[-max(1, len(train_ids)//5):]
  if not test_ids:
    test_ids = train_ids[-max(1, len(train_ids)//5):]

  # Compute per-split label breakdown
  def get_split_label_counts(split_ids_set: set) -> Dict[str, int]:
    counts = {}
    for i, s_id in enumerate(ids):
      if s_id in split_ids_set:
        lbl = labels[i]
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts

  train_counts = get_split_label_counts(set(train_ids))
  val_counts = get_split_label_counts(set(val_ids))
  test_counts = get_split_label_counts(set(test_ids))

  print(f"\nSplit Distribution:")
  print(f" - Train Split : {len(train_ids):4d} samples ({len(train_ids)/total_samples*100:.1f}%)")
  print(f" - Val Split   : {len(val_ids):4d} samples ({len(val_ids)/total_samples*100:.1f}%)")
  print(f" - Test Split  : {len(test_ids):4d} samples ({len(test_ids)/total_samples*100:.1f}%)\n")

  split_manifest = {
    "npz_path": npz_path,
    "total_samples": total_samples,
    "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
    "leakage_guard_applied": leakage_guard_applied,
    "unique_performers": list(unique_performers),
    "random_seed": random_seed,
    "train_ids": train_ids,
    "val_ids": val_ids,
    "test_ids": test_ids,
    "split_label_counts": {
      "train": train_counts,
      "val": val_counts,
      "test": test_counts,
    },
  }

  output_split_path = npz_path.replace(".npz", "_split.json")
  with open(output_split_path, "w", encoding="utf-8") as f:
    json.dump(split_manifest, f, indent=2)

  print(f"✅ Split manifest saved to: {output_split_path}")
  return split_manifest

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Build performer group-aware train/val/test split")
  parser.add_argument("--input", required=True, help="Path to processed .npz dataset")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  args = parser.parse_args()

  perform_group_aware_split(args.input, random_seed=args.seed)
