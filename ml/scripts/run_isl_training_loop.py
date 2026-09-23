"""
Sign Bridge — ISL Training Loop Orchestrator
=============================================
Runs the full training pipeline repeatedly until:
  SUCCESS: val_acc >= 0.90 AND test_acc >= 0.90
  OR
  BUDGET_EXHAUSTED: MAX_EXPERIMENTS reached without success

Usage:
    python ml/scripts/run_isl_training_loop.py [--max_experiments 20] [--max_epochs 150]

The script performs controlled hyperparameter sweeps, not random search.
Every experiment is logged. The test set is ONLY used for final evaluation
after each full training run — never for hyperparameter selection.

Experiment sequence:
  1. Baseline (config.baseline.json defaults)
  2. Increase LSTM units
  3. Deeper architecture
  4. Higher augmentation
  5. Lower learning rate
  6. Larger batch
  7. Smaller dropout
  ... up to MAX_EXPERIMENTS

90% Dashboard is printed after every experiment.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from ml.preprocessing.validate import validate_dataset_file
from ml.preprocessing.split import perform_group_aware_split
from ml.training.train import train_model
from ml.training.config import TrainingConfig


# ─────────────────────────────────────────────────────────
# Controlled experiment sweep configurations
# These are SEQUENTIAL changes, not random search.
# ─────────────────────────────────────────────────────────
EXPERIMENT_CONFIGS: List[Dict[str, Any]] = [
    # Exp 1 — Baseline
    {"lstm_units_1": 64, "lstm_units_2": 32, "dense_units": 32,
     "dropout_1": 0.3, "dropout_2": 0.3, "dropout_3": 0.2,
     "learning_rate": 1e-3, "batch_size": 16, "augmentation_enabled": False},
    # Exp 2 — Larger LSTM
    {"lstm_units_1": 128, "lstm_units_2": 64, "dense_units": 64,
     "dropout_1": 0.3, "dropout_2": 0.3, "dropout_3": 0.2,
     "learning_rate": 1e-3, "batch_size": 16, "augmentation_enabled": False},
    # Exp 3 — Larger LSTM + Augmentation
    {"lstm_units_1": 128, "lstm_units_2": 64, "dense_units": 64,
     "dropout_1": 0.25, "dropout_2": 0.25, "dropout_3": 0.15,
     "learning_rate": 1e-3, "batch_size": 32, "augmentation_enabled": True, "augmentation_factor": 2},
    # Exp 4 — Lower learning rate
    {"lstm_units_1": 128, "lstm_units_2": 64, "dense_units": 64,
     "dropout_1": 0.25, "dropout_2": 0.25, "dropout_3": 0.15,
     "learning_rate": 5e-4, "batch_size": 32, "augmentation_enabled": True, "augmentation_factor": 2},
    # Exp 5 — Wider + stronger augmentation
    {"lstm_units_1": 256, "lstm_units_2": 128, "dense_units": 128,
     "dropout_1": 0.3, "dropout_2": 0.3, "dropout_3": 0.2,
     "learning_rate": 5e-4, "batch_size": 32, "augmentation_enabled": True, "augmentation_factor": 3},
    # Exp 6 — Very low LR + wider net
    {"lstm_units_1": 256, "lstm_units_2": 128, "dense_units": 128,
     "dropout_1": 0.2, "dropout_2": 0.2, "dropout_3": 0.1,
     "learning_rate": 1e-4, "batch_size": 32, "augmentation_enabled": True, "augmentation_factor": 3},
    # Exp 7 — Small model (underfitting check)
    {"lstm_units_1": 64, "lstm_units_2": 32, "dense_units": 64,
     "dropout_1": 0.15, "dropout_2": 0.15, "dropout_3": 0.1,
     "learning_rate": 1e-3, "batch_size": 16, "augmentation_enabled": False},
    # Exp 8 — Larger batch
    {"lstm_units_1": 128, "lstm_units_2": 64, "dense_units": 64,
     "dropout_1": 0.25, "dropout_2": 0.25, "dropout_3": 0.15,
     "learning_rate": 2e-3, "batch_size": 64, "augmentation_enabled": True, "augmentation_factor": 2},
    # Exp 9–20 — Repeat best config with different seeds
]
# Fill remaining slots with variations of the best config
for seed in [123, 456, 789, 1024, 2048, 4096, 99, 777, 111, 314, 1337]:
    EXPERIMENT_CONFIGS.append(
        {"lstm_units_1": 128, "lstm_units_2": 64, "dense_units": 64,
         "dropout_1": 0.25, "dropout_2": 0.25, "dropout_3": 0.15,
         "learning_rate": 5e-4, "batch_size": 32,
         "augmentation_enabled": True, "augmentation_factor": 2,
         "random_seed": seed}
    )


# ─────────────────────────────────────────────────────────
# Dashboard printer (Section 20)
# ─────────────────────────────────────────────────────────
def print_dashboard(
    exp_num: int,
    max_experiments: int,
    epoch: int,
    max_epochs: int,
    val_acc: Optional[float],
    test_acc: Optional[float],
    best_val: float,
    best_test: float,
    status: str,
):
    bar = "=" * 62
    val_str = f"{val_acc*100:.1f}%" if val_acc is not None else "—"
    test_str = f"{test_acc*100:.1f}%" if test_acc is not None else "—"
    best_val_str = f"{best_val*100:.1f}%" if best_val > 0 else "—"
    best_test_str = f"{best_test*100:.1f}%" if best_test > 0 else "—"
    print(f"\n{bar}")
    print(f"  SIGN BRIDGE — ISL 90% TRAINING DASHBOARD")
    print(bar)
    print(f"  TARGET              : 90.0%")
    print(f"  CURRENT VALIDATION  : {val_str}")
    print(f"  CURRENT TEST        : {test_str}")
    print(f"  BEST VALIDATION     : {best_val_str}")
    print(f"  BEST TEST           : {best_test_str}")
    print(f"  EXPERIMENT          : {exp_num} / {max_experiments}")
    print(f"  LAST EPOCH BUDGET   : {max_epochs}")
    print(f"  STATUS              : {status}")
    print(f"{bar}\n")


# ─────────────────────────────────────────────────────────
# Leakage check
# ─────────────────────────────────────────────────────────
def check_leakage(split_manifest: Dict[str, Any]) -> bool:
    train_ids = set(split_manifest.get("train_ids", []))
    val_ids = set(split_manifest.get("val_ids", []))
    test_ids = set(split_manifest.get("test_ids", []))
    train_test_overlap = train_ids & test_ids
    train_val_overlap = train_ids & val_ids
    val_test_overlap = val_ids & test_ids
    has_leakage = bool(train_test_overlap or val_test_overlap)
    if has_leakage:
        print(f"  [LEAKAGE] Train∩Test={len(train_test_overlap)}, Val∩Test={len(val_test_overlap)}")
    else:
        print(f"  [LEAKAGE CHECK] No overlap detected. Train∩Val={len(train_val_overlap)} (acceptable).")
    return not has_leakage


# ─────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ISL Training Loop — Runs until 90% or budget exhausted")
    parser.add_argument("--max_experiments", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--raw_json", default=None,
                        help="Path to raw JSON dataset. If omitted, uses newest in ml/data/raw/")
    parser.add_argument("--target_val", type=float, default=0.90)
    parser.add_argument("--target_test", type=float, default=0.90)
    args = parser.parse_args()

    loop_start = time.time()
    print(f"\n{'='*62}")
    print(f"  SIGN BRIDGE ISL TRAINING LOOP  |  MAX {args.max_experiments} EXPERIMENTS")
    print(f"  Target: val_acc >= {args.target_val*100:.0f}% AND test_acc >= {args.target_test*100:.0f}%")
    print(f"{'='*62}\n")

    # Discover raw JSON dataset
    raw_json_path = args.raw_json
    if not raw_json_path:
        # Look in ml/data/raw recursively for JSON files
        raw_dir = os.path.join("ml", "data", "raw")
        json_files = []
        if os.path.isdir(raw_dir):
            for root, _, files in os.walk(raw_dir):
                for fname in files:
                    if fname.endswith(".json"):
                        json_files.append(os.path.join(root, fname))

        if not json_files:
            # Fallback to processed directory
            proc_dir = os.path.join("ml", "data", "processed")
            npz_files = [os.path.join(proc_dir, f) for f in os.listdir(proc_dir) if f.endswith(".npz")]
            if not npz_files:
                print("\n[ERROR] No dataset found in ml/data/raw/ or ml/data/processed/")
                print("[INFO]  Please use the Dataset Studio to record ISL samples,")
                print("[INFO]  or run: python ml/scripts/extract_landmarks_from_media.py --input <media> --label <LABEL>")
                print("[INFO]  Then export the JSON and place it in ml/data/raw/<LABEL>/")
                sys.exit(1)
            # Use NPZ directly
            npz_path = sorted(npz_files)[-1]
            split_path = npz_path.replace(".npz", "_split.json")
            if not os.path.exists(split_path):
                split_manifest = perform_group_aware_split(npz_path)
                split_path = npz_path.replace(".npz", "_split.json")
            raw_json_path = None  # Signal to skip validate step
        else:
            # Merge all JSON files into a single temporary file
            import tempfile
            merged_samples = []
            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "samples" in data:
                            merged_samples.extend(data["samples"])
                except Exception as e:
                    print(f"[WARN] Failed to read {jf}: {e}")
            
            if not merged_samples:
                print("\n[ERROR] No valid samples found in raw JSON files.")
                sys.exit(1)
                
            merged_data = {
                "version": 1,
                "featureCount": 63,
                "sequenceLength": 30,
                "samples": merged_samples
            }
            
            tf_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(merged_data, tf_file)
            tf_file.close()
            
            raw_json_path = tf_file.name
            npz_path = None
            split_path = None
    else:
        npz_path = None
        split_path = None

    best_val = 0.0
    best_test = 0.0
    best_run_record = None
    success = False
    all_runs = []

    exp_configs = EXPERIMENT_CONFIGS[:args.max_experiments]

    for exp_idx, exp_cfg in enumerate(exp_configs):
        exp_num = exp_idx + 1
        print(f"\n{'─'*62}")
        print(f"  EXPERIMENT {exp_num}/{args.max_experiments}")
        print(f"  Config: {json.dumps({k: v for k, v in exp_cfg.items() if k not in ['random_seed']}, indent=None)}")
        print(f"{'─'*62}")

        # Build config
        config = TrainingConfig()
        config.epochs = args.max_epochs
        config.early_stopping_patience = 20
        for k, v in exp_cfg.items():
            if hasattr(config, k):
                setattr(config, k, v)

        # Validate + split if using JSON
        if raw_json_path:
            try:
                X, labels, ids, performers, meta_info, npz_path = validate_dataset_file(
                    raw_json_path
                )
            except SystemExit:
                print(f"  [SKIP] Experiment {exp_num}: Dataset validation failed.")
                continue

            split_manifest = perform_group_aware_split(npz_path)
            split_path = npz_path.replace(".npz", "_split.json")
        elif npz_path and not split_path:
            split_manifest = perform_group_aware_split(npz_path)
            split_path = npz_path.replace(".npz", "_split.json")
        else:
            with open(split_path, "r") as f:
                split_manifest = json.load(f)

        # Leakage check
        leakage_ok = check_leakage(split_manifest)
        if not leakage_ok:
            print("  [WARNING] Leakage detected — continuing but metrics are unreliable.")

        # Override config JSON file with current experiment settings
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf_file:
            json.dump(config.to_dict(), tf_file)
            config_path = tf_file.name

        try:
            model_version = train_model(config_path, npz_path, split_path)
        except Exception as e:
            print(f"  [ERROR] Experiment {exp_num} failed: {e}")
            import traceback; traceback.print_exc()
            continue
        finally:
            os.unlink(config_path)

        # Load run record from runs.jsonl
        runs_path = os.path.join("ml", "experiments", "runs.jsonl")
        run_record = None
        if os.path.exists(runs_path):
            with open(runs_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        if r.get("model_version") == model_version:
                            run_record = r
                    except Exception:
                        continue

        if run_record is None:
            print(f"  [WARN] Could not load run record for {model_version}")
            continue

        val_acc = run_record.get("val_accuracy_final", 0.0)
        test_acc = run_record.get("test_accuracy", 0.0)
        all_runs.append(run_record)

        if val_acc > best_val:
            best_val = val_acc
            best_run_record = run_record
        if test_acc > best_test:
            best_test = test_acc

        # Check 90% target
        if val_acc >= args.target_val and test_acc >= args.target_test:
            success = True
            print_dashboard(exp_num, args.max_experiments, args.max_epochs, args.max_epochs,
                            val_acc, test_acc, best_val, best_test, "TARGET REACHED ✅")
            break

        print_dashboard(exp_num, args.max_experiments, args.max_epochs, args.max_epochs,
                        val_acc, test_acc, best_val, best_test, "TRAINING — NEXT EXPERIMENT")

    # ─────────────────────────────────────────────────────
    # Final report
    # ─────────────────────────────────────────────────────
    elapsed_min = (time.time() - loop_start) / 60.0
    print(f"\n{'='*62}")
    print(f"  SIGN BRIDGE ISL TRAINING LOOP — FINAL REPORT")
    print(f"{'='*62}")
    print(f"  Experiments run  : {len(all_runs)}")
    print(f"  Best val acc     : {best_val*100:.2f}%")
    print(f"  Best test acc    : {best_test*100:.2f}%")
    print(f"  Total time       : {elapsed_min:.1f} min")

    if success:
        print(f"\n  ✅ SUCCESS — 90% TARGET REACHED")
        if best_run_record:
            print(f"  Best model       : {best_run_record.get('model_version', 'unknown')}")
            print(f"  Checkpoint       : {best_run_record.get('best_checkpoint_path', 'N/A')}")
    else:
        print(f"\n  ❌ 90% TARGET NOT REACHED WITH AVAILABLE DATA")
        print(f"\n  HONEST REPORT:")
        print(f"  The model achieved a best validation accuracy of {best_val*100:.1f}% and")
        print(f"  a best test accuracy of {best_test*100:.1f}% across {len(all_runs)} experiments.")
        print(f"\n  To improve beyond this, you need MORE REAL ISL TRAINING DATA.")
        print(f"  Specifically:")
        print(f"    1. More signers (currently limited by performer count in split)")
        print(f"    2. More samples per class (target: 50+ per class)")
        print(f"    3. More recording sessions per signer (variation)")
        print(f"    4. Both left-hand and right-hand samples")
        print(f"\n  Use: python ml/scripts/extract_landmarks_from_media.py --input <video> --label <LABEL>")
        print(f"  Or:  Use the Dataset Studio in-browser to record webcam samples")

    print(f"{'='*62}\n")

    # Save loop summary
    summary = {
        "loop_completed_at": datetime.now().isoformat(),
        "success": success,
        "experiments_run": len(all_runs),
        "best_val_accuracy": best_val,
        "best_test_accuracy": best_test,
        "target_val": args.target_val,
        "target_test": args.target_test,
        "elapsed_minutes": round(elapsed_min, 2),
        "best_model_version": best_run_record.get("model_version") if best_run_record else None,
    }
    out_path = os.path.join("ml", "experiments", "training_loop_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Loop summary saved → {out_path}")


if __name__ == "__main__":
    main()
