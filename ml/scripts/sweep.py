import os
import sys
import yaml
import json
import hashlib
import time
import argparse
from pathlib import Path
import numpy as np

# Make sure imports work when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml.preprocessing.loader import load_and_validate_dataset
from ml.preprocessing.splitter import perform_dataset_split
from ml.models.baseline_lstm import build_baseline_lstm
from ml.training.seeding import set_deterministic_seeds
from ml.training.loop import train_model

def hash_array(arr: np.ndarray) -> str:
    """Returns MD5 hash of a numpy array to verify split determinism."""
    return hashlib.md5(arr.tobytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Sign Bridge Hyperparameter Sweep Harness")
    parser.add_argument("--sweep_dir", type=str, required=True, help="Path to sweep directory (e.g. ml/experiments/grid_<id>)")
    parser.add_argument("--dataset", type=str, default="auto", help="Path to dataset export")
    
    args = parser.parse_args()
    sweep_dir = Path(args.sweep_dir)
    configs_dir = sweep_dir / "configs"
    
    if not configs_dir.exists():
        print(f"Error: {configs_dir} does not exist.")
        sys.exit(1)
        
    config_files = sorted(list(configs_dir.glob("*.yaml")))
    if not config_files:
        print(f"No configs found in {configs_dir}")
        sys.exit(1)
        
    print(f"Found {len(config_files)} configurations to run.")
    
    # 1. Load dataset ONCE
    in_repo_path = args.dataset if args.dataset != "auto" else "../dataset_export.json"
    X, y_encoded, label_to_id, id_to_label, sample_meta, dataset_version, norm_version = load_and_validate_dataset(
        configured_in_repo_path=in_repo_path
    )
    
    num_classes = len(label_to_id)
    first_split_hashes = None
    
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        exp_id = config.get('experiment_id')
        sweep_id = config.get('sweep_id')
        seed = config.get('seed', 42)
        
        run_id = f"{sweep_id}/{exp_id}"
        
        # Check Resumability
        meta_path = Path("ml/exports") / run_id / "checkpoint_meta.json"
        if meta_path.exists():
            print(f"Skipping {exp_id} - already completed.")
            continue
            
        print(f"\n==================================================")
        print(f" RUNNING: {exp_id}")
        print(f"==================================================")
        
        # Split Determinism
        set_deterministic_seeds(seed)
        
        # We re-split per experiment as required by Section 4
        (X_train, y_train), (X_val, y_val), (X_test, y_test), strategy = perform_dataset_split(
            X, y_encoded, sample_meta, 
            test_size=config.get('test_size', 0.15), 
            val_size=config.get('val_size', 0.15),
            random_state=seed
        )
        
        current_hashes = {
            "train": hash_array(y_train),
            "val": hash_array(y_val),
            "test": hash_array(y_test)
        }
        
        if first_split_hashes is None:
            first_split_hashes = current_hashes
            print(f"Recorded split hashes for determinism verification:")
            print(f"  Train: {first_split_hashes['train']}")
            print(f"  Val:   {first_split_hashes['val']}")
            print(f"  Test:  {first_split_hashes['test']}")
        else:
            if current_hashes != first_split_hashes:
                print("FATAL ERROR: Split hashes do not match! Determinism broken.")
                print(f"Expected: {first_split_hashes}")
                print(f"Got:      {current_hashes}")
                sys.exit(1)
                
        # Build & Train
        start_time = time.time()
        model = build_baseline_lstm(num_classes=num_classes, config=config.get('model', {}))
        
        trained_model, history = train_model(
            model, X_train, y_train, X_val, y_val, 
            run_id, config.get('training', {}), seed, strategy
        )
        train_time = time.time() - start_time
        val_acc = history['val_accuracy'][-1]
        
        # Save Checkpoint Meta
        checkpoint_meta = {
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "feature_count": 63,
            "sequence_length": 30,
            "dataset_version": dataset_version,
            "normalization_version": norm_version,
            "config_snapshot": config,
            "seed": seed,
            "split_strategy_used": strategy,
            "run_id": run_id,
            "train_time_seconds": train_time,
            "epochs_run": len(history['loss'])
        }
        
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)
            
        print(f"Completed {exp_id} in {train_time:.1f}s")
        
if __name__ == "__main__":
    main()
