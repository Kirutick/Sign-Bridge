import os
import sys
import argparse
import yaml
import json
import uuid
import datetime
from pathlib import Path
import numpy as np

# Make sure imports work when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml.preprocessing.loader import load_and_validate_dataset
from ml.preprocessing.splitter import perform_dataset_split
from ml.models.baseline_lstm import build_baseline_lstm
from ml.training.seeding import set_deterministic_seeds
from ml.training.loop import train_model

def main():
    parser = argparse.ArgumentParser(description="Sign Bridge Baseline LSTM Trainer")
    parser.add_argument("--dataset", type=str, default="auto", help="Path to dataset export, or 'auto' to auto-detect")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training configuration")
    
    args = parser.parse_args()
    
    # 1. Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    seed = config.get('seed', 42)
    set_deterministic_seeds(seed)
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:4]}"
    
    # 2. Load dataset
    print("\n--- DATA LOADING ---")
    in_repo_path = args.dataset if args.dataset != "auto" else "../dataset_export.json"
    
    X, y_encoded, label_to_id, id_to_label, sample_meta, dataset_version, norm_version = load_and_validate_dataset(
        configured_in_repo_path=in_repo_path
    )
    
    num_samples = len(y_encoded)
    num_classes = len(label_to_id)
    print(f"Loaded {num_samples} samples across {num_classes} classes.")
    
    # 3. Data Splitting
    print("\n--- DATA SPLITTING ---")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), strategy = perform_dataset_split(
        X, y_encoded, sample_meta, 
        test_size=config.get('test_size', 0.15), 
        val_size=config.get('val_size', 0.15),
        random_state=seed
    )
    
    print(f"Split Strategy Used: {strategy}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # 4. Model Building
    print("\n--- MODEL BUILDING ---")
    model = build_baseline_lstm(num_classes=num_classes, config=config.get('model', {}))
    model.summary()
    
    # 5. Training
    print("\n--- TRAINING ---")
    trained_model, history = train_model(
        model, 
        X_train, y_train, 
        X_val, y_val, 
        run_id, 
        config.get('training', {}), 
        seed,
        strategy
    )
    
    best_epoch = np.argmax(history['val_accuracy']) + 1
    best_val_acc = history['val_accuracy'][best_epoch - 1]
    best_train_acc = history['accuracy'][best_epoch - 1]
    
    # 6. Export Self-Describing Checkpoint Meta
    print("\n--- EXPORTING ---")
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
        "run_id": run_id
    }
    
    meta_path = Path("ml/exports") / run_id / "checkpoint_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(checkpoint_meta, f, indent=2)
        
    print(f"Model and metadata saved to: ml/exports/{run_id}/")
    
    # 7. Final Summary (§12)
    print("\n==================================================")
    print("                TRAINING SUMMARY")
    print("==================================================")
    print(f"Classes:         {num_classes}")
    print(f"Total Samples:   {num_samples} (Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)})")
    print(f"Input Shape:     (30, 63)")
    print(f"Split Strategy:  {strategy}")
    print(f"Best Epoch:      {best_epoch}")
    print(f"Best Train Acc:  {best_train_acc:.4f}")
    print(f"Best Val Acc:    {best_val_acc:.4f}")
    print("==================================================")
    print("NOTE: This is a baseline model only. Not ready for production.")

if __name__ == "__main__":
    main()
