import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# Make sure imports work when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml.preprocessing.loader import load_and_validate_dataset
from ml.preprocessing.encoder import encode_labels

def main():
    parser = argparse.ArgumentParser(description="Sign Bridge Baseline LSTM Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Path to exported model directory (e.g. ml/exports/<run_id>)")
    parser.add_argument("--dataset", type=str, default="auto", help="Path to dataset export, or 'auto' to auto-detect")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model)
    meta_path = model_dir / "checkpoint_meta.json"
    weights_path = model_dir / "best_model.keras"
    
    if not meta_path.exists() or not weights_path.exists():
        print(f"Error: Could not find model or metadata in {model_dir}")
        sys.exit(1)
        
    # 1. Load Checkpoint Meta
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    label_to_id = meta['label_to_id']
    id_to_label = meta['id_to_label']
    seq_len = meta['sequence_length']
    feat_count = meta['feature_count']
    
    # Check if this model was trained with a safe split
    split_strategy = meta.get('split_strategy_used', 'unknown')
    if split_strategy != "session-safe":
        print("\n==================================================")
        print("WARNING: This model was trained WITHOUT session-safe splitting!")
        print("         Evaluation accuracy on in-domain data may be optimistic.")
        print("==================================================\n")
        
    # 2. Load Data (bypass normal label encoding to use the model's fixed vocabulary)
    print("Loading dataset...")
    in_repo_path = args.dataset if args.dataset != "auto" else "../dataset_export.json"
    
    # We load but IGNORE the derived label_to_id because we MUST use the model's vocabulary
    X, y_raw_encoded, _, _, sample_meta, _, _ = load_and_validate_dataset(
        configured_in_repo_path=in_repo_path
    )
    
    # Wait, the loader encoded y using its own derived mapping. 
    # To evaluate properly, we need to map the raw string labels to the MODEL'S ids.
    # Since loader doesn't easily expose raw strings, let's look at `id_to_label` from the loader
    # to reverse map back to strings, then encode using the MODEL's map.
    loader_dataset_path = args.dataset if args.dataset != "auto" else "../dataset_export.json"
    
    # Actually, a better way is to just do a direct parse or unmap.
    # We will import the loader's private parse function for evaluation to get raw strings.
    from ml.preprocessing.loader import find_dataset, _load_json_dataset, _load_npz_dataset
    dataset_path, _ = find_dataset(in_repo_path)
    
    if dataset_path.suffix.lower() == '.json':
        X, y_str, _, _, _ = _load_json_dataset(dataset_path)
    else:
        X, y_str, _, _, _ = _load_npz_dataset(dataset_path)
        
    try:
        y_model_encoded = np.array(encode_labels(y_str.tolist(), label_to_id))
    except ValueError as e:
        print(f"Dataset contains labels unknown to the model: {e}")
        sys.exit(1)
        
    # 3. Load Model
    print("Loading model weights...")
    model = tf.keras.models.load_model(weights_path)
    
    # 4. Evaluate
    print("Evaluating...")
    loss, accuracy = model.evaluate(X, y_model_encoded, verbose=1)
    
    print("\n--- EVALUATION RESULTS ---")
    print(f"Total Samples: {len(X)}")
    print(f"Loss:          {loss:.4f}")
    print(f"Accuracy:      {accuracy:.4f}")

if __name__ == "__main__":
    main()
