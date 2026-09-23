"""
Sign Bridge - Post-Fix Real Inference & Parity Verification Script
"""

import os
import json
import numpy as np
import tensorflow as tf

def check_model_topology(model_json_path):
    print(f"--- 1. Inspecting {model_json_path} ---")
    with open(model_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    raw_str = json.dumps(data)
    unsupported_ops = ["NotEqual", "Equal", "Lambda", "TFOpLambda", "TensorFlowOpLayer"]
    found_unsupported = []
    for op in unsupported_ops:
        if f'"{op}"' in raw_str:
            found_unsupported.append(op)
            
    print(f"Unsupported ops found: {found_unsupported}")
    assert len(found_unsupported) == 0, f"Model contains unsupported ops: {found_unsupported}"
    
    config = data["modelTopology"]["model_config"]["config"]
    layers = config["layers"]
    print(f"Total layers: {len(layers)}")
    for i, l in enumerate(layers):
        cname = l["class_name"]
        name = l["name"]
        print(f"  Layer {i}: {cname} (name: {name})")
        
    in_layer = layers[0]["config"]
    print(f"Input Shape: {in_layer.get('batch_shape') or in_layer.get('batch_input_shape')}")
    out_layer = layers[-1]["config"]
    print(f"Output Units: {out_layer.get('units')}")
    return True

def check_label_map(label_map_path):
    print(f"\n--- 2. Checking Label Map: {label_map_path} ---")
    with open(label_map_path, "r", encoding="utf-8") as f:
        lmap = json.load(f)
        
    expected_order = [
        "HELLO", "NO", "THANK_YOU", "YES", "HELP",
        "PLEASE", "SORRY", "GOODBYE", "NAME", "WATER"
    ]
    for i, exp in enumerate(expected_order):
        actual = lmap["index_to_label"][str(i)]
        print(f"  Index {i}: Expected={exp:<10} Actual={actual:<10} Match={exp == actual}")
        assert exp == actual, f"Mismatch at index {i}: expected {exp}, got {actual}"
    print("Label map matches expected order 100%.")

def check_python_checkpoint_predictions():
    print(f"\n--- 3. Python Checkpoint & Parity Evaluation ---")
    model_path = "models/sign_bridge_v3_asl10/model.keras"
    model = tf.keras.models.load_model(model_path)
    
    data = np.load("ml/data/processed/dataset_asl10_v1.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    meta = data["sample_meta"]
    
    # Select 20 held-out test samples (perf_09, perf_10)
    test_indices = [i for i, m in enumerate(meta) if m["performer_id"] in ["perf_09_holdout", "perf_10_holdout"]][:20]
    X_test_20 = X[test_indices]
    y_test_20 = y[test_indices]
    
    probs = model.predict(X_test_20, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    print(f"Test 20 sample predictions:")
    for i in range(20):
        conf = probs[i, preds[i]]
        print(f"  Sample {i:02d}: True={y_test_20[i]} Pred={preds[i]} Conf={conf:.4f} ProbVector sum={np.sum(probs[i]):.4f}")
        
    # Save test vector for Node TFJS comparison
    np.save("scratch/test_20_X.npy", X_test_20)
    np.save("scratch/test_20_y.npy", y_test_20)
    np.save("scratch/test_20_py_probs.npy", probs)
    print("Saved test tensors to scratch/ for TFJS parity testing.")

if __name__ == "__main__":
    os.makedirs("scratch", exist_ok=True)
    check_model_topology("public/models/sign-model/model.json")
    check_label_map("public/models/sign-model/label_map.json")
    check_python_checkpoint_predictions()
