"""
Sign Bridge - Comprehensive Per-Class Verification for All 50 ASL Signs
"""

import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=== EVALUATING ALL 50 ASL SIGNS ===")
    
    # Load 50-class dataset
    data = np.load("ml/data/processed/dataset_asl50_v1.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    meta = data["sample_meta"]
    
    with open("ml/data/processed/meta_asl50_v1.json", "r", encoding="utf-8") as f:
        meta_json = json.load(f)
        
    classes = meta_json["classes"]
    
    # Load model
    model = tf.keras.models.load_model("models/sign_bridge_v4_asl50/model.keras")
    
    # Evaluate across all 2000 samples
    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    print("\n--- PER-CLASS ACCURACY BREAKDOWN (40 samples/class) ---")
    failing_classes = []
    
    for c_idx, c_name in enumerate(classes):
        mask = (y == c_idx)
        c_acc = np.mean(preds[mask] == y[mask])
        mean_conf = np.mean(np.max(probs[mask], axis=1))
        
        status = "OK" if c_acc >= 0.80 else "NEEDS_OPTIMIZATION"
        print(f"Class {c_idx:02d} [{c_name:<10}]: Accuracy = {c_acc*100:6.2f}% | Mean Conf = {mean_conf:.4f} | [{status}]")
        
        if c_acc < 0.80:
            failing_classes.append((c_idx, c_name, c_acc, mean_conf))
            
    total_acc = np.mean(preds == y)
    print(f"\nOverall Full Dataset Accuracy: {total_acc*100:.2f}%")
    print(f"Classes needing optimization (<80%): {len(failing_classes)} / 50")
    for c_idx, c_name, acc, conf in failing_classes:
        # Find what it gets confused with
        mask = (y == c_idx)
        misclass = preds[mask][preds[mask] != c_idx]
        top_confusion = np.bincount(misclass).argmax() if len(misclass) > 0 else "None"
        confused_name = classes[top_confusion] if isinstance(top_confusion, (int, np.integer)) else "None"
        print(f"  - {c_name} (acc: {acc*100:.1f}%) -> often confused with: {confused_name}")

if __name__ == "__main__":
    main()
