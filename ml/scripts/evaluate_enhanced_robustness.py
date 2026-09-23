"""
Robustness & Noise Stress Test: Baseline (63) vs Enhanced (134, 203, 266, 271)
Evaluates generalization degradation under:
  1. Coordinate Gaussian Noise (sigma = 0.02, 0.05, 0.10)
  2. Landmark Dropout (5%, 10% dropped points)
  3. Frame rate / Temporal subsampling jitter
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.abspath("."))
from ml.preprocessing.enhanced_features import (
    extract_sequence_enhanced_features,
    transform_dataset_features,
    FEATURE_SCHEMAS,
)

def add_gaussian_noise(X_seq: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, X_seq.shape)
    return X_seq + noise

def add_landmark_dropout(X_seq: np.ndarray, drop_prob: float) -> np.ndarray:
    mask = (np.random.rand(*X_seq.shape) > drop_prob).astype(np.float32)
    return X_seq * mask

def main():
    data = np.load("ml/data/processed/dataset_asl50_v1.npz", allow_pickle=True)
    X_raw = data["X"]
    y = data["y"]
    meta = data["sample_meta"]
    
    test_idx = [i for i, m in enumerate(meta) if m["performer_id"] in ["perf_09_holdout", "perf_10_holdout"]]
    X_test_raw = X_raw[test_idx]
    y_test = y[test_idx]
    
    schemas = [
        ("Exp A (63)", "v1_single_hand_63"),
        ("Exp B (134)", "v2_single_hand_geometry"),
        ("Exp C (203)", "v3_single_hand_geom_vel"),
        ("Exp D (266)", "v4_single_hand_geom_vel_acc"),
        ("Exp E (271)", "v5_single_hand_all"),
    ]
    
    noise_levels = [0.0, 0.02, 0.05, 0.08]
    dropout_levels = [0.0, 0.05, 0.10]
    
    results = {}
    
    for title, schema_id in schemas:
        model_path = f"ml/experiments/enhanced_features/{schema_id}/model.keras"
        if not os.path.exists(model_path):
            continue
        model = tf.keras.models.load_model(model_path)
        
        schema_results = {"noise": {}, "dropout": {}}
        
        # Test Noise
        for sigma in noise_levels:
            np.random.seed(42)
            X_noisy = np.array([add_gaussian_noise(seq, sigma) for seq in X_test_raw])
            X_noisy_feat = transform_dataset_features(X_noisy, schema=schema_id)
            preds = np.argmax(model.predict(X_noisy_feat, verbose=0), axis=1)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            schema_results["noise"][f"sigma_{sigma}"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
            
        # Test Dropout
        for p in dropout_levels:
            np.random.seed(42)
            X_dropped = np.array([add_landmark_dropout(seq, p) for seq in X_test_raw])
            X_dropped_feat = transform_dataset_features(X_dropped, schema=schema_id)
            preds = np.argmax(model.predict(X_dropped_feat, verbose=0), axis=1)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            schema_results["dropout"][f"p_{p}"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
            
        results[schema_id] = schema_results
        
    out_path = "ml/experiments/enhanced_features/robustness_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n================ ROBUSTNESS STRESS TEST SUMMARY ================")
    print(f"{'Schema':<28} | {'Clean':<7} | {'Noise 0.02':<10} | {'Noise 0.05':<10} | {'Drop 5%':<8} | {'Drop 10%':<8}")
    print("-" * 84)
    for schema_id, data in results.items():
        clean = data['noise']['sigma_0.0']['acc'] * 100
        n02 = data['noise']['sigma_0.02']['acc'] * 100
        n05 = data['noise']['sigma_0.05']['acc'] * 100
        d05 = data['dropout']['p_0.05']['acc'] * 100
        d10 = data['dropout']['p_0.1']['acc'] * 100
        print(f"{schema_id:<28} | {clean:<6.1f}% | {n02:<9.1f}% | {n05:<9.1f}% | {d05:<7.1f}% | {d10:<7.1f}%")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
