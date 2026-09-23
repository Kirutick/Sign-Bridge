"""
Sign Bridge — Scientific Feature Engineering Experiment Runner
Runs controlled comparative experiments:
  Exp A: v1_single_hand_63 (63 features - Control Baseline)
  Exp B: v2_single_hand_geometry (134 features - Geometry)
  Exp C: v3_single_hand_geom_vel (203 features - Geometry + Velocity)
  Exp D: v4_single_hand_geom_vel_acc (266 features - Geometry + Vel + Accel)
  Exp E: v5_single_hand_all (271 features - All Validated Features)

Under identical datasets, splits, random seeds, training schedules, and evaluation metrics.
"""

import os
import sys
import time
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.abspath("."))
from ml.preprocessing.enhanced_features import (
    extract_sequence_enhanced_features,
    transform_dataset_features,
    FEATURE_SCHEMAS,
)

CLASSES_50 = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "NUM_0", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5", "NUM_6", "NUM_7", "NUM_8", "NUM_9",
    "HELLO", "NO", "THANK_YOU", "YES", "HELP", "PLEASE", "SORRY",
    "GOODBYE", "NAME", "WATER", "FOOD", "STOP", "GO", "MORE"
]

STATIC_SIGNS = set([
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
    "NUM_0", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5", "NUM_6", "NUM_7", "NUM_8", "NUM_9"
])
DYNAMIC_SIGNS = set([
    "J", "Z", "HELLO", "NO", "THANK_YOU", "YES", "HELP", "PLEASE", "SORRY", "GOODBYE", "NAME", "WATER", "FOOD", "STOP", "GO", "MORE"
])

def calculate_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) with equal-width confidence bins."""
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = predictions == y_true
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def build_experiment_model(input_shape: tuple, num_classes: int = 50) -> tf.keras.Model:
    """Standardized Keras architecture with dynamic feature count input shape."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape, name="input_layer"),
        tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.15, name="lstm_1"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.15, name="dropout_1"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")
    ], name=f"sign_bridge_lstm_{input_shape[1]}")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def analyze_feature_statistics(X_enhanced: np.ndarray) -> dict:
    """Computes min, max, mean, std, NaN count, Inf count across dataset."""
    N, T, D = X_enhanced.shape
    flat_X = X_enhanced.reshape(-1, D)
    
    nan_count = int(np.isnan(flat_X).sum())
    inf_count = int(np.isinf(flat_X).sum())
    mins = np.min(flat_X, axis=0)
    maxs = np.max(flat_X, axis=0)
    means = np.mean(flat_X, axis=0)
    stds = np.std(flat_X, axis=0)
    
    near_zero_variance = [int(i) for i in range(D) if stds[i] < 1e-5]
    
    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "near_zero_variance_indices": near_zero_variance,
        "feature_mins": mins.tolist(),
        "feature_maxs": maxs.tolist(),
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
    }

def analyze_feature_correlations(X_enhanced: np.ndarray, threshold: float = 0.95) -> list:
    """Identifies highly correlated/redundant feature pairs (|r| > threshold)."""
    N, T, D = X_enhanced.shape
    flat_X = X_enhanced.reshape(-1, D)
    
    # Filter out zero-std features to prevent division by zero
    stds = np.std(flat_X, axis=0)
    valid_cols = [i for i in range(D) if stds[i] > 1e-5]
    
    corr_matrix = np.corrcoef(flat_X[:, valid_cols], rowvar=False)
    redundant_pairs = []
    
    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            r_val = float(corr_matrix[i, j])
            if abs(r_val) >= threshold:
                redundant_pairs.append({
                    "feat_a": valid_cols[i],
                    "feat_b": valid_cols[j],
                    "correlation": round(r_val, 4),
                })
                
    return redundant_pairs

def benchmark_inference_latency(model: tf.keras.Model, input_shape: tuple, runs: int = 100) -> dict:
    """Measures single-sample inference latency (batch_size=1) over multiple runs."""
    dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
    # Warmup
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)
        
    start_t = time.perf_counter()
    for _ in range(runs):
        _ = model.predict(dummy_input, verbose=0)
    end_t = time.perf_counter()
    
    avg_latency_ms = ((end_t - start_t) / runs) * 1000.0
    return {
        "avg_latency_ms": round(avg_latency_ms, 3),
        "fps_throughput": round(1000.0 / avg_latency_ms, 1),
    }

def main():
    print("\n" + "="*70)
    print(" SIGN BRIDGE — ENHANCED FEATURE ENGINEERING EXPERIMENTS")
    print("="*70 + "\n")
    
    # Load dataset
    data_path = "ml/data/processed/dataset_asl50_v1.npz"
    data = np.load(data_path, allow_pickle=True)
    X_raw = data["X"] # (2000, 30, 63)
    y = data["y"]
    meta = data["sample_meta"]
    
    train_idx, val_idx, test_idx = [], [], []
    for i, m in enumerate(meta):
        p = m["performer_id"]
        if p in ["perf_01_native", "perf_02_native", "perf_03_interpreter", "perf_04_interpreter",
                 "perf_05_coda", "perf_06_coda", "perf_07_learner"]:
            train_idx.append(i)
        elif p == "perf_08_learner":
            val_idx.append(i)
        elif p in ["perf_09_holdout", "perf_10_holdout"]:
            test_idx.append(i)
            
    print(f"Dataset Partitions:")
    print(f"  Train samples: {len(train_idx)} (7 signers)")
    print(f"  Val samples:   {len(val_idx)} (1 signer: perf_08)")
    print(f"  Test samples:  {len(test_idx)} (2 unseen signers: perf_09, perf_10)")
    
    experiments = [
        ("Exp A (Baseline)", "v1_single_hand_63", "63 raw coordinates"),
        ("Exp B (Geometry)", "v2_single_hand_geometry", "63 coords + joint angles + lengths + spread + palm + orient"),
        ("Exp C (Geom + Vel)", "v3_single_hand_geom_vel", "Geometry + landmark velocities + global motion"),
        ("Exp D (Geom + Vel + Accel)", "v4_single_hand_geom_vel_acc", "Geometry + velocities + accelerations + global motion"),
        ("Exp E (All Features)", "v5_single_hand_all", "All validated geometric, velocity, accel, and temporal summaries"),
    ]
    
    results = []
    
    for exp_title, schema_id, desc in experiments:
        print(f"\n" + "-"*70)
        print(f"[+] Running {exp_title}: Schema '{schema_id}' ({FEATURE_SCHEMAS[schema_id]} features)")
        print(f"   Description: {desc}")
        print("-"*70)
        
        # 1. Feature Extraction & Transformation Timing
        t0 = time.perf_counter()
        X_enhanced = transform_dataset_features(X_raw, schema=schema_id)
        feat_time_sec = time.perf_counter() - t0
        feat_ms_per_sample = (feat_time_sec / len(X_raw)) * 1000.0
        
        # 2. Sanity Checks
        stats = analyze_feature_statistics(X_enhanced)
        print(f"   Feature Extraction: {feat_ms_per_sample:.3f} ms/sample (Total: {feat_time_sec:.2f}s)")
        print(f"   Sanity Check: NaNs = {stats['nan_count']}, Infs = {stats['inf_count']}, Near-zero variance features = {len(stats['near_zero_variance_indices'])}")
        
        # 3. Correlation Analysis
        redundant = analyze_feature_correlations(X_enhanced, threshold=0.95)
        print(f"   Correlations: {len(redundant)} feature pairs with |r| >= 0.95")
        
        # 4. Train / Val / Test Partitioning
        X_train, y_train = X_enhanced[train_idx], y[train_idx]
        X_val, y_val = X_enhanced[val_idx], y[val_idx]
        X_test, y_test = X_enhanced[test_idx], y[test_idx]
        
        # 5. Model Training
        tf.random.set_seed(42)
        np.random.seed(42)
        
        model = build_experiment_model(input_shape=(30, FEATURE_SCHEMAS[schema_id]), num_classes=50)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5)
        ]
        
        train_start = time.perf_counter()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        train_time_sec = round(time.perf_counter() - train_start, 2)
        epochs_run = len(history.history["loss"])
        
        # 6. Evaluation Metrics
        # Train
        train_probs = model.predict(X_train, verbose=0)
        train_preds = np.argmax(train_probs, axis=1)
        train_acc = accuracy_score(y_train, train_preds)
        
        # Val
        val_probs = model.predict(X_val, verbose=0)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = accuracy_score(y_val, val_preds)
        val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        
        # Test (Unseen Signers)
        test_probs = model.predict(X_test, verbose=0)
        test_preds = np.argmax(test_probs, axis=1)
        test_acc = accuracy_score(y_test, test_preds)
        test_macro_p = precision_score(y_test, test_preds, average="macro", zero_division=0)
        test_macro_r = recall_score(y_test, test_preds, average="macro", zero_division=0)
        test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        test_weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
        
        # ECE Calibration
        test_ece = calculate_ece(y_test, test_probs, n_bins=10)
        
        # Static vs Dynamic Breakdown
        test_static_mask = [CLASSES_50[y_test[i]] in STATIC_SIGNS for i in range(len(y_test))]
        test_dynamic_mask = [CLASSES_50[y_test[i]] in DYNAMIC_SIGNS for i in range(len(y_test))]
        
        static_acc = accuracy_score(y_test[test_static_mask], test_preds[test_static_mask])
        dynamic_acc = accuracy_score(y_test[test_dynamic_mask], test_preds[test_dynamic_mask])
        
        # Overfitting gap
        train_val_gap = (train_acc - val_acc) * 100.0
        train_test_gap = (train_acc - test_acc) * 100.0
        
        # Inference Latency Benchmark
        latency_info = benchmark_inference_latency(model, (30, FEATURE_SCHEMAS[schema_id]))
        param_count = model.count_params()
        
        # Print summary
        print(f"   Epochs: {epochs_run} (Train Time: {train_time_sec}s)")
        print(f"   Train Acc:      {train_acc*100:.2f}%")
        print(f"   Val Acc:        {val_acc*100:.2f}% | Val Macro F1: {val_macro_f1:.4f}")
        print(f"   Unseen Test Acc:{test_acc*100:.2f}% | Test Macro F1: {test_macro_f1:.4f} | Weighted F1: {test_weighted_f1:.4f}")
        print(f"   Test ECE:       {test_ece:.4f}")
        print(f"   Static Signs:   {static_acc*100:.2f}% ({sum(test_static_mask)} samples)")
        print(f"   Dynamic Signs:  {dynamic_acc*100:.2f}% ({sum(test_dynamic_mask)} samples)")
        print(f"   Inference Cost: {latency_info['avg_latency_ms']} ms/frame | Params: {param_count:,}")
        
        # Save model
        exp_dir = f"ml/experiments/enhanced_features/{schema_id}"
        os.makedirs(exp_dir, exist_ok=True)
        model.save(os.path.join(exp_dir, "model.keras"))
        
        res_record = {
            "experiment": exp_title,
            "schema": schema_id,
            "feature_dim": FEATURE_SCHEMAS[schema_id],
            "description": desc,
            "epochs_run": epochs_run,
            "train_time_sec": train_time_sec,
            "feat_ms_per_sample": round(feat_ms_per_sample, 3),
            "train_accuracy": round(float(train_acc), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_macro_f1": round(float(val_macro_f1), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_precision": round(float(test_macro_p), 4),
            "test_macro_recall": round(float(test_macro_r), 4),
            "test_macro_f1": round(float(test_macro_f1), 4),
            "test_weighted_f1": round(float(test_weighted_f1), 4),
            "test_ece": round(float(test_ece), 4),
            "static_accuracy": round(float(static_acc), 4),
            "dynamic_accuracy": round(float(dynamic_acc), 4),
            "train_val_gap_pct": round(float(train_val_gap), 2),
            "train_test_gap_pct": round(float(train_test_gap), 2),
            "inference_latency_ms": latency_info["avg_latency_ms"],
            "param_count": param_count,
            "nan_count": stats["nan_count"],
            "inf_count": stats["inf_count"],
            "redundant_pair_count": len(redundant),
            "redundant_pairs_top5": redundant[:5],
        }
        results.append(res_record)
        
    # Save overall experiment report JSON
    out_json = "ml/experiments/enhanced_features/experiment_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*70)
    print(" EXPERIMENT SUMMARY TABLE")
    print("="*70)
    print(f"{'Experiment':<24} | {'Dim':<4} | {'Val F1':<7} | {'Test Acc':<8} | {'Test F1':<8} | {'ECE':<6} | {'Static':<7} | {'Dynamic':<7} | {'Lat(ms)':<7}")
    print("-" * 96)
    for r in results:
        print(f"{r['experiment']:<24} | {r['feature_dim']:<4} | {r['val_macro_f1']:<7.4f} | {r['test_accuracy']*100:<7.2f}% | {r['test_macro_f1']:<8.4f} | {r['test_ece']:<6.4f} | {r['static_accuracy']*100:<6.2f}% | {r['dynamic_accuracy']*100:<6.2f}% | {r['inference_latency_ms']:<7.2f}")
    print("="*70 + "\n")
    print(f"Results saved to: {out_json}")

if __name__ == "__main__":
    main()
