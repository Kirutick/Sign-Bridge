"""
Sign Bridge - ASL-10 Independent Forensic Verification Script (§1 - §23)
Performs automated accounting, duplicate detection, label shuffling, feature scrambling,
and checkpoint re-evaluation.
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def main():
    print("=== ASL-10 INDEPENDENT FORENSIC VERIFICATION ===")
    
    # 1. Dataset Verification
    data_path = "ml/data/processed/dataset_asl10_v1.npz"
    meta_path = "ml/data/processed/meta_asl10_v1.json"
    
    assert os.path.exists(data_path), f"Missing {data_path}"
    assert os.path.exists(meta_path), f"Missing {meta_path}"
    
    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    sample_meta = data["sample_meta"]
    
    print(f"Total Samples: {len(X)}")
    print(f"Feature Tensor Shape: {X.shape} (Expected: 400, 30, 63)")
    print(f"Labels Array Shape: {y.shape}")
    
    # 2. Sample Accounting
    train_idx, val_idx, test_idx = [], [], []
    train_signers, val_signers, test_signers = set(), set(), set()
    
    for i, m in enumerate(sample_meta):
        p = m["performer_id"]
        if p in ["perf_01_native", "perf_02_native", "perf_03_interpreter", "perf_04_interpreter",
                 "perf_05_coda", "perf_06_coda", "perf_07_learner"]:
            train_idx.append(i)
            train_signers.add(p)
        elif p == "perf_08_learner":
            val_idx.append(i)
            val_signers.add(p)
        elif p in ["perf_09_holdout", "perf_10_holdout"]:
            test_idx.append(i)
            test_signers.add(p)
            
    print(f"\nSample Accounting:")
    print(f"  Train:      {len(train_idx)} samples across {len(train_signers)} signers: {sorted(train_signers)}")
    print(f"  Validation: {len(val_idx)} samples across {len(val_signers)} signers: {sorted(val_signers)}")
    print(f"  Test:       {len(test_idx)} samples across {len(test_signers)} signers: {sorted(test_signers)}")
    print(f"  Sum:        {len(train_idx) + len(val_idx) + len(test_idx)} / {len(X)}")
    
    # 3. Disjointness Check
    tv_intersect = train_signers.intersection(val_signers)
    tt_intersect = train_signers.intersection(test_signers)
    vt_intersect = val_signers.intersection(test_signers)
    print(f"\nSigner Disjointness:")
    print(f"  Train INTERSECT Val:  {tv_intersect} (Empty: {len(tv_intersect) == 0})")
    print(f"  Train INTERSECT Test: {tt_intersect} (Empty: {len(tt_intersect) == 0})")
    print(f"  Val INTERSECT Test:   {vt_intersect} (Empty: {len(vt_intersect) == 0})")
    
    # 4. Pairwise Distance & Near-Duplicate Analysis
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    flat_train = X_train.reshape(len(X_train), -1)
    flat_test = X_test.reshape(len(X_test), -1)
    
    min_test_to_train_dist = float("inf")
    exact_duplicates = 0
    
    for i in range(len(flat_test)):
        dists = np.linalg.norm(flat_train - flat_test[i], axis=1)
        min_d = np.min(dists)
        if min_d < min_test_to_train_dist:
            min_test_to_train_dist = min_d
        if min_d < 1e-5:
            exact_duplicates += 1
            
    print(f"\nDistance & Duplicate Analysis:")
    print(f"  Exact Duplicates between Train and Test: {exact_duplicates}")
    print(f"  Minimum Euclidean Distance (Test -> Train): {min_test_to_train_dist:.4f}")
    
    # 5. Independent Checkpoint Reload & Re-evaluation
    saved_model_path = "models/sign_bridge_v3_asl10/model.keras"
    model = tf.keras.models.load_model(saved_model_path)
    
    y_test = y[test_idx]
    test_probs = model.predict(X_test, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    
    acc = accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average="macro")
    print(f"\nIndependent Checkpoint Evaluation:")
    print(f"  Loaded model: {saved_model_path}")
    print(f"  Recomputed Test Accuracy: {acc*100:.2f}%")
    print(f"  Recomputed Macro F1:      {macro_f1:.4f}")
    
    # 6. Sanity Test: Label Shuffle Control Test
    print(f"\nRunning Sanity Test: Random-Label Shuffling on Training Set...")
    tf.random.set_seed(123)
    np.random.seed(123)
    y_train_shuffled = np.random.permutation(y[train_idx])
    
    ctrl_inputs = tf.keras.Input(shape=(30, 63))
    cx = tf.keras.layers.Masking(mask_value=0.0)(ctrl_inputs)
    cx = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(cx)
    cx = tf.keras.layers.Dense(48, activation="relu")(cx)
    ctrl_out = tf.keras.layers.Dense(10, activation="softmax")(cx)
    ctrl_model = tf.keras.Model(inputs=ctrl_inputs, outputs=ctrl_out)
    ctrl_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    ctrl_model.fit(X_train, y_train_shuffled, epochs=25, batch_size=16, verbose=0)
    ctrl_preds = np.argmax(ctrl_model.predict(X_test, verbose=0), axis=1)
    ctrl_acc = accuracy_score(y_test, ctrl_preds)
    ctrl_f1 = f1_score(y_test, ctrl_preds, average="macro", zero_division=0)
    
    print(f"  Shuffled-Label Test Accuracy: {ctrl_acc*100:.2f}% (Expected ~10.0% chance)")
    print(f"  Shuffled-Label Macro F1:      {ctrl_f1:.4f}")
    
    # 7. Sanity Test: Feature Scrambling Control Test
    print(f"\nRunning Sanity Test: Feature Scrambling...")
    X_test_scrambled = np.random.permutation(X_test.reshape(-1)).reshape(X_test.shape)
    scramble_preds = np.argmax(model.predict(X_test_scrambled, verbose=0), axis=1)
    scramble_acc = accuracy_score(y_test, scramble_preds)
    scramble_f1 = f1_score(y_test, scramble_preds, average="macro", zero_division=0)
    print(f"  Scrambled-Feature Test Accuracy: {scramble_acc*100:.2f}% (Expected ~10.0%)")
    print(f"  Scrambled-Feature Macro F1:      {scramble_f1:.4f}")

if __name__ == "__main__":
    main()
