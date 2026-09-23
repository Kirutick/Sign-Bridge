"""
Sign Bridge - Kinematic Model Training, Evaluation, Calibration & Parity Export (Phase 8 Troubleshooting)
"""

import os
import sys
import json
import shutil
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy import stats

def build_model():
    inputs = tf.keras.Input(shape=(30, 63))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.1)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="sign_bridge_kinematic_v1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    
    data = np.load("ml/data/processed/dataset_kinematic_v1.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    sample_meta = data["sample_meta"]
    
    # Signer-independent grouped splitting
    train_idx = []
    val_idx = []
    test_idx = []
    
    for i, meta in enumerate(sample_meta):
        p = meta["performer_id"]
        if p in ["perf_01", "perf_02", "perf_03"]:
            train_idx.append(i)
        elif p == "perf_04":
            val_idx.append(i)
        elif p == "perf_05":
            test_idx.append(i)
            
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Grouped Performer Split:")
    print(f"  Train ({X_train.shape[0]} samples): perf_01, perf_02, perf_03")
    print(f"  Val   ({X_val.shape[0]} samples): perf_04")
    print(f"  Test  ({X_test.shape[0]} samples): perf_05 (Unseen Holdout)")
    
    model = build_model()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=0
    )
    
    print(f"\nTraining finished in {len(history.history['loss'])} epochs.")
    
    # 1. Validation Metrics
    val_probs = model.predict(X_val, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    print(f"Validation (perf_04): Acc = {val_acc*100:.2f}%, Macro F1 = {val_f1:.4f}")
    
    # 2. Test Metrics (Unseen perf_05)
    test_probs = model.predict(X_test, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    
    print(f"\nUnseen Holdout Test (perf_05 - {len(y_test)} samples):")
    print(f"  Test Accuracy:   {test_acc*100:.2f}%")
    print(f"  Test Macro F1:   {test_macro_f1:.4f}")
    print(f"  Test Weighted F1:{test_weighted_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    print("\nConfusion Matrix (Test):")
    print(cm)
    
    # Per-class breakdown
    CLASSES = ["HELLO", "NO", "THANK_YOU", "YES"]
    for c in range(4):
        c_acc = np.mean(test_preds[y_test == c] == c)
        c_f1 = f1_score(y_test == c, test_preds == c, zero_division=0)
        print(f"  Class {c} ({CLASSES[c]}): Accuracy = {c_acc*100:.1f}%, F1 = {c_f1:.4f}")
        
    # Save checkpoints and update models
    ckpt_dir = "models/sign_bridge_v2"
    os.makedirs(ckpt_dir, exist_ok=True)
    h5_path = os.path.join(ckpt_dir, "temp.h5")
    model.save(h5_path)
    print(f"\nSaved updated model to {h5_path}")
    
    # Save layers model json directly
    layers_dir = os.path.join(ckpt_dir, "model_layers")
    os.makedirs(layers_dir, exist_ok=True)
    model.save(os.path.join(layers_dir, "model.keras"))
    
    # Export TFJS layers topology
    model_json_path = os.path.join(ckpt_dir, "model", "model.json")
    os.makedirs(os.path.dirname(model_json_path), exist_ok=True)
    
    # Export weights
    weights = model.get_weights()
    weight_bytes = b"".join([w.astype(np.float32).tobytes() for w in weights])
    bin_path = os.path.join(ckpt_dir, "model", "group1-shard1of1.bin")
    with open(bin_path, "wb") as f:
        f.write(weight_bytes)
        
    # Build Keras 2 / TFJS compatible model.json
    weight_manifest_entries = []
    for weight_tensor in model.weights:
        name = weight_tensor.name
        shape = list(weight_tensor.shape)
        weight_manifest_entries.append({
            "name": name,
            "shape": shape,
            "dtype": "float32"
        })
        
    config = model.get_config()
    # Ensure batch_input_shape in layer 0
    if "layers" in config and len(config["layers"]) > 0:
        if "batch_input_shape" not in config["layers"][0]["config"]:
            config["layers"][0]["config"]["batch_input_shape"] = [None, 30, 63]
            
    tfjs_manifest = {
        "format": "layers-model",
        "generatedBy": "keras v3.11.2",
        "convertedBy": "TensorFlow.js Converter v4.22.0",
        "modelTopology": {
            "keras_version": "2.15.0",
            "backend": "tensorflow",
            "model_config": {
                "class_name": "Functional",
                "config": config
            }
        },
        "weightsManifest": [
            {
                "paths": ["group1-shard1of1.bin"],
                "weights": weight_manifest_entries
            }
        ]
    }
    
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(tfjs_manifest, f, indent=2)
        
    print(f"Exported TFJS model to {model_json_path} ({len(weight_bytes)} bytes)")
    
    # Sync to public/models/sign-model/
    public_dir = "public/models/sign-model"
    os.makedirs(public_dir, exist_ok=True)
    shutil.copy2(model_json_path, os.path.join(public_dir, "model.json"))
    shutil.copy2(bin_path, os.path.join(public_dir, "group1-shard1of1.bin"))
    
    # Copy label map and metadata
    label_map_path = os.path.join(public_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "label_to_index": {"HELLO": 0, "NO": 1, "THANK_YOU": 2, "YES": 3},
            "index_to_label": {"0": "HELLO", "1": "NO", "2": "THANK_YOU", "3": "YES"},
            "label_to_id": {"HELLO": 0, "NO": 1, "THANK_YOU": 2, "YES": 3},
            "id_to_label": {"0": "HELLO", "1": "NO", "2": "THANK_YOU", "3": "YES"},
            "num_classes": 4
        }, f, indent=2)
        
    feature_spec_path = os.path.join(public_dir, "feature_spec.json")
    with open(feature_spec_path, "w", encoding="utf-8") as f:
        json.dump({
            "featureCount": 63,
            "sequenceLength": 30,
            "datasetVersion": "kinematic_v1",
            "modelVersion": "v0.7.1-kinematic"
        }, f, indent=2)
        
    print("Synced updated model artifacts to public/models/sign-model/")

if __name__ == "__main__":
    main()
