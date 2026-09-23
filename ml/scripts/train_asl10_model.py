"""
Sign Bridge - 10-Class Expanded ASL Model Training, Evaluation & TFJS Export (Pure TFJS Layers Compatible)
"""

import os
import json
import shutil
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

CLASSES_10 = [
    "HELLO", "NO", "THANK_YOU", "YES", "HELP",
    "PLEASE", "SORRY", "GOODBYE", "NAME", "WATER"
]

def build_asl10_model(input_shape=(30, 63), num_classes=10):
    # Pure TFJS-compatible functional topology without internal masking ops
    inputs = tf.keras.Input(shape=input_shape, name="input_1")
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2),
        name="bidirectional_lstm"
    )(inputs)
    x = tf.keras.layers.Dense(48, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_1")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="sign_bridge_asl10_bilstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    
    data = np.load("ml/data/processed/dataset_asl10_v1.npz", allow_pickle=True)
    X = data["X"]
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
            
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Dataset Partitions:")
    print(f"  Train: {X_train.shape[0]} samples (7 signers)")
    print(f"  Val:   {X_val.shape[0]} samples (1 signer)")
    print(f"  Test:  {X_test.shape[0]} samples (2 unseen signers)")
    
    model = build_asl10_model(input_shape=(30, 63), num_classes=10)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=16,
        callbacks=callbacks,
        verbose=0
    )
    
    print(f"\nTraining finished in {len(history.history['loss'])} epochs.")
    
    # Validation Evaluation
    val_probs = model.predict(X_val, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    print(f"Validation Score: Acc = {val_acc*100:.2f}%, Macro F1 = {val_f1:.4f}")
    
    # Test Evaluation (Unseen Signers)
    test_probs = model.predict(X_test, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    
    print(f"\n==========================================")
    print(f"UNSEEN-SIGNER 10-CLASS EVALUATION METRICS:")
    print(f"  Test Accuracy:    {test_acc*100:.2f}%")
    print(f"  Test Macro F1:    {test_macro_f1:.4f}")
    print(f"  Test Weighted F1: {test_weighted_f1:.4f}")
    print(f"==========================================")
    
    # Save model artifacts
    out_dir = "models/sign_bridge_v3_asl10"
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.keras"))
    
    # Export Pure TFJS
    model_dir = os.path.join(out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    weights = model.get_weights()
    weight_bytes = b"".join([w.astype(np.float32).tobytes() for w in weights])
    bin_path = os.path.join(model_dir, "group1-shard1of1.bin")
    with open(bin_path, "wb") as f:
        f.write(weight_bytes)
        
    weight_manifest_entries = []
    for weight_tensor in model.weights:
        weight_manifest_entries.append({
            "name": weight_tensor.name,
            "shape": list(weight_tensor.shape),
            "dtype": "float32"
        })
        
    config = model.get_config()
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
    
    model_json_path = os.path.join(model_dir, "model.json")
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(tfjs_manifest, f, indent=2)
        
    labels_manifest = {
        "label_to_index": {lbl: i for i, lbl in enumerate(CLASSES_10)},
        "index_to_label": {str(i): lbl for i, lbl in enumerate(CLASSES_10)},
        "label_to_id": {lbl: i for i, lbl in enumerate(CLASSES_10)},
        "id_to_label": {str(i): lbl for i, lbl in enumerate(CLASSES_10)},
        "num_classes": 10
    }
    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(labels_manifest, f, indent=2)
        
    feature_spec = {
        "featureCount": 63,
        "sequenceLength": 30,
        "datasetVersion": "asl10_v1",
        "modelVersion": "v0.8.0-asl10"
    }
    with open(os.path.join(out_dir, "feature_spec.json"), "w", encoding="utf-8") as f:
        json.dump(feature_spec, f, indent=2)
        
    # Sync to public/models/sign-model/
    public_dir = "public/models/sign-model"
    os.makedirs(public_dir, exist_ok=True)
    shutil.copy2(model_json_path, os.path.join(public_dir, "model.json"))
    shutil.copy2(bin_path, os.path.join(public_dir, "group1-shard1of1.bin"))
    shutil.copy2(os.path.join(out_dir, "label_map.json"), os.path.join(public_dir, "label_map.json"))
    shutil.copy2(os.path.join(out_dir, "feature_spec.json"), os.path.join(public_dir, "feature_spec.json"))
    
    print(f"\nSuccessfully trained, exported, and synced 10-class ASL model to {public_dir}")

if __name__ == "__main__":
    main()
