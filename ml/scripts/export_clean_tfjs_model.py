"""
Sign Bridge - Fix TFJS Model JSON Manifest Weight Names
"""

import os
import json
import shutil
import numpy as np
import tensorflow as tf

def fix_and_export_tfjs():
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
    
    # Simple, high-accuracy LSTM architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(30, 63), name="input_layer"),
        tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2, name="lstm_1"),
        tf.keras.layers.Dense(48, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.2, name="dropout_1"),
        tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
    ], name="sign_bridge_asl10_lstm")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        ],
        verbose=0
    )
    
    test_preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = np.mean(test_preds == y_test)
    print(f"ASL-10 Test Accuracy: {acc*100:.2f}%")
    
    out_dir = "models/sign_bridge_v3_asl10"
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.keras"))
    
    weights = model.get_weights()
    weight_manifest_entries = [
        {"name": "lstm_1/kernel", "shape": [63, 256], "dtype": "float32"},
        {"name": "lstm_1/recurrent_kernel", "shape": [64, 256], "dtype": "float32"},
        {"name": "lstm_1/bias", "shape": [256], "dtype": "float32"},
        {"name": "dense_1/kernel", "shape": [64, 48], "dtype": "float32"},
        {"name": "dense_1/bias", "shape": [48], "dtype": "float32"},
        {"name": "output_layer/kernel", "shape": [48, 10], "dtype": "float32"},
        {"name": "output_layer/bias", "shape": [10], "dtype": "float32"}
    ]
    
    weight_bytes = b"".join([w.astype(np.float32).tobytes() for w in weights])
    bin_path = os.path.join(out_dir, "model", "group1-shard1of1.bin")
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    with open(bin_path, "wb") as f:
        f.write(weight_bytes)
        
    layers_config = [
        {
            "class_name": "LSTM",
            "config": {
                "name": "lstm_1",
                "trainable": True,
                "batch_input_shape": [None, 30, 63],
                "dtype": "float32",
                "units": 64,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "use_bias": True,
                "return_sequences": False,
                "dropout": 0.2
            }
        },
        {
            "class_name": "Dense",
            "config": {
                "name": "dense_1",
                "trainable": True,
                "units": 48,
                "activation": "relu",
                "use_bias": True
            }
        },
        {
            "class_name": "Dropout",
            "config": {
                "name": "dropout_1",
                "trainable": True,
                "rate": 0.2
            }
        },
        {
            "class_name": "Dense",
            "config": {
                "name": "output_layer",
                "trainable": True,
                "units": 10,
                "activation": "softmax",
                "use_bias": True
            }
        }
    ]
    
    tfjs_manifest = {
        "format": "layers-model",
        "generatedBy": "keras v2.15.0",
        "convertedBy": "TensorFlow.js Converter v4.22.0",
        "modelTopology": {
            "keras_version": "2.15.0",
            "backend": "tensorflow",
            "model_config": {
                "class_name": "Sequential",
                "config": {
                    "name": "sign_bridge_asl10_lstm",
                    "layers": layers_config
                }
            }
        },
        "weightsManifest": [
            {
                "paths": ["group1-shard1of1.bin"],
                "weights": weight_manifest_entries
            }
        ]
    }
    
    model_json_path = os.path.join(out_dir, "model", "model.json")
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(tfjs_manifest, f, indent=2)
        
    public_dir = "public/models/sign-model"
    os.makedirs(public_dir, exist_ok=True)
    shutil.copy2(model_json_path, os.path.join(public_dir, "model.json"))
    shutil.copy2(bin_path, os.path.join(public_dir, "group1-shard1of1.bin"))
    print("Exported TFJS model with explicit weight bindings.")

if __name__ == "__main__":
    fix_and_export_tfjs()
