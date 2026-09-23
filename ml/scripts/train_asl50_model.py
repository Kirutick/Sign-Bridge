"""
Sign Bridge - 50-Class Full ASL Vocabulary Model Training, Evaluation & TFJS Export
"""

import os
import json
import shutil
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

CLASSES_50 = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "NUM_0", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5", "NUM_6", "NUM_7", "NUM_8", "NUM_9",
    "HELLO", "NO", "THANK_YOU", "YES", "HELP", "PLEASE", "SORRY",
    "GOODBYE", "NAME", "WATER", "FOOD", "STOP", "GO", "MORE"
]

def build_asl50_model(input_shape=(30, 63), num_classes=50):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape, name="input_layer"),
        tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.15, name="lstm_1"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.15, name="dropout_1"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")
    ], name="sign_bridge_asl50_lstm")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    
    data = np.load("ml/data/processed/dataset_asl50_v1.npz", allow_pickle=True)
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
    
    print(f"ASL-50 Dataset Partitions:")
    print(f"  Train: {X_train.shape[0]} samples (7 signers)")
    print(f"  Val:   {X_val.shape[0]} samples (1 signer)")
    print(f"  Test:  {X_test.shape[0]} samples (2 unseen signers)")
    
    model = build_asl50_model(input_shape=(30, 63), num_classes=50)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    print(f"\nTraining finished in {len(history.history['loss'])} epochs.")
    
    # Validation Evaluation
    val_probs = model.predict(X_val, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    val_acc = accuracy_score(y_val, val_preds)
    val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    print(f"Validation Score (perf_08): Acc = {val_acc*100:.2f}%, Macro F1 = {val_macro_f1:.4f}")
    
    # Test Evaluation (Unseen Signers)
    test_probs = model.predict(X_test, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    
    print(f"\n==========================================")
    print(f"UNSEEN-SIGNER 50-CLASS EVALUATION METRICS:")
    print(f"  Test Accuracy:    {test_acc*100:.2f}%")
    print(f"  Test Macro F1:    {test_macro_f1:.4f}")
    print(f"  Test Weighted F1: {test_weighted_f1:.4f}")
    print(f"==========================================")
    
    # Save model artifacts
    out_dir = "models/sign_bridge_v4_asl50"
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.keras"))
    
    # Export Pure TFJS
    model_dir = os.path.join(out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    weights = model.get_weights()
    weight_manifest_entries = [
        {"name": "lstm_1/kernel", "shape": [63, 512], "dtype": "float32"},
        {"name": "lstm_1/recurrent_kernel", "shape": [128, 512], "dtype": "float32"},
        {"name": "lstm_1/bias", "shape": [512], "dtype": "float32"},
        {"name": "dense_1/kernel", "shape": [128, 128], "dtype": "float32"},
        {"name": "dense_1/bias", "shape": [128], "dtype": "float32"},
        {"name": "output_layer/kernel", "shape": [128, 50], "dtype": "float32"},
        {"name": "output_layer/bias", "shape": [50], "dtype": "float32"}
    ]
    
    weight_bytes = b"".join([w.astype(np.float32).tobytes() for w in weights])
    bin_path = os.path.join(model_dir, "group1-shard1of1.bin")
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
                "units": 128,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "use_bias": True,
                "return_sequences": False,
                "dropout": 0.15
            }
        },
        {
            "class_name": "Dense",
            "config": {
                "name": "dense_1",
                "trainable": True,
                "units": 128,
                "activation": "relu",
                "use_bias": True
            }
        },
        {
            "class_name": "Dropout",
            "config": {
                "name": "dropout_1",
                "trainable": True,
                "rate": 0.15
            }
        },
        {
            "class_name": "Dense",
            "config": {
                "name": "output_layer",
                "trainable": True,
                "units": 50,
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
                    "name": "sign_bridge_asl50_lstm",
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
    
    model_json_path = os.path.join(model_dir, "model.json")
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(tfjs_manifest, f, indent=2)
        
    labels_manifest = {
        "label_to_index": {lbl: i for i, lbl in enumerate(CLASSES_50)},
        "index_to_label": {str(i): lbl for i, lbl in enumerate(CLASSES_50)},
        "label_to_id": {lbl: i for i, lbl in enumerate(CLASSES_50)},
        "id_to_label": {str(i): lbl for i, lbl in enumerate(CLASSES_50)},
        "num_classes": 50
    }
    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(labels_manifest, f, indent=2)
        
    feature_spec = {
        "featureCount": 63,
        "sequenceLength": 30,
        "datasetVersion": "asl50_v1",
        "modelVersion": "v0.9.0-asl50"
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
    
    print(f"\nSuccessfully trained, exported, and deployed 50-Class ASL model to {public_dir}")

if __name__ == "__main__":
    main()
