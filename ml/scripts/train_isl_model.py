"""
Sign Bridge — 50-Class Indian Sign Language (ISL) Model Training & Pure TFJS Export
"""

import os
import sys
import json
import shutil
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SIGN_LANGUAGE = "ISL"
SIGN_LANGUAGE_NAME = "Indian Sign Language"

OUTPUT_LANGUAGE = "English"
OUTPUT_LANGUAGE_CODE = "en"

CLASSES_ISL_50 = [
    # 26 Letters (ISL Alphabet)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    # 10 Numbers
    "NUM_0", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5", "NUM_6", "NUM_7", "NUM_8", "NUM_9",
    # 14 ISL Signs (Represented with English Concept Labels)
    "HELLO", "THANK_YOU", "YES", "NO", "HELP", "PLEASE", "SORRY",
    "GOODBYE", "WATER", "FOOD", "STOP", "GO", "AND", "INDIA"
]

def build_isl50_model(input_shape=(30, 63), num_classes=50):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape, name="input_layer"),
        tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.15, name="lstm_1"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.15, name="dropout_1"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")
    ], name="sign_bridge_isl50_lstm")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    print("\n" + "="*70)
    print(" TRAINING INDIAN SIGN LANGUAGE (ISL) 50-CLASS MODEL")
    print("="*70 + "\n")
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    dataset_path = "ml/data/processed/dataset_isl50_v1.npz"
    if len(sys.argv) > 1 and sys.argv[1].endswith(".npz"):
        dataset_path = sys.argv[1]

    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    meta = data["sample_meta"]

    is_synthetic = True
    if len(meta) > 0 and isinstance(meta[0], dict):
        is_synthetic = meta[0].get("isSynthetic", True)

    if is_synthetic:
        print("*" * 70)
        print(" [!] NOTICE: TRAINING ON SYNTHETIC DEMO DATASET")
        print(" [!] This dataset was procedurally generated using kinematic sine curves.")
        print(" [!] Real-world recognition accuracy on native ISL signers will be 0.0%.")
        print(" [!] Use ml/scripts/ingest_real_isl_dataset.py to ingest real human recordings.")
        print("*" * 70 + "\n")
    else:
        print("[+] Training on verified REAL Indian Sign Language dataset.\n")
    
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
    
    print(f"ISL Partitions: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    
    model = build_isl50_model(input_shape=(30, 63), num_classes=50)
    
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
    
    epochs_run = len(history.history["loss"])
    
    # Evaluation
    test_probs = model.predict(X_test, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    
    metric_label = "DEMO / SYNTHETIC METRICS (Real ISL Accuracy: 0.0% Pending Real Data)" if is_synthetic else "REAL ISL EVALUATION METRICS"
    print(f"\n[+] ISL Model Training Finished in {epochs_run} epochs:")
    print(f"    Evaluation Type:        {metric_label}")
    print(f"    Held-Out Test Accuracy: {test_acc*100:.2f}%")
    print(f"    Held-Out Test Macro F1: {test_macro_f1:.4f}")
    
    # Save Artifacts
    out_dir = "models/sign_bridge_v1_isl50"
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.keras"))
    
    # Export Pure TFJS
    public_dir = "public/models/sign-model-isl"
    os.makedirs(public_dir, exist_ok=True)
    
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
    bin_path = os.path.join(public_dir, "group1-shard1of1.bin")
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
                    "name": "sign_bridge_isl50_lstm",
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
    
    model_json_path = os.path.join(public_dir, "model.json")
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(tfjs_manifest, f, indent=2)
        
    labels_manifest = {
        "signLanguage": SIGN_LANGUAGE_NAME,
        "signLanguageCode": SIGN_LANGUAGE,
        "outputLanguage": OUTPUT_LANGUAGE,
        "outputLanguageCode": OUTPUT_LANGUAGE_CODE,
        "isSynthetic": is_synthetic,
        "datasetType": "DEMO_SYNTHETIC" if is_synthetic else "REAL_HUMAN_RECORDINGS",
        "validationStatus": "UNVERIFIED_DEMO" if is_synthetic else "VALIDATED_ISL",
        "modelStatus": "PROTOTYPE_DEMO" if is_synthetic else "VALIDATED_PRODUCTION",
        "notice": "Trained on synthetic demo kinematics. Real ISL training data required for production recognition." if is_synthetic else "Trained on verified real human ISL recordings.",
        "language": "Indian Sign Language (ISL)",
        "language_code": "isl",
        "label_to_index": {lbl: i for i, lbl in enumerate(CLASSES_ISL_50)},
        "index_to_label": {str(i): lbl for i, lbl in enumerate(CLASSES_ISL_50)},
        "label_to_id": {lbl: i for i, lbl in enumerate(CLASSES_ISL_50)},
        "id_to_label": {str(i): lbl for i, lbl in enumerate(CLASSES_ISL_50)},
        "num_classes": 50
    }
    with open(os.path.join(public_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(labels_manifest, f, indent=2)
        
    feature_spec = {
        "featureCount": 63,
        "sequenceLength": 30,
        "datasetVersion": "isl50_v1",
        "modelVersion": "v1.0.0-isl50-en",
        "signLanguage": SIGN_LANGUAGE_NAME,
        "signLanguageCode": SIGN_LANGUAGE,
        "outputLanguage": OUTPUT_LANGUAGE,
        "outputLanguageCode": OUTPUT_LANGUAGE_CODE,
        "isSynthetic": is_synthetic,
        "datasetType": "DEMO_SYNTHETIC" if is_synthetic else "REAL_HUMAN_RECORDINGS",
        "validationStatus": "UNVERIFIED_DEMO" if is_synthetic else "VALIDATED_ISL",
        "modelStatus": "PROTOTYPE_DEMO" if is_synthetic else "VALIDATED_PRODUCTION",
        "notice": "Trained on synthetic demo kinematics. Real ISL training data required for production recognition." if is_synthetic else "Trained on verified real human ISL recordings.",
        "language": "ISL"
    }
    with open(os.path.join(public_dir, "feature_spec.json"), "w", encoding="utf-8") as f:
        json.dump(feature_spec, f, indent=2)
        
    print(f"✅ Exported pure TFJS Indian Sign Language model to {public_dir}\n")

if __name__ == "__main__":
    main()
