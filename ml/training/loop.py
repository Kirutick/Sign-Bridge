import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from sklearn.metrics import f1_score # type: ignore

logger = logging.getLogger(__name__)

class SweepMetricsCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        X_val, y_val = self.validation_data
        
        # Predict on validation data
        y_pred_probs = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate Macro F1
        val_f1 = f1_score(y_val, y_pred, average='macro')
        logs['val_f1_macro'] = val_f1
        # It's helpful to print it so we can see it during training
        print(f" - val_f1_macro: {val_f1:.4f}")

def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    run_id: str,
    config: Dict[str, Any],
    seed: int,
    split_strategy: str
) -> Tuple[Sequential, Dict[str, Any]]:
    """
    Compiles and trains the Keras model.
    Saves best weights to ml/exports/<run_id>/best_model.keras.
    Saves training history to ml/experiments/<run_id>/history.json.
    """
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 50)
    learning_rate = config.get('learning_rate', 0.001)
    
    # Paths
    export_dir = Path("ml/exports") / run_id
    export_dir.mkdir(parents=True, exist_ok=True)
    
    exp_dir = Path("ml/experiments") / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = export_dir / "best_model.keras"
    history_path = exp_dir / "history.json"
    config_path = exp_dir / "config_snapshot.json"
    
    # Save a snapshot of the config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(weights_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        ),
        SweepMetricsCallback(validation_data=(X_val, y_val))
    ]
    
    logger.info(f"Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save history
    hist_dict = history.history
    
    # Convert numpy types to native Python types for JSON serialization
    for k, v in hist_dict.items():
        hist_dict[k] = [float(val) for val in v]
        
    hist_dict['_meta'] = {
        'seed': seed,
        'split_strategy': split_strategy,
        'parameter_count': model.count_params()
    }
    
    with open(history_path, 'w') as f:
        json.dump(hist_dict, f, indent=2)
        
    return model, hist_dict
