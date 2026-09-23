import os
import tensorflow as tf
from tensorflow.keras import layers, models
from ml.training.config import TrainingConfig

def build_lstm_model(num_classes: int, config: TrainingConfig) -> tf.keras.Model:
  """
  Constructs baseline Keras LSTM classifier matching specification §5.
  """
  model = models.Sequential([
    layers.Input(shape=(config.sequence_length, config.feature_count), name="input_sequence"),
    layers.LSTM(config.lstm_units_1, return_sequences=True, name="lstm_layer_1"),
    layers.Dropout(config.dropout_1, name="dropout_1"),
    layers.LSTM(config.lstm_units_2, return_sequences=False, name="lstm_layer_2"),
    layers.Dropout(config.dropout_2, name="dropout_2"),
    layers.Dense(config.dense_units, activation="relu", name="dense_dense_1"),
    layers.Dropout(config.dropout_3, name="dropout_3"),
    layers.Dense(num_classes, activation="softmax", name="output_classifier"),
  ], name="SignBridge_LSTM_Baseline")

  optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
  model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
  )

  return model

def save_model_summary(model: tf.keras.Model, output_path: str):
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  string_list = []
  model.summary(print_fn=lambda x: string_list.append(x))
  summary_string = "\n".join(string_list)
  
  with open(output_path, "w", encoding="utf-8") as f:
    f.write(summary_string)
