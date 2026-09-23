from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from typing import Dict, Any

def build_baseline_lstm(
    num_classes: int, 
    sequence_length: int = 30, 
    feature_count: int = 63,
    config: Dict[str, Any] = None
) -> Sequential:
    """
    Builds the baseline LSTM architecture.
    Configurable via the provided config dictionary.
    """
    if config is None:
        config = {}
        
    lstm_units = config.get('lstm_units', 64)
    dropout_rate = config.get('dropout_rate', 0.5)
    dense_units = config.get('dense_units', 64)
    layers = config.get('layers', 1)
    
    model = Sequential()
    model.add(Input(shape=(sequence_length, feature_count)))
    
    for i in range(layers):
        return_seq = (i < layers - 1)
        model.add(LSTM(lstm_units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model
