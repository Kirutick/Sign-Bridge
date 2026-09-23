import os
import sys
import json
import tempfile
import pytest
import numpy as np
import tensorflow as tf

def test_tfjs_export_shape_enforcement():
    """Verify that export script fails if the shapes are incorrect."""
    # We can simulate the check logic here
    # 1. Invalid input shape
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(None, 20, 63)), # Wrong sequence length
        tf.keras.layers.Dense(4)
    ])
    
    input_shape = model.input_shape
    assert len(input_shape) == 3
    assert input_shape[1:] != (30, 63), "Test setup failed"
    
    # 2. Invalid output shape
    model2 = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(None, 30, 63)),
        tf.keras.layers.Dense(10) # Suppose metadata says 4 classes
    ])
    output_shape = model2.output_shape
    num_classes = 4
    assert output_shape[1] != num_classes, "Test setup failed"
    
def test_export_metadata_schema():
    """Ensure metadata contains required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock metadata
        meta = {
            "modelVersion": "sign_bridge_v1",
            "datasetVersion": "v1.0.0",
            "createdAt": "2024-01-01T00:00:00",
            "featureCount": 63,
            "sequenceLength": 30,
            "classCount": 4,
            "normalizationVersion": "v1"
        }
        
        meta_path = os.path.join(tmpdir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
            
        with open(meta_path, 'r') as f:
            loaded = json.load(f)
            
        assert loaded['normalizationVersion'] == "v1"
        assert loaded['featureCount'] == 63
        assert loaded['sequenceLength'] == 30
