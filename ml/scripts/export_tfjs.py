import os
import sys
import json
import argparse
import shutil
import datetime
from pathlib import Path

import tensorflow as tf
import sys
import numpy as np
np.object = np.object_
np.bool = np.bool_
class MockHub:
    pass
sys.modules['tensorflow_hub'] = MockHub()
import tensorflow as tf
import tensorflowjs as tfjs

def get_next_version_dir(base_dir: Path) -> Path:
    version = 1
    while True:
        v_dir = base_dir / f"sign_bridge_v{version}"
        if not v_dir.exists():
            return v_dir
        version += 1

def main():
    parser = argparse.ArgumentParser(description="Export Keras model to TensorFlow.js")
    parser.add_argument("--source", type=str, required=True, help="Path to the model directory (e.g. from Training 3)")
    parser.add_argument("--outdir", type=str, default="models", help="Base output directory for models")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    meta_path = source_dir / "checkpoint_meta.json"
    weights_path = source_dir / "best_model.keras"
    eval_report_path = source_dir / "EVALUATION_REPORT.md"
    
    if not meta_path.exists() or not weights_path.exists():
        print(f"Error: source directory must contain checkpoint_meta.json and best_model.keras")
        sys.exit(1)
        
    base_out_dir = Path(args.outdir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    target_dir = get_next_version_dir(base_out_dir)
    model_out_dir = target_dir / "model"
    labels_out_dir = target_dir / "labels"
    meta_out_dir = target_dir / "metadata"
    
    # 1. Load the model and verify shapes
    print(f"Loading Keras model from {weights_path}...")
    model = tf.keras.models.load_model(weights_path)
    
    input_shape = model.input_shape
    output_shape = model.output_shape
    
    print(f"Model Input Shape: {input_shape}")
    print(f"Model Output Shape: {output_shape}")
    
    if len(input_shape) != 3 or input_shape[1:] != (30, 63):
        print(f"ERROR: Expected input shape (None, 30, 63), got {input_shape}")
        sys.exit(1)
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    num_classes = len(meta['label_to_id'])
    
    if len(output_shape) != 2 or output_shape[1] != num_classes:
        print(f"ERROR: Expected output shape (None, {num_classes}), got {output_shape}")
        sys.exit(1)
        
    # 2. Export to TF.js via H5 intermediate
    print(f"Exporting to TF.js format at {model_out_dir}...")
    temp_h5 = target_dir / "temp.h5"
    
    # Save as Keras 2 H5 format (using Keras 3 legacy export)
    model.save(str(temp_h5))
    
    import h5py
    with h5py.File(str(temp_h5), 'r+') as f:
        if 'keras_version' in f.attrs:
            f.attrs['keras_version'] = b'2.15.0'
    
    from tensorflowjs.converters.converter import convert
    
    # Needs np.object and np.bool monkey patches to work
    import numpy as np
    np.object = getattr(np, 'object_', object)
    np.bool = getattr(np, 'bool_', bool)
    
    convert(['--input_format=keras', str(temp_h5), str(model_out_dir)])
    
    import os
    os.remove(str(temp_h5))
    
    # 2.5 Fix the batch_input_shape mapping for Keras 3 layers
    model_json_path = model_out_dir / 'model.json'
    with open(model_json_path, 'r') as f:
        model_data = json.load(f)
        
    for layer in model_data.get('modelTopology', {}).get('model_config', {}).get('config', {}).get('layers', []):
        if layer.get('class_name') == 'InputLayer':
            if 'batch_shape' in layer['config'] and 'batch_input_shape' not in layer['config']:
                layer['config']['batch_input_shape'] = layer['config']['batch_shape']
                
    with open(model_json_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    # 3. Write Labels
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    with open(labels_out_dir / "label_map.json", "w") as f:
        json.dump({
            "label_to_id": meta['label_to_id'],
            "id_to_label": meta['id_to_label']
        }, f, indent=2)
        
    # 4. Extract config
    # Keras models can be serialized to get their architecture
    try:
        architecture = json.loads(model.to_json())
    except Exception as e:
        architecture = "Failed to serialize architecture: " + str(e)
        
    # 5. Create metadata.json
    meta_out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "modelVersion": target_dir.name,
        "datasetVersion": meta.get('dataset_version', 'unknown'),
        "createdAt": datetime.datetime.now().isoformat(),
        "featureCount": 63,
        "sequenceLength": 30,
        "classCount": num_classes,
        "normalizationVersion": meta.get('normalization_version', 'unknown'),  # Dynamically propagated
        "architecture": architecture,
        "trainingConfiguration": meta.get('config_snapshot', {}),
        "evaluationMetrics": f"See {target_dir.name}/README.md for full report."
    }
    
    with open(meta_out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    # 6. Initialize README
    readme_path = target_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# Sign Bridge Model Release: {target_dir.name}\n\n")
        f.write("## Lineage\n")
        f.write(f"- Source Experiment: `{source_dir.name}`\n")
        f.write("- Recommendation passed via `final_eval.py` test-set verification.\n")
        f.write("\n## Preprocessing Specification\n")
        f.write("To ensure compatibility with this model, the frontend MUST perform the following steps identically:\n")
        f.write("1. **Landmark Extraction**: Extract exactly 21 hand landmarks per frame.\n")
        f.write("2. **Wrist Reference**: Landmark index 0 (Wrist) is the origin.\n")
        f.write("3. **Normalization Formula**: For each point `i`:\n")
        f.write("   `tx_i = x_i - x_0`\n")
        f.write("   `ty_i = y_i - y_0`\n")
        f.write("   `tz_i = z_i - z_0`\n")
        f.write("4. **Scaling**: Compute scale factor `S = distance(wrist, middle_finger_mcp)`. \n")
        f.write("   `S = sqrt(tx_9^2 + ty_9^2 + tz_9^2)`\n")
        f.write("   Divide every translated coordinate by `S`.\n")
        f.write("5. **Flattening Order**: For each landmark 0 through 20 sequentially, append `[nx, ny, nz]`, resulting in a flat 63-element vector per frame.\n")
        f.write("\n*Any change to the above requires a bump of `NORMALIZATION_VERSION` in both frontend and backend!*\n\n")
        
        f.write("## Evaluation Report\n")
        if eval_report_path.exists():
            with open(eval_report_path, 'r') as er:
                f.write(er.read())
        else:
            f.write("*Evaluation report not found at source.*\n")
            
        f.write("\n## Parity Test Report\n")
        f.write("PENDING (Run parity check to append results)\n")
        
    print(f"Export successful. Artifacts located in {target_dir}")

if __name__ == "__main__":
    main()
