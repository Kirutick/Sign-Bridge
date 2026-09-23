import os
import json
import argparse
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description="Generate Parity Test Fixtures")
    parser.add_argument("--model", type=str, required=True, help="Path to keras model (best_model.keras)")
    parser.add_argument("--out", type=str, required=True, help="Output json fixture path")
    args = parser.parse_args()
    
    print(f"Loading Keras model from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    
    # Generate 5 random sequences per class (4 classes = 20 sequences)
    # The shapes are (1, 30, 63)
    
    fixtures = []
    
    for i in range(20):
        # We generate random values between -5 and 5 to simulate normalized coordinates
        input_tensor = np.random.uniform(-5, 5, size=(1, 30, 63)).astype(np.float32)
        
        # Predict using the Python model
        pred_probs = model.predict(input_tensor, verbose=0)
        
        fixtures.append({
            "id": f"seq_{i}",
            "input": input_tensor.tolist(),
            "expected_probs": pred_probs.tolist()
        })
        
    with open(args.out, "w") as f:
        json.dump(fixtures, f, indent=2)
        
    print(f"Generated {len(fixtures)} fixtures at {args.out}")

if __name__ == "__main__":
    main()
