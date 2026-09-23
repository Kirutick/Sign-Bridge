# Sign Bridge - Machine Learning Training Subsystem

This directory (`ml/`) contains the standalone ML training environment for Sign Bridge. It is fully decoupled from the React frontend.

## Framework Choice: TensorFlow / Keras

**TensorFlow (Keras)** was chosen over PyTorch for this baseline for the following primary reason:
The ultimate goal of this pipeline (Phase 5) is to load and run this model inside the browser via TensorFlow.js for real-time edge inference. Using Keras allows us to natively convert the saved model to TFJS format using `tfjs-converter`. PyTorch would require a much more fragile and complex ONNX -> TF -> TFJS conversion pipeline, which often suffers from missing operator implementations.

## Setup Instructions

1. Create a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install exact dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Workflow

- **Train Baseline**:

  ```bash
  python scripts/train.py --dataset auto --config configs/baseline.yaml
  ```

- **Evaluate**:

  ```bash
  python scripts/evaluate.py --model exports/<run_id> --dataset auto
  ```
