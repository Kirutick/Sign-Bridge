# Model Card: Sign Bridge (v2)

## Model Details
- **Architecture**: LSTM (Long Short-Term Memory) sequence classifier.
- **Task**: Real-time sign language gesture recognition from MediaPipe Hand Landmarks.
- **Format**: TensorFlow.js LayersModel (exported from Keras 3).

## Intended Use
Designed to run entirely client-side in the browser via `@tensorflow/tfjs`. Used to predict a gesture from a 30-frame sequence of 63-dimensional normalized hand landmarks.

## Inputs
- **Shape**: `[batch_size, 30, 63]` (sequence_length=30, features=63)
- **Feature Space**: 21 3D landmarks (x, y, z) per frame, translated relative to the wrist and scaled by the wrist-to-middle-MCP distance. See `NORMALIZATION_VERSION = "v1"` in `src/services/landmarkProcessor.ts`.

## Outputs
- **Shape**: `[batch_size, 4]` (assuming 4 classes)
- **Interpretation**: Softmax probabilities for each gesture class.

## Evaluation Data & Metrics
See `README.md` in this directory for the full test-set evaluation report and error analysis.

## Parity
This model was evaluated against strict mathematical parity (1e-4 tolerance) between the Python runtime and the JavaScript runtime, ensuring that conversion to JS did not introduce numerical drift.

## Maintenance
Model updates should be placed in incrementally versioned directories (`sign_bridge_v3`, etc.). Any change to normalization logic REQUIRES bumping `NORMALIZATION_VERSION`.
