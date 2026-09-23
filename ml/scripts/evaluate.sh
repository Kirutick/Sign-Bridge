#!/usr/bin/env bash
# CLI wrapper for evaluating a trained Sign Bridge model on held-out test split

MODEL_VERSION=${1}

if [ -z "$MODEL_VERSION" ]; then
  MODEL_VERSION=$(ls -t ml/models/ | head -n 1)
  echo "ℹ️ No model version specified. Using latest model: $MODEL_VERSION"
fi

echo "🧪 Running Evaluation for Model Version: $MODEL_VERSION..."
python ml/evaluation/evaluate.py --model-version "$MODEL_VERSION"
