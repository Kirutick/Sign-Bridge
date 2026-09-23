#!/usr/bin/env bash
# CLI wrapper for training Sign Bridge Keras LSTM Baseline model

CONFIG=${1:-"ml/training/config.baseline.json"}
RAW_INPUT=${2:-"ml/data/raw/dataset_mock.json"}

echo "🚀 Step 1: Validating Raw Dataset..."
python ml/preprocessing/validate.py --input "$RAW_INPUT"

PROCESSED_NPZ=$(ls -t ml/data/processed/*.npz | head -n 1)

echo "🔒 Step 2: Building Performer Group-Aware Split..."
python ml/preprocessing/split.py --input "$PROCESSED_NPZ"

echo "🏋️ Step 3: Running Model Training..."
python ml/training/train.py --config "$CONFIG" --npz "$PROCESSED_NPZ"
