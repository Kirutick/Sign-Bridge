#!/usr/bin/env bash
# Bundles model folder artifacts into a self-contained zip archive for deployment handoff (§8)

MODEL_VERSION=${1}

if [ -z "$MODEL_VERSION" ]; then
  MODEL_VERSION=$(ls -t ml/models/ | head -n 1)
fi

MODEL_DIR="ml/models/$MODEL_VERSION"

if [ ! -d "$MODEL_DIR" ]; then
  echo "❌ ERROR: Model folder not found: $MODEL_DIR"
  exit 1
fi

ZIP_PATH="ml/models/${MODEL_VERSION}_bundle.zip"

echo "📦 Bundling model artifacts from $MODEL_DIR into $ZIP_PATH..."

python -c "
import zipfile, os
model_dir = '$MODEL_DIR'
zip_path = '$ZIP_PATH'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, model_dir)
            zipf.write(full_path, arcname)
print(f'✅ Successfully created deployment artifact bundle: {zip_path}')
"
