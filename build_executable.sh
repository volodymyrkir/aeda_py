#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "  AEDA - Building Standalone Executable"
echo "=========================================="

python -m pip install --upgrade pip
python -m pip install -U pyinstaller

if [ -f "requirements.txt" ]; then
  echo "Installing dependencies..."
  python -m pip install -r requirements.txt
fi

echo "Building executable..."

PYI_ADD_DATA_SEP=":"
if [[ "${OS:-}" == "Windows_NT" ]]; then
  PYI_ADD_DATA_SEP=";"
fi

pyinstaller \
  --onedir \
  --windowed \
  --name "AEDA" \
  --add-data "core${PYI_ADD_DATA_SEP}core" \
  --add-data "preprocessing${PYI_ADD_DATA_SEP}preprocessing" \
  --add-data "report_components${PYI_ADD_DATA_SEP}report_components" \
  --add-data "utils${PYI_ADD_DATA_SEP}utils" \
  --collect-all torch \
  --collect-all transformers \
  --collect-all accelerate \
  --collect-all tokenizers \
  --hidden-import sklearn.ensemble._iforest \
  --hidden-import sklearn.tree._tree \
  --hidden-import shap.explainers._tree \
  --noconfirm \
  --clean \
  ui_app.py

if [ -d "dist/AEDA" ]; then
  echo ""
  echo "=========================================="
  echo "  ✅ BUILD SUCCESSFUL!"
  echo "=========================================="
  echo ""
  echo "  Your app is at: dist/AEDA/"
  echo ""
  echo "  To share:"
  echo "  1. cd dist && zip -r AEDA_linux.zip AEDA/"
  echo "  2. Send AEDA_linux.zip"
  echo "  3. Extract and runs: ./AEDA/AEDA"
  echo ""
  echo "  Note: the LLM weights are NOT bundled by default."
  echo "  They’ll download on first run and be cached on your machine."
  echo "=========================================="
else
  echo "❌ Build failed"
  exit 1
fi
