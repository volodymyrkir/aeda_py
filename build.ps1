$ErrorActionPreference = "Stop"

Write-Host "=========================================="
Write-Host "  AEDA - Building Standalone Executable"
Write-Host "=========================================="

python -m pip install --upgrade pip
python -m pip install -U pyinstaller

if (Test-Path "requirements.txt") {
  Write-Host "Installing dependencies..."
  python -m pip install -r requirements.txt
}

Write-Host "Building executable..."

pyinstaller `
  --onedir `
  --windowed `
  --name "AEDA" `
  --add-data "core;core" `
  --add-data "preprocessing;preprocessing" `
  --add-data "report_components;report_components" `
  --add-data "utils;utils" `
  --collect-all torch `
  --collect-all transformers `
  --collect-all accelerate `
  --collect-all tokenizers `
  --hidden-import sklearn.ensemble._iforest `
  --hidden-import sklearn.tree._tree `
  --hidden-import shap.explainers._tree `
  --noconfirm `
  --clean `
  ui_app.py

if (Test-Path "dist\AEDA") {
  Write-Host ""
  Write-Host "=========================================="
  Write-Host "  BUILD SUCCESSFUL!"
  Write-Host "=========================================="
  Write-Host ""
  Write-Host "Your app is at: dist\AEDA\"
} else {
  Write-Host "Build failed"
  exit 1
}
