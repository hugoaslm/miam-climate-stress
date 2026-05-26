#!/bin/bash
# setup_colab.sh — Run as first cell in Colab
# Clones MIAM, installs deps, downloads checkpoints, and sets up directories.

set -e

echo "=== MIAM Climate Stress Test Setup ==="

# Clone MIAM repo
if [ ! -d "MIAM" ]; then
    echo "Cloning MIAM..."
    git clone https://github.com/zbirobin/MIAM.git
fi
cd MIAM

# Create venv and install
echo "Installing dependencies..."
pip install -q -r requirements.txt
pip install -q -e .

# Additional deps
pip install -q huggingface_hub kagglehub jsonlines seaborn verde

# Download checkpoints
echo "Downloading pretrained checkpoints..."
hf download zbirobin/MIAM \
    geoplant_miam.pt \
    geoplant_dropout.pt \
    geoplant_constant.pt

# Organize checkpoints
mkdir -p models/{miam,dropout,constant}
mv geoplant_miam.pt models/miam/miam.pt 2>/dev/null || true
mv geoplant_dropout.pt models/dropout/dropout.pt 2>/dev/null || true
mv geoplant_constant.pt models/constant/constant.pt 2>/dev/null || true

# Create project directories
cd ..
mkdir -p src scripts results figures notebooks

# Copy scripts if not present (assumes scripts/ exists alongside this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../src/stress_transforms.py" ]; then
    cp "$SCRIPT_DIR/../src/stress_transforms.py" src/
    cp "$SCRIPT_DIR/../scripts/evaluate_climate_stress.py" scripts/
    cp "$SCRIPT_DIR/../scripts/plot_climate_stress.py" scripts/
fi

echo ""
echo "=== Setup Complete ==="
echo "Next: Download GeoPlant data (CELL 2) or skip to fallback (CELL 8)"
echo "Checkpoints at: MIAM/models/{miam,dropout,constant}/"
