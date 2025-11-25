#!/bin/bash
# Setup script for Experiment 3 - Run once when you have internet connection

echo "=========================================="
echo "Experiment 3 Setup: Downloading required data"
echo "=========================================="

# Activate conda environment
source /home/medal/anupam.rawat/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# 1. Download CIFAR-10 dataset
echo ""
echo "Step 1/2: Downloading CIFAR-10 dataset..."
python download_cifar10.py

if [ $? -ne 0 ]; then
    echo "✗ Failed to download CIFAR-10"
    exit 1
fi

# 2. Download CLIP model
echo ""
echo "Step 2/2: Downloading CLIP model..."
python download_clip_model.py

if [ $? -ne 0 ]; then
    echo "✗ Failed to download CLIP model"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Verification:"
echo "  CIFAR-10: $(ls -d data/cifar-10-batches-py 2>/dev/null && echo '✓ Found' || echo '✗ Not found')"
echo "  CLIP model: $(ls -d /home/medal/anupam.rawat/clip_model/config.json 2>/dev/null && echo '✓ Found' || echo '✗ Not found')"
echo ""
echo "You can now run: sbatch job.sh"
