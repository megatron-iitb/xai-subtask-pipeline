#!/bin/bash
#SBATCH --job-name=Exp3_OptA
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/medal/anupam.rawat/Experiment_3/logs/optA_%j.log
#SBATCH --error=/home/medal/anupam.rawat/Experiment_3/logs/optA_%j.err

# OPTION A: Best from Run 1 (batch=128, lr=1e-3) + Edge Complexity
# Expected: 95.5-96% accuracy, ECE < 2%, complexity ~65%

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Setup CUDA paths
export CUDA_HOME=/home/medal/anupam.rawat/miniconda3/envs/myenv
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Activate conda environment
source /home/medal/anupam.rawat/miniconda3/etc/profile.d/conda.sh
conda activate myenv

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Enable offline mode for HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set cache directories
export HF_HOME=/home/medal/anupam.rawat/.cache/huggingface
export TORCH_HOME=/home/medal/anupam.rawat/.cache/torch
export HF_HUB_CACHE=/home/medal/anupam.rawat/.cache/huggingface/hub

# Path to local models
CLIP_MODEL_PATH="/home/medal/anupam.rawat/clip_model"
CIFAR_DATA_PATH="/home/medal/anupam.rawat/Experiment_3/data"

# Navigate to experiment directory
cd /home/medal/anupam.rawat/Experiment_3

# Create artifacts directory
mkdir -p artifacts_optA

echo "=========================================="
echo "OPTION A: Optimal Hyperparameters"
echo "Configuration:"
echo "  - Strategy: Best from Run 1 + Edge complexity"
echo "  - Batch size: 128"
echo "  - Learning rate: 1e-3"
echo "  - Epochs: 12 (early stop patience=4)"
echo "  - MC-Dropout samples: 25"
echo "  - Hidden size: 384"
echo "  - Expected: 95.5-96% acc, ECE < 2%"
echo "=========================================="

# Check for required files
if [ ! -d "$CIFAR_DATA_PATH/cifar-10-batches-py" ]; then
    echo "✗ ERROR: CIFAR-10 dataset not found"
    exit 1
fi

if [ ! -d "$CLIP_MODEL_PATH" ]; then
    echo "✗ WARNING: CLIP model not found"
fi

echo "Using local CIFAR-10: $CIFAR_DATA_PATH"
echo "Using local CLIP: $CLIP_MODEL_PATH"

# Run Option A
python experiment_3_optA.py \
    --epochs 12 \
    --batch_size 128 \
    --lr 1e-3 \
    --device cuda \
    --data_root "$CIFAR_DATA_PATH" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --patience 4 \
    --mc_samples 25

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Option A completed successfully!"
    echo "✓ Check artifacts_optA/ for outputs"
    
    # Display final metrics
    echo ""
    echo "Final Results:"
    cat artifacts_optA/audit_log.jsonl | grep -o '"accuracy":[^,]*' | tail -1
    cat artifacts_optA/audit_log.jsonl | grep -o '"ece":[^,]*' | tail -1
    cat artifacts_optA/audit_log.jsonl | grep -o '"mean_uncertainty":[^,]*' | tail -1
else
    echo "✗ Option A failed with exit code: $EXIT_CODE"
fi
echo "Job ended at: $(date)"
echo "=========================================="

exit $EXIT_CODE
