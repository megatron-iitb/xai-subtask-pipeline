#!/bin/bash
#SBATCH --job-name=Exp_3_Improved
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/medal/anupam.rawat/Experiment_3/logs/exp3_improved_%j.log
#SBATCH --error=/home/medal/anupam.rawat/Experiment_3/logs/exp3_improved_%j.err

# Experiment 3 Improved: Accountable XAI Pipeline with CIFAR-10
# Improvements: Better subtasks, validation split, deeper meta-learner, MC-dropout, optimized SHAP

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

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

# Set cache directories to use local pre-downloaded models
export HF_HOME=/home/medal/anupam.rawat/.cache/huggingface
export TORCH_HOME=/home/medal/anupam.rawat/.cache/torch
export HF_HUB_CACHE=/home/medal/anupam.rawat/.cache/huggingface/hub

# Path to local CLIP model (if available)
CLIP_MODEL_PATH="/home/medal/anupam.rawat/clip_model"

# Path to local CIFAR-10 dataset
CIFAR_DATA_PATH="/home/medal/anupam.rawat/Experiment_3/data"

# Navigate to experiment directory
cd /home/medal/anupam.rawat/Experiment_3

# Create artifacts directory
mkdir -p artifacts

echo "=========================================="
echo "Starting Experiment 3 Improved (Optimized Hyperparameters)..."
echo "Configuration:"
echo "  - Epochs: 15"
echo "  - Batch size: 256"
echo "  - Learning rate: 5e-4"
echo "  - MC-Dropout samples: 30"
echo "  - Early stopping patience: 5"
echo "  - Hidden layer size: 384"
echo "  - Improvements: Edge-based complexity, larger capacity"
echo "=========================================="

# Check for required local files
if [ ! -d "$CIFAR_DATA_PATH/cifar-10-batches-py" ]; then
    echo "✗ ERROR: CIFAR-10 dataset not found at: $CIFAR_DATA_PATH"
    echo "Please run: python download_cifar10.py (when online)"
    exit 1
fi

if [ ! -d "$CLIP_MODEL_PATH" ]; then
    echo "✗ WARNING: CLIP model not found at: $CLIP_MODEL_PATH"
    echo "Attempting to use cached model from HuggingFace cache..."
fi

echo "Using local CIFAR-10 dataset from: $CIFAR_DATA_PATH"
echo "Using local CLIP model from: $CLIP_MODEL_PATH"

# Run the improved experiment with optimized hyperparameters
python experiment_3_improved.py \
    --epochs 15 \
    --batch_size 256 \
    --lr 5e-4 \
    --device cuda \
    --data_root "$CIFAR_DATA_PATH" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --patience 5 \
    --mc_samples 30

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment completed successfully!"
    echo "✓ Check artifacts/ directory for outputs"
    echo "✓ Model artifacts: artifacts/model/"
    echo "✓ Audit log: artifacts/audit_log.jsonl"
    echo "✓ Registry: artifacts/registry.json"
else
    echo "✗ Experiment failed with exit code: $EXIT_CODE"
fi
echo "Job ended at: $(date)"
echo "=========================================="

exit $EXIT_CODE
