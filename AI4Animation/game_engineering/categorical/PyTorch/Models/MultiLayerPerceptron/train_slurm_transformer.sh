#!/usr/bin/bash
#SBATCH -J trackerbody-transformer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g8
#SBATCH -t 1-0
#SBATCH -o /data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch/Models/logs/slurm-transformer-%A.out

echo "=========================================="
echo "   TrackerBody Transformer Training"
echo "   (GMD + SlowFast - EgoPoser Style)"
echo "=========================================="

hostname
nvidia-smi

# =============================================================================
# Step 1: Setup Directories
# =============================================================================
echo ""
echo "[Step 1] Setting up directories..."

WORK_DIR=/data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch
LOG_DIR=$WORK_DIR/Models/logs
mkdir -p $LOG_DIR

LOCAL_DIR=/local_datasets/ktw3389
mkdir -p $LOCAL_DIR

echo "Work directory: $WORK_DIR"
echo "Log directory: $LOG_DIR"
echo "Local cache: $LOCAL_DIR"

# =============================================================================
# Step 2: Extract Dataset to Local Disk
# =============================================================================
echo ""
echo "[Step 2] Verifying/Extracting dataset on local disk..."

DATASET_TAR=/data/datasets/trackerbody_data.tar
DATASET_LOCAL=$LOCAL_DIR/Trackerbodypredictor

if [ ! -f "$DATASET_TAR" ]; then
    echo "ERROR: Dataset tar not found at $DATASET_TAR"
    echo ""
    echo "Please upload the dataset first:"
    echo "  1. Create tar on local machine:"
    echo "     cd c:\\Users\\KTW\\GameEngineering\\AI4Animation\\game_engineering\\categorical\\PyTorch\\Datasets"
    echo "     tar -cvf trackerbody_data.tar Trackerbodypredictor/"
    echo ""
    echo "  2. Upload to server:"
    echo "     scp trackerbody_data.tar ktw3389@aurora.khu.ac.kr:/data/datasets/"
    echo ""
    exit 1
fi

if [ -d "$DATASET_LOCAL" ]; then
    echo "Dataset already extracted at $DATASET_LOCAL"
else
    echo "Extracting trackerbody_data.tar to $LOCAL_DIR..."
    tar -xf $DATASET_TAR -C $LOCAL_DIR/
    echo "Extraction complete"
fi

# =============================================================================
# Step 3: Verify Dataset Files
# =============================================================================
echo ""
echo "[Step 3] Verifying dataset files..."

REQUIRED_FILES=(
    "Input.bin"
    "Output.bin"
    "InputShape.txt"
    "OutputShape.txt"
    "InputNormalization.txt"
    "OutputNormalization.txt"
)

ALL_OK=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DATASET_LOCAL/$file" ]; then
        SIZE=$(ls -lh "$DATASET_LOCAL/$file" | awk '{print $5}')
        echo "  $file ($SIZE)"
    else
        echo "  $file NOT found"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ERROR: Missing required dataset files!"
    exit 1
fi

INPUT_SIZE=$(stat -c%s "$DATASET_LOCAL/Input.bin")
if [ $INPUT_SIZE -lt 1000000000 ]; then
    echo ""
    echo "ERROR: Input.bin is too small ($INPUT_SIZE bytes)"
    exit 1
fi
echo ""
echo "Input.bin size verified ($INPUT_SIZE bytes)"

echo ""
echo "=== Shape Information ==="
echo -n "Input Shape: "
cat "$DATASET_LOCAL/InputShape.txt"
echo ""
echo -n "Output Shape: "
cat "$DATASET_LOCAL/OutputShape.txt"
echo ""

# =============================================================================
# Step 4: Setup Python Environment
# =============================================================================
echo ""
echo "[Step 4] Setting up Python environment..."

cd $WORK_DIR/Models/MultiLayerPerceptron
pwd

source /data/ktw3389/anaconda3/etc/profile.d/conda.sh
conda activate manikin

echo "Checking required packages..."
pip install onnx onnxscript --quiet

export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
export TRACKERBODY_DATASET_PATH="$DATASET_LOCAL"

echo ""
echo "=== Environment Check ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
import os
print(f'Dataset path: {os.environ.get(\"TRACKERBODY_DATASET_PATH\", \"NOT SET\")}')"
echo "==========================="

# =============================================================================
# Step 5: Run Training
# =============================================================================
echo ""
echo "[Step 5] Starting training..."
echo ""
echo "=========================================="
echo "Script: Network_TrackerBody_Transformer.py"
echo "Dataset: $DATASET_LOCAL"
echo "Architecture: Transformer (EgoPoser Style)"
echo "Features: GMD + SlowFast"
echo ""
echo "Hyperparameters:"
echo "  - Embed dim: 256"
echo "  - Heads: 4"
echo "  - Layers: 4"
echo "  - Epochs: 150"
echo "  - Batch size: 64"
echo "=========================================="
echo ""

export PYTHONUNBUFFERED=1
python Network_TrackerBody_Transformer.py

# =============================================================================
# Step 6: Cleanup
# =============================================================================
echo ""
echo "[Step 6] Training complete!"
echo ""

TRAINING_DIR=$(ls -td $DATASET_LOCAL/Training_*Transformer* 2>/dev/null | head -1)
if [ -n "$TRAINING_DIR" ]; then
    echo "=== Training Results ==="
    echo "Output directory: $TRAINING_DIR"
    ls -la "$TRAINING_DIR" | tail -5
    echo ""
fi

echo "=========================================="
echo "   Transformer Training Complete!"
echo "=========================================="
exit 0
