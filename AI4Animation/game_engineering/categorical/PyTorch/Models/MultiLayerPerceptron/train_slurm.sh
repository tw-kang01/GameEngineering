#!/usr/bin/bash
#SBATCH -J trackerbody-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g8
#SBATCH -t 1-0
#SBATCH -o /data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch/Models/logs/slurm-%A.out

echo "=========================================="
echo "   TrackerBody MLP Training - SLURM"
echo "=========================================="

hostname
nvidia-smi

# =============================================================================
# Step 1: Setup Directories
# =============================================================================
echo ""
echo "[Step 1] Setting up directories..."

# NAS 작업 디렉토리 (코드 실행)
WORK_DIR=/data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch
LOG_DIR=$WORK_DIR/Models/logs
mkdir -p $LOG_DIR

# 로컬 디스크 (빠른 I/O를 위한 데이터셋 캐시)
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

# TrackerBody 데이터셋 확인 및 압축 해제
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
    echo "✓ Dataset already extracted at $DATASET_LOCAL"
else
    echo "Extracting trackerbody_data.tar to $LOCAL_DIR..."
    tar -xf $DATASET_TAR -C $LOCAL_DIR/
    echo "✓ Extraction complete"
fi

# =============================================================================
# Step 3: Verify Dataset Files
# =============================================================================
echo ""
echo "[Step 3] Verifying dataset files..."

echo ""
echo "=== Dataset Details ==="

# 필수 파일 확인
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
        echo "✓ $file ($SIZE)"
    else
        echo "✗ $file NOT found"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ERROR: Missing required dataset files!"
    exit 1
fi

# 파일 크기 검증 (Input.bin은 최소 1GB 이상이어야 함)
INPUT_SIZE=$(stat -c%s "$DATASET_LOCAL/Input.bin")
if [ $INPUT_SIZE -lt 1000000000 ]; then
    echo ""
    echo "ERROR: Input.bin is too small ($INPUT_SIZE bytes)"
    echo "Expected: ~1.6GB (1644387840 bytes)"
    echo ""
    echo "The dataset may be corrupted. Please re-upload."
    exit 1
fi
echo ""
echo "✓ Input.bin size verified ($INPUT_SIZE bytes)"

# Shape 파일 내용 출력
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

# Conda 활성화
source /data/ktw3389/anaconda3/etc/profile.d/conda.sh
conda activate manikin

# 필수 패키지 설치
echo "Checking required packages..."
pip install onnx onnxscript --quiet

# PYTHONPATH 설정
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# 데이터셋 경로를 환경 변수로 설정
export TRACKERBODY_DATASET_PATH="$DATASET_LOCAL"

# CUDA 확인
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
echo "Script: Network_TrackerBody.py"
echo "Dataset: $DATASET_LOCAL"
echo "Features: GMD + SlowFast Enhancement"
echo "=========================================="
echo ""

# PYTHONUNBUFFERED: 실시간 로그 출력 (버퍼링 비활성화)
export PYTHONUNBUFFERED=1
python Network_TrackerBody.py

# =============================================================================
# Step 6: Cleanup (Optional)
# =============================================================================
echo ""
echo "[Step 6] Training complete!"
echo ""

# 결과 파일 확인
TRAINING_DIR=$(ls -td $DATASET_LOCAL/Training_* 2>/dev/null | head -1)
if [ -n "$TRAINING_DIR" ]; then
    echo "=== Training Results ==="
    echo "Output directory: $TRAINING_DIR"
    ls -la "$TRAINING_DIR" | tail -5
    echo ""
    
    # NAS로 결과 복사 (옵션)
    # cp -r "$TRAINING_DIR" "$WORK_DIR/Datasets/Trackerbodypredictor/"
fi

echo "=========================================="
echo "   Training Complete!"
echo "=========================================="
exit 0
