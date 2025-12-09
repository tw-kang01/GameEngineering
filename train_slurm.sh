#!/usr/bin/bash
#SBATCH -J trackerbody-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g8
#SBATCH -t 1-0
#SBATCH -o /data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch/Models/logs/slurm-%A.out

hostname
nvidia-smi

# =============================================================================
# TrackerBody MLP (GMD + SlowFast) Training Script
# =============================================================================

# 로그 디렉토리 생성
mkdir -p /data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch/Models/logs

# 로컬 디스크에 압축 해제 (필요한 경우)
LOCAL_DIR=/local_datasets/ktw3389
mkdir -p $LOCAL_DIR

# TrackerBody 데이터셋 압축 해제 (필요한 경우)
# 구조: Trackerbodypredictor/Input.bin, Output.bin, *Shape.txt, *Normalization.txt
if [ -f "/data/datasets/trackerbody_data.tar" ] && [ ! -d "$LOCAL_DIR/Trackerbodypredictor" ]; then
    echo "Extracting trackerbody_data.tar..."
    tar -xf /data/datasets/trackerbody_data.tar -C $LOCAL_DIR/
fi

# body_models.tar 풀기 (SMPL-H 모델, 필요한 경우)
if [ -f "/data/datasets/body_models.tar" ] && [ ! -d "$LOCAL_DIR/AvatarPoser/support_data/body_models" ]; then
    echo "Extracting body_models.tar..."
    tar -xvf /data/datasets/body_models.tar -C $LOCAL_DIR/
fi

# 디버깅: 파일 확인
echo "=== Checking workspace structure ==="
ls -la /data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch/
echo "=== End check ==="

# 코드는 NAS에서 실행 (Network_TrackerBody.py가 있는 디렉토리)
cd /data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch/Models/MultiLayerPerceptron
pwd

# Conda 활성화
source /data/ktw3389/anaconda3/etc/profile.d/conda.sh
conda activate manikin

# 필수 패키지 설치 (없으면 설치)
echo "=== Checking/Installing Required Packages ==="
pip install onnx onnxscript --quiet
echo "=== End Package Check ==="

# PYTHONPATH 설정 (Library 모듈 접근을 위해)
export PYTHONPATH="/data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch:$PYTHONPATH"

# CUDA 확인
echo "=== CUDA Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "=== End CUDA Check ===" 

# 데이터셋 경로 설정
# 옵션 1: NAS 데이터셋 사용 (기본)
DATASET_PATH="/data/ktw3389/repos/codebook/AI4Animation/game_engineering/categorical/PyTorch/Datasets/Trackerbodypredictor"

# 옵션 2: 로컬 디스크 데이터셋 사용 (더 빠른 I/O)
# DATASET_PATH="$LOCAL_DIR/Trackerbodypredictor"

# 데이터셋 존재 확인
echo "=== Checking Dataset ==="
if [ -d "$DATASET_PATH" ]; then
    echo "Dataset found at: $DATASET_PATH"
    ls -la $DATASET_PATH/
else
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please ensure the dataset exists or update DATASET_PATH"
    exit 1
fi
echo "=== End Dataset Check ==="

# 학습 실행
echo "=== Starting TrackerBody MLP Training ==="
echo "Script: Network_TrackerBody.py"
echo "Dataset: $DATASET_PATH"
echo "Features: GMD + SlowFast Enhancement"
echo "=========================================="

python Network_TrackerBody.py

echo "=== Training Complete ==="
exit 0
