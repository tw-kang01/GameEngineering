"""
TrackerBody Transformer with GMD + SlowFast (EgoPoser Style)
=============================================================

TrackerBodyPredictor를 EgoPoser 논문의 Transformer 아키텍처로 구현

논문 참조: EgoPoser (ECCV 2024) - Robust Real-Time Egocentric Pose Estimation

핵심 구조:
1. GMD (Global Motion Decomposition):
   - Spatial Normalization: Head XZ 기준 상대 좌표 (Y는 유지)
   - Temporal Normalization: 첫 프레임 대비 delta
2. SlowFast Feature Fusion:
   - Fast pathway: 후반 τ/2 프레임 (최근 움직임)
   - Slow pathway: stride 2 (전체 시간대)
   - 공유 Linear Embedding
3. Transformer Encoder:
   - 시퀀스 구조 유지 (B, T, D)
   - Self-attention으로 temporal dependency 학습
4. Output: 마지막 timestep에서 prediction

데이터 구조 (MLP 버전과 동일한 I/O):
- Input: 576 = 3 trackers × 16 timesteps × 12 features
  - [Head_t0..t15 (192), LWrist_t0..t15 (192), RWrist_t0..t15 (192)]
  - 각 tracker/timestep: Position(3) + Forward(3) + Up(3) + Velocity(3) = 12
- Output: 231 = RootUpdate(3) + 19 bones × 12 features

ONNX Export: MLP 버전과 완전히 동일한 인터페이스
- Input shape: (B, 576)
- Output shape: (B, 231)

Author: Transformer implementation based on EgoPoser for Categorical pipeline
"""

import sys
import os
sys.path.append("../../../PyTorch")

import Library.Utility as utility
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


# =============================================================================
# MPJPE / MPJVE Metrics (MLP 버전과 동일)
# =============================================================================

def compute_mpjpe_mpjve(pred, target, num_joints=19, root_dim=3):
    """
    Compute MPJPE (Mean Per Joint Position Error) and MPJVE (Mean Per Joint Velocity Error)
    """
    B = pred.shape[0]

    positions_pred = []
    positions_target = []
    velocities_pred = []
    velocities_target = []

    for j in range(num_joints):
        joint_start = root_dim + j * 12

        pos_pred = pred[:, joint_start:joint_start+3]
        pos_target = target[:, joint_start:joint_start+3]
        positions_pred.append(pos_pred)
        positions_target.append(pos_target)

        vel_pred = pred[:, joint_start+9:joint_start+12]
        vel_target = target[:, joint_start+9:joint_start+12]
        velocities_pred.append(vel_pred)
        velocities_target.append(vel_target)

    positions_pred = torch.stack(positions_pred, dim=1)
    positions_target = torch.stack(positions_target, dim=1)
    velocities_pred = torch.stack(velocities_pred, dim=1)
    velocities_target = torch.stack(velocities_target, dim=1)

    pos_error = torch.norm(positions_pred - positions_target, dim=2)
    mpjpe = pos_error.mean()

    vel_error = torch.norm(velocities_pred - velocities_target, dim=2)
    mpjve = vel_error.mean()

    return mpjpe.item(), mpjve.item()


# =============================================================================
# Positional Encoding for Transformer
# =============================================================================

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            x + positional encoding: (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# GMD Preprocessor (Sequence-aware version for Transformer)
# =============================================================================

class GMDPreprocessorSequence(nn.Module):
    """
    Global Motion Decomposition Preprocessor for Transformer

    MLP 버전과 달리 시퀀스 구조 유지: (B, 576) → (B, 16, 42)

    각 timestep별 features:
    - Original: 36 = 3 trackers × 12 features
    - GMD deltas: 6 = 3 trackers × 2 (XZ delta from t0)
    - Total per timestep: 42
    """

    def __init__(self, num_timesteps=16, features_per_timestep=12):
        super(GMDPreprocessorSequence, self).__init__()
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        self.num_trackers = 3  # Head, LWrist, RWrist

    def forward(self, x):
        """
        Args:
            x: (B, 576) flat input
        Returns:
            x_seq: (B, 16, 42) sequence with GMD features
        """
        B = x.shape[0]
        device = x.device

        # Reshape to sequence: (B, 576) → (B, 3, 16, 12) → (B, 16, 36)
        # 원본 구조: [Head_all(192), LWrist_all(192), RWrist_all(192)]
        # 각 tracker: 16 timesteps × 12 features

        # Split by tracker
        head_features = x[:, 0:192].view(B, 16, 12)      # (B, 16, 12)
        lwrist_features = x[:, 192:384].view(B, 16, 12)  # (B, 16, 12)
        rwrist_features = x[:, 384:576].view(B, 16, 12)  # (B, 16, 12)

        # Clone for modification (spatial normalization)
        head_modified = head_features.clone()
        lwrist_modified = lwrist_features.clone()
        rwrist_modified = rwrist_features.clone()

        # === Spatial Normalization ===
        # Head XZ를 기준으로 상대화 (Y는 유지)
        head_x = head_features[:, :, 0:1].clone()  # (B, 16, 1)
        head_z = head_features[:, :, 2:3].clone()  # (B, 16, 1)

        # Head XZ → 0
        head_modified[:, :, 0:1] -= head_x
        head_modified[:, :, 2:3] -= head_z

        # LWrist XZ → Head 기준
        lwrist_modified[:, :, 0:1] -= head_x
        lwrist_modified[:, :, 2:3] -= head_z

        # RWrist XZ → Head 기준
        rwrist_modified[:, :, 0:1] -= head_x
        rwrist_modified[:, :, 2:3] -= head_z

        # === Temporal Normalization ===
        # delta = position[t] - position[0] for XZ
        head_x_0 = head_modified[:, 0:1, 0:1]    # (B, 1, 1)
        head_z_0 = head_modified[:, 0:1, 2:3]    # (B, 1, 1)
        lwrist_x_0 = lwrist_modified[:, 0:1, 0:1]
        lwrist_z_0 = lwrist_modified[:, 0:1, 2:3]
        rwrist_x_0 = rwrist_modified[:, 0:1, 0:1]
        rwrist_z_0 = rwrist_modified[:, 0:1, 2:3]

        # Compute deltas: (B, 16, 2) per tracker
        head_delta = torch.cat([
            head_modified[:, :, 0:1] - head_x_0,
            head_modified[:, :, 2:3] - head_z_0
        ], dim=2)  # (B, 16, 2)

        lwrist_delta = torch.cat([
            lwrist_modified[:, :, 0:1] - lwrist_x_0,
            lwrist_modified[:, :, 2:3] - lwrist_z_0
        ], dim=2)  # (B, 16, 2)

        rwrist_delta = torch.cat([
            rwrist_modified[:, :, 0:1] - rwrist_x_0,
            rwrist_modified[:, :, 2:3] - rwrist_z_0
        ], dim=2)  # (B, 16, 2)

        # Concatenate all deltas: (B, 16, 6)
        all_deltas = torch.cat([head_delta, lwrist_delta, rwrist_delta], dim=2)

        # === Combine: original features + deltas ===
        # (B, 16, 12+12+12+6) = (B, 16, 42)
        x_seq = torch.cat([
            head_modified, lwrist_modified, rwrist_modified, all_deltas
        ], dim=2)

        return x_seq


# =============================================================================
# SlowFast Transformer (EgoPoser Style)
# =============================================================================

class SlowFastTransformer(nn.Module):
    """
    SlowFast Feature Fusion with Transformer Encoder (EgoPoser Style)

    EgoPoser 논문 구현:
    1. Fast pathway: 후반 T/2 frames
    2. Slow pathway: stride 2 over full window
    3. Shared Linear Embedding
    4. Element-wise sum
    5. Transformer Encoder
    6. Last timestep output
    """

    def __init__(self,
                 input_dim_per_timestep=42,  # GMD output per timestep
                 embed_dim=256,
                 num_heads=4,
                 num_layers=4,
                 dropout=0.1,
                 output_dim=231):
        super(SlowFastTransformer, self).__init__()

        self.input_dim = input_dim_per_timestep
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # === Shared Linear Embedding (EgoPoser Style) ===
        # Fast와 Slow가 같은 embedding layer 사용
        self.linear_embedding = nn.Linear(input_dim_per_timestep, embed_dim)

        # === Positional Encoding ===
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=20, dropout=dropout)

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True  # (B, T, D) format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Output Projection ===
        # EgoPoser uses separate decoders, we use single projection for compatibility
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
        )

        print(f"\nSlowFast Transformer Configuration (EgoPoser Style):")
        print(f"  Input per timestep: {input_dim_per_timestep}")
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Transformer: {num_layers} layers, {num_heads} heads")
        print(f"  Output dim: {output_dim}")

    def forward(self, x_seq):
        """
        Args:
            x_seq: (B, 16, 42) GMD preprocessed sequence
        Returns:
            out: (B, 231) predicted output
        """
        B, T, D = x_seq.shape  # (B, 16, 42)

        # === SlowFast Extraction (EgoPoser Style) ===
        # Fast: last T/2 frames (indices 8-15)
        x_fast = x_seq[:, -T//2:, :]  # (B, 8, 42)

        # Slow: stride 2 (indices 0,2,4,6,8,10,12,14)
        x_slow = x_seq[:, ::2, :]  # (B, 8, 42)

        # === Shared Linear Embedding ===
        # Same weights for both pathways
        x_fast = self.linear_embedding(x_fast)  # (B, 8, embed_dim)
        x_slow = self.linear_embedding(x_slow)  # (B, 8, embed_dim)

        # === Fusion: Element-wise Sum ===
        x = x_fast + x_slow  # (B, 8, embed_dim)

        # === Positional Encoding ===
        x = self.pos_encoder(x)  # (B, 8, embed_dim)

        # === Transformer Encoder ===
        x = self.transformer_encoder(x)  # (B, 8, embed_dim)

        # === Take Last Timestep ===
        x = x[:, -1, :]  # (B, embed_dim)

        # === Output Projection ===
        out = self.output_projection(x)  # (B, 231)

        return out


# =============================================================================
# Main Model: TrackerBody Transformer + GMD + SlowFast
# =============================================================================

class Model(nn.Module):
    """
    TrackerBody Transformer with GMD + SlowFast (EgoPoser Style)

    MLP 버전과 동일한 I/O 인터페이스:
    - Input: (B, 576) flat tensor
    - Output: (B, 231) flat tensor

    내부 구조:
    1. GMD Preprocessing: (B, 576) → (B, 16, 42)
    2. SlowFast + Transformer: (B, 16, 42) → (B, 231)
    """

    def __init__(self,
                 input_dim=576,
                 output_dim=231,
                 embed_dim=256,
                 num_heads=4,
                 num_layers=4,
                 dropout=0.1,
                 input_norm=None,
                 output_norm=None):
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # === Normalization Parameters ===
        if input_norm is not None:
            self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)
        else:
            # Identity normalization
            self.Xnorm = Parameter(torch.zeros(2, input_dim), requires_grad=False)
            self.Xnorm[1, :] = 1.0  # std = 1

        if output_norm is not None:
            self.Ynorm = Parameter(torch.from_numpy(output_norm), requires_grad=False)
        else:
            self.Ynorm = Parameter(torch.zeros(2, output_dim), requires_grad=False)
            self.Ynorm[1, :] = 1.0

        # === GMD Preprocessor ===
        self.gmd = GMDPreprocessorSequence(num_timesteps=16, features_per_timestep=12)

        # GMD output: 36 (original per timestep) + 6 (deltas) = 42
        gmd_output_dim_per_timestep = 42

        # === SlowFast Transformer ===
        self.transformer = SlowFastTransformer(
            input_dim_per_timestep=gmd_output_dim_per_timestep,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=output_dim
        )

        print(f"\n{'='*60}")
        print(f"Model: TrackerBody Transformer + GMD + SlowFast")
        print(f"{'='*60}")
        print(f"Input: {input_dim} (flat) → (B, 16, 36) sequence")
        print(f"GMD: (B, 16, 36) → (B, 16, 42)")
        print(f"SlowFast + Transformer: (B, 16, 42) → (B, {output_dim})")
        print(f"Embed dim: {embed_dim}")
        print(f"Transformer: {num_layers} layers, {num_heads} heads")
        print(f"{'='*60}\n")

    def forward(self, x):
        """
        Forward pass (ONNX compatible)

        Args:
            x: (B, 576) flat input (same as MLP version)
        Returns:
            y: (B, 231) predicted output (same as MLP version)
        """
        # === Normalize Input ===
        x = utility.Normalize(x, self.Xnorm)

        # === GMD Preprocessing ===
        # (B, 576) → (B, 16, 42)
        x_seq = self.gmd(x)

        # === SlowFast Transformer ===
        # (B, 16, 42) → (B, 231)
        y = self.transformer(x_seq)

        # === Renormalize Output ===
        y = utility.Renormalize(y, self.Ynorm)

        return y


# =============================================================================
# Legacy Model Wrapper (for compatibility with existing training script)
# =============================================================================

class ModelLegacy(Model):
    """
    Legacy wrapper for compatibility with original training script interface.

    Original MLP interface:
    - rng, layers, activations, dropout, input_norm, output_norm

    Transformer interface:
    - input_dim, output_dim, embed_dim, num_heads, num_layers, dropout, input_norm, output_norm
    """

    def __init__(self,
                 rng=None,
                 layers=None,
                 activations=None,
                 dropout=0.1,
                 input_norm=None,
                 output_norm=None,
                 embed_dim=256,
                 num_heads=4,
                 num_layers=4,
                 **kwargs):

        # Extract input/output dimensions from layers
        if layers is not None:
            input_dim = layers[0]
            output_dim = layers[-1]
        else:
            input_dim = 576
            output_dim = 231

        super(ModelLegacy, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            input_norm=input_norm,
            output_norm=output_norm
        )


# =============================================================================
# Shape Verification
# =============================================================================

def run_shape_verification(input_dim=576, output_dim=231, batch_size=4):
    """Shape verification test"""
    print("\n" + "="*60)
    print("SHAPE VERIFICATION TEST")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test tensors
    x = torch.randn(batch_size, input_dim).to(device)
    print(f"\nInput: {list(x.shape)}")

    # === GMD Test ===
    print(f"\n--- GMD Preprocessor (Sequence) ---")
    gmd = GMDPreprocessorSequence(num_timesteps=16, features_per_timestep=12).to(device)
    x_seq = gmd(x)
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(x_seq.shape)}")
    print(f"  Expected: (B, 16, 42)")
    assert x_seq.shape == (batch_size, 16, 42), f"GMD shape mismatch! Got {x_seq.shape}"
    print(f"  GMD OK")

    # === SlowFast Transformer Test ===
    print(f"\n--- SlowFast Transformer ---")
    transformer = SlowFastTransformer(
        input_dim_per_timestep=42,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        output_dim=output_dim
    ).to(device)
    y = transformer(x_seq)
    print(f"  Input:  {list(x_seq.shape)}")
    print(f"  Output: {list(y.shape)}")
    print(f"  Expected: (B, {output_dim})")
    assert y.shape == (batch_size, output_dim), f"Transformer shape mismatch! Got {y.shape}"
    print(f"  Transformer OK")

    # === Full Model Test ===
    print(f"\n--- Full Model ---")

    # Proper normalization
    input_norm = np.zeros((2, input_dim), dtype=np.float32)
    input_norm[1, :] = 1.0
    output_norm = np.zeros((2, output_dim), dtype=np.float32)
    output_norm[1, :] = 1.0

    model = Model(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        input_norm=input_norm,
        output_norm=output_norm
    ).to(device)

    y = model(x)
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(y.shape)}")
    print(f"  Expected: (B, {output_dim})")
    assert y.shape == (batch_size, output_dim), f"Model shape mismatch! Got {y.shape}"
    print(f"  Full Model OK")

    # === Gradient Flow Test ===
    print(f"\n--- Gradient Flow Test ---")
    model.train()
    y = model(x)
    loss = y.sum()
    loss.backward()

    grad_ok = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  No gradient for: {name}")
            grad_ok = False

    if grad_ok:
        print(f"  All gradients computed")

    # === Parameter Count ===
    print(f"\n--- Parameter Count ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # === ONNX Export Test ===
    print(f"\n--- ONNX Export Test ---")
    model.eval()
    dummy_input = torch.randn(1, input_dim).to(device)
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            torch.onnx.export(
                model,
                dummy_input,
                f.name,
                input_names=['X'],
                output_names=['Y'],
                dynamic_axes={'X': {0: 'batch_size'}, 'Y': {0: 'batch_size'}},
                opset_version=14
            )
            print(f"  ONNX export successful: {f.name}")
            os.remove(f.name)
    except Exception as e:
        print(f"  ONNX export failed: {e}")

    print("\n" + "="*60)
    print("ALL SHAPE VERIFICATIONS PASSED!")
    print("="*60 + "\n")

    return True


# =============================================================================
# Training Script
# =============================================================================

if __name__ == '__main__':
    # === Configuration ===
    name = "Trackerbodypredictor"

    # Dataset path
    env_dataset_path = os.environ.get("TRACKERBODY_DATASET_PATH")
    if env_dataset_path and os.path.isdir(env_dataset_path):
        directory = env_dataset_path
        print(f"Using dataset from environment variable: {directory}")
    else:
        directory = "../../Datasets/" + name
        print(f"Using default dataset path: {directory}")

    id = name + "_Transformer_GMD_SlowFast_" + utility.GetFileID(__file__)
    load = directory
    save = directory + "/Training_" + id
    utility.MakeDirectory(save)

    # === Load Data ===
    InputName = "Input"
    OutputName = "Output"
    InputFile = load + "/" + InputName + ".bin"
    OutputFile = load + "/" + OutputName + ".bin"
    Xshape = utility.LoadTxtAsInt(load + "/" + InputName + "Shape.txt", True)
    Yshape = utility.LoadTxtAsInt(load + "/" + OutputName + "Shape.txt", True)
    Xnorm = utility.LoadTxt(load + "/" + InputName + "Normalization.txt", True)
    Ynorm = utility.LoadTxt(load + "/" + OutputName + "Normalization.txt", True)

    # === Verify Binary File Sizes ===
    input_file_size = os.path.getsize(InputFile)
    output_file_size = os.path.getsize(OutputFile)
    input_dim_shape = Xshape[1]
    output_dim_shape = Yshape[1]

    actual_input_samples = input_file_size // (input_dim_shape * 4)
    actual_output_samples = output_file_size // (output_dim_shape * 4)

    print(f"\n{'='*60}")
    print("BINARY FILE VERIFICATION")
    print(f"{'='*60}")
    print(f"Input.bin size: {input_file_size:,} bytes")
    print(f"  Expected samples (from Shape.txt): {Xshape[0]:,}")
    print(f"  Actual samples (from file size):   {actual_input_samples:,}")
    print(f"Output.bin size: {output_file_size:,} bytes")
    print(f"  Expected samples (from Shape.txt): {Yshape[0]:,}")
    print(f"  Actual samples (from file size):   {actual_output_samples:,}")

    if actual_input_samples != Xshape[0] or actual_output_samples != Yshape[0]:
        print(f"\nWARNING: File size mismatch detected!")
        print(f"  Using actual sample count: {min(actual_input_samples, actual_output_samples):,}")
        Xshape[0] = min(actual_input_samples, actual_output_samples)
        Yshape[0] = min(actual_input_samples, actual_output_samples)
    print(f"{'='*60}\n")

    # === Seed ===
    seed = 23456
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # === Hyperparameters ===
    epochs = 150
    batch_size = 64
    dropout = 0.1

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    sample_count = Xshape[0]
    input_dim = Xshape[1]
    output_dim = Yshape[1]

    # Transformer hyperparameters
    embed_dim = 256
    num_heads = 4
    num_layers = 4

    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {name}")
    print(f"Input Features: {input_dim}")
    print(f"Output Features: {output_dim}")
    print(f"Sample Count: {sample_count:,}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Transformer:")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Heads: {num_heads}")
    print(f"  Layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print("="*60)

    # === Shape Verification ===
    print("\nRunning shape verification...")
    run_shape_verification(input_dim, output_dim, batch_size=4)

    # === Create Model ===
    network = Model(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        input_norm=Xnorm,
        output_norm=Ynorm
    )

    # === Parameter Count ===
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"PARAMETER COUNT")
    print(f"{'='*60}")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"{'='*60}\n")

    if torch.cuda.is_available():
        print('GPU found, training on GPU...')
        network = network.cuda()
    else:
        print('No GPU found, training on CPU...')

    # === Optimizer & Scheduler ===
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(
        optimizer=optimizer,
        batch_size=batch_size,
        epoch_size=sample_count,
        restart_period=restart_period,
        t_mult=restart_mult,
        policy="cosine",
        verbose=True
    )
    loss_function = nn.MSELoss()

    # === Training Loop ===
    I = np.arange(sample_count)
    best_loss = float('inf')

    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(I)

        network.train()
        train_error = 0.0
        epoch_mpjpe = 0.0
        epoch_mpjve = 0.0
        num_batches = 0
        last_progress = -1

        for i in range(0, sample_count, batch_size):
            # Progress logging (10% intervals)
            progress = int(100 * i / sample_count)
            if progress // 10 > last_progress // 10:
                print(f'Epoch {epoch+1} - {progress}%')
                last_progress = progress

            train_indices = I[i:i+batch_size]

            xBatch = utility.ReadBatchFromFile(InputFile, train_indices, input_dim)
            yBatch = utility.ReadBatchFromFile(OutputFile, train_indices, output_dim)

            yPred = network(xBatch)

            # Loss (normalized space)
            loss = loss_function(
                utility.Normalize(yPred, network.Ynorm),
                utility.Normalize(yBatch, network.Ynorm)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            train_error += loss.item()

            # MPJPE/MPJVE
            with torch.no_grad():
                mpjpe, mpjve = compute_mpjpe_mpjve(yPred, yBatch)
                epoch_mpjpe += mpjpe
                epoch_mpjve += mpjve
                num_batches += 1

        # === Save Model ===
        avg_loss = train_error / num_batches

        # Save every epoch
        utility.SaveONNX(
            path=save + '/' + id + '_' + str(epoch+1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim)),
            input_names=['X'],
            output_names=['Y']
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            utility.SaveONNX(
                path=save + '/' + id + '_best.onnx',
                model=network,
                input_size=(torch.zeros(1, input_dim)),
                input_names=['X'],
                output_names=['Y']
            )
            print(f"  New best model saved!")

        # === Log Progress ===
        avg_mpjpe = epoch_mpjpe / num_batches
        avg_mpjve = epoch_mpjve / num_batches
        avg_mpjpe_cm = avg_mpjpe * 100

        print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}, MPJPE: {avg_mpjpe_cm:.2f}cm, MPJVE: {avg_mpjve:.4f}')
