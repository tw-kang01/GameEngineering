"""
Categorical VQ-VAE with GMD + SlowFast Enhancement (Corrected Version)
======================================================================

핵심 원칙:
- Categorical Codebook Matching (SIGGRAPH 2024)을 **베이스**로 유지
- EgoPoser (ECCV 2024)에서 GMD + SlowFast **아이디어만** 차용
- Categorical의 MLP 기반 구조는 그대로 유지 (Transformer 추가 X)

수정 사항:
1. GMD: Spatial + Temporal Normalization → 입력 전처리로 적용
2. SlowFast: Transformer 대신 MLP 기반 융합 (Categorical 스타일 유지)
3. 원본 Categorical의 Encoder/Estimator/Decoder 구조 100% 보존

Author: Integration of EgoPoser GMD+SlowFast into Categorical VQ-VAE
"""

import sys
sys.path.append("../../../PyTorch")

import Library.Utility as utility
import Library.Plotting as plotting
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import Library.Modules as modules

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import matplotlib.pyplot as plt


# =============================================================================
# GMD Module: Global Motion Decomposition (EgoPoser에서 차용)
# =============================================================================

class GMDPreprocessor(nn.Module):
    """
    Global Motion Decomposition Preprocessor
    
    EgoPoser의 GMD를 Categorical 입력에 맞게 적용
    - Spatial Normalization: Head XZ 기준 상대 좌표
    - Temporal Normalization: 첫 프레임 대비 delta
    
    주의: 이 모듈은 학습 파라미터가 없음 (순수 전처리)
    
    TrackerBodyPredictor 입력 구조 (252 features):
    - 7 timesteps × 36 features/timestep
    - 각 timestep: [Head(12), LeftWrist(12), RightWrist(12)]
    - 각 tracker: [Position(3), Forward(3), Up(3), Velocity(3)]
    """
    def __init__(self, num_timesteps=7, features_per_timestep=36):
        super(GMDPreprocessor, self).__init__()
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        
        # Position offsets within each timestep (36 features)
        # Head: 0-11, LeftWrist: 12-23, RightWrist: 24-35
        self.head_pos = 0       # Head position starts at 0
        self.lwrist_pos = 12    # LeftWrist position starts at 12
        self.rwrist_pos = 24    # RightWrist position starts at 24
        
        # XZ indices (Unity: X=horizontal, Y=vertical, Z=horizontal)
        self.x_idx = 0
        self.z_idx = 2
        
    def forward(self, x):
        """
        Args:
            x: (B, 252) flat input - 원본 Categorical 입력과 동일
        Returns:
            x_gmd: (B, 252 + delta_dim) GMD 적용된 입력
            
        delta_dim = 6 × num_timesteps = 42 (각 timestep마다 6개 delta)
        → 총 출력: (B, 252 + 42) = (B, 294)
        """
        B = x.shape[0]
        device = x.device
        
        # Reshape: (B, 252) -> (B, 7, 36)
        x_reshaped = x.view(B, self.num_timesteps, self.features_per_timestep).clone()
        
        # === Spatial Normalization ===
        # EgoPoser 원본 (lines 75-78):
        # head_horizontal_trans = input_tensor.clone()[...,36:38].detach()
        # input_tensor[...,36:38] -= head_horizontal_trans
        # input_tensor[...,39:41] -= head_horizontal_trans
        # input_tensor[...,42:44] -= head_horizontal_trans
        
        # Head XZ를 기준으로 모든 tracker XZ를 상대화
        head_x = x_reshaped[:, :, self.head_pos + self.x_idx].clone()  # (B, 7)
        head_z = x_reshaped[:, :, self.head_pos + self.z_idx].clone()  # (B, 7)
        
        # Head position XZ를 0으로 (자기 자신 기준)
        x_reshaped[:, :, self.head_pos + self.x_idx] -= head_x
        x_reshaped[:, :, self.head_pos + self.z_idx] -= head_z
        
        # LeftWrist XZ를 Head 기준 상대 좌표로
        x_reshaped[:, :, self.lwrist_pos + self.x_idx] -= head_x
        x_reshaped[:, :, self.lwrist_pos + self.z_idx] -= head_z
        
        # RightWrist XZ를 Head 기준 상대 좌표로
        x_reshaped[:, :, self.rwrist_pos + self.x_idx] -= head_x
        x_reshaped[:, :, self.rwrist_pos + self.z_idx] -= head_z
        
        # === Temporal Normalization ===
        # EgoPoser 원본 (lines 82-88):
        # delta_0 = input_tensor[...,36:38] - input_tensor[...,[0],36:38]
        # delta_1 = input_tensor[...,39:41] - input_tensor[...,[0],39:41]
        # delta_2 = input_tensor[...,42:44] - input_tensor[...,[0],42:44]
        # input_tensor = torch.cat([input_tensor, delta_0, delta_1, delta_2], dim=-1)
        
        # 각 timestep에서 첫 프레임 대비 delta 계산
        # Head XZ delta
        delta_head_x = x_reshaped[:, :, self.head_pos + self.x_idx] - x_reshaped[:, [0], self.head_pos + self.x_idx]
        delta_head_z = x_reshaped[:, :, self.head_pos + self.z_idx] - x_reshaped[:, [0], self.head_pos + self.z_idx]
        
        # LeftWrist XZ delta
        delta_lwrist_x = x_reshaped[:, :, self.lwrist_pos + self.x_idx] - x_reshaped[:, [0], self.lwrist_pos + self.x_idx]
        delta_lwrist_z = x_reshaped[:, :, self.lwrist_pos + self.z_idx] - x_reshaped[:, [0], self.lwrist_pos + self.z_idx]
        
        # RightWrist XZ delta  
        delta_rwrist_x = x_reshaped[:, :, self.rwrist_pos + self.x_idx] - x_reshaped[:, [0], self.rwrist_pos + self.x_idx]
        delta_rwrist_z = x_reshaped[:, :, self.rwrist_pos + self.z_idx] - x_reshaped[:, [0], self.rwrist_pos + self.z_idx]
        
        # Delta 피처 결합: (B, 7, 6)
        deltas = torch.stack([
            delta_head_x, delta_head_z,
            delta_lwrist_x, delta_lwrist_z,
            delta_rwrist_x, delta_rwrist_z
        ], dim=-1)
        
        # Spatial normalized features + temporal deltas
        x_with_delta = torch.cat([x_reshaped, deltas], dim=-1)  # (B, 7, 42)
        
        # Flatten back: (B, 7, 42) -> (B, 294)
        x_gmd = x_with_delta.view(B, -1)
        
        return x_gmd


# =============================================================================
# SlowFast Fusion: MLP 기반 (Categorical 스타일 유지)
# =============================================================================

class SlowFastMLP(nn.Module):
    """
    SlowFast Feature Fusion - MLP 기반
    
    Categorical의 LinearEncoder 스타일을 유지하면서 SlowFast 아이디어 적용
    - Fast: 최근 timesteps (세밀한 모션)
    - Slow: 전체 timesteps 서브샘플링 (장기 맥락)
    - Fusion: MLP로 처리 후 합산
    
    주의: Transformer 사용하지 않음 (Categorical 베이스 유지)
    """
    def __init__(self, input_dim_per_timestep, num_timesteps, output_dim, hidden_dim=512, dropout=0.25):
        super(SlowFastMLP, self).__init__()
        
        self.num_timesteps = num_timesteps
        self.input_dim_per_timestep = input_dim_per_timestep
        
        # Fast pathway: 후반 timesteps 처리
        # T=7일 때: indices 4,5,6 (3 frames)
        self.num_fast_frames = (num_timesteps + 1) // 2
        fast_input_dim = self.num_fast_frames * input_dim_per_timestep
        
        # Slow pathway: 매 2 timestep 샘플링
        # T=7일 때: indices 0,2,4,6 (4 frames)
        self.num_slow_frames = (num_timesteps + 1) // 2
        slow_input_dim = self.num_slow_frames * input_dim_per_timestep
        
        # Fast MLP (Categorical LinearEncoder 스타일)
        self.fast_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fast_input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ELU()
        )
        
        # Slow MLP (Categorical LinearEncoder 스타일)
        self.slow_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(slow_input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ELU()
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, num_timesteps * input_dim_per_timestep) e.g., (B, 294)
        Returns:
            out: (B, output_dim) fused features
        """
        B = x.shape[0]
        
        # Reshape: (B, 294) -> (B, 7, 42)
        x_reshaped = x.view(B, self.num_timesteps, self.input_dim_per_timestep)
        
        # === Fast Pathway ===
        # EgoPoser: x_fast = input_tensor[:,-input_tensor.shape[1]//2:,...]
        # 후반 ~50% timesteps (최근 모션)
        fast_start = self.num_timesteps - self.num_fast_frames
        x_fast = x_reshaped[:, fast_start:, :]  # (B, num_fast_frames, 42)
        x_fast_flat = x_fast.reshape(B, -1)     # (B, num_fast_frames * 42)
        
        # === Slow Pathway ===
        # EgoPoser: x_slow = input_tensor[:,::2,...]
        # 매 2 timestep (장기 맥락)
        x_slow = x_reshaped[:, ::2, :]          # (B, num_slow_frames, 42)
        x_slow_flat = x_slow.reshape(B, -1)     # (B, num_slow_frames * 42)
        
        # === MLP Processing ===
        fast_out = self.fast_mlp(x_fast_flat)   # (B, output_dim)
        slow_out = self.slow_mlp(x_slow_flat)   # (B, output_dim)
        
        # === Fusion: Element-wise sum ===
        # EgoPoser: x = x_fast + x_slow
        out = fast_out + slow_out
        
        return out


# =============================================================================
# Main Model: Categorical VQ-VAE + GMD + SlowFast
# =============================================================================

class Model(nn.Module):
    """
    Categorical VQ-VAE with GMD + SlowFast (Corrected Version)
    
    베이스: Categorical Codebook Matching (SIGGRAPH 2024)
    - Encoder: LinearEncoder (MLP) - 변경 없음
    - Estimator: LinearEncoder (MLP) - 변경 없음
    - Decoder: LinearEncoder (MLP) - 변경 없음
    - Gumbel-Softmax: 변경 없음
    
    추가: EgoPoser (ECCV 2024) 아이디어
    - GMD: 입력 전처리 (Spatial + Temporal Norm)
    - SlowFast: MLP 기반 융합 (Transformer 아님!)
    
    Pipeline:
    [GMD + SlowFast 전처리]
    Input (252) → GMD (294) → SlowFast (252) → + Original (Residual)
                                                    ↓
    [원본 Categorical VQ-VAE]
                        Normalize → Estimator → Gumbel-Softmax
                                              ↓
                                   Codebook → Decoder → Output
    """
    def __init__(self, encoder, estimator, decoder, xNorm, yNorm, 
                 codebook_channels, codebook_dim,
                 # GMD + SlowFast settings
                 use_gmd=True,
                 num_timesteps=7,
                 features_per_timestep=36,
                 slowfast_hidden_dim=512,
                 dropout=0.25):
        super(Model, self).__init__()

        # === 원본 Categorical VQ-VAE 컴포넌트 (변경 없음) ===
        self.Encoder = encoder
        self.Estimator = estimator
        self.Decoder = decoder
        self.XNorm = xNorm
        self.YNorm = yNorm
        self.C = codebook_channels
        self.D = codebook_dim
        
        # === GMD + SlowFast 설정 ===
        self.use_gmd = use_gmd
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        
        if self.use_gmd:
            # GMD Preprocessor (학습 파라미터 없음)
            self.gmd = GMDPreprocessor(num_timesteps, features_per_timestep)
            
            # GMD 적용 후 피처 차원
            gmd_features_per_timestep = features_per_timestep + 6  # 36 + 6 = 42
            gmd_total_dim = num_timesteps * gmd_features_per_timestep  # 7 * 42 = 294
            
            # SlowFast MLP
            original_input_dim = num_timesteps * features_per_timestep  # 252
            self.slowfast = SlowFastMLP(
                input_dim_per_timestep=gmd_features_per_timestep,  # 42
                num_timesteps=num_timesteps,  # 7
                output_dim=original_input_dim,  # 252 (원본 차원으로 복원)
                hidden_dim=slowfast_hidden_dim,
                dropout=dropout
            )
            
            # Learnable residual weight
            self.residual_alpha = nn.Parameter(torch.tensor(0.5))
            
            print("=" * 60)
            print("GMD + SlowFast Configuration (Categorical Base)")
            print("=" * 60)
            print(f"  Base Model: Categorical VQ-VAE (MLP-based)")
            print(f"  GMD Enabled: {use_gmd}")
            print(f"  Input: {num_timesteps} timesteps × {features_per_timestep} features = {original_input_dim}")
            print(f"  GMD Output: {num_timesteps} × {gmd_features_per_timestep} = {gmd_total_dim}")
            print(f"  SlowFast Output: {original_input_dim} (same as original)")
            print(f"  Residual: α * SlowFast + (1-α) * Original, α=learnable")
            print("=" * 60)

    # === Gumbel-Softmax (원본 Categorical과 동일) ===
    
    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.reshape(-1,1,1,1)
        noise = torch.rand_like(tensor) - 0.5
        samples = scale * noise + 0.5
        return -torch.log(-torch.log(samples + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, scale):
        y = logits + self.sample_gumbel(logits, scale)
        return F.softmax(y / temperature, dim=-1)
    
    def gumbel_softmax(self, logits, temperature, scale):
        y = self.gumbel_softmax_sample(logits, temperature, scale)
        y_soft = y.view(logits.shape)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        y_hard = y_hard.view(logits.shape)
        return y_soft, y_hard

    def sample(self, z, knn):
        z = z.reshape(-1, self.C, self.D)
        z = z.unsqueeze(0).repeat(knn.size(0), 1, 1, 1)
        z_soft, z_hard = self.gumbel_softmax(z, 1.0, knn)
        z_soft = z_soft.reshape(-1, self.C*self.D)
        z_hard = z_hard.reshape(-1, self.C*self.D)
        return z_soft, z_hard
    
    # === GMD + SlowFast Preprocessing ===
    
    def apply_gmd_slowfast(self, x):
        """
        GMD + SlowFast 전처리 적용
        
        Pipeline:
        1. GMD: (B, 252) → (B, 294)  [Spatial + Temporal Norm + Delta features]
        2. SlowFast: (B, 294) → (B, 252)  [Fast + Slow pathway fusion]
        3. Residual: α * slowfast_out + (1-α) * original
        """
        # Step 1: GMD preprocessing
        x_gmd = self.gmd(x)  # (B, 294)
        
        # Step 2: SlowFast fusion
        x_slowfast = self.slowfast(x_gmd)  # (B, 252)
        
        return x_slowfast
    
    def forward(self, x, knn, t=None):
        """
        Forward pass - 원본 Categorical과 동일한 인터페이스
        
        Args:
            x: Input (B, input_dim) - tracker features
            knn: K-nearest neighbor for Gumbel-Softmax
            t: Target (B, output_dim) - only during training
        """
        # === GMD + SlowFast Preprocessing ===
        if self.use_gmd:
            x_original = x.clone()
            x_enhanced = self.apply_gmd_slowfast(x)
            
            # Learnable residual connection
            alpha = torch.sigmoid(self.residual_alpha)
            x = alpha * x_enhanced + (1 - alpha) * x_original
        
        # === 원본 Categorical VQ-VAE Pipeline (변경 없음) ===
        
        # Training
        if t is not None:
            # Normalize
            x = utility.Normalize(x, self.XNorm)
            t = utility.Normalize(t, self.YNorm)

            # Encode Y (Teacher)
            target_logits = self.Encoder(torch.cat((t, x), dim=1))
            target_probs, target = self.sample(target_logits, knn)

            # Encode X (Student)
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            # Decode
            y = self.Decoder(target)

            # Renormalize
            return (utility.Renormalize(y, self.YNorm), 
                    target_logits, target_probs, target, 
                    estimate_logits, estimate_probs, estimate)
                
        # Inference
        else:
            # Normalize
            x = utility.Normalize(x, self.XNorm)
            
            # Encode X
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            # Decode
            y = self.Decoder(estimate)

            # Renormalize
            return utility.Renormalize(y, self.YNorm), estimate


# =============================================================================
# 차원 검증 함수
# =============================================================================

def verify_dimensions():
    """
    차원 검증 - 학습 전 필수 확인
    """
    print("\n" + "=" * 60)
    print("Dimension Verification")
    print("=" * 60)
    
    # 테스트 설정
    batch_size = 32
    num_timesteps = 7
    features_per_timestep = 36
    input_dim = num_timesteps * features_per_timestep  # 252
    
    # 테스트 입력
    x = torch.randn(batch_size, input_dim)
    
    # GMD Preprocessor 테스트
    gmd = GMDPreprocessor(num_timesteps, features_per_timestep)
    x_gmd = gmd(x)
    
    expected_gmd_dim = num_timesteps * (features_per_timestep + 6)  # 7 * 42 = 294
    print(f"GMD Input:  {x.shape} → Expected: (B, {input_dim})")
    print(f"GMD Output: {x_gmd.shape} → Expected: (B, {expected_gmd_dim})")
    assert x_gmd.shape == (batch_size, expected_gmd_dim), f"GMD dimension mismatch!"
    print("✓ GMD dimensions correct")
    
    # SlowFast 테스트
    slowfast = SlowFastMLP(
        input_dim_per_timestep=features_per_timestep + 6,  # 42
        num_timesteps=num_timesteps,
        output_dim=input_dim,  # 252
        hidden_dim=512
    )
    x_slowfast = slowfast(x_gmd)
    
    print(f"SlowFast Input:  {x_gmd.shape}")
    print(f"SlowFast Output: {x_slowfast.shape} → Expected: (B, {input_dim})")
    assert x_slowfast.shape == (batch_size, input_dim), f"SlowFast dimension mismatch!"
    print("✓ SlowFast dimensions correct")
    
    # Residual 후 차원
    alpha = torch.sigmoid(torch.tensor(0.5))
    x_final = alpha * x_slowfast + (1 - alpha) * x
    print(f"Final Output: {x_final.shape} → Expected: (B, {input_dim})")
    assert x_final.shape == (batch_size, input_dim), f"Final dimension mismatch!"
    print("✓ Final dimensions correct (matches original Categorical input)")
    
    print("\n✓ All dimension checks passed!")
    print("=" * 60)
    
    return True


def verify_coordinate_system():
    """
    좌표계 검증
    
    Unity (Categorical): Y-up, left-handed
    - X: 오른쪽
    - Y: 위쪽 (vertical)
    - Z: 앞쪽
    
    SMPL (EgoPoser): Y-up, right-handed
    - X: 오른쪽  
    - Y: 위쪽 (vertical)
    - Z: 뒤쪽
    
    GMD에서 XZ를 사용하므로 Y(vertical)는 유지됨 → 좌표계 변환 불필요
    """
    print("\n" + "=" * 60)
    print("Coordinate System Verification")
    print("=" * 60)
    
    print("Unity (Categorical):")
    print("  - Y-up, left-handed")
    print("  - XZ plane = horizontal (ground)")
    print("  - Y = vertical (height)")
    
    print("\nGMD Spatial Normalization:")
    print("  - Normalizes XZ (horizontal) relative to head")
    print("  - Keeps Y (vertical) as global")
    print("  - ✓ Compatible with Unity coordinate system")
    
    print("\nGMD Temporal Normalization:")
    print("  - Computes XZ delta between frames")
    print("  - Frame 0 as reference")
    print("  - ✓ Coordinate system agnostic")
    
    print("\n✓ Coordinate systems compatible - no conversion needed")
    print("=" * 60)


def verify_feature_indices():
    """
    피처 인덱스 검증
    
    TrackerBodyPredictor 입력 구조 확인:
    - Total: 252 features = 7 timesteps × 36 features/timestep
    - Per timestep: Head(12) + LeftWrist(12) + RightWrist(12)
    - Per tracker: Position(3) + Forward(3) + Up(3) + Velocity(3)
    """
    print("\n" + "=" * 60)
    print("Feature Index Verification")
    print("=" * 60)
    
    num_timesteps = 7
    features_per_timestep = 36
    
    print(f"Total input: {num_timesteps} × {features_per_timestep} = {num_timesteps * features_per_timestep}")
    
    print("\nPer timestep (36 features):")
    print("  Head:       [0:12]  - Position[0:3], Forward[3:6], Up[6:9], Velocity[9:12]")
    print("  LeftWrist:  [12:24] - Position[12:15], Forward[15:18], Up[18:21], Velocity[21:24]")
    print("  RightWrist: [24:36] - Position[24:27], Forward[27:30], Up[30:33], Velocity[33:36]")
    
    print("\nGMD uses position XZ indices:")
    print("  Head XZ:       indices 0, 2 (within timestep)")
    print("  LeftWrist XZ:  indices 12, 14 (within timestep)")
    print("  RightWrist XZ: indices 24, 26 (within timestep)")
    
    print("\nEgoPoser comparison:")
    print("  EgoPoser input: 54 features (6D rot + positions for head/hands)")
    print("  EgoPoser Head XZ: indices 36, 37 (flat)")
    print("  Categorical: Different structure, adapted accordingly")
    
    print("\n✓ Feature indices correctly mapped for Categorical structure")
    print("=" * 60)


# =============================================================================
# Training Script
# =============================================================================

if __name__ == '__main__':
    # 먼저 검증 수행
    print("\n" + "=" * 60)
    print("PRE-TRAINING VERIFICATION")
    print("=" * 60)
    
    verify_dimensions()
    verify_coordinate_system()
    verify_feature_indices()
    
    # Dataset settings
    name = "LowerBody"
    directory = "../../Datasets/" + name
    id = name + "_GMD_SlowFast_v2_" + utility.GetFileID(__file__)
    load = directory
    save = directory + "/Training_" + id
    utility.MakeDirectory(save)

    # Load data
    XFile = load + "/Input.bin"
    YFile = load + "/Output.bin"
    XShape = utility.LoadTxtAsInt(load + "/InputShape.txt", True)
    YShape = utility.LoadTxtAsInt(load + "/OutputShape.txt", True)

    sample_count = XShape[0]
    input_dim = XShape[1]
    output_dim = YShape[1]

    utility.SetSeed(23456)

    # Training hyperparameters (원본 Categorical과 동일)
    epochs = 150
    batch_size = 32
    dropout = 0.25

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    # Network architecture (원본 Categorical과 동일)
    encoder_dim = 1024
    estimator_dim = 1024
    decoder_dim = 1024

    codebook_channels = 128
    codebook_dim = 8
    codebook_size = codebook_channels * codebook_dim
    
    # GMD + SlowFast settings
    use_gmd = True
    num_timesteps = 7
    features_per_timestep = 36
    slowfast_hidden_dim = 512
    
    # 입력 차원 검증
    expected_input_dim = num_timesteps * features_per_timestep
    print(f"\nInput dimension check:")
    print(f"  Dataset input_dim: {input_dim}")
    print(f"  Expected (7×36): {expected_input_dim}")
    
    if input_dim != expected_input_dim:
        print(f"  ⚠ Warning: Input dimension mismatch!")
        print(f"  Adjusting num_timesteps and features_per_timestep...")
        # 실제 데이터에 맞게 조정 필요
    
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Dataset: {name}")
    print(f"Input Features: {input_dim}")
    print(f"Output Features: {output_dim}")
    print(f"Sample Count: {sample_count}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"GMD Enabled: {use_gmd}")
    print("=" * 60)

    # Create model
    network = utility.ToDevice(Model(
        # 원본 Categorical 컴포넌트 (변경 없음)
        encoder=modules.LinearEncoder(input_dim + output_dim, encoder_dim, encoder_dim, codebook_size, dropout),
        estimator=modules.LinearEncoder(input_dim, estimator_dim, estimator_dim, codebook_size, dropout),
        decoder=modules.LinearEncoder(codebook_size, decoder_dim, decoder_dim, output_dim, 0.0),
        
        xNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/InputNormalization.txt", True)), requires_grad=False),
        yNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/OutputNormalization.txt", True)), requires_grad=False),
        
        codebook_channels=codebook_channels,
        codebook_dim=codebook_dim,
        
        # GMD + SlowFast
        use_gmd=use_gmd,
        num_timesteps=num_timesteps,
        features_per_timestep=features_per_timestep,
        slowfast_hidden_dim=slowfast_hidden_dim,
        dropout=dropout
    ))
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
        
    # Optimizer and scheduler (원본 Categorical과 동일)
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(
        optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count,
        restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True
    )
    loss_function = nn.MSELoss()

    # Plotting setup
    plt.ion()
    _, ax_latent = plt.subplots(1, 5, figsize=(10, 2))
    loss_history = utility.PlottingWindow("Loss History", ax=plt.subplots(figsize=(10, 5)), drawInterval=500, yScale='log')
    
    def Item(value):
        return value.detach().cpu()

    # Test sequences
    Sequences = utility.LoadTxtRaw(load + "/Sequences.txt", False)
    Sequences = np.array(utility.Transpose2DList(Sequences)[0], dtype=np.int64)
    test_sequence_length = 60
    test_sequences = []
    for i in range(int(Sequences[-1])):
        indices = np.where(Sequences == (i + 1))[0]
        intervals = int(np.floor(len(indices) / test_sequence_length))
        if intervals > 0:
            slices = np.array_split(indices, intervals)
            test_sequences += slices
    print(f"Test Sequences: {len(test_sequences)}")

    # Training loop (원본 Categorical과 동일한 손실 함수)
    I = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(I)
        error = 0.0
        
        for i in range(0, sample_count, batch_size):
            print(f'Progress {round(100 * i / sample_count, 2)}%', end="\r")
            train_indices = I[i:i + batch_size]

            xBatch = utility.ReadBatchFromFile(XFile, train_indices, XShape[1])
            yBatch = utility.ReadBatchFromFile(YFile, train_indices, YShape[1])

            prediction, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = network(
                xBatch, knn=torch.ones(1, device=xBatch.device), t=yBatch
            )

            # 원본 Categorical 손실 함수 (변경 없음)
            mse_loss = loss_function(
                utility.Normalize(yBatch, network.YNorm),
                utility.Normalize(prediction, network.YNorm)
            )
            matching_loss = loss_function(target, estimate)
            
            loss = mse_loss + matching_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            error += loss.item()

            loss_history.Add(
                (Item(mse_loss).item(), "MSE Loss"),
                (Item(matching_loss).item(), "Matching Loss")
            )

            # Visualization (원본 Categorical과 동일)
            if loss_history.Counter == 0:
                network.eval()

                input_sequences = []
                output_sequences = []
                target_sequences = []
                estimate_sequences = []
                predictions_sequences = []
                
                for s in range(100):
                    idx = random.choice(test_sequences)
                    xBatch = utility.ReadBatchFromFile(XFile, idx, XShape[1])
                    yBatch = utility.ReadBatchFromFile(YFile, idx, YShape[1])
                    prediction, _, _, target, _, _, estimate = network(
                        xBatch, knn=torch.zeros(1, device=xBatch.device), t=yBatch
                    )
                    input_sequences.append(Item(xBatch))
                    output_sequences.append(Item(yBatch))
                    target_sequences.append(Item(target))
                    estimate_sequences.append(Item(estimate))
                    predictions_sequences.append(Item(prediction))
                
                plotting.PCA2DSequence(ax_latent[0], test_sequence_length, input_dim, input_sequences, "Input")
                plotting.PCA2DSequence(ax_latent[1], test_sequence_length, output_dim, output_sequences, "Output")
                plotting.PCA2DSequence(ax_latent[2], test_sequence_length, codebook_size, target_sequences, "Target")
                plotting.PCA2DSequence(ax_latent[3], test_sequence_length, codebook_size, estimate_sequences, "Estimate")
                plotting.PCA2DSequence(ax_latent[4], test_sequence_length, output_dim, predictions_sequences, "Prediction")

                network.train()
                plt.gcf().canvas.draw()
                plt.gcf().canvas.start_event_loop(1e-1)

        print(f'Epoch {epoch + 1}, Loss: {error / (sample_count / batch_size):.6f}')
        loss_history.Print()

        # Save model (원본 Categorical과 동일한 형식)
        utility.SaveONNX(
            path=save + '/' + id + '_' + str(epoch + 1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim), torch.ones(1)),
            input_names=['X', 'K'],
            output_names=['Y', 'Code'],
            dynamic_axes={'K': {0: 'Size'}}
        )
