"""
Categorical VQ-VAE with GMD + SlowFast Enhancement
===================================================

최종 검증 및 통합 버전
- Shape 로깅으로 디버깅 용이
- Categorical VQ-VAE 베이스 100% 유지
- EgoPoser GMD + SlowFast 아이디어만 차용

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
# Shape Logger - 디버깅용
# =============================================================================

class ShapeLogger:
    """학습 중 텐서 shape를 로깅하는 유틸리티"""
    
    def __init__(self, enabled=True, log_interval=1000):
        self.enabled = enabled
        self.log_interval = log_interval
        self.step_count = 0
        self.logged_once = False
        
    def log(self, name, tensor, force=False):
        """텐서 shape 로깅"""
        if not self.enabled:
            return
            
        # 첫 번째 스텝이거나 force=True일 때만 로깅
        if not self.logged_once or force:
            if isinstance(tensor, torch.Tensor):
                print(f"  [{name}] shape: {list(tensor.shape)}, dtype: {tensor.dtype}, device: {tensor.device}")
            else:
                print(f"  [{name}] type: {type(tensor)}, value: {tensor}")
                
    def step(self):
        """스텝 카운터 증가"""
        self.step_count += 1
        if self.step_count == 1:
            self.logged_once = True
            
    def header(self, title):
        """섹션 헤더 출력"""
        if not self.enabled or self.logged_once:
            return
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")


# =============================================================================
# GMD Preprocessor - 학습 파라미터 없음 (순수 전처리)
# =============================================================================

class GMDPreprocessor(nn.Module):
    """
    Global Motion Decomposition Preprocessor
    
    EgoPoser의 GMD를 Categorical 입력에 맞게 적용:
    - Spatial Normalization: Head XZ 기준 상대 좌표
    - Temporal Normalization: 첫 프레임 대비 delta
    
    주의: 이 모듈은 학습 파라미터가 없음!
    
    입력 구조 (TrackerBodyPredictor 기준):
    - 7 timesteps × 36 features/timestep = 252
    - 각 timestep: [Head(12), LeftWrist(12), RightWrist(12)]
    - 각 tracker: [Position(3), Forward(3), Up(3), Velocity(3)]
    """
    def __init__(self, num_timesteps, features_per_timestep, logger=None):
        super(GMDPreprocessor, self).__init__()
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        self.logger = logger
        
        # Position XZ indices within each timestep
        # Head: pos at 0,1,2 → X=0, Z=2
        # LeftWrist: pos at 12,13,14 → X=12, Z=14
        # RightWrist: pos at 24,25,26 → X=24, Z=26
        self.head_x, self.head_z = 0, 2
        self.lwrist_x, self.lwrist_z = 12, 14
        self.rwrist_x, self.rwrist_z = 24, 26
        
    def forward(self, x):
        """
        Args:
            x: (B, input_dim) flat input tensor
        Returns:
            x_gmd: (B, input_dim + delta_dim) GMD 적용된 입력
        """
        B = x.shape[0]
        device = x.device
        
        if self.logger:
            self.logger.header("GMD Preprocessor")
            self.logger.log("Input", x)
        
        # Reshape: (B, flat) -> (B, T, F)
        x_reshaped = x.view(B, self.num_timesteps, self.features_per_timestep).clone()
        
        if self.logger:
            self.logger.log("Reshaped", x_reshaped)
        
        # === Spatial Normalization ===
        # Head XZ를 기준으로 모든 tracker의 XZ를 상대화
        head_x_vals = x_reshaped[:, :, self.head_x].clone()  # (B, T)
        head_z_vals = x_reshaped[:, :, self.head_z].clone()  # (B, T)
        
        # Head XZ → 0
        x_reshaped[:, :, self.head_x] -= head_x_vals
        x_reshaped[:, :, self.head_z] -= head_z_vals
        
        # LeftWrist XZ → Head 기준 상대
        x_reshaped[:, :, self.lwrist_x] -= head_x_vals
        x_reshaped[:, :, self.lwrist_z] -= head_z_vals
        
        # RightWrist XZ → Head 기준 상대
        x_reshaped[:, :, self.rwrist_x] -= head_x_vals
        x_reshaped[:, :, self.rwrist_z] -= head_z_vals
        
        if self.logger:
            self.logger.log("After Spatial Norm", x_reshaped)
        
        # === Temporal Normalization ===
        # delta = frame[t] - frame[0]
        delta_head_x = x_reshaped[:, :, self.head_x] - x_reshaped[:, [0], self.head_x]
        delta_head_z = x_reshaped[:, :, self.head_z] - x_reshaped[:, [0], self.head_z]
        delta_lwrist_x = x_reshaped[:, :, self.lwrist_x] - x_reshaped[:, [0], self.lwrist_x]
        delta_lwrist_z = x_reshaped[:, :, self.lwrist_z] - x_reshaped[:, [0], self.lwrist_z]
        delta_rwrist_x = x_reshaped[:, :, self.rwrist_x] - x_reshaped[:, [0], self.rwrist_x]
        delta_rwrist_z = x_reshaped[:, :, self.rwrist_z] - x_reshaped[:, [0], self.rwrist_z]
        
        # Stack deltas: (B, T, 6)
        deltas = torch.stack([
            delta_head_x, delta_head_z,
            delta_lwrist_x, delta_lwrist_z,
            delta_rwrist_x, delta_rwrist_z
        ], dim=-1)
        
        if self.logger:
            self.logger.log("Temporal Deltas", deltas)
        
        # Concat: (B, T, F) + (B, T, 6) -> (B, T, F+6)
        x_with_delta = torch.cat([x_reshaped, deltas], dim=-1)
        
        if self.logger:
            self.logger.log("After Temporal Norm", x_with_delta)
        
        # Flatten: (B, T, F+6) -> (B, T*(F+6))
        x_gmd = x_with_delta.view(B, -1)
        
        if self.logger:
            self.logger.log("GMD Output (flat)", x_gmd)
        
        return x_gmd


# =============================================================================
# SlowFast MLP - Categorical LinearEncoder 스타일 유지
# =============================================================================

class SlowFastMLP(nn.Module):
    """
    SlowFast Feature Fusion - MLP 기반
    
    Categorical의 LinearEncoder 스타일 유지:
    - Fast: 최근 timesteps (세밀한 모션)
    - Slow: 전체 timesteps 서브샘플링 (장기 맥락)
    - Fusion: MLP 처리 후 합산
    
    주의: Transformer 사용하지 않음!
    """
    def __init__(self, input_dim_per_timestep, num_timesteps, output_dim, 
                 hidden_dim=512, dropout=0.25, logger=None):
        super(SlowFastMLP, self).__init__()
        
        self.num_timesteps = num_timesteps
        self.input_dim_per_timestep = input_dim_per_timestep
        self.logger = logger
        
        # Fast pathway: 후반 timesteps
        # T=7일 때: indices 4,5,6 (3 frames) 또는 3,4,5,6 (4 frames)
        self.fast_frames = max(num_timesteps // 2, 1)
        fast_input_dim = self.fast_frames * input_dim_per_timestep
        
        # Slow pathway: 매 2 timestep
        # T=7일 때: indices 0,2,4,6 (4 frames)
        self.slow_frames = (num_timesteps + 1) // 2
        slow_input_dim = self.slow_frames * input_dim_per_timestep
        
        # Fast MLP (Categorical LinearEncoder 스타일: 3-layer with ELU)
        self.fast_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fast_input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Slow MLP (동일 구조)
        self.slow_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(slow_input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        print(f"SlowFast MLP Configuration:")
        print(f"  Fast: {self.fast_frames} frames × {input_dim_per_timestep} = {fast_input_dim} → {output_dim}")
        print(f"  Slow: {self.slow_frames} frames × {input_dim_per_timestep} = {slow_input_dim} → {output_dim}")
        
    def forward(self, x):
        """
        Args:
            x: (B, num_timesteps * input_dim_per_timestep)
        Returns:
            out: (B, output_dim)
        """
        B = x.shape[0]
        
        if self.logger:
            self.logger.header("SlowFast MLP")
            self.logger.log("Input", x)
        
        # Reshape: (B, T*F) -> (B, T, F)
        x_reshaped = x.view(B, self.num_timesteps, self.input_dim_per_timestep)
        
        if self.logger:
            self.logger.log("Reshaped", x_reshaped)
        
        # === Fast Pathway ===
        # EgoPoser: x_fast = input_tensor[:,-input_tensor.shape[1]//2:,...]
        x_fast = x_reshaped[:, -self.fast_frames:, :]  # (B, fast_frames, F)
        x_fast_flat = x_fast.reshape(B, -1)            # (B, fast_frames * F)
        
        if self.logger:
            self.logger.log("Fast (selected frames)", x_fast)
            self.logger.log("Fast (flat)", x_fast_flat)
        
        # === Slow Pathway ===
        # EgoPoser: x_slow = input_tensor[:,::2,...]
        x_slow = x_reshaped[:, ::2, :]                 # (B, slow_frames, F)
        x_slow_flat = x_slow.reshape(B, -1)            # (B, slow_frames * F)
        
        if self.logger:
            self.logger.log("Slow (selected frames)", x_slow)
            self.logger.log("Slow (flat)", x_slow_flat)
        
        # === MLP Processing ===
        fast_out = self.fast_mlp(x_fast_flat)
        slow_out = self.slow_mlp(x_slow_flat)
        
        if self.logger:
            self.logger.log("Fast MLP output", fast_out)
            self.logger.log("Slow MLP output", slow_out)
        
        # === Fusion: Element-wise sum ===
        out = fast_out + slow_out
        
        if self.logger:
            self.logger.log("Fused output", out)
        
        return out


# =============================================================================
# Main Model: Categorical VQ-VAE + GMD + SlowFast
# =============================================================================

class Model(nn.Module):
    """
    Categorical VQ-VAE with GMD + SlowFast
    
    베이스: Categorical Codebook Matching (SIGGRAPH 2024) - 100% 유지
    추가: EgoPoser (ECCV 2024) GMD + SlowFast 아이디어
    
    변경사항:
    - Encoder/Estimator/Decoder: 변경 없음 (LinearEncoder)
    - Gumbel-Softmax: 변경 없음
    - 손실 함수: 변경 없음 (MSE + Matching)
    - 추가: GMD 전처리 + SlowFast 융합
    """
    def __init__(self, encoder, estimator, decoder, xNorm, yNorm, 
                 codebook_channels, codebook_dim,
                 # GMD + SlowFast 설정
                 use_gmd=True,
                 num_timesteps=7,
                 features_per_timestep=36,
                 slowfast_hidden_dim=512,
                 dropout=0.25,
                 debug_shapes=True):
        super(Model, self).__init__()

        # === 원본 Categorical VQ-VAE (변경 없음) ===
        self.Encoder = encoder
        self.Estimator = estimator
        self.Decoder = decoder
        self.XNorm = xNorm
        self.YNorm = yNorm
        self.C = codebook_channels
        self.D = codebook_dim
        
        # === GMD + SlowFast ===
        self.use_gmd = use_gmd
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        
        # Shape logger
        self.logger = ShapeLogger(enabled=debug_shapes) if debug_shapes else None
        
        if self.use_gmd:
            # GMD: 36 features → 42 features per timestep
            self.gmd = GMDPreprocessor(num_timesteps, features_per_timestep, self.logger)
            
            gmd_features_per_timestep = features_per_timestep + 6  # 42
            gmd_total_dim = num_timesteps * gmd_features_per_timestep  # 294
            
            # SlowFast: 294 → 252 (원본 차원으로 복원)
            original_input_dim = num_timesteps * features_per_timestep  # 252
            self.slowfast = SlowFastMLP(
                input_dim_per_timestep=gmd_features_per_timestep,
                num_timesteps=num_timesteps,
                output_dim=original_input_dim,
                hidden_dim=slowfast_hidden_dim,
                dropout=dropout,
                logger=self.logger
            )
            
            # Learnable residual weight
            self.residual_alpha = nn.Parameter(torch.tensor(0.5))
            
            print("\n" + "="*60)
            print("Model Configuration: Categorical VQ-VAE + GMD + SlowFast")
            print("="*60)
            print(f"Base: Categorical VQ-VAE (MLP-based, unchanged)")
            print(f"GMD: Enabled")
            print(f"  Input:  ({num_timesteps} × {features_per_timestep}) = {original_input_dim}")
            print(f"  Output: ({num_timesteps} × {gmd_features_per_timestep}) = {gmd_total_dim}")
            print(f"SlowFast: MLP-based (no Transformer)")
            print(f"  Output: {original_input_dim} (restored to original)")
            print(f"Residual: α={self.residual_alpha.item():.2f} (learnable)")
            print("="*60 + "\n")

    # === Gumbel-Softmax (원본과 동일) ===
    
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
    
    def forward(self, x, knn, t=None):
        """
        Forward pass
        
        Args:
            x: (B, input_dim) tracker features
            knn: K-nearest neighbor for Gumbel-Softmax
            t: (B, output_dim) target (training only)
        """
        if self.logger:
            self.logger.header("Forward Pass Start")
            self.logger.log("Input x", x)
            self.logger.log("knn", knn)
            if t is not None:
                self.logger.log("Target t", t)
        
        # === GMD + SlowFast Preprocessing ===
        if self.use_gmd:
            x_original = x.clone()
            
            # GMD preprocessing
            x_gmd = self.gmd(x)  # (B, 294)
            
            # SlowFast fusion
            x_enhanced = self.slowfast(x_gmd)  # (B, 252)
            
            # Learnable residual
            alpha = torch.sigmoid(self.residual_alpha)
            x = alpha * x_enhanced + (1 - alpha) * x_original
            
            if self.logger:
                self.logger.log("GMD output", x_gmd)
                self.logger.log("SlowFast output", x_enhanced)
                self.logger.log("Residual alpha", alpha)
                self.logger.log("After residual", x)
        
        # === 원본 Categorical VQ-VAE (변경 없음) ===
        
        # Training
        if t is not None:
            if self.logger:
                self.logger.header("Categorical VQ-VAE (Training)")
            
            # Normalize
            x_norm = utility.Normalize(x, self.XNorm)
            t_norm = utility.Normalize(t, self.YNorm)
            
            if self.logger:
                self.logger.log("x normalized", x_norm)
                self.logger.log("t normalized", t_norm)

            # Encode Y (Teacher)
            encoder_input = torch.cat((t_norm, x_norm), dim=1)
            if self.logger:
                self.logger.log("Encoder input (t+x)", encoder_input)
            
            target_logits = self.Encoder(encoder_input)
            if self.logger:
                self.logger.log("Encoder output (logits)", target_logits)
            
            target_probs, target = self.sample(target_logits, knn)
            if self.logger:
                self.logger.log("Target (sampled)", target)

            # Encode X (Student)
            estimate_logits = self.Estimator(x_norm)
            if self.logger:
                self.logger.log("Estimator output (logits)", estimate_logits)
            
            estimate_probs, estimate = self.sample(estimate_logits, knn)
            if self.logger:
                self.logger.log("Estimate (sampled)", estimate)

            # Decode
            y = self.Decoder(target)
            if self.logger:
                self.logger.log("Decoder output", y)

            # Renormalize
            y_output = utility.Renormalize(y, self.YNorm)
            if self.logger:
                self.logger.log("Output (renormalized)", y_output)
                self.logger.step()

            return (y_output, target_logits, target_probs, target, 
                    estimate_logits, estimate_probs, estimate)
                
        # Inference
        else:
            if self.logger:
                self.logger.header("Categorical VQ-VAE (Inference)")
            
            x_norm = utility.Normalize(x, self.XNorm)
            
            estimate_logits = self.Estimator(x_norm)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            y = self.Decoder(estimate)
            y_output = utility.Renormalize(y, self.YNorm)
            
            if self.logger:
                self.logger.log("Output", y_output)
                self.logger.step()

            return y_output, estimate


# =============================================================================
# 검증 함수들
# =============================================================================

def run_shape_verification(input_dim, output_dim, batch_size=4):
    """
    Shape 검증 - 전체 파이프라인 테스트
    """
    print("\n" + "="*60)
    print("SHAPE VERIFICATION TEST")
    print("="*60)
    
    # 설정
    num_timesteps = 7
    features_per_timestep = 36
    expected_input_dim = num_timesteps * features_per_timestep  # 252
    
    print(f"\nExpected input_dim: {expected_input_dim}")
    print(f"Actual input_dim: {input_dim}")
    
    if input_dim != expected_input_dim:
        print(f"⚠️  Warning: input_dim mismatch!")
        print(f"    Adjusting num_timesteps or features_per_timestep may be needed")
        # 실제 데이터에 맞게 조정
        if input_dim % num_timesteps == 0:
            features_per_timestep = input_dim // num_timesteps
            print(f"    Adjusted: {num_timesteps} timesteps × {features_per_timestep} features")
        else:
            print(f"    Cannot evenly divide. Using input_dim directly.")
            num_timesteps = 1
            features_per_timestep = input_dim
    
    # 테스트 텐서
    x = torch.randn(batch_size, input_dim)
    t = torch.randn(batch_size, output_dim)
    knn = torch.ones(1)
    
    print(f"\nTest tensors:")
    print(f"  x: {list(x.shape)}")
    print(f"  t: {list(t.shape)}")
    print(f"  knn: {list(knn.shape)}")
    
    # GMD 테스트
    print(f"\n--- GMD Preprocessor ---")
    gmd = GMDPreprocessor(num_timesteps, features_per_timestep)
    x_gmd = gmd(x)
    gmd_expected_dim = num_timesteps * (features_per_timestep + 6)
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(x_gmd.shape)}")
    print(f"  Expected: (B, {gmd_expected_dim})")
    assert x_gmd.shape == (batch_size, gmd_expected_dim), "GMD shape mismatch!"
    print(f"  ✓ GMD OK")
    
    # SlowFast 테스트
    print(f"\n--- SlowFast MLP ---")
    slowfast = SlowFastMLP(
        input_dim_per_timestep=features_per_timestep + 6,
        num_timesteps=num_timesteps,
        output_dim=input_dim
    )
    x_sf = slowfast(x_gmd)
    print(f"  Input:  {list(x_gmd.shape)}")
    print(f"  Output: {list(x_sf.shape)}")
    print(f"  Expected: (B, {input_dim})")
    assert x_sf.shape == (batch_size, input_dim), "SlowFast shape mismatch!"
    print(f"  ✓ SlowFast OK")
    
    # 전체 모델 테스트
    print(f"\n--- Full Model ---")
    codebook_channels = 128
    codebook_dim = 8
    codebook_size = codebook_channels * codebook_dim
    
    model = Model(
        encoder=modules.LinearEncoder(input_dim + output_dim, 512, 512, codebook_size, 0.25),
        estimator=modules.LinearEncoder(input_dim, 512, 512, codebook_size, 0.25),
        decoder=modules.LinearEncoder(codebook_size, 512, 512, output_dim, 0.0),
        xNorm=Parameter(torch.zeros(2, input_dim), requires_grad=False),
        yNorm=Parameter(torch.zeros(2, output_dim), requires_grad=False),
        codebook_channels=codebook_channels,
        codebook_dim=codebook_dim,
        use_gmd=True,
        num_timesteps=num_timesteps,
        features_per_timestep=features_per_timestep,
        debug_shapes=False  # 검증 중에는 로깅 끄기
    )
    
    # Forward (training)
    y, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = model(x, knn, t)
    print(f"  Output y: {list(y.shape)}")
    print(f"  target: {list(target.shape)}")
    print(f"  estimate: {list(estimate.shape)}")
    print(f"  ✓ Full model OK")
    
    # Forward (inference)
    y_inf, estimate_inf = model(x, knn)
    print(f"  Inference y: {list(y_inf.shape)}")
    print(f"  Inference estimate: {list(estimate_inf.shape)}")
    print(f"  ✓ Inference OK")
    
    print("\n" + "="*60)
    print("ALL SHAPE VERIFICATIONS PASSED!")
    print("="*60 + "\n")
    
    return True


# =============================================================================
# Training Script
# =============================================================================

if __name__ == '__main__':
    # Dataset
    name = "LowerBody"
    directory = "../../Datasets/" + name
    id = name + "_GMD_SlowFast_Final_" + utility.GetFileID(__file__)
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

    # === 먼저 Shape 검증 수행 ===
    print("\n" + "="*60)
    print("PRE-TRAINING SHAPE VERIFICATION")
    print("="*60)
    run_shape_verification(input_dim, output_dim, batch_size=4)
    
    # Hyperparameters (원본 Categorical과 동일)
    epochs = 150
    batch_size = 32
    dropout = 0.25

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    encoder_dim = 1024
    estimator_dim = 1024
    decoder_dim = 1024

    codebook_channels = 128
    codebook_dim = 8
    codebook_size = codebook_channels * codebook_dim
    
    # GMD + SlowFast 설정
    use_gmd = True
    
    # 입력 차원에 따라 timesteps 자동 계산
    num_timesteps = 7
    features_per_timestep = 36
    if input_dim != num_timesteps * features_per_timestep:
        if input_dim % 7 == 0:
            features_per_timestep = input_dim // 7
            print(f"Adjusted: {num_timesteps} timesteps × {features_per_timestep} features = {input_dim}")
        else:
            print(f"Warning: Cannot divide input_dim={input_dim} by 7. Using direct input.")
            num_timesteps = 1
            features_per_timestep = input_dim
    
    slowfast_hidden_dim = 512
    debug_shapes = True  # 첫 번째 배치에서 shape 로깅
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {name}")
    print(f"Input Features: {input_dim}")
    print(f"Output Features: {output_dim}")
    print(f"Sample Count: {sample_count}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"GMD: {use_gmd}")
    print(f"Timesteps: {num_timesteps}")
    print(f"Features/Timestep: {features_per_timestep}")
    print("="*60)

    # Create model
    network = utility.ToDevice(Model(
        encoder=modules.LinearEncoder(input_dim + output_dim, encoder_dim, encoder_dim, codebook_size, dropout),
        estimator=modules.LinearEncoder(input_dim, estimator_dim, estimator_dim, codebook_size, dropout),
        decoder=modules.LinearEncoder(codebook_size, decoder_dim, decoder_dim, output_dim, 0.0),
        
        xNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/InputNormalization.txt", True)), requires_grad=False),
        yNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/OutputNormalization.txt", True)), requires_grad=False),
        
        codebook_channels=codebook_channels,
        codebook_dim=codebook_dim,
        
        use_gmd=use_gmd,
        num_timesteps=num_timesteps,
        features_per_timestep=features_per_timestep,
        slowfast_hidden_dim=slowfast_hidden_dim,
        dropout=dropout,
        debug_shapes=debug_shapes
    ))
    
    # 모델 파라미터 수
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
        
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(
        optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count,
        restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True
    )
    loss_function = nn.MSELoss()

    # Plotting
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

    # Training loop
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

            # Forward
            prediction, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = network(
                xBatch, knn=torch.ones(1, device=xBatch.device), t=yBatch
            )

            # Losses (원본 Categorical과 동일)
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

            # Visualization
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
                    pred, _, _, tgt, _, _, est = network(
                        xBatch, knn=torch.zeros(1, device=xBatch.device), t=yBatch
                    )
                    input_sequences.append(Item(xBatch))
                    output_sequences.append(Item(yBatch))
                    target_sequences.append(Item(tgt))
                    estimate_sequences.append(Item(est))
                    predictions_sequences.append(Item(pred))
                
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

        # Save
        utility.SaveONNX(
            path=save + '/' + id + '_' + str(epoch + 1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim), torch.ones(1)),
            input_names=['X', 'K'],
            output_names=['Y', 'Code'],
            dynamic_axes={'K': {0: 'Size'}}
        )
