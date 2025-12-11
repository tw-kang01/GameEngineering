"""
TrackerBody MLP with GMD + SlowFast Enhancement
================================================

TrackerBodyPredictor용 Simple MLP에 EgoPoser의 GMD + SlowFast 적용

데이터 구조:
- Input: 576 = 3 trackers × 16 timesteps × 12 features
  - [Head_t0..t15 (192), LWrist_t0..t15 (192), RWrist_t0..t15 (192)]
  - 각 tracker/timestep: Position(3) + Forward(3) + Up(3) + Velocity(3) = 12
- Output: 231 = RootUpdate(3) + 19 bones × 12 features

GMD Index 매핑 (현재 데이터 구조):
- Head Position X at t: t*12 + 0, Z: t*12 + 2 (t=0..15, indices 0-191)
- LWrist Position X at t: 192 + t*12 + 0, Z: 192 + t*12 + 2
- RWrist Position X at t: 384 + t*12 + 0, Z: 384 + t*12 + 2

Author: Integration of EgoPoser GMD+SlowFast into Categorical MLP
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


# =============================================================================
# Shape Logger - 디버깅용
# =============================================================================

class ShapeLogger:
    """학습 중 텐서 shape를 로깅하는 유틸리티"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.logged_once = False
        
    def log(self, name, tensor, force=False):
        if not self.enabled:
            return
        if not self.logged_once or force:
            if isinstance(tensor, torch.Tensor):
                print(f"  [{name}] shape: {list(tensor.shape)}, dtype: {tensor.dtype}")
            else:
                print(f"  [{name}] type: {type(tensor)}, value: {tensor}")
                
    def step(self):
        if not self.logged_once:
            self.logged_once = True
            
    def header(self, title):
        if not self.enabled or self.logged_once:
            return
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")


# =============================================================================
# MPJPE / MPJVE Metrics
# =============================================================================

def compute_mpjpe_mpjve(pred, target, num_joints=19, root_dim=3):
    """
    Compute MPJPE (Mean Per Joint Position Error) and MPJVE (Mean Per Joint Velocity Error)

    Output structure: 231 = RootUpdate(3) + 19 bones × 12 features
    Each bone: Position(3) + Forward(3) + Up(3) + Velocity(3)

    Args:
        pred: (B, 231) predicted output
        target: (B, 231) ground truth output
        num_joints: number of joints (19)
        root_dim: root update dimension (3)

    Returns:
        mpjpe: Mean Per Joint Position Error (meters)
        mpjve: Mean Per Joint Velocity Error (m/s)
    """
    B = pred.shape[0]

    # Extract positions and velocities for all joints
    # Skip root update (first 3), then each joint has 12 features
    positions_pred = []
    positions_target = []
    velocities_pred = []
    velocities_target = []

    for j in range(num_joints):
        # Joint j starts at index: root_dim + j * 12
        joint_start = root_dim + j * 12

        # Position: offset 0-2
        pos_pred = pred[:, joint_start:joint_start+3]
        pos_target = target[:, joint_start:joint_start+3]
        positions_pred.append(pos_pred)
        positions_target.append(pos_target)

        # Velocity: offset 9-11
        vel_pred = pred[:, joint_start+9:joint_start+12]
        vel_target = target[:, joint_start+9:joint_start+12]
        velocities_pred.append(vel_pred)
        velocities_target.append(vel_target)

    # Stack: (B, num_joints, 3)
    positions_pred = torch.stack(positions_pred, dim=1)
    positions_target = torch.stack(positions_target, dim=1)
    velocities_pred = torch.stack(velocities_pred, dim=1)
    velocities_target = torch.stack(velocities_target, dim=1)

    # MPJPE: mean of L2 norm of position errors
    pos_error = torch.norm(positions_pred - positions_target, dim=2)  # (B, num_joints)
    mpjpe = pos_error.mean()  # scalar

    # MPJVE: mean of L2 norm of velocity errors
    vel_error = torch.norm(velocities_pred - velocities_target, dim=2)  # (B, num_joints)
    mpjve = vel_error.mean()  # scalar

    return mpjpe.item(), mpjve.item()


# =============================================================================
# GMD Preprocessor for TrackerBody (16 timesteps, 새로운 데이터 구조)
# =============================================================================

class GMDPreprocessor(nn.Module):
    """
    Global Motion Decomposition Preprocessor for TrackerBody
    
    TrackerBody 데이터 구조:
    - 576 features = [Head_all_timesteps(192), LWrist_all_timesteps(192), RWrist_all_timesteps(192)]
    - 각 tracker: 16 timesteps × 12 features
    - 각 timestep: Position(3) + Forward(3) + Up(3) + Velocity(3)
    
    GMD 적용:
    - Spatial Normalization: Head XZ 기준 상대 좌표
    - Temporal Normalization: 첫 프레임 대비 delta
    
    주의: 학습 파라미터 없음 (순수 전처리)
    """
    def __init__(self, num_timesteps=16, features_per_tracker_timestep=12, logger=None):
        super(GMDPreprocessor, self).__init__()
        self.num_timesteps = num_timesteps  # 16
        self.features_per_tracker_timestep = features_per_tracker_timestep  # 12
        self.logger = logger
        
        # Tracker별 시작 인덱스 (flat 기준)
        self.head_start = 0                  # 0-191
        self.lwrist_start = 192              # 192-383
        self.rwrist_start = 384              # 384-575
        
        # Position 인덱스 (각 tracker 내에서)
        # Position(3) + Forward(3) + Up(3) + Velocity(3) = 12
        self.pos_x_offset = 0
        self.pos_y_offset = 1
        self.pos_z_offset = 2
        
    def _get_position_indices(self, tracker_start, timestep, axis_offset):
        """특정 tracker, timestep, axis의 인덱스 계산"""
        return tracker_start + timestep * self.features_per_tracker_timestep + axis_offset
    
    def forward(self, x):
        """
        Args:
            x: (B, 576) flat input tensor
        Returns:
            x_gmd: (B, 576 + 96) = (B, 672) GMD 적용된 입력
                   96 = 3 trackers × 16 timesteps × 2 (XZ delta)
        """
        B = x.shape[0]
        device = x.device
        
        if self.logger:
            self.logger.header("GMD Preprocessor (TrackerBody)")
            self.logger.log("Input", x)
        
        # Clone for modification
        x_modified = x.clone()
        
        # === Spatial Normalization ===
        # 각 timestep에서 Head XZ를 기준으로 모든 tracker의 XZ를 상대화
        for t in range(self.num_timesteps):
            # Head position at timestep t
            head_x_idx = self._get_position_indices(self.head_start, t, self.pos_x_offset)
            head_z_idx = self._get_position_indices(self.head_start, t, self.pos_z_offset)
            
            head_x = x_modified[:, head_x_idx].clone()  # (B,)
            head_z = x_modified[:, head_z_idx].clone()  # (B,)
            
            # Head XZ → 0
            x_modified[:, head_x_idx] -= head_x
            x_modified[:, head_z_idx] -= head_z
            
            # LWrist XZ → Head 기준 상대
            lwrist_x_idx = self._get_position_indices(self.lwrist_start, t, self.pos_x_offset)
            lwrist_z_idx = self._get_position_indices(self.lwrist_start, t, self.pos_z_offset)
            x_modified[:, lwrist_x_idx] -= head_x
            x_modified[:, lwrist_z_idx] -= head_z
            
            # RWrist XZ → Head 기준 상대
            rwrist_x_idx = self._get_position_indices(self.rwrist_start, t, self.pos_x_offset)
            rwrist_z_idx = self._get_position_indices(self.rwrist_start, t, self.pos_z_offset)
            x_modified[:, rwrist_x_idx] -= head_x
            x_modified[:, rwrist_z_idx] -= head_z
        
        if self.logger:
            self.logger.log("After Spatial Norm", x_modified)
        
        # === Temporal Normalization ===
        # delta = position[t] - position[0] (각 tracker별로)
        deltas = []
        
        for t in range(self.num_timesteps):
            # Head delta
            head_x_t = x_modified[:, self._get_position_indices(self.head_start, t, self.pos_x_offset)]
            head_z_t = x_modified[:, self._get_position_indices(self.head_start, t, self.pos_z_offset)]
            head_x_0 = x_modified[:, self._get_position_indices(self.head_start, 0, self.pos_x_offset)]
            head_z_0 = x_modified[:, self._get_position_indices(self.head_start, 0, self.pos_z_offset)]
            
            # LWrist delta
            lwrist_x_t = x_modified[:, self._get_position_indices(self.lwrist_start, t, self.pos_x_offset)]
            lwrist_z_t = x_modified[:, self._get_position_indices(self.lwrist_start, t, self.pos_z_offset)]
            lwrist_x_0 = x_modified[:, self._get_position_indices(self.lwrist_start, 0, self.pos_x_offset)]
            lwrist_z_0 = x_modified[:, self._get_position_indices(self.lwrist_start, 0, self.pos_z_offset)]
            
            # RWrist delta
            rwrist_x_t = x_modified[:, self._get_position_indices(self.rwrist_start, t, self.pos_x_offset)]
            rwrist_z_t = x_modified[:, self._get_position_indices(self.rwrist_start, t, self.pos_z_offset)]
            rwrist_x_0 = x_modified[:, self._get_position_indices(self.rwrist_start, 0, self.pos_x_offset)]
            rwrist_z_0 = x_modified[:, self._get_position_indices(self.rwrist_start, 0, self.pos_z_offset)]
            
            # Stack deltas for this timestep: 6 values
            delta_t = torch.stack([
                head_x_t - head_x_0,
                head_z_t - head_z_0,
                lwrist_x_t - lwrist_x_0,
                lwrist_z_t - lwrist_z_0,
                rwrist_x_t - rwrist_x_0,
                rwrist_z_t - rwrist_z_0
            ], dim=1)  # (B, 6)
            
            deltas.append(delta_t)
        
        # Stack all timesteps: (B, 16, 6) → flatten to (B, 96)
        all_deltas = torch.stack(deltas, dim=1)  # (B, 16, 6)
        all_deltas_flat = all_deltas.view(B, -1)  # (B, 96)
        
        if self.logger:
            self.logger.log("Temporal Deltas", all_deltas_flat)
        
        # Concat: (B, 576) + (B, 96) → (B, 672)
        x_gmd = torch.cat([x_modified, all_deltas_flat], dim=1)
        
        if self.logger:
            self.logger.log("GMD Output", x_gmd)
        
        return x_gmd


# =============================================================================
# SlowFast MLP for TrackerBody (16 timesteps) - EgoPoser Style
# =============================================================================

class SlowFastMLP(nn.Module):
    """
    SlowFast Feature Fusion for TrackerBody (EgoPoser Style)

    EgoPoser 원본 구현을 따름:
    - Fast pathway: 후반 절반 frames (indices 8-15) - 최근 움직임
    - Slow pathway: stride 2 (indices 0,2,4,6,8,10,12,14) - 전체 시간대
    - 공유 embedding 후 합산 → 공통 MLP 처리

    핵심 변경 (EgoPoser 스타일):
    1. Fast/Slow가 같은 shared_embedding 사용 (파라미터 공유)
    2. Embedding 후 합산 → fusion_mlp로 처리
    3. 개별 MLP 대신 공유 구조

    입력: GMD 적용 후 (B, 672) = (B, 576 + 96)
    출력: (B, 576) - 원본 차원으로 복원
    """
    def __init__(self, gmd_dim=672, original_dim=576, num_timesteps=16,
                 hidden_dim=512, dropout=0.25, logger=None):
        super(SlowFastMLP, self).__init__()

        self.num_timesteps = num_timesteps
        self.gmd_dim = gmd_dim
        self.original_dim = original_dim
        self.logger = logger

        # GMD 적용 후 구조:
        # - 원본 features: 576 = 3 trackers × 16 timesteps × 12 features
        # - Delta features: 96 = 16 timesteps × 6 deltas
        # - Total: 672

        # 각 tracker의 feature 수
        self.features_per_tracker = 192  # 16 × 12
        self.delta_features = 96  # 16 × 6

        # Fast pathway: 후반 8 frames (indices 8-15)
        self.fast_frames = 8
        # Slow pathway: stride 2 (indices 0,2,4,6,8,10,12,14) = 8 frames
        self.slow_frames = 8

        # Fast/Slow 입력 차원 (동일)
        # 각 frame: 3 trackers × 12 features + 6 deltas = 42
        # 8 frames: 8 × 42 = 336
        pathway_input_dim = 8 * (12 * 3 + 6)  # 336

        # === EgoPoser Style: 공유 Embedding ===
        # Fast와 Slow가 같은 embedding layer 사용
        self.shared_embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pathway_input_dim, hidden_dim),
            nn.ELU()
        )

        # === Fusion MLP: embedding 합산 후 처리 ===
        # EgoPoser는 Transformer 사용, 여기선 MLP로 대체
        self.fusion_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, original_dim)
        )

        print(f"SlowFast MLP Configuration (EgoPoser Style):")
        print(f"  Shared Embedding: {pathway_input_dim} → {hidden_dim}")
        print(f"  Fusion MLP: {hidden_dim} → {original_dim}")
        print(f"  Fast: frames 8-15 (recent {self.fast_frames} frames)")
        print(f"  Slow: frames 0,2,4,...,14 (strided {self.slow_frames} frames)")
        
    def _extract_frames(self, x_gmd, frame_indices):
        """
        GMD 출력에서 특정 프레임들의 features 추출
        
        Args:
            x_gmd: (B, 672) GMD output
            frame_indices: list of frame indices to extract
        Returns:
            extracted: (B, num_frames * (12*3 + 6))
        """
        B = x_gmd.shape[0]
        
        extracted_parts = []
        
        for t in frame_indices:
            # Head features at frame t: indices [t*12 : (t+1)*12] within head block
            head_start = t * 12
            head_end = (t + 1) * 12
            extracted_parts.append(x_gmd[:, head_start:head_end])
            
            # LWrist features at frame t
            lwrist_start = 192 + t * 12
            lwrist_end = 192 + (t + 1) * 12
            extracted_parts.append(x_gmd[:, lwrist_start:lwrist_end])
            
            # RWrist features at frame t
            rwrist_start = 384 + t * 12
            rwrist_end = 384 + (t + 1) * 12
            extracted_parts.append(x_gmd[:, rwrist_start:rwrist_end])
            
            # Delta features at frame t: indices [576 + t*6 : 576 + (t+1)*6]
            delta_start = 576 + t * 6
            delta_end = 576 + (t + 1) * 6
            extracted_parts.append(x_gmd[:, delta_start:delta_end])
        
        return torch.cat(extracted_parts, dim=1)
        
    def forward(self, x_gmd):
        """
        EgoPoser Style Forward Pass

        Args:
            x_gmd: (B, 672) GMD processed input
        Returns:
            out: (B, 576) restored to original dimension

        Flow (EgoPoser Style):
            1. Extract Fast/Slow frames
            2. Shared embedding (same weights for both)
            3. Sum embeddings
            4. Fusion MLP
        """
        B = x_gmd.shape[0]

        if self.logger:
            self.logger.header("SlowFast MLP (EgoPoser Style)")
            self.logger.log("Input (GMD)", x_gmd)

        # === Fast Pathway ===
        # 후반 8 frames: indices 8-15 (EgoPoser: input[:, -T//2:, ...])
        fast_indices = list(range(8, 16))  # [8,9,10,11,12,13,14,15]
        x_fast = self._extract_frames(x_gmd, fast_indices)

        if self.logger:
            self.logger.log("Fast input", x_fast)

        # === Slow Pathway ===
        # stride 2: indices 0,2,4,6,8,10,12,14 (EgoPoser: input[:, ::2, ...])
        slow_indices = list(range(0, 16, 2))  # [0,2,4,6,8,10,12,14]
        x_slow = self._extract_frames(x_gmd, slow_indices)

        if self.logger:
            self.logger.log("Slow input", x_slow)

        # === Shared Embedding (EgoPoser Style) ===
        # 같은 embedding layer로 처리 → 파라미터 공유
        fast_embed = self.shared_embedding(x_fast)
        slow_embed = self.shared_embedding(x_slow)

        if self.logger:
            self.logger.log("Fast embedding", fast_embed)
            self.logger.log("Slow embedding", slow_embed)

        # === Fusion: Sum after embedding (EgoPoser Style) ===
        # EgoPoser: x = x_fast + x_slow (embedding 후 합산)
        fused = fast_embed + slow_embed

        if self.logger:
            self.logger.log("Fused embedding", fused)

        # === Fusion MLP (replaces Transformer) ===
        out = self.fusion_mlp(fused)

        if self.logger:
            self.logger.log("Output", out)

        return out


# =============================================================================
# Main Model: TrackerBody MLP + GMD + SlowFast
# =============================================================================

class Model(nn.Module):
    """
    TrackerBody MLP with GMD + SlowFast Enhancement
    
    베이스: Categorical Simple MLP (변경 없음)
    추가: EgoPoser GMD + SlowFast
    
    구조:
    1. GMD Preprocessing: (B, 576) → (B, 672)
    2. SlowFast Fusion: (B, 672) → (B, 576)
    3. Residual Connection: α * SlowFast + (1-α) * Original
    4. Simple MLP: (B, 576) → (B, 231)
    """
    def __init__(self, rng, layers, activations, dropout, input_norm, output_norm,
                 use_gmd=True, use_residual=True, num_timesteps=16, hidden_dim=512, debug_shapes=True):
        super(Model, self).__init__()
        
        self.rng = rng
        self.layers = layers
        self.activations = activations
        self.dropout = dropout
        self.use_gmd = use_gmd
        self.use_residual = use_residual
        
        # Normalization parameters
        self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)
        self.Ynorm = Parameter(torch.from_numpy(output_norm), requires_grad=False)
        
        # MLP weights and biases (original Categorical style)
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for i in range(len(layers) - 1):
            self.W.append(self._init_weights([layers[i], layers[i+1]]))
            self.b.append(self._init_bias([1, layers[i+1]]))
        
        # Shape logger
        self.logger = ShapeLogger(enabled=debug_shapes) if debug_shapes else None
        
        # GMD + SlowFast modules
        if self.use_gmd:
            input_dim = layers[0]  # 576
            
            self.gmd = GMDPreprocessor(
                num_timesteps=num_timesteps,
                features_per_tracker_timestep=12,
                logger=self.logger
            )
            
            gmd_output_dim = input_dim + num_timesteps * 6  # 576 + 96 = 672
            
            self.slowfast = SlowFastMLP(
                gmd_dim=gmd_output_dim,
                original_dim=input_dim,
                num_timesteps=num_timesteps,
                hidden_dim=hidden_dim,
                dropout=dropout,
                logger=self.logger
            )
            
            # Learnable residual weight (initialized to 0.5)
            if self.use_residual:
                self.residual_alpha = nn.Parameter(torch.tensor(0.5))
            else:
                self.residual_alpha = None
            
            print(f"\n{'='*60}")
            print(f"Model: TrackerBody MLP + GMD + SlowFast")
            print(f"{'='*60}")
            print(f"GMD: Enabled")
            print(f"  Input: {input_dim} → {gmd_output_dim}")
            print(f"SlowFast: MLP-based fusion")
            print(f"  Output: {gmd_output_dim} → {input_dim}")
            if self.use_residual:
                print(f"Residual: α = {self.residual_alpha.item():.2f} (learnable)")
            else:
                print(f"Residual: DISABLED (pure GMD+SlowFast)")
            print(f"MLP Layers: {layers}")
            print(f"{'='*60}\n")
    
    def _init_weights(self, shape):
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(
            self.rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32
        )
        return Parameter(torch.from_numpy(alpha), requires_grad=True)
    
    def _init_bias(self, shape):
        return Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (B, 576) tracker features
        Returns:
            y: (B, 231) predicted upper body + root update
        """
        if self.logger:
            self.logger.header("Forward Pass")
            self.logger.log("Input", x)
        
        # === GMD + SlowFast Preprocessing ===
        if self.use_gmd:
            x_original = x.clone()
            
            # GMD preprocessing: (B, 576) → (B, 672)
            x_gmd = self.gmd(x)
            
            # SlowFast fusion: (B, 672) → (B, 576)
            x_enhanced = self.slowfast(x_gmd)
            
            # Residual connection (optional)
            if self.use_residual and self.residual_alpha is not None:
                alpha = torch.sigmoid(self.residual_alpha)
                x = alpha * x_enhanced + (1 - alpha) * x_original
                if self.logger:
                    self.logger.log("Residual alpha", alpha)
            else:
                # Pure GMD+SlowFast output (no residual)
                x = x_enhanced
                if self.logger:
                    self.logger.log("Pure GMD+SlowFast", "No residual")
            
            if self.logger:
                self.logger.log("After preprocessing", x)
        
        # === Original MLP Forward (Categorical style) ===
        x = utility.Normalize(x, self.Xnorm)
        
        y = x
        for i in range(len(self.layers) - 1):
            y = F.dropout(y, self.dropout, training=self.training)
            y = y.matmul(self.W[i]) + self.b[i]
            if self.activations[i] is not None:
                y = self.activations[i](y)
        
        y = utility.Renormalize(y, self.Ynorm)
        
        if self.logger:
            self.logger.log("Output", y)
            self.logger.step()
        
        return y


# =============================================================================
# Shape Verification
# =============================================================================

def run_shape_verification(input_dim=576, output_dim=231, batch_size=4):
    """Shape 검증 테스트"""
    print("\n" + "="*60)
    print("SHAPE VERIFICATION TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test tensors
    x = torch.randn(batch_size, input_dim).to(device)
    print(f"\nInput: {list(x.shape)}")
    
    # === GMD Test ===
    print(f"\n--- GMD Preprocessor ---")
    gmd = GMDPreprocessor(num_timesteps=16, features_per_tracker_timestep=12).to(device)
    x_gmd = gmd(x)
    expected_gmd_dim = input_dim + 16 * 6  # 576 + 96 = 672
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(x_gmd.shape)}")
    print(f"  Expected: (B, {expected_gmd_dim})")
    assert x_gmd.shape == (batch_size, expected_gmd_dim), f"GMD shape mismatch! Got {x_gmd.shape}"
    print(f"  ✓ GMD OK")
    
    # === SlowFast Test ===
    print(f"\n--- SlowFast MLP ---")
    slowfast = SlowFastMLP(
        gmd_dim=expected_gmd_dim,
        original_dim=input_dim,
        num_timesteps=16
    ).to(device)
    x_sf = slowfast(x_gmd)
    print(f"  Input:  {list(x_gmd.shape)}")
    print(f"  Output: {list(x_sf.shape)}")
    print(f"  Expected: (B, {input_dim})")
    assert x_sf.shape == (batch_size, input_dim), f"SlowFast shape mismatch! Got {x_sf.shape}"
    print(f"  ✓ SlowFast OK")
    
    # === Full Model Test ===
    print(f"\n--- Full Model ---")
    rng = np.random.RandomState(23456)
    layers = [input_dim, 512, 512, output_dim]
    activations = [F.elu, F.elu, None]
    
    # Proper normalization: mean=0, std=1 (identity transform)
    input_norm = np.zeros((2, input_dim), dtype=np.float32)
    input_norm[1, :] = 1.0  # std = 1
    output_norm = np.zeros((2, output_dim), dtype=np.float32)
    output_norm[1, :] = 1.0  # std = 1
    
    model = Model(
        rng=rng,
        layers=layers,
        activations=activations,
        dropout=0.25,
        input_norm=input_norm,
        output_norm=output_norm,
        use_gmd=True,
        num_timesteps=16,
        hidden_dim=512,
        debug_shapes=False
    ).to(device)
    
    y = model(x)
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(y.shape)}")
    print(f"  Expected: (B, {output_dim})")
    assert y.shape == (batch_size, output_dim), f"Model shape mismatch! Got {y.shape}"
    print(f"  ✓ Full Model OK")
    
    # === Gradient Flow Test ===
    print(f"\n--- Gradient Flow Test ---")
    model.train()
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Check gradients
    grad_ok = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  ✗ No gradient for: {name}")
            grad_ok = False
    
    if grad_ok:
        print(f"  ✓ All gradients computed")
    
    # Check residual_alpha gradient
    if model.use_gmd:
        if model.residual_alpha.grad is not None:
            print(f"  ✓ residual_alpha gradient: {model.residual_alpha.grad.item():.6f}")
        else:
            print(f"  ✗ residual_alpha has no gradient")
    
    # === Parameter Count ===
    print(f"\n--- Parameter Count ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("ALL SHAPE VERIFICATIONS PASSED!")
    print("="*60 + "\n")
    
    return True


# =============================================================================
# Training Script
# =============================================================================

if __name__ == '__main__':
    # === Configuration ===
    name = "Trackerbodypredictor"  # 폴더명에 맞게
    
    # 데이터셋 경로: 환경변수 우선, 없으면 기본 경로 사용
    env_dataset_path = os.environ.get("TRACKERBODY_DATASET_PATH")
    if env_dataset_path and os.path.isdir(env_dataset_path):
        directory = env_dataset_path
        print(f"Using dataset from environment variable: {directory}")
    else:
        directory = "../../Datasets/" + name
        print(f"Using default dataset path: {directory}")
    
    id = name + "_GMD_SlowFast_" + utility.GetFileID(__file__)
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
    import os
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
        print(f"\n⚠️  WARNING: File size mismatch detected!")
        print(f"  Using actual sample count: {min(actual_input_samples, actual_output_samples):,}")
        # Override sample count with actual value
        Xshape[0] = min(actual_input_samples, actual_output_samples)
        Yshape[0] = min(actual_input_samples, actual_output_samples)
    print(f"{'='*60}\n")
    
    # === Seed ===
    seed = 23456
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    
    # === Hyperparameters ===
    epochs = 50
    batch_size = 32
    dropout = 0.25
    
    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2
    
    sample_count = Xshape[0]
    input_dim = Xshape[1]
    output_dim = Yshape[1]
    
    hidden_dim = 512
    num_timesteps = 16  # TrackerBody uses 16 timesteps
    
    # GMD + SlowFast 설정
    use_gmd = True
    use_residual = False  # True: α * enhanced + (1-α) * original, False: pure GMD+SlowFast
    debug_shapes = True  # 첫 번째 배치에서 shape 로깅
    
    # MLP layers (원본과 동일)
    layers = [input_dim, hidden_dim, hidden_dim, output_dim]
    activations = [F.elu, F.elu, None]
    
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
    print(f"Residual: {use_residual}")
    print(f"Timesteps: {num_timesteps}")
    print(f"Network Structure: {layers}")
    print("="*60)
    
    # === Shape Verification ===
    print("\nRunning shape verification...")
    run_shape_verification(input_dim, output_dim, batch_size=4)
    
    # === Create Model ===
    network = Model(
        rng=rng,
        layers=layers,
        activations=activations,
        dropout=dropout,
        input_norm=Xnorm,
        output_norm=Ynorm,
        use_gmd=use_gmd,
        use_residual=use_residual,
        num_timesteps=num_timesteps,
        hidden_dim=hidden_dim,
        debug_shapes=debug_shapes
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

    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(I)
        error = 0.0
        epoch_mpjpe = 0.0
        epoch_mpjve = 0.0
        num_batches = 0
        last_progress = -1  # 중복 출력 방지

        for i in range(0, sample_count, batch_size):
            # 10% 단위로만 progress 출력 (로그 파일 크기 감소)
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

            error += loss.item()

            # MPJPE/MPJVE (original space, no grad)
            with torch.no_grad():
                mpjpe, mpjve = compute_mpjpe_mpjve(yPred, yBatch)
                epoch_mpjpe += mpjpe
                epoch_mpjve += mpjve
                num_batches += 1
        
        # === Save Model ===
        utility.SaveONNX(
            path=save + '/' + id + '_' + str(epoch+1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim)),
            input_names=['X'],
            output_names=['Y']
        )
        
        # === Log Progress ===
        avg_loss = error / num_batches
        avg_mpjpe = epoch_mpjpe / num_batches
        avg_mpjve = epoch_mpjve / num_batches

        # MPJPE in cm for readability (assuming data is in meters)
        avg_mpjpe_cm = avg_mpjpe * 100

        if use_gmd and network.residual_alpha is not None:
            alpha_val = torch.sigmoid(network.residual_alpha).item()
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}, MPJPE: {avg_mpjpe_cm:.2f}cm, MPJVE: {avg_mpjve:.4f}, α: {alpha_val:.4f}')
        elif use_gmd:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}, MPJPE: {avg_mpjpe_cm:.2f}cm, MPJVE: {avg_mpjve:.4f} (pure GMD+SlowFast)')
        else:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}, MPJPE: {avg_mpjpe_cm:.2f}cm, MPJVE: {avg_mpjve:.4f}')
