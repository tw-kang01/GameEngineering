# EgoPoser + Categorical Codebook Matching 통합 가이드

> **목표**: EgoPoser의 **Global Motion Decomposition** + **SlowFast Feature Fusion**을  
> Categorical Codebook Matching의 **VQ-VAE 기반 모션 생성 네트워크**에 통합하여  
> **Position-Invariant하고 Temporally-Rich한 Upper Body + Root Trajectory Prediction** 구현

---

## 목차

1. [배경 및 동기](#1-배경-및-동기)
2. [Categorical Codebook Matching 아키텍처](#2-categorical-codebook-matching-아키텍처)
3. [EgoPoser의 핵심 모듈](#3-egoposer의-핵심-모듈)
4. [통합 설계](#4-통합-설계)
5. [구현 계획 및 코드 수정 위치](#5-구현-계획-및-코드-수정-위치)
6. [Contribution 및 차별점](#6-contribution-및-차별점)
7. [참고문헌](#7-참고문헌)

---

## 1. 배경 및 동기

### 1.1 문제 상황

| 문제 | 원인 | 영향 |
|------|------|------|
| **위치 일반화 실패** | 실내 데이터로 학습 → 절대 좌표 의존 | 실외/대규모 환경에서 성능 저하 |
| **Temporal Context 부족** | 단일 프레임 입력 | 모션 ambiguity, 부자연스러운 전환 |
| **Codebook Collapse** | Motion prior와 control 분리 학습 | 일부 코드만 과사용, 다양성 감소 |

### 1.2 해결 방향

- **EgoPoser**: Global Motion Decomposition으로 position-invariant 표현 학습
- **EgoPoser**: SlowFast로 효율적인 temporal context 확장
- **Categorical Codebook Matching**: Control-aware motion manifold로 codebook 활용도 극대화

---

## 2. Categorical Codebook Matching 아키텍처

> **논문**: *Categorical Codebook Matching for Embodied Character Controllers* (SIGGRAPH 2024)

### 2.1 핵심 Task: Upper Body + Root Future Trajectory Prediction

**Categorical Codebook Matching의 핵심 역할**:
- **입력**: 3-Point Tracking (Head, Left Hand, Right Hand) 히스토리
- **출력**: **Upper Body Motion + Root(Pelvis) Future Trajectory** (0.5초)
- **목적**: Sparse tracking에서 상체 포즈와 루트 이동 경로 예측

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CATEGORICAL CODEBOOK MATCHING                             │
│                    Upper Body + Root Trajectory Prediction                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input X:                                                                    │
│    - Tracker History (past 0.5s): Head, Left Hand, Right Hand               │
│    - Current State: positions, rotations, velocities                        │
│                                                                              │
│  Output Y:                                                                   │
│    - Upper Body Motion Sequence (future 0.5s)                               │
│    - Root (Pelvis) Future Trajectory                                        │
│    - K개의 다양한 모션 샘플 생성 가능                                         │
│                                                                              │
│  ★ Lower Body는 별도 시스템에서 처리 (IK 또는 다른 네트워크)                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 전체 파이프라인 (3-Stage Character Control)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Control Prediction (TrackerBody)                                  │
│  ───────────────────────────────────────────                                │
│  Input:  T_P = Tracker History (past 0.5s, Head + Hands)                   │
│  Output: T_F = Future Trajectories (next 0.5s)                              │
│          - Root trajectory (2D position, rotation, velocity)                │
│          - Upper body motion trajectories (K개)                             │
│  Network: Simple MLP                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Upper Body + Root Regression ★ CODEBOOK MATCHING ★                │
│  ───────────────────────────────────────────                                │
│  Input X = (S_i, C_F):                                                      │
│    - S_i: Current state (tracker data, upper body pose)                    │
│    - C_F: Control signals from Stage 1                                      │
│                                                                             │
│  Output Y:                                                                   │
│    - Future Upper Body Motion Sequence (0.5s)                               │
│    - Root (Pelvis) Future Trajectory                                        │
│                                                                             │
│  Network: Encoder-Estimator-Decoder with Categorical Codebook              │
│                                                                             │
│  ★ EgoPoser GMD + SlowFast 적용 위치 ★                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Full-Body Fusion                                                  │
│  ───────────────────────────────────────────                                │
│  - Upper body (Stage 2) + Lower body (IK/별도 시스템)                       │
│  - Apply IK for end effectors                                               │
│  Output: Full Body Pose                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Codebook Matching 메커니즘

#### 핵심 구조: Teacher-Student 학습

```
                    ┌──────────────────┐
  Y (정답 출력) ────►│  Output Encoder  │────► Z_Y (최적 코드)
  X (입력)      ────►│  (Teacher)       │      "정답을 알면 최적 선택 가능"
                    └──────────────────┘
                              │
                              │ Matching Loss: MSE(Z_Y, Z_X)
                              │ "Student가 Teacher를 모방"
                              ▼
                    ┌──────────────────┐
  X (입력)      ────►│  Input Encoder   │────► Z_X (추정 코드)
                    │  (Student)       │      "입력만으로 코드 예측"
                    └──────────────────┘
```

#### 왜 이 구조가 필요한가?

1. **단순 Autoencoder 문제**: `Encoder(Y, X) → Z → Decoder → Y'`
   - Y → Y' identity mapping 학습 → X 무시됨
   - Inference 시 Y 없으면 동작 불가

2. **Codebook Matching 해결책**:
   - Output Encoder: Y+X → Z_Y (최적 코드 학습)
   - Input Encoder: X → Z_X (X만으로 Z_Y 근사)
   - Matching Loss로 Z_X ≈ Z_Y 강제
   - **Inference: Z_X만으로 동작 가능**

### 2.4 Categorical Codebook 구조

```
Network Output: [B, C×D] = [B, 1024]
        │
        ▼ reshape
     [B, C, D] = [B, 128, 8]
        │
        │   각 채널(row)에서 1개 선택 (Gumbel-Softmax)
        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Channel 1:   [0.2, 0.8, ...]  → [0, 1, 0, ..., 0]  │
   │  Channel 2:   [0.7, 0.3, ...]  → [1, 0, 0, ..., 0]  │
   │  ...                                                 │
   │  Channel 128: [0.1, 0.9, ...]  → [0, 1, 0, ..., 0]  │
   └─────────────────────────────────────────────────────┘
        │
        ▼ flatten
   Codebook Vector: [B, C×D] = [B, 1024]
```

**파라미터 vs 용량**:
| 방식 | 파라미터 | 표현 용량 |
|------|----------|-----------|
| 전통 VQ-VAE (K entries) | K × D | K |
| **Categorical (C × D)** | C × D | **D^C** |

예: C=128, D=8 → 파라미터 1024개로 **8^128 ≈ 10^115** 조합 표현

### 2.5 Gumbel-Softmax: 미분 가능한 이산 선택

```python
# Gumbel-Max Trick
G ~ Gumbel(0,1) = -log(-log(U)),  U ~ Uniform(0,1)
argmax_i (log π_i + G_i) ~ Categorical(π)

# Straight-Through Estimator (STE)
y_hard = (y_hard - y_soft).detach() + y_soft
# Forward: y_hard (discrete one-hot)
# Backward: gradient through y_soft (continuous)
```

### 2.6 Loss 함수

```python
L_Rec = MSE(Y, Y')           # Reconstruction Loss
L_Match = MSE(Z_X, Z_Y)      # Codebook Matching Loss

L_total = L_Rec + L_Match    # VQ Loss 불필요!
```

---

## 3. EgoPoser의 핵심 모듈

> **논문**: *EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere* (ECCV 2024)

### 3.1 Global Motion Decomposition

#### 문제: Position Dependency

```
실내 학습 데이터:              실외 테스트:
┌──────────────┐              ┌─────────────────────────────┐
│   ○ (origin) │              │                             │
│   ↑          │              │              ○              │
│   User       │              │              ↑              │
│ (0~3m 범위)  │              │        User (100m+ 위치)   │
└──────────────┘              └─────────────────────────────┘

→ 절대 좌표 학습 → 분포 불일치 → 일반화 실패
```

#### 해결: Spatial + Temporal Normalization

##### A. Spatial Normalization (SN)

**핵심**: 수평(XY) 좌표만 reference joint 기준 상대좌표로 변환, **수직(Z)은 유지**

```python
# 논문 Eq. (2)
p_SN,h = p_W,h(hand) - p_W,h(head)  # horizontal only

# 코드 구현
head_xy = input[..., 36:38].clone().detach()
input[..., 36:38] -= head_xy  # head → (0, 0)
input[..., 39:41] -= head_xy  # left hand
input[..., 42:44] -= head_xy  # right hand
# Z 좌표 (38, 41, 44)는 유지 → 중력 방향 정보 보존
```

##### B. Temporal Normalization (TN)

**핵심**: 시퀀스 내 위치 변화량(trajectory delta) 계산

```python
# 논문 Eq. (2)
p_TN = p_W(t) - p_W(t_0)  # 첫 프레임 기준 delta

# 코드 구현
delta_head = input[..., 36:38] - input[..., [0], 36:38]
delta_left = input[..., 39:41] - input[..., [0], 39:41]
delta_right = input[..., 42:44] - input[..., [0], 42:44]

input = torch.cat([input, delta_head, delta_left, delta_right], dim=-1)
# [B, T, 54] → [B, T, 60]
```

#### 시각화

```
World Frame                      After GMD
┌────────────────────┐          ┌────────────────────┐
│  t=0    t=40  t=79 │          │ t=0   t=40  t=79   │
│   ○──────○──────○  │    SN    │  ○     ○     ○     │  ← Head: (0,0,z)
│  /|     /|     /|  │   ───►   │ /|    /|    /|     │
│ ○ ○    ○ ○    ○ ○  │          │○ ○   ○ ○   ○ ○     │  ← Hands: relative
│                    │          │                    │
│ Global positions   │          │ Head-relative XY   │
└────────────────────┘          └────────────────────┘
                                         │
                                         │ TN (delta concat)
                                         ▼
                                ┌────────────────────┐
                                │ Original features  │
                                │ + Δhead_xy         │
                                │ + Δleft_xy         │
                                │ + Δright_xy        │
                                └────────────────────┘
```

### 3.2 SlowFast Feature Fusion

#### 문제: Computational Cost vs Temporal Context

```
단일 프레임: 모호함 (무한한 가능 포즈)
긴 시퀀스:  O(T²) attention → 계산량 폭발
```

#### 해결: Dual Pathway

```
Input: T = 80 frames
┌─────────────────────────────────────────────────────────────────────────────┐
│  Frame: 0  1  2  3 ... 38 39 40 41 42 ... 76 77 78 79                      │
│         ▲     ▲     ▲           ▲                 ▲                         │
│         └─────┴─────┴───────────┴─────────────────┘                         │
│                    SLOW: stride=2, 40 frames                                │
│                    (전체 temporal context)                                  │
│                                                                             │
│                            ┌────────────────────────────┐                   │
│                            │ 40 41 42 ... 77 78 79     │                   │
│                            │ FAST: last 40 frames       │                   │
│                            │ (고해상도 최근 모션)        │                   │
│                            └────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 코드 구현

```python
# SlowFast Split
T = input.shape[1]  # 80
x_fast = input[:, -T//2:, :]  # [B, 40, D] - last 40 frames
x_slow = input[:, ::2, :]     # [B, 40, D] - every 2nd frame

# Embedding (shared weights)
x_fast = self.linear_embedding(x_fast)  # [B, 40, embed_dim]
x_slow = self.linear_embedding(x_slow)  # [B, 40, embed_dim]

# Fusion
x = x_fast + x_slow  # [B, 40, embed_dim]
```

#### 이점

| Pathway | 시간 범위 | 해상도 | 캡처하는 정보 |
|---------|-----------|--------|---------------|
| **Fast** | 최근 0.5s | 원본 | 세밀한 모션, 급격한 변화 |
| **Slow** | 전체 1.0s | 1/2 | 장기 트렌드, 전반적 패턴 |

**결과**: 80 frames → 40 frames (계산량 절반) + 전체 temporal context 유지

---

## 4. 통합 설계

### 4.1 통합 아키텍처

```
┌────────────────────────────────────────────────────────────────────────────────┐
│              통합 모델: Upper Body + Root Trajectory Prediction                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Input X [B, T=80, D]                                                          │
│  (과거 80 프레임: 3-Point Tracker History)                                     │
│                                                                                 │
│         ┌─────────────────────────────────────────────────────────────┐        │
│         │ [head_pos, head_rot, head_vel,                              │        │
│         │  left_hand_pos, left_hand_rot, left_hand_vel,               │        │
│         │  right_hand_pos, right_hand_rot, right_hand_vel]            │        │
│         └─────────────────────────────────────────────────────────────┘        │
│                              │                                                  │
│                              ▼                                                  │
│                    ┌─────────────────────┐                                     │
│                    │  Global Motion      │  ← NEW (EgoPoser)                   │
│                    │  Decomposition      │                                     │
│                    │  - Spatial Norm     │  head 기준 XY 상대좌표              │
│                    │  - Temporal Norm    │  첫 프레임 기준 delta               │
│                    └──────────┬──────────┘                                     │
│                              │ [B, 80, D+delta]                                │
│                              ▼                                                  │
│                    ┌─────────────────────┐                                     │
│                    │  SlowFast Fusion    │  ← NEW (EgoPoser)                   │
│                    │  Fast: last 40      │                                     │
│                    │  Slow: stride 2     │                                     │
│                    └──────────┬──────────┘                                     │
│                              │ [B, 40, embed_dim]                              │
│                              ▼                                                  │
│                    ┌─────────────────────┐                                     │
│                    │  Transformer        │  ← NEW (Optional)                   │
│                    │  (3 layers, 8 head) │                                     │
│                    └──────────┬──────────┘                                     │
│                              │ [B, embed_dim]                                  │
│                              ▼                                                  │
│         ┌────────────────────┴────────────────────┐                            │
│         │                                          │                            │
│         ▼                                          ▼                            │
│  ┌─────────────┐                           ┌─────────────┐                     │
│  │  Estimator  │                           │  Encoder    │ (training only)     │
│  │  X → Z_X    │                           │ (Y+X) → Z_Y │                     │
│  └──────┬──────┘                           └──────┬──────┘                     │
│         │                                          │                            │
│         │ ←────────── Matching Loss ─────────────► │                            │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────┐                                                          │
│  │ Gumbel-Softmax   │  ← EXISTING (Categorical)                                │
│  │ Categorical      │                                                          │
│  │ Sampling (K개)   │                                                          │
│  └────────┬─────────┘                                                          │
│           │ [B, C×D]                                                           │
│           ▼                                                                     │
│  ┌─────────────┐                                                               │
│  │  Decoder    │  ← EXISTING (Categorical)                                     │
│  └──────┬──────┘                                                               │
│         │                                                                       │
│         ▼                                                                       │
│  Output Y [B, N, D_out]                                                        │
│  ┌─────────────────────────────────────────────────────────┐                   │
│  │ - Upper Body Motion Sequence (future 0.5s)              │                   │
│  │ - Root (Pelvis) Future Trajectory                       │                   │
│  │ - 상체 관절 회전 + 루트 위치/방향/속도                   │                   │
│  └─────────────────────────────────────────────────────────┘                   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 입출력 상세

#### Input Features (X)
```
3-Point Tracking × 80 frames:
├── Head:       position(3) + rotation_6d(6) + velocity(6) = 15
├── Left Hand:  position(3) + rotation_6d(6) + velocity(6) = 15  
├── Right Hand: position(3) + rotation_6d(6) + velocity(6) = 15
└── Total: 45 features/frame × 80 frames

+ GMD Delta: 6 features (Δhead_xy, Δleft_xy, Δright_xy)
= 51 features/frame
```

#### Output Features (Y)
```
Upper Body + Root Trajectory × N frames:
├── Root Trajectory: position_2d(2) + rotation(1) + velocity(3) = 6
├── Upper Body:      joint_rotations × num_upper_joints
└── Total: Future 0.5s sequence
```

---

## 5. 구현 계획 및 코드 수정 위치

### 5.1 파일별 수정 사항

#### A. `Library/Modules.py` - 새 클래스 추가

```python
# ═══════════════════════════════════════════════════════════════════════════════
# 추가할 위치: 파일 끝 (ConvolutionalEncoder 클래스 다음)
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalMotionDecomposition(nn.Module):
    """
    EgoPoser Section 3.3: Spatial + Temporal Normalization
    
    Position-invariant representation for large-scale environment generalization.
    
    Args:
        position_indices: List of slice objects for XY positions to normalize
                         e.g., [slice(0,2), slice(15,17), slice(30,32)] for head, left, right
        reference_idx: Index of reference position (default: 0 = head)
        use_spatial_norm: Whether to apply spatial normalization
    """
    def __init__(self, position_indices, reference_idx=0, use_spatial_norm=True):
        super(GlobalMotionDecomposition, self).__init__()
        self.position_indices = position_indices  # List of slice objects for XY
        self.reference_idx = reference_idx
        self.use_spatial_norm = use_spatial_norm
        self.num_positions = len(position_indices)
    
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        
        # Spatial Normalization: XY relative to reference
        if self.use_spatial_norm and self.num_positions > 0:
            ref_slice = self.position_indices[self.reference_idx]
            ref_xy = x[..., ref_slice].clone().detach()  # [B, T, 2]
            
            for pos_slice in self.position_indices:
                x[..., pos_slice] = x[..., pos_slice] - ref_xy
        
        # Temporal Normalization: delta from first frame
        deltas = []
        for pos_slice in self.position_indices:
            delta = x[..., pos_slice] - x[:, [0], pos_slice]  # [B, T, 2]
            deltas.append(delta)
        
        if deltas:
            x = torch.cat([x] + deltas, dim=-1)  # [B, T, D + num_positions*2]
        
        return x


class SlowFastEncoder(nn.Module):
    """
    EgoPoser Section 3.4: SlowFast Feature Fusion
    
    Efficient temporal context expansion with dual pathway.
    
    Args:
        input_dim: Input feature dimension per frame
        embed_dim: Output embedding dimension
    """
    def __init__(self, input_dim, embed_dim):
        super(SlowFastEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.linear_embedding = nn.Linear(input_dim, embed_dim)
    
    def forward(self, x):
        # x: [B, T, D]
        T = x.shape[1]
        
        # Fast pathway: last half (high temporal resolution)
        x_fast = x[:, -T//2:, :]  # [B, T//2, D]
        
        # Slow pathway: stride 2 (long temporal context)
        x_slow = x[:, ::2, :]     # [B, T//2, D]
        
        # Shared embedding
        x_fast = self.linear_embedding(x_fast)  # [B, T//2, embed_dim]
        x_slow = self.linear_embedding(x_slow)  # [B, T//2, embed_dim]
        
        # Additive fusion
        return x_fast + x_slow  # [B, T//2, embed_dim]


class TransformerSlowFastEncoder(nn.Module):
    """
    Combined GMD + SlowFast + Transformer Encoder
    
    Replaces LinearEncoder for sequence input processing.
    
    Args:
        input_dim: Per-frame input dimension
        embed_dim: Transformer embedding dimension
        output_dim: Output dimension (codebook_size)
        num_layers: Number of transformer layers
        nhead: Number of attention heads
        dropout: Dropout rate
        position_indices: For GlobalMotionDecomposition
        use_spatial_norm: Whether to use spatial normalization
    """
    def __init__(self, input_dim, embed_dim, output_dim, 
                 num_layers=3, nhead=8, dropout=0.25,
                 position_indices=None, use_spatial_norm=True):
        super(TransformerSlowFastEncoder, self).__init__()
        
        self.position_indices = position_indices or []
        num_delta_features = len(self.position_indices) * 2
        augmented_dim = input_dim + num_delta_features
        
        # GMD Module
        self.gmd = GlobalMotionDecomposition(
            position_indices=position_indices,
            reference_idx=0,
            use_spatial_norm=use_spatial_norm
        ) if position_indices else None
        
        # SlowFast Module
        self.slowfast = SlowFastEncoder(augmented_dim, embed_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
        )
    
    def forward(self, x):
        # x: [B, T, D]
        
        # Global Motion Decomposition
        if self.gmd is not None:
            x = self.gmd(x)
        
        # SlowFast Fusion
        x = self.slowfast(x)  # [B, T//2, embed_dim]
        
        # Transformer (expects [T, B, D])
        x = x.permute(1, 0, 2)  # [T//2, B, embed_dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, T//2, embed_dim]
        
        # Take last frame
        x = x[:, -1, :]  # [B, embed_dim]
        
        # Project to output
        return self.output_proj(x)  # [B, output_dim]
```

#### B. `Models/CodebookMatching/Network.py` - Model 클래스 수정

```python
# ═══════════════════════════════════════════════════════════════════════════════
# 수정할 위치: class Model 의 __init__ 메서드
# Line 19-31 부근
# ═══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    def __init__(self, encoder, estimator, decoder, xNorm, yNorm, 
                 codebook_channels, codebook_dim,
                 # NEW: EgoPoser modules (optional)
                 use_sequence_input=False,
                 gmd_position_indices=None,
                 use_spatial_norm=True):
        super(Model, self).__init__()

        self.Encoder = encoder
        self.Estimator = estimator
        self.Decoder = decoder

        self.XNorm = xNorm
        self.YNorm = yNorm

        self.C = codebook_channels
        self.D = codebook_dim
        
        # NEW: Sequence processing flags
        self.use_sequence_input = use_sequence_input
        self.gmd_position_indices = gmd_position_indices
        self.use_spatial_norm = use_spatial_norm


# ═══════════════════════════════════════════════════════════════════════════════
# 수정할 위치: forward 메서드
# Line 69-99 부근
# ═══════════════════════════════════════════════════════════════════════════════

    def forward(self, x, knn, t=None):
        # NEW: Handle sequence input [B, T, D] vs legacy [B, D]
        if self.use_sequence_input and x.dim() == 3:
            # Estimator handles GMD + SlowFast internally (TransformerSlowFastEncoder)
            # Just normalize per-frame features before processing
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
            x_flat = utility.Normalize(x_flat, self.XNorm)
            x = x_flat.view(B, T, D)
        else:
            x = utility.Normalize(x, self.XNorm)
        
        # ... (rest of forward logic)


# ═══════════════════════════════════════════════════════════════════════════════
# 수정할 위치: 학습 스크립트 (if __name__ == '__main__' 부분)
# Line 145-160 부근 - 네트워크 생성 부분
# ═══════════════════════════════════════════════════════════════════════════════

    # Option 1: Legacy mode (single frame)
    network = utility.ToDevice(Model(
        encoder=modules.LinearEncoder(input_dim + output_dim, ...),
        estimator=modules.LinearEncoder(input_dim, ...),
        decoder=modules.LinearEncoder(codebook_size, ...),
        ...
    ))
    
    # Option 2: NEW - Sequence mode with GMD + SlowFast
    window_size = 80
    frame_dim = input_dim // window_size  # or define explicitly
    embed_dim = 256
    
    # Define position indices for GMD (example for 3-point tracking)
    # Adjust based on your actual data format
    position_indices = [
        slice(0, 2),    # head XY
        slice(15, 17),  # left hand XY
        slice(30, 32),  # right hand XY
    ]
    
    network = utility.ToDevice(Model(
        encoder=modules.LinearEncoder(embed_dim + output_dim, encoder_dim, encoder_dim, codebook_size, dropout),
        
        estimator=modules.TransformerSlowFastEncoder(
            input_dim=frame_dim,
            embed_dim=embed_dim,
            output_dim=codebook_size,
            num_layers=3,
            nhead=8,
            dropout=dropout,
            position_indices=position_indices,
            use_spatial_norm=True
        ),
        
        decoder=modules.LinearEncoder(codebook_size, decoder_dim, decoder_dim, output_dim, 0.0),
        
        xNorm=Parameter(...),
        yNorm=Parameter(...),
        
        codebook_channels=codebook_channels,
        codebook_dim=codebook_dim,
        
        # NEW flags
        use_sequence_input=True,
        gmd_position_indices=position_indices,
        use_spatial_norm=True
    ))
```

#### C. 데이터 로딩 수정

```python
# ═══════════════════════════════════════════════════════════════════════════════
# 수정할 위치: 학습 루프 내 데이터 로딩
# Line 190-195 부근
# ═══════════════════════════════════════════════════════════════════════════════

# 기존: 단일 프레임
xBatch = utility.ReadBatchFromFile(XFile, train_indices, XShape[1])  # [B, D]

# 수정: 시퀀스 윈도우 (새 함수 필요)
window_size = 80
xBatch = utility.ReadSequenceWindowBatch(XFile, train_indices, window_size, frame_dim)  # [B, T, D]
```

### 5.2 새로 추가할 유틸리티 함수 (`Library/Utility.py`)

```python
def ReadSequenceWindowBatch(file, center_indices, window_size, frame_dim):
    """
    Read sequence windows centered at given indices.
    
    Args:
        file: Binary file path
        center_indices: Center frame indices for each batch item
        window_size: Number of frames in window (e.g., 80)
        frame_dim: Features per frame
    
    Returns:
        Tensor of shape [B, window_size, frame_dim]
    """
    batch = []
    half_window = window_size // 2
    
    for idx in center_indices:
        start = max(0, idx - half_window)
        end = start + window_size
        
        # Read window_size consecutive frames
        window_indices = np.arange(start, end)
        frames = ReadBatchFromFile(file, window_indices, frame_dim)
        batch.append(frames)
    
    return torch.stack(batch, dim=0)  # [B, T, D]
```

### 5.3 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `window_size` | 80 | 입력 시퀀스 길이 (약 1초 @ 80fps) |
| `embed_dim` | 256 | SlowFast/Transformer 임베딩 차원 |
| `num_layers` | 3 | Transformer 레이어 수 |
| `nhead` | 8 | Attention heads |
| `codebook_channels` | 128 | Categorical codebook C |
| `codebook_dim` | 8 | Categorical codebook D |
| `dropout` | 0.25 | Dropout rate |

---

## 6. Contribution 및 차별점

### 6.1 주요 Contribution

| # | Contribution | 설명 |
|---|--------------|------|
| **C1** | **Position-Invariant Upper Body Prediction** | GMD를 통해 실내/실외 환경에 관계없이 일관된 상체 모션 예측 |
| **C2** | **Efficient Temporal Modeling** | SlowFast로 계산량 절반 유지하면서 2배 긴 temporal context 활용 |
| **C3** | **Control-Aware Motion Manifold** | Codebook Matching으로 motion prior와 control mapping 동시 학습 |
| **C4** | **Multi-Modal Motion Sampling** | 같은 입력에서 K개의 다양한 upper body + root trajectory 생성 |

### 6.2 기존 방법 대비 차별점

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         방법론 비교                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  기존 EgoPoser:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ + Position-invariant (GMD)                                          │   │
│  │ + Efficient temporal (SlowFast)                                     │   │
│  │ - Deterministic output (단일 결과)                                   │   │
│  │ - Full body 직접 예측 (복잡)                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  기존 Categorical Codebook Matching:                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ + Multi-modal sampling (K개 샘플)                                    │   │
│  │ + Control-aware manifold                                            │   │
│  │ - Position dependency (절대 좌표)                                    │   │
│  │ - Limited temporal context (단일 프레임)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ★ 통합 모델 (Ours):                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ✓ Position-invariant (GMD from EgoPoser)                            │   │
│  │ ✓ Efficient temporal (SlowFast from EgoPoser)                       │   │
│  │ ✓ Multi-modal sampling (Categorical Codebook)                       │   │
│  │ ✓ Control-aware manifold (Codebook Matching)                        │   │
│  │ ✓ Upper body + Root trajectory 특화                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 예상 성능 향상

| 메트릭 | Baseline | +GMD | +SlowFast | +Both |
|--------|----------|------|-----------|-------|
| Position Error (실내) | 기준 | 동등 | -5% | -5% |
| Position Error (실외) | 기준 | **-30%** | -5% | **-35%** |
| Motion Smoothness | 기준 | 동등 | **+15%** | **+15%** |
| Inference Speed | 기준 | 동등 | 동등 | 동등 |
| Motion Diversity | 기준 | 동등 | 동등 | 동등 |

---

## 7. 참고문헌

1. **Categorical Codebook Matching for Embodied Character Controllers**  
   Starke et al., ACM SIGGRAPH 2024  
   [Paper](https://github.com/sebastianstarke/AI4Animation)

2. **EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere**  
   Jiang et al., ECCV 2024  
   [Paper](https://arxiv.org/abs/2308.06493) | [Code](https://github.com/eth-siplab/EgoPoser)

3. **Categorical Reparameterization with Gumbel-Softmax**  
   Jang et al., ICLR 2017

4. **SlowFast Networks for Video Recognition**  
   Feichtenhofer et al., ICCV 2019

---

## 라이선스

이 프로젝트는 연구 목적으로 작성되었습니다.

---

## Appendix: Quick Start

```bash
# 1. 환경 설정
cd categorical/PyTorch

# 2. 기존 모델 학습 (baseline)
python Models/CodebookMatching/Network.py

# 3. 통합 모델 학습 (GMD + SlowFast)
# Network.py 수정 후
python Models/CodebookMatching/Network.py --use_sequence_input --window_size 80
```
