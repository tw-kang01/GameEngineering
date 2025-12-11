# EgoPoser + Categorical Codebook Matching 통합 가이드

> **목표**: EgoPoser의 **Global Motion Decomposition** + **SlowFast Feature Fusion**을
> Categorical Codebook Matching의 **TrackerBody 네트워크**에 통합하여
> **Position-Invariant하고 Temporally-Rich한 VR Body Pose Prediction** 구현

---

## 목차

1. [배경 및 동기](#1-배경-및-동기)
2. [Categorical Codebook Matching 아키텍처](#2-categorical-codebook-matching-아키텍처)
3. [EgoPoser의 핵심 모듈](#3-egoposer의-핵심-모듈)
4. [통합 구현](#4-통합-구현)
5. [파일 구조](#5-파일-구조)
6. [학습 결과](#6-학습-결과)
7. [Contribution 및 차별점](#7-contribution-및-차별점)
8. [참고문헌](#8-참고문헌)

---

## 1. 배경 및 동기

### 1.1 문제 상황

| 문제 | 원인 | 영향 |
|------|------|------|
| **위치 일반화 실패** | 절대 좌표 의존 | 실외/대규모 환경에서 성능 저하 |
| **Temporal Context 부족** | Flat input (576) | 시간 구조 손실, 모션 ambiguity |
| **평가 지표 부재** | MSE Loss만 사용 | MPJPE/MPJVE 미제공 |

### 1.2 해결 방향

- **EgoPoser**: Global Motion Decomposition으로 position-invariant 표현 학습
- **EgoPoser**: SlowFast로 효율적인 temporal context 활용
- **Transformer**: Self-attention으로 temporal dependency 학습

---

## 2. Categorical Codebook Matching 아키텍처

> **논문**: *Categorical Codebook Matching for Embodied Character Controllers* (SIGGRAPH 2024)

### 2.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: TrackerBody Prediction  ★ GMD + SlowFast 적용 완료 ★              │
│  ───────────────────────────────────────                                    │
│  Input:  3-Point Tracking History (Head + Hands) × 16 timesteps            │
│          (B, 576) = (B, 16 × 36)                                           │
│  Output: Body Joint States (positions, rotations, velocities)              │
│          (B, 231) = Root(3) + 19 joints × 12                               │
│  Network: Transformer + GMD + SlowFast                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Codebook Matching (Motion Generation)                            │
│  ───────────────────────────────────────                                    │
│  - Categorical VQ-VAE로 다양한 모션 생성                                    │
│  - Teacher-Student 학습 (Encoder-Estimator 구조)                            │
│  - K개의 다양한 모션 샘플 생성 가능                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Full-Body Integration (Unity)                                    │
│  ───────────────────────────────────────                                    │
│  - ONNX 모델로 Unity Sentis에서 추론                                        │
│  - IK로 end effector 정밀 조정                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 TrackerBody 네트워크 (Stage 1)

**기존 구조 (Network.py - MLP)**
```
Input (576) → Normalize → Linear(512) → ELU → Linear(512) → ELU → Linear(231) → Renormalize → Output
```

**개선 구조 (Network_TrackerBody_Transformer.py)**
```
Input (B, 576) flat
    ↓
Normalize (Xnorm)
    ↓
GMDPreprocessorSequence
  ├─ Reshape: (B, 576) → (B, 16, 36)
  ├─ Spatial Norm: Head XZ 기준 상대 좌표
  └─ Temporal Norm: t0 대비 delta 추가 → (B, 16, 42)
    ↓
SlowFast Extraction
  ├─ Fast: frames [8-15]    → (B, 8, 42)
  └─ Slow: frames [0,2,4,...,14] → (B, 8, 42)
    ↓
Shared Linear Embedding → (B, 8, 256)
    ↓
x = x_fast + x_slow (element-wise sum)
    ↓
Positional Encoding
    ↓
Transformer Encoder (4 layers, 4 heads)
    ↓
x[:, -1] (last timestep) → (B, 256)
    ↓
Output Projection → (B, 231)
    ↓
Renormalize (Ynorm)
```

---

## 3. EgoPoser의 핵심 모듈

> **논문**: *EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere* (ECCV 2024)

### 3.1 Global Motion Decomposition

#### Spatial Normalization

수평(XZ) 좌표만 Head 기준 상대좌표로 변환, 수직(Y)은 유지:

```python
# Codebook 데이터 구조에 맞게 재구현
head_x = head_features[:, :, 0:1]  # (B, 16, 1)
head_z = head_features[:, :, 2:3]

# Head → (0, y, 0)
head_modified[:, :, 0:1] -= head_x
head_modified[:, :, 2:3] -= head_z

# Wrists → Head 기준 상대 좌표
lwrist_modified[:, :, 0:1] -= head_x
lwrist_modified[:, :, 2:3] -= head_z
rwrist_modified[:, :, 0:1] -= head_x
rwrist_modified[:, :, 2:3] -= head_z
```

#### Temporal Normalization

첫 프레임 대비 XZ 변화량 추가:

```python
# 6 features per timestep 추가 → 36 + 6 = 42
head_delta = [head_x_t - head_x_0, head_z_t - head_z_0]
lwrist_delta = [lwrist_x_t - lwrist_x_0, lwrist_z_t - lwrist_z_0]
rwrist_delta = [rwrist_x_t - rwrist_x_0, rwrist_z_t - rwrist_z_0]
```

### 3.2 SlowFast Feature Fusion

```python
# Fast pathway: 최근 움직임 (후반 8 frames)
x_fast = x_seq[:, -T//2:, :]  # (B, 8, 42)

# Slow pathway: 전체 맥락 (stride 2)
x_slow = x_seq[:, ::2, :]     # (B, 8, 42)

# Shared embedding (같은 weights)
x_fast = self.linear_embedding(x_fast)
x_slow = self.linear_embedding(x_slow)

# Fusion (element-wise sum)
x = x_fast + x_slow  # (B, 8, 256)
```

| Pathway | 시간 범위 | 해상도 | 캡처하는 정보 |
|---------|-----------|--------|---------------|
| **Fast** | 최근 8 frames | 원본 | 세밀한 모션, 급격한 변화 |
| **Slow** | 전체 16 frames | 1/2 | 장기 트렌드, 전반적 패턴 |

---

## 4. 통합 구현

### 4.1 아키텍처 비교 (3-Way)

| 항목 | Codebook (Network.py) | EgoPoser (egoposer.py) | Ours |
|------|----------------------|------------------------|------|
| **Input 형식** | `(B, 576)` flat | `dict{'sparse_input', 'fov_l', 'fov_r'}` | `(B, 576)` flat |
| **Output 형식** | `(B, 231)` flat | `dict{'root_orient', 'pose_body', 'betas'}` | `(B, 231)` flat |
| **아키텍처** | MLP | Transformer | Transformer |
| **GMD** | X | O (forward 내부) | O (별도 모듈) |
| **SlowFast** | X | O | O |
| **Positional Encoding** | X | X | O |
| **Target** | Unity ONNX | SMPL body model | Unity ONNX |

### 4.2 핵심 클래스

#### GMDPreprocessorSequence

```python
class GMDPreprocessorSequence(nn.Module):
    """별도 모듈로 분리하여 재사용성 향상"""

    def forward(self, x):
        # (B, 576) → (B, 16, 36) reshape
        head_features = x[:, 0:192].view(B, 16, 12)
        lwrist_features = x[:, 192:384].view(B, 16, 12)
        rwrist_features = x[:, 384:576].view(B, 16, 12)

        # Spatial + Temporal Normalization
        # ... (상세 구현은 CHANGELOG_TrackerBody.md 참조)

        return torch.cat([head_modified, lwrist_modified, rwrist_modified, all_deltas], dim=2)
        # Output: (B, 16, 42)
```

#### SlowFastTransformer

```python
class SlowFastTransformer(nn.Module):
    def __init__(self, input_dim_per_timestep, embed_dim, output_dim, ...):
        self.linear_embedding = nn.Linear(input_dim_per_timestep, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=20, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True  # EgoPoser와 다름: permute 불필요
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_seq):
        # SlowFast extraction
        x_fast = x_seq[:, -T//2:, :]
        x_slow = x_seq[:, ::2, :]

        x_fast = self.linear_embedding(x_fast)
        x_slow = self.linear_embedding(x_slow)
        x = x_fast + x_slow

        # Positional Encoding (EgoPoser에 없음)
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # last timestep

        return self.output_projection(x)
```

### 4.3 입출력 상세

#### Input (576 features)

```
3-Point Tracking × 16 timesteps:
├── Head (0-191):       16 × 12 features
│   └── position(3) + forward(3) + up(3) + velocity(3)
├── Left Wrist (192-383): 16 × 12 features
│   └── position(3) + forward(3) + up(3) + velocity(3)
├── Right Wrist (384-575): 16 × 12 features
│   └── position(3) + forward(3) + up(3) + velocity(3)
└── Total: 576 features
```

#### Output (231 features)

```
Root + 19 Body Joints:
├── Root (0-2): position(3)
├── Joint 0-18 (3-230): 19 × 12 features each
│   └── position(3) + forward(3) + up(3) + velocity(3)
└── Total: 231 features
```

### 4.4 하이퍼파라미터

| 파라미터 | 기존 MLP | Transformer (Ours) |
|----------|----------|---------------------|
| Architecture | [576, 512, 512, 231] | Transformer (4L, 4H) |
| Embed dim | 512 (hidden) | 256 |
| Batch size | 32 | 64 |
| Epochs | 150 | 150 |
| Learning rate | 1e-4 | 1e-4 |
| Dropout | 0.25 | 0.1 |
| Parameters | ~0.6M | ~3.3M |

---

## 5. 파일 구조

```
AI4Animation/game_engineering/
├── categorical/
│   └── PyTorch/
│       ├── Library/
│       │   ├── Utility.py
│       │   └── Modules.py
│       ├── Models/
│       │   ├── MultiLayerPerceptron/
│       │   │   ├── Network.py                        # 기존 MLP (baseline)
│       │   │   ├── Network_TrackerBody.py            # GMD+SlowFast MLP 버전
│       │   │   ├── Network_TrackerBody_Transformer.py # ★ Transformer 버전 ★
│       │   │   ├── train_slurm.sh                    # MLP 학습 스크립트
│       │   │   ├── train_slurm_transformer.sh        # Transformer 학습 스크립트
│       │   │   └── CHANGELOG_TrackerBody.md          # 상세 변경 기록
│       │   └── CodebookMatching/
│       │       └── Network.py
│       └── Datasets/
│           └── Trackerbodypredictor/
│               ├── Input.bin
│               ├── Output.bin
│               ├── InputShape.txt
│               ├── OutputShape.txt
│               ├── InputNormalization.txt
│               └── OutputNormalization.txt
├── egoposer/
│   └── networks/
│       └── egoposer.py                              # 원본 EgoPoser 참조
└── INTEGRATION_GUIDE.md                             # 이 문서
```

---

## 6. 학습 결과

### 6.1 Transformer + GMD + SlowFast

| Metric | 값 |
|--------|-----|
| Loss | 0.059360 |
| **MPJPE** | **2.75 cm** |
| MPJVE | 0.1644 |
| Parameters | ~3.3M |

### 6.2 ONNX 호환성

```python
# Input/Output 인터페이스 동일
Input:  (B, 576) float32
Output: (B, 231) float32

# Unity Sentis에서 그대로 사용 가능
# 기존 파이프라인 수정 불필요
```

---

## 7. Contribution 및 차별점

### 7.1 주요 Contribution

| # | Contribution | 설명 |
|---|--------------|------|
| 1 | **GMD 적용** | Spatial + Temporal normalization으로 위치 불변성 향상 |
| 2 | **SlowFast 적용** | Fast (최근) + Slow (맥락) 정보 동시 활용 |
| 3 | **Transformer 아키텍처** | Self-attention으로 temporal dependency 학습 |
| 4 | **Positional Encoding** | EgoPoser에 없는 위치 정보 명시적 인코딩 |
| 5 | **MPJPE/MPJVE 메트릭** | 표준화된 포즈 추정 평가 지표 추가 |
| 6 | **ONNX 호환성** | 기존 Unity 파이프라인 수정 없이 사용 가능 |

### 7.2 EgoPoser 대비 차별점

| 컴포넌트 | EgoPoser | Ours |
|----------|----------|------|
| **I/O 인터페이스** | dict I/O (SMPL용) | flat tensor (ONNX 호환) |
| **GMD** | forward 내부 인라인 | 별도 모듈 분리 |
| **Positional Encoding** | 없음 | 추가 |
| **Transformer** | batch_first=False (permute 필요) | batch_first=True |
| **Output Decoder** | 3개 분리 (root, body, betas) | 단일 통합 |
| **FOV Masking** | Hand tracking용 | 제거 (VR Controller는 항상 추적) |

### 7.3 Codebook (기존 MLP) 대비 개선

| 컴포넌트 | 기존 MLP | Ours |
|----------|----------|------|
| **전처리** | 없음 | GMD (Spatial + Temporal) |
| **아키텍처** | MLP | Transformer |
| **Temporal 정보** | Flatten으로 손실 | SlowFast로 보존 |
| **평가 지표** | Loss만 | Loss + MPJPE + MPJVE |
| **Best Model** | 없음 | 자동 저장 |

---

## 8. 참고문헌

1. **Categorical Codebook Matching for Embodied Character Controllers**
   Starke et al., ACM SIGGRAPH 2024

2. **EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere**
   Jiang et al., ECCV 2024

3. **SlowFast Networks for Video Recognition**
   Feichtenhofer et al., ICCV 2019

---

## Quick Start

```bash
# SLURM 클러스터에서 학습
cd categorical/PyTorch/Models/MultiLayerPerceptron

# Transformer 버전 학습
sbatch train_slurm_transformer.sh

# 또는 로컬에서 직접 실행
python Network_TrackerBody_Transformer.py
```

---

*Last Updated: 2024-12-11*
