
# TrackerBody Network 개선 기록

## 아키텍쳐

본 연구에서는 EgoPoser의 Global Motion Decomposition(GMD)과 SlowFast 기법을 CCM 파이프라인에 통합하고, Self-Attention 메커니즘을 활용한 Transformer Encoder 아키텍처를 제안한다.

**Figure 1: 전체 파이프라인 아키텍처**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        VR 3-Point Tracking Input                                │
│                    (Head + Left Controller + Right Controller)                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Input Tensor                                       │
│                                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                            │
│   │    Head     │  │   LWrist    │  │   RWrist    │   × 16 timesteps           │
│   │  (192 dim)  │  │  (192 dim)  │  │  (192 dim)  │                            │
│   └─────────────┘  └─────────────┘  └─────────────┘                            │
│                         ↓                                                       │
│                  Total: 576 features                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     GMD Preprocessor (Global Motion Decomposition)              │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                      Spatial Normalization                              │    │
│  │                                                                         │    │
│  │   Head XZ ────────────────► (0, 0) Reference Point                     │    │
│  │   LWrist XZ ─────────────► Relative to Head                            │    │
│  │   RWrist XZ ─────────────► Relative to Head                            │    │
│  │   Y-axis (vertical) ─────► Preserved (gravity alignment)               │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                        │
│                                        ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                      Temporal Normalization                             │    │
│  │                                                                         │    │
│  │   Δpos[t] = pos[t] - pos[0]                                            │    │
│  │                                                                         │    │
│  │   +6 delta features: [H_Δx, H_Δz, L_Δx, L_Δz, R_Δx, R_Δz]              │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                        │
│                         Output: (B, 16, 42) Sequence                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SlowFast Feature Extraction                            │
│                                                                                 │
│   ┌───────────────────────────┐      ┌───────────────────────────┐             │
│   │      Fast Pathway         │      │      Slow Pathway          │             │
│   │                           │      │                            │             │
│   │  Frames: [8,9,10,11,      │      │  Frames: [0,2,4,6,        │             │
│   │          12,13,14,15]     │      │          8,10,12,14]      │             │
│   │                           │      │                            │             │
│   │  Recent 8 frames          │      │  Stride 2 (8 frames)       │             │
│   │  (~0.27 sec)              │      │  (~0.53 sec coverage)      │             │
│   │                           │      │                            │             │
│   │  ► Fine gestures          │      │  ► Global motion flow      │             │
│   │  ► Quick head movement    │      │  ► Walking pattern         │             │
│   └─────────────┬─────────────┘      └─────────────┬──────────────┘             │
│                 │                                   │                           │
│                 ▼                                   ▼                           │
│   ┌─────────────────────────────────────────────────────────────────┐          │
│   │              Shared Linear Embedding: 42 → 256                   │          │
│   └─────────────────────────────────────────────────────────────────┘          │
│                 │                                   │                           │
│                 ▼                                   ▼                           │
│   ┌─────────────────────────┐      ┌─────────────────────────┐                 │
│   │  + Positional Encoding  │      │  + Positional Encoding  │                 │
│   │      (Sinusoidal)       │      │      (Sinusoidal)       │                 │
│   └─────────────┬───────────┘      └─────────────┬───────────┘                 │
│                 │                                 │                             │
│                 └────────────────┬────────────────┘                             │
│                                  ▼                                              │
│                    ┌─────────────────────────┐                                  │
│                    │    Element-wise Sum     │                                  │
│                    │     E_fast ⊕ E_slow     │                                  │
│                    └─────────────────────────┘                                  │
│                                  │                                              │
│                         Output: (B, 8, 256)                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Transformer Encoder                                   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐      │
│   │                     Multi-Head Self-Attention                        │      │
│   │                         (4 heads, d_k=64)                            │      │
│   │                                                                      │      │
│   │     ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                           │      │
│   │     │Head1│   │Head2│   │Head3│   │Head4│                           │      │
│   │     └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘                           │      │
│   │        └─────────┴─────────┴─────────┘                              │      │
│   │                          │                                           │      │
│   │                    Concat + Linear                                   │      │
│   └─────────────────────────────────────────────────────────────────────┘      │
│                              │                                                  │
│                     ┌────────┴────────┐                                        │
│                     │  Add & LayerNorm │                                        │
│                     └────────┬────────┘                                        │
│                              │                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐      │
│   │                      Feed-Forward Network                            │      │
│   │                                                                      │      │
│   │         Linear(256 → 1024) → ReLU → Linear(1024 → 256)              │      │
│   └─────────────────────────────────────────────────────────────────────┘      │
│                              │                                                  │
│                     ┌────────┴────────┐                                        │
│                     │  Add & LayerNorm │                                        │
│                     └────────┬────────┘                                        │
│                              │                                                  │
│                         × 4 layers                                             │
│                              │                                                  │
│                    Output: (B, 8, 256)                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Output Projection                                     │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐      │
│   │  Select Last Timestep: Z[:, -1, :] → (B, 256)                       │      │
│   └─────────────────────────────────────────────────────────────────────┘      │
│                              │                                                  │
│                              ▼                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐      │
│   │  MLP: Linear(256→256) → ReLU → Dropout → Linear(256→231)            │      │
│   └─────────────────────────────────────────────────────────────────────┘      │
│                              │                                                  │
│                    Output: (B, 231) Upper Body Pose                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Output Structure                                   │
│                                                                                 │
│   ┌───────────────┐  ┌─────────────────────────────────────────────────┐       │
│   │  Root Update  │  │              19 Body Joints                      │       │
│   │    (3 dim)    │  │         (19 × 12 = 228 dim)                     │       │
│   │               │  │                                                  │       │
│   │  Δx, Δy, Δz   │  │  Per Joint: Position(3) + Forward(3)            │       │
│   │               │  │            + Up(3) + Velocity(3)                 │       │
│   └───────────────┘  └─────────────────────────────────────────────────┘       │
│                                                                                 │
│                         Total: 3 + 228 = 231 features                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Figure 2: 데이터 차원 변환 흐름**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Dimension Transformation Flow                         │
└──────────────────────────────────────────────────────────────────────────────┘

  Input                    GMD                SlowFast            Transformer
    │                       │                    │                     │
    ▼                       ▼                    ▼                     ▼

(B, 576)              (B, 16, 42)            (B, 8, 256)          (B, 8, 256)
    │                       │                    │                     │
    │   ┌───────────────────┘                    │                     │
    │   │   Reshape +                            │                     │
    │   │   Spatial Norm +                       │                     │
    │   │   Temporal Norm                        │                     │
    │   │   (+6 delta features)                  │                     │
    │   │                                        │                     │
    │   │         ┌──────────────────────────────┘                     │
    │   │         │  Fast: frames 8-15                                 │
    │   │         │  Slow: frames 0,2,4...14                          │
    │   │         │  → Shared Embed (42→256)                          │
    │   │         │  → PE + Element-wise Sum                          │
    │   │         │                                                    │
    │   │         │               ┌────────────────────────────────────┘
    │   │         │               │  4× Transformer Encoder Layers
    │   │         │               │  (Self-Attention + FFN)
    │   │         │               │
    ▼   ▼         ▼               ▼

┌─────┐  ┌─────────┐  ┌───────────┐  ┌───────────┐  ┌─────────┐  ┌─────┐
│ 576 │─►│ 16 × 42 │─►│  8 × 256  │─►│  8 × 256  │─►│   256   │─►│ 231 │
└─────┘  └─────────┘  └───────────┘  └───────────┘  └─────────┘  └─────┘
  Flat    Sequence     SlowFast      Transformer    Last Step   Output
 Input     + GMD        Fused         Encoded       Selected    Pose
```

**Figure 3: SlowFast 시간 샘플링 패턴**

```
                        16 Input Frames (t=0 to t=15)
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │ t6 │ t7 │ t8 │ t9 │t10 │t11 │t12 │t13 │t14 │t15 │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
   │         │         │         │         │         │         │         │
   ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼

 SLOW PATHWAY (stride 2):  Global motion context (~0.53 sec)
┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐
│ t0 │     │ t2 │     │ t4 │     │ t6 │     │ t8 │     │t10 │     │t12 │     │t14 │
└────┘     └────┘     └────┘     └────┘     └────┘     └────┘     └────┘     └────┘
  ▲                                                                            ▲
  │                                                                            │
  └──────────────────── 8 frames spanning full history ────────────────────────┘


                                                 FAST PATHWAY (recent):  Fine motion detail (~0.27 sec)
                                              ┌────┬────┬────┬────┬────┬────┬────┬────┐
                                              │ t8 │ t9 │t10 │t11 │t12 │t13 │t14 │t15 │
                                              └────┴────┴────┴────┴────┴────┴────┴────┘
                                                ▲                                    ▲
                                                │                                    │
                                                └──── 8 consecutive recent frames ───┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ FUSION: Element-wise Addition                                                   │
│                                                                                 │
│  Slow (8 frames) ───────┐                                                       │
│        embed(42→256)    │                                                       │
│        + PE             ├──► E_slow ⊕ E_fast ──► Combined Features (8, 256)    │
│                         │                                                       │
│  Fast (8 frames) ───────┘                                                       │
│        embed(42→256)                                                            │
│        + PE                                                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Figure 4: 관절별 오차 히트맵 (시각화)**

```
                        Per-Joint Position Error (cm)

     LOW ERROR                                              HIGH ERROR
         │◄─────────────────────────────────────────────────────►│
         0                    2                    4              5

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │                            HEAD                                  │
  │                           ┌───┐                                  │
  │                           │2.4│ ██████████                       │
  │                           └───┘                                  │
  │                             │                                    │
  │                           ┌───┐                                  │
  │                      Neck │2.0│ ████████                         │
  │                           └───┘                                  │
  │                             │                                    │
  │    ┌───┐               ┌─────────┐               ┌───┐          │
  │LS  │2.6│ ██████████    │  Spine  │               │3.7│ RS       │
  │    └───┘               │   1.6   │ ██████        └───┘          │
  │      │                 │  ████   │                 │ ████████████│
  │    ┌───┐               └─────────┘               ┌───┐          │
  │LA  │2.2│ █████████                               │4.3│ RA       │
  │    └───┘                   │                     └───┘          │
  │      │                 ┌───────┐                   │ █████████████
  │    ┌───┐               │ Hips  │               ┌───┐            │
  │LFA │2.6│ ██████████    │  1.9  │ ███████       │4.5│ RFA       │
  │    └───┘               └───────┘               └───┘            │
  │      │                  /     \                   │ ██████████████
  │    ┌───┐            ┌───┐   ┌───┐             ┌───┐            │
  │LH  │2.6│ ██████████ │2.6│   │3.8│             │2.2│ RH         │
  │    └───┘       LUL  └───┘   └───┘ RUL         └───┘            │
  │                       │       │               █████████         │
  │                     ┌───┐   ┌───┐                               │
  │                 LL  │2.6│   │4.4│ RL                            │
  │                     └───┘   └───┘                               │
  │                 ████████   █████████████                        │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  Legend:
  LS=LeftShoulder, RS=RightShoulder, LA=LeftArm, RA=RightArm
  LFA=LeftForeArm, RFA=RightForeArm, LH=LeftHand, RH=RightHand
  LUL=LeftUpLeg, RUL=RightUpLeg, LL=LeftLeg, RL=RightLeg

  Key Observations:
  ├─ Spine/Trunk:  LOW error (0.9-1.9 cm) - close to tracked Head
  ├─ Left limbs:   MEDIUM error (2.2-2.6 cm)
  └─ Right limbs:  HIGH error (3.7-4.5 cm) - data distribution bias
```
## 개요

EgoPoser (ECCV 2024) 논문의 핵심 기법인 **GMD (Global Motion Decomposition)**와 **SlowFast Feature Fusion**을 Categorical Codebook Matching 파이프라인의 TrackerBody 네트워크에 적용한 기록입니다.

| 항목 | 기존 | 개선 |
|------|------|------|
| 파일 | `Network.py` | `Network_TrackerBody_Transformer.py` |
| 아키텍처 | Simple MLP | Transformer Encoder |
| 전처리 | 없음 | GMD (Spatial + Temporal) |
| Feature Fusion | 없음 | SlowFast |
| 평가 지표 | Loss만 | Loss + MPJPE + MPJVE |

---

## 1. 기존 코드 (MultiLayer Perceptron Network.py)

### 1.1 구조

```
Input (576) → Normalize → Linear(512) → ELU → Linear(512) → ELU → Linear(231) → Renormalize → Output
```

### 1.2 코드

```python
class Model(torch.nn.Module):
    def __init__(self, rng, layers, activations, dropout, input_norm, output_norm):
        super(Model, self).__init__()
        self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)
        self.Ynorm = Parameter(torch.from_numpy(output_norm), requires_grad=False)
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for i in range(len(layers)-1):
            self.W.append(self.weights([self.layers[i], self.layers[i+1]]))
            self.b.append(self.bias([1, self.layers[i+1]]))

    def forward(self, x):
        x = utility.Normalize(x, self.Xnorm)
        y = x
        for i in range(len(self.layers)-1):
            y = F.dropout(y, self.dropout, training=self.training)
            y = y.matmul(self.W[i]) + self.b[i]
            if self.activations[i] != None:
                y = self.activations[i](y)
        return utility.Renormalize(y, self.Ynorm)
```

### 1.3 한계점

| 한계 | 설명 |
|------|------|
| **Temporal 정보 손실** | 576 = 16 timesteps × 36 features를 flatten하여 시간 구조 손실 |
| **Global 좌표계** | 절대 좌표 사용으로 사용자 위치에 민감 |
| **Long-range dependency** | MLP는 입력 간 관계를 명시적으로 모델링하지 못함 |
| **평가 지표 부재** | MSE Loss만 사용, MPJPE/MPJVE 미제공 |

---

## 2. 개선 코드 (Network_TrackerBody_Transformer.py)

### 2.1 구조

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

### 2.2 핵심 변경점

#### 2.2.1 GMD (Global Motion Decomposition)

EgoPoser 논문의 전처리 기법:

**Spatial Normalization**
```python
# 각 timestep에서 Head XZ를 기준으로 상대화
head_x = head_features[:, :, 0:1]  # (B, 16, 1)
head_z = head_features[:, :, 2:3]

# Head → 0
head_modified[:, :, 0:1] -= head_x
head_modified[:, :, 2:3] -= head_z

# Wrists → Head 기준
lwrist_modified[:, :, 0:1] -= head_x
lwrist_modified[:, :, 2:3] -= head_z
rwrist_modified[:, :, 0:1] -= head_x
rwrist_modified[:, :, 2:3] -= head_z
```

**Temporal Normalization**
```python
# 첫 프레임 대비 XZ 변화량
head_delta = [head_x_t - head_x_0, head_z_t - head_z_0]
lwrist_delta = [lwrist_x_t - lwrist_x_0, lwrist_z_t - lwrist_z_0]
rwrist_delta = [rwrist_x_t - rwrist_x_0, rwrist_z_t - rwrist_z_0]

# 6 features per timestep 추가 → 36 + 6 = 42
```

#### 2.2.2 SlowFast Feature Fusion

```python
# Fast pathway: 최근 움직임 (후반 8 frames)
x_fast = x_seq[:, -T//2:, :]  # (B, 8, 42)

# Slow pathway: 전체 맥락 (stride 2)
x_slow = x_seq[:, ::2, :]     # (B, 8, 42)

# Shared embedding (같은 weights)
x_fast = self.linear_embedding(x_fast)
x_slow = self.linear_embedding(x_slow)

# Fusion
x = x_fast + x_slow
```

#### 2.2.3 Transformer Encoder

```python
# Positional Encoding
self.pos_encoder = PositionalEncoding(embed_dim, max_len=20, dropout=dropout)

# Transformer Encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    dim_feedforward=embed_dim * 4,
    dropout=dropout,
    batch_first=True
)
self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# Forward
x = self.pos_encoder(x)
x = self.transformer_encoder(x)
x = x[:, -1, :]  # last timestep
```

#### 2.2.4 MPJPE/MPJVE 메트릭

```python
def compute_mpjpe_mpjve(pred, target, num_joints=19, root_dim=3):
    """
    MPJPE: Mean Per Joint Position Error (위치 오차)
    MPJVE: Mean Per Joint Velocity Error (속도 오차)
    """
    for j in range(num_joints):
        joint_start = root_dim + j * 12
        pos_pred = pred[:, joint_start:joint_start+3]
        vel_pred = pred[:, joint_start+9:joint_start+12]
        ...

    mpjpe = torch.norm(pos_pred - pos_target, dim=2).mean()
    mpjve = torch.norm(vel_pred - vel_target, dim=2).mean()
    return mpjpe, mpjve
```

---

## 3. 코드 비교

### 3.1 Model 클래스 비교

| 항목 | Network.py | Network_TrackerBody_Transformer.py |
|------|------------|-------------------------------------|
| `__init__` 파라미터 | `rng, layers, activations, dropout, input_norm, output_norm` | `input_dim, output_dim, embed_dim, num_heads, num_layers, dropout, input_norm, output_norm` |
| 네트워크 구조 | Manual Linear layers (W, b) | `nn.TransformerEncoder` |
| 전처리 | 없음 | `GMDPreprocessorSequence` |
| Feature fusion | 없음 | `SlowFastTransformer` |

### 3.2 Forward Pass 비교

**기존 (Network.py)**
```python
def forward(self, x):
    x = utility.Normalize(x, self.Xnorm)
    y = x
    for i in range(len(self.layers)-1):
        y = F.dropout(y, self.dropout, training=self.training)
        y = y.matmul(self.W[i]) + self.b[i]
        if self.activations[i] != None:
            y = self.activations[i](y)
    return utility.Renormalize(y, self.Ynorm)
```

**개선 (Network_TrackerBody_Transformer.py)**
```python
def forward(self, x):
    x = utility.Normalize(x, self.Xnorm)

    # GMD Preprocessing: (B, 576) → (B, 16, 42)
    x_seq = self.gmd(x)

    # SlowFast Transformer: (B, 16, 42) → (B, 231)
    y = self.transformer(x_seq)

    return utility.Renormalize(y, self.Ynorm)
```

### 3.3 학습 루프 비교

**기존**
```python
for epoch in range(epochs):
    for i in range(0, sample_count, batch_size):
        print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
        ...
        loss = loss_function(...)
        ...
    print('Epoch', epoch+1, error/(sample_count/batch_size))
```

**개선**
```python
for epoch in range(epochs):
    for i in range(0, sample_count, batch_size):
        # 10% 단위 progress
        if progress // 10 > last_progress // 10:
            print(f'Epoch {epoch+1} - {progress}%')
        ...
        # MPJPE/MPJVE 계산
        with torch.no_grad():
            mpjpe, mpjve = compute_mpjpe_mpjve(yPred, yBatch)

    # Best model 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        utility.SaveONNX(..., '_best.onnx')

    print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}, MPJPE: {avg_mpjpe_cm:.2f}cm, MPJVE: {avg_mpjve:.4f}')
```

---

## 4. 새로 추가된 클래스/함수

| 이름 | 타입 | 설명 |
|------|------|------|
| `GMDPreprocessorSequence` | Class | GMD 전처리 (sequence output) |
| `SlowFastTransformer` | Class | SlowFast + Transformer Encoder |
| `PositionalEncoding` | Class | Sinusoidal positional encoding |
| `compute_mpjpe_mpjve()` | Function | 표준 평가 지표 계산 |
| `run_shape_verification()` | Function | Shape 검증 테스트 |

---

## 5. 하이퍼파라미터

| 파라미터 | 기존 | 개선 |
|----------|------|------|
| Architecture | MLP [576, 512, 512, 231] | Transformer (4 layers, 4 heads) |
| Embed dim | 512 (hidden) | 256 |
| Batch size | 32 | 64 |
| Epochs | 150 | 150 |
| Learning rate | 1e-4 | 1e-4 |
| Dropout | 0.25 | 0.1 |

---

## 6. 학습 결과

### Transformer + GMD + SlowFast

| Metric | 값 |
|--------|-----|
| Loss | 0.059360 |
| **MPJPE** | **2.75 cm** |
| MPJVE | 0.1644 |
| Parameters | ~3.3M |

---

## 7. ONNX 호환성

입출력 인터페이스 동일 유지:

```python
# Input/Output (동일)
Input:  (B, 576)
Output: (B, 231)

# Unity Sentis에서 그대로 사용 가능
```

---

## 8. 파일 구조

```
Models/MultiLayerPerceptron/
├── Network.py                           # 기존 Simple MLP
├── Network_TrackerBody_Transformer.py   # [NEW] Transformer + GMD + SlowFast
├── train_slurm_transformer.sh           # [NEW] SLURM 학습 스크립트
└── CHANGELOG_TrackerBody.md             # [NEW] 이 문서
```

---

## 9. 핵심 기여 요약

| # | 기여 | 설명 |
|---|------|------|
| 1 | **GMD 적용** | Spatial + Temporal normalization으로 위치 불변성 향상 |
| 2 | **SlowFast 적용** | Fast (최근) + Slow (맥락) 정보 동시 활용 |
| 3 | **Transformer 아키텍처** | Self-attention으로 temporal dependency 학습 |
| 4 | **MPJPE/MPJVE 메트릭** | 표준화된 포즈 추정 평가 지표 추가 |
| 5 | **ONNX 호환성** | 기존 Unity 파이프라인 수정 없이 사용 가능 |

---

## 10. 3-Way 코드 비교 (Codebook vs EgoPoser vs Ours)

### 10.1 전체 아키텍처 비교

| 항목 | Codebook (Network.py) | EgoPoser (egoposer.py) | Ours |
|------|----------------------|------------------------|------|
| **Input 형식** | `(B, 576)` flat | `dict{'sparse_input', 'fov_l', 'fov_r'}` | `(B, 576)` flat |
| **Output 형식** | `(B, 231)` flat | `dict{'root_orient', 'pose_body', 'betas'}` | `(B, 231)` flat |
| **아키텍처** | MLP | Transformer | Transformer |
| **GMD** | X | O (forward 내부) | O (별도 모듈) |
| **SlowFast** | X | O | O |
| **Positional Encoding** | X | X | O |
| **Target** | Unity ONNX | SMPL body model | Unity ONNX |

---

## 11. 컴포넌트별 차별점 및 Contribution

### 11.1 Input/Output 인터페이스

#### Codebook (기존)
```python
def forward(self, x):  # x: (B, 576) flat tensor
    ...
    return y  # (B, 231) flat tensor
```

#### EgoPoser (원본)
```python
def forward(self, x):
    input_tensor = x['sparse_input']  # dict 입력
    fov_l = x['fov_l']
    fov_r = x['fov_r']
    ...
    output = {}
    output['root_orient'] = root_orient  # (B, 6)
    output['pose_body'] = pose_body      # (B, 126)
    output['betas'] = betas              # (B, 16)
    return output  # dict 출력
```

#### Ours
```python
def forward(self, x):  # x: (B, 576) flat tensor - Codebook 호환
    x = utility.Normalize(x, self.Xnorm)
    x_seq = self.gmd(x)
    y = self.transformer(x_seq)
    return utility.Renormalize(y, self.Ynorm)  # (B, 231) - Codebook 호환
```

| Contribution | 설명 |
|--------------|------|
| **Codebook 호환성 유지** | EgoPoser의 dict I/O 대신 flat tensor 유지로 기존 Unity 파이프라인 수정 불필요 |
| **Normalize/Renormalize** | Codebook의 정규화 방식 그대로 적용하여 학습 안정성 유지 |

---

### 11.2 GMD (Global Motion Decomposition)

#### Codebook (기존)
```python
# 전처리 없음
x = utility.Normalize(x, self.Xnorm)  # 단순 정규화만
```

#### EgoPoser (원본)
```python
# Spatial Normalization - 하드코딩된 인덱스
head_horizontal_trans = input_tensor.clone()[...,36:38].detach()
input_tensor[...,36:38] -= head_horizontal_trans
input_tensor[...,39:41] -= head_horizontal_trans
input_tensor[...,42:44] -= head_horizontal_trans

# Temporal Normalization
delta_0 = input_tensor[...,36:38] - input_tensor[...,[0],36:38]
delta_1 = input_tensor[...,39:41] - input_tensor[...,[0],39:41]
delta_2 = input_tensor[...,42:44] - input_tensor[...,[0],42:44]
input_tensor = torch.cat([input_tensor, delta_0, delta_1, delta_2], dim=-1)
```

#### Ours
```python
class GMDPreprocessorSequence(nn.Module):
    """별도 모듈로 분리하여 재사용성 향상"""

    def forward(self, x):
        # (B, 576) → (B, 16, 36) reshape
        head_features = x[:, 0:192].view(B, 16, 12)
        lwrist_features = x[:, 192:384].view(B, 16, 12)
        rwrist_features = x[:, 384:576].view(B, 16, 12)

        # Spatial Normalization - 명시적 인덱싱
        head_x = head_features[:, :, 0:1]  # Position X
        head_z = head_features[:, :, 2:3]  # Position Z

        head_modified[:, :, 0:1] -= head_x
        head_modified[:, :, 2:3] -= head_z
        lwrist_modified[:, :, 0:1] -= head_x
        lwrist_modified[:, :, 2:3] -= head_z
        rwrist_modified[:, :, 0:1] -= head_x
        rwrist_modified[:, :, 2:3] -= head_z

        # Temporal Normalization
        head_delta = torch.cat([
            head_modified[:, :, 0:1] - head_x_0,
            head_modified[:, :, 2:3] - head_z_0
        ], dim=2)
        # ... lwrist_delta, rwrist_delta 동일

        return torch.cat([head_modified, lwrist_modified, rwrist_modified, all_deltas], dim=2)
```

| Contribution | 설명 |
|--------------|------|
| **모듈화** | EgoPoser는 forward 내부에 인라인, Ours는 별도 클래스로 분리 |
| **데이터 구조 적응** | EgoPoser의 하드코딩 인덱스([36:38]) 대신 Codebook 데이터 구조에 맞게 재구현 |
| **명시적 reshape** | `(B, 576)` flat → `(B, 16, 36)` sequence로 변환하여 temporal 구조 복원 |
| **가독성** | 각 tracker(Head, LWrist, RWrist)별로 명시적 처리 |

---

### 11.3 SlowFast Feature Fusion

#### Codebook (기존)
```python
# SlowFast 없음 - 단순 MLP
y = x.matmul(self.W[i]) + self.b[i]
```

#### EgoPoser (원본)
```python
x_fast = input_tensor[:,-input_tensor.shape[1]//2:,...]
x_slow = input_tensor[:,::2,...]

x_fast = self.linear_embedding(x_fast)
x_slow = self.linear_embedding(x_slow)
x = x_fast + x_slow
```

#### Ours
```python
class SlowFastTransformer(nn.Module):
    def __init__(self, ...):
        # Shared embedding (EgoPoser와 동일)
        self.linear_embedding = nn.Linear(input_dim_per_timestep, embed_dim)

        # Positional Encoding 추가 (EgoPoser에 없음)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=20, dropout=dropout)

        # Transformer with batch_first=True (EgoPoser는 False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True  # 간결한 코드
        )

    def forward(self, x_seq):
        # SlowFast extraction (EgoPoser와 동일)
        x_fast = x_seq[:, -T//2:, :]
        x_slow = x_seq[:, ::2, :]

        x_fast = self.linear_embedding(x_fast)
        x_slow = self.linear_embedding(x_slow)
        x = x_fast + x_slow

        # Positional Encoding 추가
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # last timestep

        return self.output_projection(x)
```

| Contribution | 설명 |
|--------------|------|
| **Positional Encoding 추가** | EgoPoser에 없는 positional encoding 추가로 temporal 위치 정보 명시적 인코딩 |
| **batch_first=True** | EgoPoser는 permute(1,0,2) 필요, Ours는 batch_first로 간결 |
| **단일 output_projection** | EgoPoser의 3개 분리 decoder 대신 단일 projection으로 ONNX 호환 |

---

### 11.4 Transformer Encoder

#### Codebook (기존)
```python
# Transformer 없음 - MLP만 사용
for i in range(len(self.layers)-1):
    y = F.dropout(y, self.dropout, training=self.training)
    y = y.matmul(self.W[i]) + self.b[i]
```

#### EgoPoser (원본)
```python
encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

# Forward - permute 필요
x = x.permute(1,0,2)  # (B,T,D) → (T,B,D)
x = self.transformer_encoder(x)
x = x.permute(1,0,2)  # (T,B,D) → (B,T,D)
x = x[:, -1]
```

#### Ours
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    dim_feedforward=embed_dim * 4,  # 명시적 FFN 크기
    dropout=dropout,
    batch_first=True
)
self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# Forward - permute 불필요
x = self.pos_encoder(x)
x = self.transformer_encoder(x)
x = x[:, -1, :]
```

| Contribution | 설명 |
|--------------|------|
| **batch_first=True** | permute 제거로 코드 간결화 및 잠재적 성능 향상 |
| **명시적 dim_feedforward** | EgoPoser는 default(2048), Ours는 embed_dim*4로 모델 크기 조절 |
| **Positional Encoding** | Transformer에 필수적인 위치 정보 추가 |

---

### 11.5 Output Decoder

#### Codebook (기존)
```python
# 단일 출력
return utility.Renormalize(y, self.Ynorm)  # (B, 231)
```

#### EgoPoser (원본)
```python
# 3개 분리 decoder (SMPL 모델용)
self.global_orientation_decoder = nn.Sequential(
    nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 6)
)
self.joint_rotation_decoder = nn.Sequential(
    nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 126)
)
self.shape_decoder = nn.Sequential(
    nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 16)
)

# Forward
output = {}
output['root_orient'] = self.global_orientation_decoder(x)  # 6D rotation
output['pose_body'] = self.joint_rotation_decoder(x)        # 21 joints × 6
output['betas'] = self.shape_decoder(x)                     # body shape
```

#### Ours
```python
# 단일 통합 decoder (ONNX 호환)
self.output_projection = nn.Sequential(
    nn.Linear(embed_dim, embed_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(embed_dim, output_dim)  # 231
)

# Forward
out = self.output_projection(x)  # (B, 231)
return utility.Renormalize(out, self.Ynorm)
```

| Contribution | 설명 |
|--------------|------|
| **단일 출력** | EgoPoser의 SMPL용 3분리 decoder를 Codebook 호환 단일 출력으로 변경 |
| **Dropout 추가** | output projection에 dropout 추가로 과적합 방지 |
| **Renormalize** | Codebook 방식의 출력 정규화 유지 |

---

### 11.6 FOV Masking

#### EgoPoser (원본)
```python
def input_masking(self, input_tensor, fov_l, fov_r):
    """Hand tracking이 FOV 밖일 때 마스킹"""
    lefthand_idx = [*range(6,12),*range(24,30),*range(39,42),*range(48,51)]
    righthand_idx = [*range(12,18),*range(30,36),*range(42,45),*range(51,54)]

    lefthand_out_fov_selector = torch.zeros(input_tensor.shape, dtype=torch.bool)
    lefthand_out_fov_selector[fov_l==False] = True
    # ... masking logic
    input_tensor[lefthand_out_fov_selector] = 0
```

#### Ours
```python
# FOV Masking 제거
# 이유: VR Controller는 항상 tracking됨 (Hand tracking과 다름)
```

| Contribution | 설명 |
|--------------|------|
| **FOV Masking 제거** | EgoPoser는 hand tracking용으로 FOV 밖 손 마스킹 필요, VR Controller는 항상 추적되므로 불필요 |
| **코드 단순화** | 불필요한 기능 제거로 inference 속도 향상 |

---

### 11.7 평가 메트릭

#### Codebook (기존)
```python
# Loss만 출력
print('Epoch', epoch+1, error/(sample_count/batch_size))
```

#### EgoPoser (원본)
```python
# 별도 evaluation 스크립트에서 MPJPE 계산
```

#### Ours
```python
def compute_mpjpe_mpjve(pred, target, num_joints=19, root_dim=3):
    """학습 중 실시간 MPJPE/MPJVE 계산"""
    for j in range(num_joints):
        joint_start = root_dim + j * 12
        pos_pred = pred[:, joint_start:joint_start+3]
        pos_target = target[:, joint_start:joint_start+3]
        vel_pred = pred[:, joint_start+9:joint_start+12]
        vel_target = target[:, joint_start+9:joint_start+12]

    mpjpe = torch.norm(pos_pred - pos_target, dim=2).mean()
    mpjve = torch.norm(vel_pred - vel_target, dim=2).mean()
    return mpjpe, mpjve

# 학습 중 출력
print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}, MPJPE: {avg_mpjpe_cm:.2f}cm, MPJVE: {avg_mpjve:.4f}')
```

| Contribution | 설명 |
|--------------|------|
| **실시간 MPJPE/MPJVE** | 학습 중 표준 pose estimation 메트릭 실시간 모니터링 |
| **cm 단위 출력** | MPJPE를 cm로 변환하여 직관적 이해 |
| **Velocity 평가** | MPJVE로 움직임 부드러움 평가 |

---

### 11.8 학습 개선사항

#### Codebook (기존)
```python
# 모든 epoch 저장
torch.save(network, save + '/' + id + '_' + str(epoch+1) + '.pt')
utility.SaveONNX(...)
print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
```

#### Ours
```python
# Best model 별도 저장
if avg_loss < best_loss:
    best_loss = avg_loss
    utility.SaveONNX(path=save + '/' + id + '_best.onnx', ...)
    print(f"  New best model saved!")

# 10% 단위 progress (로그 파일 크기 감소)
if progress // 10 > last_progress // 10:
    print(f'Epoch {epoch+1} - {progress}%')
```

| Contribution | 설명 |
|--------------|------|
| **Best model 저장** | 가장 좋은 모델 자동 저장 |
| **로그 최적화** | 10% 단위 출력으로 SLURM 로그 파일 크기 감소 |

---

## 12. 전체 기여 요약

| # | 컴포넌트 | Codebook 대비 | EgoPoser 대비 |
|---|----------|---------------|---------------|
| 1 | **I/O 인터페이스** | 동일 유지 | flat tensor로 변경 (ONNX 호환) |
| 2 | **GMD** | 새로 추가 | 모듈화 + 데이터 구조 적응 |
| 3 | **SlowFast** | 새로 추가 | 동일 로직 유지 |
| 4 | **Positional Encoding** | 새로 추가 | 새로 추가 |
| 5 | **Transformer** | MLP→Transformer | batch_first=True |
| 6 | **Output Decoder** | 동일 구조 | 3분리→단일 통합 |
| 7 | **FOV Masking** | N/A | 제거 (불필요) |
| 8 | **MPJPE/MPJVE** | 새로 추가 | 학습 중 실시간 계산 |
| 9 | **Best Model** | 새로 추가 | N/A |

---

## 13. 참고 문헌

- Jiang, J., et al. "EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere." ECCV 2024.
- Feichtenhofer, C., et al. "SlowFast Networks for Video Recognition." ICCV 2019.
- Starke, S., et al. "Categorical Codebook Matching for Embodied Character Controllers." SIGGRAPH 2024.

---

*Last Updated: 2024-12-11*
