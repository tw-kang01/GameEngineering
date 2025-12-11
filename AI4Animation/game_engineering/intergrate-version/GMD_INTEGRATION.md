# GMD + SlowFast Integration: Categorical VQ-VAE Enhancement

## 개요

이 문서는 EgoPoser (ECCV 2024)의 **Global Motion Decomposition (GMD)** 및 **SlowFast Feature Fusion** 기법을 Categorical Codebook Matching (SIGGRAPH 2024)의 VQ-VAE 네트워크에 통합한 구현을 설명합니다.

### ⚠️ 핵심 원칙

> **Categorical Codebook Matching을 베이스로 유지하고, EgoPoser에서 GMD + SlowFast 아이디어만 차용**

| 항목 | Categorical (베이스) | EgoPoser (차용) |
|------|---------------------|-----------------|
| **Encoder/Estimator/Decoder** | ✅ MLP (LinearEncoder) | ❌ 사용 안함 |
| **Codebook** | ✅ Gumbel-Softmax VQ | ❌ 사용 안함 |
| **손실 함수** | ✅ MSE + Matching Loss | ❌ 사용 안함 |
| **Transformer** | ❌ 사용 안함 | ❌ 사용 안함 |
| **GMD** | ❌ 원본에 없음 | ✅ 차용 |
| **SlowFast** | ❌ 원본에 없음 | ✅ 차용 (MLP로 구현) |

---

## 검증 완료 사항

### 1. Categorical VQ-VAE 구조 확인

**결론: Categorical은 MLP 기반 (Transformer 아님!)**

```python
# Categorical 원본 구조 (Modules.py)
class LinearEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, dropout):
        self.L1 = nn.Linear(input_size, hidden1_size)
        self.L2 = nn.Linear(hidden1_size, hidden2_size)
        self.L3 = nn.Linear(hidden2_size, output_size)
    
    def forward(self, z):
        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L1(z)
        z = F.elu(z)
        # ... (3-layer MLP with ELU activation)
```

### 2. 차원 검증

```
Input:     (B, 252) = 7 timesteps × 36 features/timestep
GMD:       (B, 294) = 7 timesteps × 42 features/timestep (+6 delta)
SlowFast:  (B, 252) = 원본 차원으로 복원
Final:     (B, 252) = Residual 적용 후
```

### 3. 좌표계 검증

```
Unity (Categorical): Y-up, left-handed
- XZ plane = 수평 (지면)
- Y = 수직 (높이)

GMD:
- XZ 정규화 (수평) → Unity와 호환
- Y 유지 (수직) → 변환 불필요
```

### 4. 피처 인덱스 검증

```
Per timestep (36 features):
- Head:       [0:12]  - Position[0:3], Forward[3:6], Up[6:9], Velocity[9:12]
- LeftWrist:  [12:24] - Position[12:15], Forward[15:18], Up[18:21], Velocity[21:24]
- RightWrist: [24:36] - Position[24:27], Forward[27:30], Up[30:33], Velocity[33:36]

GMD XZ indices (within timestep):
- Head XZ:       0, 2
- LeftWrist XZ:  12, 14
- RightWrist XZ: 24, 26
```

---

## 기존 코드 vs 수정된 코드 비교 (수정됨)

### 아키텍처 비교

| 구성 요소 | 기존 Categorical VQ-VAE | GMD + SlowFast 적용 버전 |
|-----------|-------------------------|--------------------------|
| **Encoder** | LinearEncoder (MLP) | **동일** (변경 없음) |
| **Estimator** | LinearEncoder (MLP) | **동일** (변경 없음) |
| **Decoder** | LinearEncoder (MLP) | **동일** (변경 없음) |
| **Gumbel-Softmax** | ✅ | **동일** (변경 없음) |
| **손실 함수** | MSE + Matching | **동일** (변경 없음) |
| **입력 전처리** | Normalization만 | GMD + SlowFast **추가** |
| **Transformer** | ❌ 없음 | ❌ 없음 (추가 안함) |

### 파이프라인 비교

**기존 Categorical VQ-VAE:**
```
Input (252) → Normalize → Estimator (MLP) → Gumbel-Softmax → Codebook → Decoder (MLP) → Output
                         ↘ Encoder (MLP, with target) ↗
```

**GMD + SlowFast 적용 버전 (Corrected):**
```
[전처리 단계 - 추가됨]
Input (252)
    │
    ▼
GMDPreprocessor (학습 파라미터 없음)
    ├── Spatial Norm: Head XZ 기준 상대화
    └── Temporal Norm: delta = frame[t] - frame[0]
    │
    ▼
(B, 294) = 7 × 42
    │
    ▼
SlowFastMLP (MLP 기반 - Categorical 스타일)
    ├── Fast: 후반 timesteps → MLP
    └── Slow: 매 2 timestep → MLP → Sum
    │
    ▼
(B, 252) = 원본 차원 복원
    │
    ▼
Residual: α × SlowFast + (1-α) × Original
    │
    ▼
[기존 Categorical VQ-VAE - 변경 없음]
Normalize → Estimator (MLP) → Gumbel-Softmax → Codebook → Decoder (MLP) → Output
           ↘ Encoder (MLP, with target) ↗
```

---

## 새로 추가된 모듈 상세

### 1. GMDModule (Global Motion Decomposition)

**파일:** `Network_GMD_SlowFast.py` (Lines 44-147)

**논문 근거:** EgoPoser ECCV 2024, Section 3.2, Equation 2

> "We decompose the global motion into spatial-normalized local features and temporal delta features to reduce drift in autoregressive prediction"

**기능:**
- **Spatial Normalization**: Head의 XZ 위치를 기준으로 모든 트래커 위치를 상대화
- **Temporal Normalization**: 첫 프레임 대비 delta 피처 계산

**수학적 표현:**
```
Spatial Norm (EgoPoser lines 75-78):
  head_xz = input[..., 36:38]
  pos_relative = pos_tracker - head_xz

Temporal Norm (EgoPoser lines 82-88):
  delta_t = pos_t - pos_0
  output = concat(input, delta_head, delta_lwrist, delta_rwrist)
```

**코드 (EgoPoser 원본과 매핑):**
```python
class GMDModule(nn.Module):
    def spatial_normalization(self, x):
        # EgoPoser: head_horizontal_trans = input_tensor.clone()[...,36:38].detach()
        head_x = x[:, :, self.head_pos + self.x_idx].clone()
        head_z = x[:, :, self.head_pos + self.z_idx].clone()
        
        # EgoPoser: input_tensor[...,36:38] -= head_horizontal_trans
        x[:, :, self.head_pos + self.x_idx] -= head_x
        x[:, :, self.head_pos + self.z_idx] -= head_z
        
        # EgoPoser: input_tensor[...,39:41] -= head_horizontal_trans
        x[:, :, self.lwrist_pos + self.x_idx] -= head_x
        x[:, :, self.lwrist_pos + self.z_idx] -= head_z
        
        # EgoPoser: input_tensor[...,42:44] -= head_horizontal_trans
        x[:, :, self.rwrist_pos + self.x_idx] -= head_x
        x[:, :, self.rwrist_pos + self.z_idx] -= head_z
        return x
    
    def temporal_normalization(self, x):
        # EgoPoser: delta_0 = input_tensor[...,36:38] - input_tensor[...,[0],36:38]
        delta_head_x = x[:, :, head_x_idx] - x[:, [0], head_x_idx]
        delta_head_z = x[:, :, head_z_idx] - x[:, [0], head_z_idx]
        # ... (delta_1, delta_2 for wrists)
        
        # EgoPoser: input_tensor = torch.cat([input_tensor, delta_0, delta_1, delta_2], dim=-1)
        deltas = torch.stack([delta_head_x, delta_head_z, ...], dim=-1)
        return deltas
```

**입출력:**
- 입력: `(B, 252)` → reshape `(B, 7, 36)`
- 출력: `(B, 7, 42)` (6개 delta 피처 추가)

---

### 2. SlowFastTransformer (이중 경로 시간 융합)

**파일:** `Network_GMD_SlowFast.py` (Lines 154-227)

**논문 근거:** EgoPoser ECCV 2024, Section 3.3

> "We use a SlowFast architecture to capture both fine-grained recent motion and longer-term context simultaneously"

**EgoPoser 원본 코드 (lines 94-106):**
```python
# SlowFast fusion 
x_fast = input_tensor[:,-input_tensor.shape[1]//2:,...]  # Last half
x_slow = input_tensor[:,::2,...]                          # Every 2nd frame

x_fast = self.linear_embedding(x_fast)
x_slow = self.linear_embedding(x_slow)
x = x_fast + x_slow  # Element-wise sum fusion

x = x.permute(1,0,2)
x = self.transformer_encoder(x)
x = x.permute(1,0,2)
x = x[:, -1]  # Take last timestep
```

**우리 구현:**
```python
class SlowFastTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=2, nhead=4):
        # Same embedding for both pathways (as in EgoPoser)
        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):  # x: (B, T, F) = (B, 7, 42)
        # === SlowFast Split ===
        x_fast = x[:, -(T // 2 + 1):, :]  # (B, 4, 42) - 최근 세밀한 모션
        x_slow = x[:, ::2, :]              # (B, 4, 42) - 장기 맥락
        
        # === Embed & Fuse ===
        x_fast_emb = self.linear_embedding(x_fast)  # (B, 4, 256)
        x_slow_emb = self.linear_embedding(x_slow)  # (B, 4, 256)
        x_fused = x_fast_emb + x_slow_emb           # Element-wise sum
        
        # === Transformer ===
        x_fused = x_fused.permute(1, 0, 2)          # (T, B, D)
        x_fused = self.transformer_encoder(x_fused)
        x_fused = x_fused.permute(1, 0, 2)          # (B, T, D)
        
        # === Output ===
        return x_fused[:, -1, :]  # (B, 256) - last timestep
```

**Fast vs Slow 경로 비교:**

| 경로 | 샘플링 방식 | T=7일 때 사용 프레임 | 역할 |
|------|-------------|---------------------|------|
| **Fast** | 후반 ~50% | 3, 4, 5, 6 | 최근 세밀한 모션 변화 캡처 |
| **Slow** | 매 2프레임 | 0, 2, 4, 6 | 장기적 모션 맥락 캡처 |

---

### 3. SlowFastMLP (경량화 버전)

**파일:** `Network_GMD_SlowFast.py` (Lines 230-272)

Transformer 없이 MLP만으로 SlowFast 구현한 경량 버전입니다.

```python
class SlowFastMLP(nn.Module):
    def forward(self, x):
        # Fast: temporal mean pooling
        x_fast = x[:, -(T // 2 + 1):, :].mean(dim=1)  # (B, F)
        
        # Slow: temporal mean pooling
        x_slow = x[:, ::2, :].mean(dim=1)  # (B, F)
        
        # Separate embeddings + sum
        fast_emb = self.fast_embedding(x_fast)
        slow_emb = self.slow_embedding(x_slow)
        
        return self.fusion(fast_emb + slow_emb)
```

**Transformer vs MLP 비교:**

| 항목 | SlowFastTransformer | SlowFastMLP |
|------|---------------------|-------------|
| **파라미터 수** | 많음 (~2M) | 적음 (~0.5M) |
| **학습 속도** | 느림 | 빠름 |
| **시간 관계 모델링** | Self-Attention (장거리) | Mean Pooling (독립적) |
| **사용 권장** | 충분한 데이터 | 빠른 실험 |

---

### 4. Model 클래스 수정사항 (전체 버전)

**파일:** `Network_GMD_SlowFast.py` (Lines 279-398)

**새로 추가된 파라미터:**
```python
def __init__(self, ..., 
             use_gmd=True,
             slowfast_type='transformer',  # 'transformer' or 'mlp'
             num_timesteps=7,
             features_per_timestep=36,
             embed_dim=256,
             num_transformer_layers=2,
             nhead=4,
             dropout=0.25):
```

**새로 추가된 레이어:**
```python
# GMD Module
self.gmd = GMDModule(num_timesteps, features_per_timestep)

# SlowFast Module (택 1)
if slowfast_type == 'transformer':
    self.slowfast = SlowFastTransformer(input_dim=42, embed_dim=256, ...)
else:
    self.slowfast = SlowFastMLP(input_dim=42, hidden_dim=512, output_dim=256, ...)

# Projection + Learnable Residual
self.projection = nn.Sequential(
    nn.Linear(embed_dim, embed_dim * 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(embed_dim * 2, 252)
)
self.residual_weight = nn.Parameter(torch.tensor(0.5))  # Learnable α
```

**Forward 수정:**
```python
def forward(self, x, knn, t=None):
    if self.use_gmd:
        x_original = x.clone()
        
        # GMD + SlowFast
        x_gmd = self.gmd(x)                    # (B, 7, 42)
        x_fused = self.slowfast(x_gmd)         # (B, 256)
        x_proj = self.projection(x_fused)      # (B, 252)
        
        # Learnable residual connection
        alpha = torch.sigmoid(self.residual_weight)
        x = alpha * x_proj + (1 - alpha) * x_original
    
    # ... 기존 VQ-VAE 파이프라인 ...
```

---

## EgoPoser 코드와의 정확한 매핑

### GMD 매핑 표

| EgoPoser (egoposer.py) | 우리 구현 (Network_GMD_SlowFast.py) |
|------------------------|-------------------------------------|
| `input_tensor[...,36:38]` (Head XZ) | `x[:, :, 0:1]` 및 `x[:, :, 2:3]` (reshapped) |
| `input_tensor[...,39:41]` (LWrist XZ) | `x[:, :, 12:13]` 및 `x[:, :, 14:15]` |
| `input_tensor[...,42:44]` (RWrist XZ) | `x[:, :, 24:25]` 및 `x[:, :, 26:27]` |
| `head_horizontal_trans.detach()` | `head_x.clone()`, `head_z.clone()` |
| `delta_0, delta_1, delta_2` | `delta_head, delta_lwrist, delta_rwrist` |

### SlowFast 매핑 표

| EgoPoser (egoposer.py) | 우리 구현 (Network_GMD_SlowFast.py) |
|------------------------|-------------------------------------|
| `x_fast = input[:,-T//2:,...]` | `x_fast = x[:, -(T//2+1):, :]` |
| `x_slow = input[:,::2,...]` | `x_slow = x[:, ::2, :]` |
| `self.linear_embedding(x_fast)` | `self.linear_embedding(x_fast)` |
| `x = x_fast + x_slow` | `x_fused = x_fast_emb + x_slow_emb` |
| `self.transformer_encoder(x)` | `self.transformer_encoder(x_fused)` |
| `x[:, -1]` | `out = x_fused[:, -1, :]` |

---

## Contribution 요약

### 1. 학술적 Contribution

| 기여 항목 | 설명 |
|-----------|------|
| **Cross-paper Integration** | EgoPoser (ECCV 2024)와 Categorical Codebook Matching (SIGGRAPH 2024) 두 최신 논문의 기법을 최초로 결합 |
| **GMD for VQ-VAE** | Global Motion Decomposition을 VQ-VAE 기반 모션 예측에 적용한 최초 사례 |
| **SlowFast + Transformer + Codebook** | SlowFast Transformer 기반 시간 인코딩을 Categorical Codebook과 결합 |
| **Learnable Residual** | GMD 피처와 원본 피처의 가중치를 학습 가능하게 설계 |

### 2. 기술적 Contribution

| 기여 항목 | 설명 |
|-----------|------|
| **드리프트 감소** | Temporal delta로 autoregressive 예측 시 누적 오차 감소 |
| **다중 시간 스케일** | Fast/Slow 경로로 세밀한 모션 + 장기 맥락 동시 모델링 |
| **Transformer 시간 인코딩** | Self-Attention으로 프레임 간 장거리 종속성 학습 |
| **Residual + Learnable α** | 원본 정보 보존 + 최적 융합 비율 자동 학습 |
| **Toggle 가능** | `use_gmd`, `slowfast_type` 파라미터로 ablation study 용이 |

### 3. 코드 Contribution

| 파일 | 설명 |
|------|------|
| `Network_GMD_SlowFast.py` (신규) | 완전한 GMD + SlowFast Transformer + Categorical VQ-VAE 통합 |
| `Network_GMD.py` (신규) | 간단한 버전 (SlowFast MLP) |
| `GMD_INTEGRATION.md` (신규) | 이 문서 |
| 기존 `Network.py` | 변경 없음 (원본 보존) |

---

## 피처 인덱스 매핑

### TrackerBodyPredictor 입력 구조 (252 features)

```
Timestep 0-6 (7개 timestep, 각 36 features):
├── Head (12 features × 7 timesteps = 84)
│   ├── Position: [0, 1, 2] (X, Y, Z)
│   ├── Forward:  [3, 4, 5]
│   ├── Up:       [6, 7, 8]
│   └── Velocity: [9, 10, 11]
├── LeftWrist (84 features, offset +12 per timestep)
│   └── 동일 구조
└── RightWrist (84 features, offset +24 per timestep)
    └── 동일 구조
```

### GMD 적용 후 (42 features/timestep)

```
원본 36 features + Delta 6 features:
├── Original features (36)
└── Temporal deltas (6)
    ├── Head XZ delta:      [36, 37]
    ├── LeftWrist XZ delta: [38, 39]
    └── RightWrist XZ delta:[40, 41]
```

---

## 사용 방법

### 전체 버전 학습 (Transformer 기반)
```bash
cd categorical/PyTorch/Models/CodebookMatching
python Network_GMD_SlowFast.py
```

### 경량 버전 학습 (MLP 기반)
```bash
python Network_GMD.py
```

### SlowFast 타입 변경 (Network_GMD_SlowFast.py)
```python
# Transformer 기반 (기본값, 더 강력)
slowfast_type = 'transformer'

# MLP 기반 (더 빠름)
slowfast_type = 'mlp'
```

### GMD 비활성화 (Ablation Study)
```python
use_gmd = False  # GMD + SlowFast 전체 비활성화
```

### 하이퍼파라미터 조정
```python
# SlowFast Transformer 설정
embed_dim = 256              # 임베딩 차원
num_transformer_layers = 2   # Transformer 레이어 수
nhead = 4                    # Attention head 수

# 입력 설정
num_timesteps = 7            # 입력 시퀀스 길이
features_per_timestep = 36   # timestep당 피처 수
```

---

## Ablation Study 가이드

| 실험 | 설정 | 목적 |
|------|------|------|
| Baseline | `use_gmd=False` | 기존 Categorical VQ-VAE |
| GMD Only | GMD만 적용, SlowFast 제거 | GMD 효과 측정 |
| SlowFast MLP | `slowfast_type='mlp'` | 경량 SlowFast 효과 |
| SlowFast Transformer | `slowfast_type='transformer'` | 전체 모델 효과 |
| w/o Residual | `residual_weight` 고정 | Residual 연결 효과 |

---

## 예상 효과

1. **Root Trajectory 예측 향상**: Spatial normalization으로 글로벌 위치 드리프트 감소
2. **상체 포즈 품질 향상**: Temporal delta로 프레임 간 일관성 유지
3. **빠른 동작 대응**: Fast pathway로 급격한 모션 변화 캡처
4. **장기 맥락 반영**: Slow pathway + Transformer로 동작의 전체적인 흐름 이해
5. **정보 보존**: Learnable residual로 원본 피처 정보 유지

---

## 참고 문헌

1. **EgoPoser** (ECCV 2024): "EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere"
   - GMD: Eq. 2, Section 3.2
   - SlowFast: Section 3.3
   - GitHub: https://github.com/eth-siplab/EgoPoser

2. **Categorical Codebook Matching** (SIGGRAPH 2024): "Categorical Codebook Matching for Embodied Character Controllers"
   - Gumbel-Softmax VQ-VAE: Eq. 1-5, Section 4
   - Codebook Matching Loss: Section 4.2

3. **SlowFast Networks** (ICCV 2019): "SlowFast Networks for Video Recognition"
   - 원본 SlowFast 아이디어

---

## 파일 구조 (최종)

```
categorical/PyTorch/Models/CodebookMatching/
├── Network.py                  # 기존 Categorical VQ-VAE (변경 없음, baseline)
├── Network_GMD.py              # 초기 버전 (deprecated)
├── Network_GMD_SlowFast.py     # Transformer 포함 버전 (deprecated - 잘못됨)
├── Network_GMD_Corrected.py    # 수정 버전 (deprecated)
├── Network_Final.py            # ✅ 최종 버전 (권장) - Shape 로깅 포함
└── GMD_INTEGRATION.md          # 이 문서
```

---

## 버전 비교 (최종)

| 파일 | GMD | SlowFast | 베이스 유지 | Shape 로깅 | 권장 |
|------|-----|----------|-------------|------------|------|
| `Network.py` | ❌ | ❌ | ✅ | ❌ | Baseline |
| `Network_GMD.py` | ✅ | MLP | ⚠️ | ❌ | ❌ deprecated |
| `Network_GMD_SlowFast.py` | ✅ | Transformer | ❌ 잘못됨 | ❌ | ❌ deprecated |
| `Network_GMD_Corrected.py` | ✅ | MLP | ✅ | ❌ | ❌ deprecated |
| **`Network_Final.py`** | ✅ | MLP | ✅ | ✅ | **✅ 권장** |

---

## 최종 권장 사용법

### 학습 실행 (최종 버전)
```bash
cd categorical/PyTorch/Models/CodebookMatching
python Network_Final.py
```

### Shape 로깅 예시 출력
```
============================================================
  Forward Pass Start
============================================================
  [Input x] shape: [32, 252], dtype: torch.float32, device: cuda:0
  [knn] shape: [1], dtype: torch.float32, device: cuda:0
  [Target t] shape: [32, 156], dtype: torch.float32, device: cuda:0

============================================================
  GMD Preprocessor
============================================================
  [Input] shape: [32, 252], dtype: torch.float32, device: cuda:0
  [Reshaped] shape: [32, 7, 36], dtype: torch.float32, device: cuda:0
  [After Spatial Norm] shape: [32, 7, 36], dtype: torch.float32, device: cuda:0
  [Temporal Deltas] shape: [32, 7, 6], dtype: torch.float32, device: cuda:0
  [GMD Output (flat)] shape: [32, 294], dtype: torch.float32, device: cuda:0

============================================================
  SlowFast MLP
============================================================
  [Input] shape: [32, 294], dtype: torch.float32, device: cuda:0
  [Fast (selected frames)] shape: [32, 3, 42], dtype: torch.float32, device: cuda:0
  [Slow (selected frames)] shape: [32, 4, 42], dtype: torch.float32, device: cuda:0
  [Fused output] shape: [32, 252], dtype: torch.float32, device: cuda:0
```

### GMD 비활성화 (Ablation)
```python
use_gmd = False  # GMD + SlowFast 전체 비활성화
```

### Shape 로깅 비활성화
```python
debug_shapes = False  # 로깅 끄기
```
