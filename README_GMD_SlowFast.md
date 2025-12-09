# TrackerBody MLP with GMD + SlowFast Enhancement

## 개요

이 문서는 **Categorical Motion Controlled (SIGGRAPH 2024)** 파이프라인의 TrackerBodyPredictor에 **EgoPoser (ECCV 2024)**의 Global Motion Decomposition (GMD)와 SlowFast Feature Fusion을 통합한 작업을 기술합니다.

---

## 1. 배경 및 동기

### 1.1 문제 정의

VR 환경에서 3-Point Tracking (Head + Left Controller + Right Controller)만으로 전신 모션을 예측하는 것은 challenging한 문제입니다:

- **Under-constrained problem**: 3개 포인트로 22+ 관절 예측
- **Position drift**: 절대 위치 의존성으로 인한 일반화 문제
- **Multi-scale motion**: 빠른 손동작과 느린 보행이 동시에 발생

### 1.2 기존 접근법

| 논문 | 접근법 | 한계 |
|------|--------|------|
| **Categorical (SIGGRAPH 2024)** | Simple MLP + Codebook Matching | Tracker 전처리 없음 |
| **EgoPoser (ECCV 2024)** | GMD + SlowFast + Transformer | Transformer 기반 (무거움) |

### 1.3 우리의 접근

**EgoPoser의 핵심 아이디어 (GMD + SlowFast)를 Categorical의 경량 MLP 아키텍처에 통합**

- GMD: 위치 불변성 확보 (Spatial + Temporal Normalization)
- SlowFast: 다중 시간 스케일 캡처 (Fast + Slow pathways)
- MLP 기반: Transformer 대비 4-10배 빠른 학습/추론

---

## 2. Model Pipeline

### 2.1 전체 Categorical 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Categorical Motion Pipeline                          │
└─────────────────────────────────────────────────────────────────────────────┘

[VR 3-Point Tracking: Head + LController + RController]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: TrackerBodyPredictor (★ 우리가 수정한 부분)                         │
│                                                                             │
│ Network: MultiLayerPerceptron/Network_TrackerBody.py                        │
│ Input:  576 features (16 timesteps × 3 trackers × 12 features)             │
│ Output: 231 features (RootUpdate + 19 upper body bones)                    │
│                                                                             │
│ 수정 전: Simple MLP [576 → 512 → 512 → 231]                                 │
│ 수정 후: GMD → SlowFast → Residual → MLP [576 → 512 → 512 → 231]           │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 1b: FutureBodyPredictor                                               │
│                                                                             │
│ Network: MultiLayerPerceptron/Network.py                                    │
│ Input:  현재 상체 포즈                                                       │
│ Output: 미래 루트 궤적 (trajectory)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: LowerBodyPredictor                                                 │
│                                                                             │
│ Network: CodebookMatching/Network.py (VQ-VAE with Gumbel-Softmax)          │
│ Input:  168 features (Upper body + Trajectory)                             │
│ Output: 1856 features (Lower body sequence)                                │
│                                                                             │
│ Architecture: Encoder (Teacher) + Estimator (Student) + Decoder            │
│ Codebook: C=128 channels, D=8 dimensions, 1024 total codes                 │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: UpperBodyPredictor                                                 │
│                                                                             │
│ Network: MultiLayerPerceptron/Network.py                                    │
│ Input:  Lower body + Tracker                                               │
│ Output: Final upper body pose (refined)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [Full Body Animation Output]
```

### 2.2 수정된 TrackerBodyPredictor 상세 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Network_TrackerBody.py: GMD + SlowFast + MLP                   │
└─────────────────────────────────────────────────────────────────────────────┘

[Input: 3PT Tracker History]
Shape: (Batch, 576)
Structure: [Head_t0..t15 (192), LWrist_t0..t15 (192), RWrist_t0..t15 (192)]
Per tracker/timestep: Position(3) + Forward(3) + Up(3) + Velocity(3) = 12
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ GMDPreprocessor (학습 파라미터 없음, 순수 전처리)                             │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Spatial Normalization (매 timestep에서)                                 │ │
│ │                                                                         │ │
│ │ 목적: 절대 위치 의존성 제거, 상대적 움직임에 집중                          │ │
│ │                                                                         │ │
│ │ 1. Head XZ → (0, 0) 으로 이동                                           │ │
│ │ 2. LWrist XZ → Head 기준 상대 좌표                                      │ │
│ │ 3. RWrist XZ → Head 기준 상대 좌표                                      │ │
│ │                                                                         │ │
│ │ Index Mapping:                                                          │ │
│ │   Head Position X at t:   t*12 + 0  (t=0..15)                          │ │
│ │   Head Position Z at t:   t*12 + 2                                     │ │
│ │   LWrist Position X at t: 192 + t*12 + 0                               │ │
│ │   LWrist Position Z at t: 192 + t*12 + 2                               │ │
│ │   RWrist Position X at t: 384 + t*12 + 0                               │ │
│ │   RWrist Position Z at t: 384 + t*12 + 2                               │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                              │
│                              ▼                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Temporal Normalization                                                  │ │
│ │                                                                         │ │
│ │ 목적: 시간적 변화량 명시적 제공                                           │ │
│ │                                                                         │ │
│ │ delta_XZ[t] = position[t] - position[0]  (첫 프레임 대비 변화량)         │ │
│ │                                                                         │ │
│ │ 추가되는 features: 16 timesteps × 3 trackers × 2 (X, Z) = 96           │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ Output: (Batch, 672) = 576 (원본) + 96 (delta)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SlowFastMLP (학습 파라미터 있음)                                             │
│                                                                             │
│ ┌───────────────────────────────┐  ┌───────────────────────────────┐       │
│ │     Fast Pathway (단기)       │  │     Slow Pathway (장기)        │       │
│ │                               │  │                               │       │
│ │ 선택: frames 8-15             │  │ 선택: frames 0,2,4,6,8,10,12,14│       │
│ │       (최근 8 frames)         │  │       (전체에서 8 frames)      │       │
│ │       (~0.27초)               │  │       (전체 0.5초 커버)        │       │
│ │                               │  │                               │       │
│ │ 용도: 세밀한 손/머리 움직임    │  │ 용도: 전체 이동 방향/트렌드    │       │
│ │       빠른 제스처             │  │       걷기, 방향 전환          │       │
│ │                               │  │                               │       │
│ │ Input:  336 features          │  │ Input:  336 features          │       │
│ │         (8 × (12×3 + 6))      │  │         (8 × (12×3 + 6))      │       │
│ │                               │  │                               │       │
│ │ MLP:    336 → 512 → 512 → 576 │  │ MLP:    336 → 512 → 512 → 576 │       │
│ │         (3-layer, ELU)        │  │         (3-layer, ELU)        │       │
│ └───────────────┬───────────────┘  └───────────────┬───────────────┘       │
│                 │                                  │                       │
│                 └──────────────┬───────────────────┘                       │
│                                ▼                                           │
│                    ┌───────────────────────┐                               │
│                    │ Element-wise Sum      │                               │
│                    │ Fast_out + Slow_out   │                               │
│                    └───────────────────────┘                               │
│                                                                             │
│ Output: (Batch, 576)                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Residual Connection (학습 가능한 α)                                         │
│                                                                             │
│ output = α × SlowFast_output + (1-α) × Original_input                      │
│                                                                             │
│ α: learnable parameter, initialized to 0.5                                 │
│    sigmoid(α)로 0~1 범위 유지                                               │
│                                                                             │
│ 효과:                                                                       │
│   - GMD+SlowFast가 도움되면 α ↑ (자동으로)                                  │
│   - 도움 안되면 α ↓ (원본 데이터 더 사용)                                    │
│   - Gradient flow 보장 (학습 안정성)                                        │
│                                                                             │
│ Output: (Batch, 576)                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Z-Score Normalization                                                       │
│                                                                             │
│ x_norm = (x - mean) / std                                                  │
│                                                                             │
│ 사용: InputNormalization.txt (Unity export 시 계산된 통계값)                │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Simple MLP (원본 Categorical과 동일)                                        │
│                                                                             │
│ Layer 1: Linear(576, 512) → ELU → Dropout(0.25)                            │
│ Layer 2: Linear(512, 512) → ELU → Dropout(0.25)                            │
│ Layer 3: Linear(512, 231) → (no activation)                                │
│                                                                             │
│ Initialization: Xavier uniform                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Renormalization                                                             │
│                                                                             │
│ y = y_norm × std + mean                                                    │
│                                                                             │
│ 사용: OutputNormalization.txt                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
[Output: Upper Body Pose]
Shape: (Batch, 231)
Structure: RootUpdate(3) + 19 bones × 12 features (Position, Forward, Up, Velocity)
```

---

## 3. Contribution

### 3.1 기술적 기여

| 기여 | 설명 | 효과 |
|------|------|------|
| **GMD Integration** | EgoPoser의 Global Motion Decomposition을 Categorical MLP에 적용 | 절대 위치 의존성 제거, VR 환경에서 일반화 향상 |
| **SlowFast MLP** | Transformer 대신 MLP 기반 SlowFast 구현 | 빠른 학습/추론, Categorical 아키텍처 일관성 유지 |
| **Adaptive Residual** | 학습 가능한 α로 GMD 효과 자동 조절 | 데이터에 따라 최적 균형 자동 탐색 |
| **TrackerBody 특화** | 16 timesteps × 3 trackers 구조에 맞춘 index mapping | 정확한 spatial/temporal normalization |

### 3.2 아키텍처 비교

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Parameter Comparison                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Original MLP (Network.py):                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Layer 1: 576 × 512 = 294,912                                        │  │
│  │ Layer 2: 512 × 512 = 262,144                                        │  │
│  │ Layer 3: 512 × 231 = 118,272                                        │  │
│  │ Biases:  512 + 512 + 231 = 1,255                                    │  │
│  │ ─────────────────────────────────────────                           │  │
│  │ Total:   ~676K parameters                                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  GMD+SlowFast MLP (Network_TrackerBody.py):                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ GMD Preprocessor:        0 (no learnable params)                    │  │
│  │ SlowFast Fast MLP:  336×512 + 512×512 + 512×576 = ~730K            │  │
│  │ SlowFast Slow MLP:  336×512 + 512×512 + 512×576 = ~730K            │  │
│  │ Residual α:              1                                          │  │
│  │ Main MLP:           576×512 + 512×512 + 512×231 = ~676K            │  │
│  │ ─────────────────────────────────────────                           │  │
│  │ Total:   ~2.14M parameters (~3.2x increase)                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 학습 시간 비교

| Model | Parameters | Time/Epoch | Total (150 epochs) |
|-------|------------|------------|-------------------|
| Original MLP | ~676K | ~20-30s | ~1-1.5 hours |
| GMD+SlowFast MLP | ~2.14M | ~30-45s | ~1.5-2 hours |
| CodebookMatching | ~3M+ | ~60-120s | ~3-5 hours |

---

## 4. 파일 구조

```
categorical/PyTorch/Models/MultiLayerPerceptron/
├── Network.py                  # 원본 Simple MLP (수정: .pt 저장 추가)
├── Network_TrackerBody.py      # ★ 새로 생성: GMD + SlowFast + MLP
├── visualize_model.py          # ★ 새로 생성: 학습 결과 시각화
└── README_GMD_SlowFast.md      # ★ 이 문서

categorical/PyTorch/Models/CodebookMatching/
├── Network.py                  # 원본 VQ-VAE (LowerBody용)
└── Network_Final.py            # GMD+SlowFast + VQ-VAE (참고용)

categorical/PyTorch/Datasets/
├── Trackerbodypredictor/       # TrackerBody 데이터셋
│   ├── Input.bin               # 576 features × 713,710 samples
│   ├── Output.bin              # 231 features × 713,710 samples
│   ├── InputNormalization.txt  # Z-score 통계 (mean, std)
│   ├── OutputNormalization.txt
│   ├── InputLabels.txt         # Feature 이름 (디버깅용)
│   ├── OutputLabels.txt
│   └── Sequences.txt           # 시퀀스 구분 정보
├── Lowerbodypredictor/         # LowerBody 데이터셋 (CodebookMatching용)
└── Futurebodypredictor/        # FutureBody 데이터셋
```

---

## 5. 사용법

### 5.1 학습

```bash
cd categorical/PyTorch/Models/MultiLayerPerceptron

# GMD + SlowFast 모델 학습
python Network_TrackerBody.py

# 원본 MLP 학습 (비교용)
python Network.py
```

### 5.2 시각화

```bash
# 학습된 모델 시각화
python visualize_model.py --model "../../Datasets/Trackerbodypredictor/Training_xxx/xxx_150.pt"

# 특정 시퀀스 시각화
python visualize_model.py --model "path/to/model.pt" --sequence 5

# 데이터셋만 확인 (untrained)
python visualize_model.py --dataset Trackerbodypredictor
```

### 5.3 Unity 배포

학습 완료 후 생성된 `.onnx` 파일을 Unity 프로젝트에 복사:

```
Training_xxx/xxx_150.onnx → Unity/Assets/Models/TrackerBodyPredictor.onnx
```

---

## 6. 기대 효과

### 6.1 정량적 개선 (예상)

- **MSE Loss**: 5-15% 감소 (GMD로 위치 불변성 확보)
- **Position Drift**: 감소 (Spatial Normalization)
- **Fast Motion Quality**: 향상 (SlowFast Fast pathway)
- **Walking Direction**: 향상 (SlowFast Slow pathway)

### 6.2 정성적 개선 (예상)

- VR 사용자가 방 안 어디에 서있든 일관된 예측
- 빠른 손 제스처와 느린 보행 동시 처리
- 갑작스러운 방향 전환에 더 부드러운 반응

---

## 7. 향후 작업

1. **정량적 평가**: Original MLP vs GMD+SlowFast MLP 비교 실험
2. **Ablation Study**: GMD만, SlowFast만, 둘 다 적용 비교
3. **Unity Integration**: 실제 VR 환경에서 실시간 테스트
4. **LowerBody 개선**: CodebookMatching에도 유사한 전처리 적용 검토

---

## 8. 참고 논문

1. **Categorical Motion Controlled Character (SIGGRAPH 2024)**
   - Codebook Matching with Gumbel-Softmax
   - Multi-stage motion prediction pipeline

2. **EgoPoser: Robust Pose Estimation from Sparse Ego-Views (ECCV 2024)**
   - Global Motion Decomposition (GMD)
   - SlowFast Feature Fusion
   - Transformer-based architecture

---

## 9. 저자

Integration by: GMD+SlowFast into Categorical MLP
Date: December 2024
