"""
ONNX → PyTorch 변환 후 MPJPE/MPJVE 평가
========================================

사용법:
    pip install onnx onnx2torch
    python evaluate_pt.py --model <onnx_path> --dataset <dataset_path>
"""

import sys
sys.path.append("../../../PyTorch")

import Library.Utility as utility

import numpy as np
import torch
import torch.nn.functional as F
import argparse
import os
from collections import defaultdict


# =============================================================================
# TrackerBody 데이터 구조 정의
# =============================================================================

class TrackerBodyStructure:
    """
    TrackerBody Output 구조 (231 features):
    - RootUpdate: 3 (deltaX, deltaAngle, deltaZ)
    - 19 Upper Body Bones × 12 features each
      - Position (3) + Forward (3) + Up (3) + Velocity (3)
    """
    
    # Unity Blueman.UpperBodyNames 순서와 일치
    BONE_NAMES = [
        "b_root",           # 0
        "b_spine0",         # 1
        "b_spine1",         # 2
        "b_spine2",         # 3
        "b_spine3",         # 4
        "b_neck0",          # 5
        "b_head",           # 6
        "b_l_shoulder",     # 7
        "p_l_scap",         # 8
        "b_l_arm",          # 9
        "b_l_forearm",      # 10
        "b_l_wrist_twist",  # 11
        "b_l_wrist",        # 12
        "b_r_shoulder",     # 13
        "p_r_scap",         # 14
        "b_r_arm",          # 15
        "b_r_forearm",      # 16
        "b_r_wrist_twist",  # 17
        "b_r_wrist"         # 18
    ]
    
    NUM_BONES = 19
    FEATURES_PER_BONE = 12
    ROOT_UPDATE_DIM = 3
    
    @staticmethod
    def parse_output(y):
        """Output tensor를 구조화된 형태로 파싱"""
        B = y.shape[0]
        
        result = {
            'root_update': y[:, :3],
            'positions': [],
            'forwards': [],
            'ups': [],
            'velocities': []
        }
        
        offset = 3
        for bone_idx in range(TrackerBodyStructure.NUM_BONES):
            start = offset + bone_idx * TrackerBodyStructure.FEATURES_PER_BONE
            
            result['positions'].append(y[:, start:start+3])
            result['forwards'].append(y[:, start+3:start+6])
            result['ups'].append(y[:, start+6:start+9])
            result['velocities'].append(y[:, start+9:start+12])
        
        result['positions'] = torch.stack(result['positions'], dim=1)
        result['forwards'] = torch.stack(result['forwards'], dim=1)
        result['ups'] = torch.stack(result['ups'], dim=1)
        result['velocities'] = torch.stack(result['velocities'], dim=1)
        
        return result


# =============================================================================
# 평가 메트릭
# =============================================================================

def compute_mpjpe(pred_positions, gt_positions):
    """Mean Per Joint Position Error"""
    diff = pred_positions - gt_positions
    dist = torch.norm(diff, dim=-1)
    per_joint = dist.mean(dim=0)
    mpjpe = dist.mean()
    return mpjpe, per_joint


def compute_mpjve(pred_velocities, gt_velocities):
    """Mean Per Joint Velocity Error"""
    diff = pred_velocities - gt_velocities
    dist = torch.norm(diff, dim=-1)
    per_joint = dist.mean(dim=0)
    mpjve = dist.mean()
    return mpjve, per_joint


def compute_rotation_error(pred_forwards, pred_ups, gt_forwards, gt_ups):
    """Rotation Error (degrees)"""
    cos_sim_forward = F.cosine_similarity(pred_forwards, gt_forwards, dim=-1)
    cos_sim_forward = torch.clamp(cos_sim_forward, -1, 1)
    angle_forward = torch.acos(cos_sim_forward) * 180 / np.pi
    
    cos_sim_up = F.cosine_similarity(pred_ups, gt_ups, dim=-1)
    cos_sim_up = torch.clamp(cos_sim_up, -1, 1)
    angle_up = torch.acos(cos_sim_up) * 180 / np.pi
    
    rotation_error = (angle_forward + angle_up) / 2
    per_joint = rotation_error.mean(dim=0)
    mean_error = rotation_error.mean()
    
    return mean_error, per_joint


def compute_root_error(pred_root, gt_root):
    """Root Update Error"""
    pos_diff = pred_root[:, [0, 2]] - gt_root[:, [0, 2]]
    position_error = torch.norm(pos_diff, dim=-1).mean()
    angle_diff = torch.abs(pred_root[:, 1] - gt_root[:, 1])
    angle_error = angle_diff.mean()
    return position_error, angle_error


# =============================================================================
# ONNX → PyTorch 변환 또는 ONNX Runtime 사용
# =============================================================================

def convert_onnx_to_pytorch(onnx_path):
    """ONNX 모델을 PyTorch로 변환 (실패 시 ONNX Runtime fallback)"""
    import onnx
    
    print(f"Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    
    # onnx2torch로 변환 시도
    try:
        from onnx2torch import convert
        pytorch_model = convert(onnx_model)
        pytorch_model.eval()
        print("✅ Converted to PyTorch using onnx2torch")
        return pytorch_model, "pytorch"
    except Exception as e:
        print(f"⚠️ onnx2torch failed: {e}")
        print("📌 Falling back to ONNX Runtime...")
        
        # ONNX Runtime으로 fallback
        return create_onnx_runtime_model(onnx_path), "onnxruntime"


def create_onnx_runtime_model(onnx_path):
    """ONNX Runtime 기반 모델 래퍼 생성"""
    import onnxruntime as ort
    
    # GPU 사용 시도
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        if 'CUDAExecutionProvider' in session.get_providers():
            print("  ✅ ONNX Runtime using CUDA")
        else:
            print("  ⚠️ ONNX Runtime using CPU")
    except Exception:
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        print("  ⚠️ ONNX Runtime using CPU (CUDA failed)")
    
    return ONNXRuntimeModel(session)


class ONNXRuntimeModel:
    """ONNX Runtime 래퍼 (PyTorch 인터페이스 호환)"""
    
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.device = 'cpu'  # ONNX Runtime 출력은 항상 CPU
    
    def __call__(self, x):
        """배치 추론 (batch=1 ONNX도 지원)"""
        import numpy as np
        
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy().astype(np.float32)
        else:
            x_np = x.astype(np.float32)
        
        batch_size = x_np.shape[0]
        
        # 전체 배치로 시도
        try:
            outputs = self.session.run([self.output_name], {self.input_name: x_np})
            return torch.from_numpy(outputs[0])
        except Exception:
            # batch=1인 ONNX면 하나씩 처리
            results = []
            for i in range(batch_size):
                output = self.session.run(
                    [self.output_name],
                    {self.input_name: x_np[i:i+1]}
                )
                results.append(output[0])
            return torch.from_numpy(np.concatenate(results, axis=0))
    
    def to(self, device):
        return self
    
    def eval(self):
        return self


# =============================================================================
# 평가 함수
# =============================================================================

def evaluate_dataset(model, dataset_path, batch_size=256, device='cuda', model_type='pytorch'):
    """전체 데이터셋에 대해 평가 수행"""
    
    # 데이터셋 로드
    InputFile = os.path.join(dataset_path, "Input.bin")
    OutputFile = os.path.join(dataset_path, "Output.bin")
    Xshape = utility.LoadTxtAsInt(os.path.join(dataset_path, "InputShape.txt"), True)
    Yshape = utility.LoadTxtAsInt(os.path.join(dataset_path, "OutputShape.txt"), True)
    Xnorm = utility.LoadTxt(os.path.join(dataset_path, "InputNormalization.txt"), True)
    Ynorm = utility.LoadTxt(os.path.join(dataset_path, "OutputNormalization.txt"), True)
    
    sample_count = Xshape[0]
    input_dim = Xshape[1]
    output_dim = Yshape[1]
    
    # ONNX Runtime은 CPU에서 동작
    if model_type == 'onnxruntime':
        device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"DATASET INFO")
    print(f"{'='*60}")
    print(f"Path: {dataset_path}")
    print(f"Samples: {sample_count:,}")
    print(f"Input dim: {input_dim}")
    print(f"Output dim: {output_dim}")
    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    print(f"{'='*60}\n")
    
    # PyTorch 모델이면 디바이스로 이동
    if model_type == 'pytorch':
        model = model.to(device)
        model.eval()
    
    # Normalization tensors (CPU로 유지 - ONNX Runtime 호환)
    Ynorm_tensor = torch.from_numpy(Ynorm)
    
    # 메트릭 수집
    all_mpjpe = []
    all_mpjve = []
    all_rotation_error = []
    all_root_pos_error = []
    all_root_angle_error = []
    all_mse = []
    
    per_joint_position = defaultdict(list)
    per_joint_velocity = defaultdict(list)
    per_joint_rotation = defaultdict(list)
    
    print("Evaluating...")
    with torch.no_grad():
        for i in range(0, sample_count, batch_size):
            end_idx = min(i + batch_size, sample_count)
            indices = np.arange(i, end_idx)
            
            try:
                xBatch = utility.ReadBatchFromFile(InputFile, indices, input_dim)
                yBatch = utility.ReadBatchFromFile(OutputFile, indices, output_dim)
            except (ValueError, Exception) as e:
                print(f"\n  Warning: Skipping batch at index {i} due to: {e}")
                continue
            
            # PyTorch 모델이면 GPU로
            if model_type == 'pytorch' and device == 'cuda':
                xBatch = xBatch.to(device)
                yBatch = yBatch.to(device)
                Ynorm_tensor_dev = Ynorm_tensor.to(device)
            else:
                Ynorm_tensor_dev = Ynorm_tensor
            
            # 예측
            yPred = model(xBatch)
            
            # ONNX Runtime 출력은 CPU tensor
            if model_type == 'onnxruntime':
                yBatch = yBatch.cpu()
            
            # MSE (normalized space)
            mse = F.mse_loss(
                utility.Normalize(yPred, Ynorm_tensor_dev),
                utility.Normalize(yBatch, Ynorm_tensor_dev)
            )
            all_mse.append(mse.item())
            
            # 출력 파싱
            pred_parsed = TrackerBodyStructure.parse_output(yPred)
            gt_parsed = TrackerBodyStructure.parse_output(yBatch)
            
            # MPJPE
            mpjpe, per_joint_pos = compute_mpjpe(
                pred_parsed['positions'], 
                gt_parsed['positions']
            )
            all_mpjpe.append(mpjpe.item())
            
            # MPJVE
            mpjve, per_joint_vel = compute_mpjve(
                pred_parsed['velocities'],
                gt_parsed['velocities']
            )
            all_mpjve.append(mpjve.item())
            
            # Rotation Error
            rot_err, per_joint_rot = compute_rotation_error(
                pred_parsed['forwards'], pred_parsed['ups'],
                gt_parsed['forwards'], gt_parsed['ups']
            )
            all_rotation_error.append(rot_err.item())
            
            # Root Error
            root_pos_err, root_angle_err = compute_root_error(
                pred_parsed['root_update'],
                gt_parsed['root_update']
            )
            all_root_pos_error.append(root_pos_err.item())
            all_root_angle_error.append(root_angle_err.item())
            
            # Per-joint 저장
            for j, bone_name in enumerate(TrackerBodyStructure.BONE_NAMES):
                per_joint_position[bone_name].append(per_joint_pos[j].item())
                per_joint_velocity[bone_name].append(per_joint_vel[j].item())
                per_joint_rotation[bone_name].append(per_joint_rot[j].item())
            
            # Progress
            if (i // batch_size) % 50 == 0:
                progress = 100 * i / sample_count
                print(f"  Progress: {i:,}/{sample_count:,} ({progress:.1f}%)")
    
    # 결과 집계
    results = {
        'MSE': np.mean(all_mse),
        'MPJPE': np.mean(all_mpjpe),
        'MPJVE': np.mean(all_mpjve),
        'Rotation_Error': np.mean(all_rotation_error),
        'Root_Position_Error': np.mean(all_root_pos_error),
        'Root_Angle_Error': np.mean(all_root_angle_error),
        'Per_Joint_Position': {k: np.mean(v) for k, v in per_joint_position.items()},
        'Per_Joint_Velocity': {k: np.mean(v) for k, v in per_joint_velocity.items()},
        'Per_Joint_Rotation': {k: np.mean(v) for k, v in per_joint_rotation.items()},
    }
    
    return results


def print_results(results, scale_to_mm=True):
    """결과 출력"""
    scale = 1000 if scale_to_mm else 1
    unit = "mm" if scale_to_mm else "units"
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\n[Overall Metrics]")
    print(f"  MSE (normalized):     {results['MSE']:.6f}")
    print(f"  MPJPE:                {results['MPJPE']*scale:.2f} {unit}")
    print(f"  MPJVE:                {results['MPJVE']*scale:.2f} {unit}/s")
    print(f"  Rotation Error:       {results['Rotation_Error']:.2f}°")
    print(f"  Root Position Error:  {results['Root_Position_Error']*scale:.2f} {unit}")
    print(f"  Root Angle Error:     {results['Root_Angle_Error']:.2f}°")
    
    print(f"\n[Per-Joint Position Error ({unit})]")
    for bone, error in results['Per_Joint_Position'].items():
        print(f"  {bone:20s}: {error*scale:8.2f}")
    
    print(f"\n[Per-Joint Velocity Error ({unit}/s)]")
    for bone, error in results['Per_Joint_Velocity'].items():
        print(f"  {bone:20s}: {error*scale:8.2f}")
    
    print(f"\n{'='*60}")


def save_results(results, save_path):
    """결과를 파일로 저장"""
    import json
    
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj
    
    results_converted = convert(results)
    
    with open(save_path, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TrackerBody Model (ONNX→PyTorch)')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--dataset', type=str, default='../../Datasets/Trackerbodypredictor', help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--save_pt', type=str, default=None, help='Save converted PyTorch model')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # ONNX → PyTorch 변환 (또는 ONNX Runtime fallback)
    model, model_type = convert_onnx_to_pytorch(args.model)
    
    # 변환된 PyTorch 모델 저장 (옵션, PyTorch 변환 성공 시만)
    if args.save_pt and model_type == 'pytorch':
        torch.save(model, args.save_pt)
        print(f"✅ Saved PyTorch model to: {args.save_pt}")
    elif args.save_pt and model_type == 'onnxruntime':
        print(f"⚠️ Cannot save .pt file - model is using ONNX Runtime")
    
    # 평가 수행
    results = evaluate_dataset(
        model=model,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        device=device,
        model_type=model_type
    )
    
    # 결과 출력
    print_results(results, scale_to_mm=True)
    
    # 결과 저장
    if args.output:
        save_results(results, args.output)
    else:
        model_name = os.path.basename(args.model).replace('.onnx', '')
        output_path = os.path.join(os.path.dirname(args.model), f'{model_name}_evaluation.json')
        save_results(results, output_path)
