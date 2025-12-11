"""
Model Evaluation Script for TrackerBodyPredictor
=================================================

평가 지표:
- MPJPE (Mean Per Joint Position Error): 관절 위치 오차 (mm)
- MPJVE (Mean Per Joint Velocity Error): 관절 속도 오차 (mm/s)
- MSE (Mean Squared Error): 전체 MSE
- Per-bone breakdown: 각 bone별 오차

사용법:
    python evaluate_model.py --model <onnx_path> --dataset <dataset_path>
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
    
    # Upper Body Bone 이름 (Unity Blueman 기준)
    BONE_NAMES = [
        "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg",
        "Root"  # 19번째
    ]
    
    NUM_BONES = 19
                               FEATURES_PER_BONE = 12  # Position(3) + Forward(3) + Up(3) + Velocity(3)
    ROOT_UPDATE_DIM = 3
    
    @staticmethod
    def parse_output(y):
        """
        Output tensor를 구조화된 형태로 파싱
        
        Args:
            y: (B, 231) output tensor
        Returns:
            dict with 'root_update', 'positions', 'forwards', 'ups', 'velocities'
        """
        B = y.shape[0]
        
        result = {
            'root_update': y[:, :3],  # (B, 3)
            'positions': [],
            'forwards': [],
            'ups': [],
            'velocities': []
        }
        
        offset = 3  # Skip root update
        for bone_idx in range(TrackerBodyStructure.NUM_BONES):
            start = offset + bone_idx * TrackerBodyStructure.FEATURES_PER_BONE
            
            result['positions'].append(y[:, start:start+3])
            result['forwards'].append(y[:, start+3:start+6])
            result['ups'].append(y[:, start+6:start+9])
            result['velocities'].append(y[:, start+9:start+12])
        
        # Stack: (B, NUM_BONES, 3)
        result['positions'] = torch.stack(result['positions'], dim=1)
        result['forwards'] = torch.stack(result['forwards'], dim=1)
        result['ups'] = torch.stack(result['ups'], dim=1)
        result['velocities'] = torch.stack(result['velocities'], dim=1)
        
        return result


# =============================================================================
# 평가 메트릭
# =============================================================================

def compute_mpjpe(pred_positions, gt_positions):
    """
    Mean Per Joint Position Error (mm 단위로 반환)
    
    Args:
        pred_positions: (B, J, 3) 예측 관절 위치
        gt_positions: (B, J, 3) GT 관절 위치
    Returns:
        mpjpe: scalar (mm)
        per_joint: (J,) 각 관절별 오차
    """
    # L2 distance per joint
    diff = pred_positions - gt_positions  # (B, J, 3)
    dist = torch.norm(diff, dim=-1)  # (B, J)
    
    per_joint = dist.mean(dim=0)  # (J,)
    mpjpe = dist.mean()  # scalar
    
    # Unity 단위가 meter라면 mm로 변환 (×1000)
    # 데이터가 이미 특정 스케일이라면 조정 필요
    return mpjpe, per_joint


def compute_mpjve(pred_velocities, gt_velocities):
    """
    Mean Per Joint Velocity Error (mm/s 단위로 반환)
    
    Args:
        pred_velocities: (B, J, 3) 예측 관절 속도
        gt_velocities: (B, J, 3) GT 관절 속도
    Returns:
        mpjve: scalar (mm/s)
        per_joint: (J,) 각 관절별 오차
    """
    diff = pred_velocities - gt_velocities
    dist = torch.norm(diff, dim=-1)
    
    per_joint = dist.mean(dim=0)
    mpjve = dist.mean()
    
    return mpjve, per_joint


def compute_rotation_error(pred_forwards, pred_ups, gt_forwards, gt_ups):
    """
    Rotation Error (도 단위)
    Forward와 Up 벡터를 이용한 각도 오차
    
    Returns:
        mean_error: 평균 회전 오차 (degrees)
        per_joint: 각 관절별 오차
    """
    # Forward 벡터 각도 오차
    cos_sim_forward = F.cosine_similarity(pred_forwards, gt_forwards, dim=-1)  # (B, J)
    cos_sim_forward = torch.clamp(cos_sim_forward, -1, 1)
    angle_forward = torch.acos(cos_sim_forward) * 180 / np.pi  # degrees
    
    # Up 벡터 각도 오차
    cos_sim_up = F.cosine_similarity(pred_ups, gt_ups, dim=-1)
    cos_sim_up = torch.clamp(cos_sim_up, -1, 1)
    angle_up = torch.acos(cos_sim_up) * 180 / np.pi
    
    # 평균
    rotation_error = (angle_forward + angle_up) / 2
    
    per_joint = rotation_error.mean(dim=0)
    mean_error = rotation_error.mean()
    
    return mean_error, per_joint


def compute_root_error(pred_root, gt_root):
    """
    Root Update Error
    
    Args:
        pred_root: (B, 3) - [deltaX, deltaAngle, deltaZ]
        gt_root: (B, 3)
    Returns:
        position_error: XZ 위치 오차
        angle_error: 각도 오차 (degrees)
    """
    # Position (X, Z)
    pos_diff = pred_root[:, [0, 2]] - gt_root[:, [0, 2]]
    position_error = torch.norm(pos_diff, dim=-1).mean()
    
    # Angle (이미 degree일 수 있음)
    angle_diff = torch.abs(pred_root[:, 1] - gt_root[:, 1])
    angle_error = angle_diff.mean()
    
    return position_error, angle_error


# =============================================================================
# 모델 로드 및 평가
# =============================================================================

def load_onnx_model(onnx_path, use_gpu=True):
    """
    ONNX Runtime으로 모델 로드 (GPU 지원)
    
    Args:
        onnx_path: ONNX 파일 경로
        use_gpu: GPU 사용 여부
    Returns:
        ONNXInferenceSession wrapper
    """
    import onnxruntime as ort
    
    print(f"Loading ONNX model from: {onnx_path}")
    print(f"  Available providers: {ort.get_available_providers()}")
    
    # Provider 설정 (GPU 우선)
    if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("  Using: CUDA (GPU)")
    else:
        providers = ['CPUExecutionProvider']
        print("  Using: CPU")
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # 입출력 정보 출력
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"  Input:  {input_info.name} {input_info.shape}")
    print(f"  Output: {output_info.name} {output_info.shape}")
    
    return ONNXModel(session, use_gpu)


class ONNXModel:
    """ONNX Runtime 세션 래퍼 (PyTorch 스타일 인터페이스)"""
    
    def __init__(self, session, use_gpu=False):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.use_gpu = use_gpu
    
    def __call__(self, x):
        """
        PyTorch tensor 입력 → PyTorch tensor 출력
        배치 크기가 1로 고정된 ONNX 모델도 처리 가능
        """
        # PyTorch → NumPy
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy().astype(np.float32)
        else:
            x_np = x.astype(np.float32)
        
        batch_size = x_np.shape[0]
        
        # ONNX 모델이 batch=1로 고정된 경우, 하나씩 처리
        try:
            # 먼저 전체 배치로 시도
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: x_np}
            )
            result = torch.from_numpy(outputs[0])
        except Exception:
            # 실패하면 하나씩 처리
            results = []
            for i in range(batch_size):
                single_input = x_np[i:i+1]  # (1, input_dim)
                output = self.session.run(
                    [self.output_name],
                    {self.input_name: single_input}
                )
                results.append(output[0])
            
            result = torch.from_numpy(np.concatenate(results, axis=0))
        
        return result
    
    def to(self, device):
        """호환성을 위한 더미 메서드 (ONNX Runtime이 자동 처리)"""
        return self
    
    def eval(self):
        """호환성을 위한 더미 메서드"""
        return self


def load_pytorch_model(model_class, weights_path, **kwargs):
    """PyTorch 체크포인트 로드"""
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def evaluate_dataset(model, dataset_path, batch_size=256, device='cuda', use_normalization=True):
    """
    전체 데이터셋에 대해 평가 수행
    
    Args:
        model: PyTorch 모델 (또는 ONNX 변환된 모델)
        dataset_path: 데이터셋 경로
        batch_size: 배치 크기
        device: 'cuda' or 'cpu'
        use_normalization: 정규화 적용 여부
    """
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
    
    print(f"\n{'='*60}")
    print(f"DATASET INFO")
    print(f"{'='*60}")
    print(f"Path: {dataset_path}")
    print(f"Samples: {sample_count:,}")
    print(f"Input dim: {input_dim}")
    print(f"Output dim: {output_dim}")
    print(f"{'='*60}\n")
    
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
    
    # ONNX는 CPU에서 실행되므로 모든 텐서를 CPU로 통일
    device = 'cpu'  # ONNX Runtime은 자체적으로 GPU 처리
    
    # Normalization tensors (CPU로 유지)
    Xnorm_tensor = torch.from_numpy(Xnorm)
    Ynorm_tensor = torch.from_numpy(Ynorm)
    
    print("Evaluating...")
    with torch.no_grad():
        for i in range(0, sample_count, batch_size):
            end_idx = min(i + batch_size, sample_count)
            indices = np.arange(i, end_idx)
            
            # 마지막 배치가 불완전할 수 있음 - 예외 처리
            try:
                # 배치 로드 (CPU로)
                xBatch = utility.ReadBatchFromFile(InputFile, indices, input_dim)
                yBatch = utility.ReadBatchFromFile(OutputFile, indices, output_dim)
            except (ValueError, Exception) as e:
                print(f"\n  Warning: Skipping batch at index {i} due to: {e}")
                continue
            
            # CPU로 이동 (이미 CPU일 수 있음)
            if hasattr(xBatch, 'is_cuda') and xBatch.is_cuda:
                xBatch = xBatch.cpu()
            if hasattr(yBatch, 'is_cuda') and yBatch.is_cuda:
                yBatch = yBatch.cpu()
            
            # 예측
            yPred = model(xBatch)
            
            # MSE (normalized space)
            mse = F.mse_loss(
                utility.Normalize(yPred, Ynorm_tensor),
                utility.Normalize(yBatch, Ynorm_tensor)
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
            if (i // batch_size) % 100 == 0:
                print(f"  Progress: {i}/{sample_count} ({100*i/sample_count:.1f}%)")
    
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
    
    print(f"\n[Per-Joint Rotation Error (°)]")
    for bone, error in results['Per_Joint_Rotation'].items():
        print(f"  {bone:20s}: {error:8.2f}")
    
    print(f"\n{'='*60}")


def save_results(results, save_path):
    """결과를 파일로 저장"""
    import json
    
    # Convert numpy types to Python types
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
    parser = argparse.ArgumentParser(description='Evaluate TrackerBody Model')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--dataset', type=str, default='../../Datasets/Trackerbodypredictor', help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--scale_mm', action='store_true', default=True, help='Scale to mm')
    
    args = parser.parse_args()
    
    # ONNX 모델 로드 (ONNX Runtime 사용)
    model = load_onnx_model(args.model, use_gpu=(args.device == 'cuda'))
    
    # 디바이스 설정
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 평가 수행
    results = evaluate_dataset(
        model=model,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        device=device
    )
    
    # 결과 출력
    print_results(results, scale_to_mm=args.scale_mm)
    
    # 결과 저장
    if args.output:
        save_results(results, args.output)
    else:
        # 기본 저장 경로
        model_name = os.path.basename(args.model).replace('.onnx', '')
        output_path = os.path.join(os.path.dirname(args.model), f'{model_name}_evaluation.json')
        save_results(results, output_path)
