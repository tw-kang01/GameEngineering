"""
학습된 모델 시각화 스크립트
- .pt 파일 또는 현재 데이터셋으로 예측 결과 확인
- Input/Output/Prediction PCA 시각화
- Loss curve 확인
"""

import sys
sys.path.append("../../../PyTorch")

import Library.Utility as utility
import Library.Plotting as plotting

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Network.py의 Model 클래스 import
from Network import Model


def load_dataset(name="Trackerbodypredictor"):
    """데이터셋 로드"""
    directory = "../../Datasets/" + name
    
    InputFile = directory + "/Input.bin"
    OutputFile = directory + "/Output.bin"
    Xshape = utility.LoadTxtAsInt(directory + "/InputShape.txt", True)
    Yshape = utility.LoadTxtAsInt(directory + "/OutputShape.txt", True)
    Xnorm = utility.LoadTxt(directory + "/InputNormalization.txt", True)
    Ynorm = utility.LoadTxt(directory + "/OutputNormalization.txt", True)
    
    # Sequences 로드 (있으면)
    try:
        Sequences = utility.LoadTxtRaw(directory + "/Sequences.txt", False)
        Sequences = np.array(utility.Transpose2DList(Sequences)[0], dtype=np.int64)
    except:
        Sequences = None
    
    return {
        'InputFile': InputFile,
        'OutputFile': OutputFile,
        'Xshape': Xshape,
        'Yshape': Yshape,
        'Xnorm': Xnorm,
        'Ynorm': Ynorm,
        'Sequences': Sequences
    }


def load_model_from_pt(pt_path, data):
    """저장된 .pt 파일에서 모델 로드"""
    model = torch.load(pt_path)
    model.eval()
    return model


def create_model(data):
    """새 모델 생성 (weight 없이)"""
    input_dim = data['Xshape'][1]
    output_dim = data['Yshape'][1]
    
    seed = 23456
    rng = np.random.RandomState(seed)
    
    layers = [input_dim, 512, 512, output_dim]
    activations = [F.elu, F.elu, None]
    
    model = Model(
        rng=rng,
        layers=layers,
        activations=activations,
        dropout=0.25,
        input_norm=data['Xnorm'],
        output_norm=data['Ynorm']
    )
    return model


def get_test_sequences(sequences, sequence_length=60, num_sequences=100):
    """테스트용 시퀀스 인덱스 추출"""
    if sequences is None:
        return None
    
    test_sequences = []
    for i in range(int(sequences[-1])):
        indices = np.where(sequences == (i + 1))[0]
        intervals = int(np.floor(len(indices) / sequence_length))
        if intervals > 0:
            slices = np.array_split(indices, intervals)
            test_sequences += slices
    
    return test_sequences[:num_sequences] if len(test_sequences) > num_sequences else test_sequences


def visualize_predictions(model, data, num_samples=500, save_path=None):
    """예측 결과 PCA 시각화"""
    model.eval()
    device = next(model.parameters()).device
    
    input_dim = data['Xshape'][1]
    output_dim = data['Yshape'][1]
    sample_count = data['Xshape'][0]
    
    # 랜덤 샘플 선택
    indices = np.random.choice(sample_count, min(num_samples, sample_count), replace=False)
    
    # 데이터 로드
    xBatch = utility.ReadBatchFromFile(data['InputFile'], indices, input_dim)
    yBatch = utility.ReadBatchFromFile(data['OutputFile'], indices, output_dim)
    
    # 예측
    with torch.no_grad():
        yPred = model(xBatch.to(device))
    
    # CPU로 이동
    xBatch = xBatch.cpu().numpy()
    yBatch = yBatch.cpu().numpy()
    yPred = yPred.cpu().numpy()
    
    # PCA 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input PCA
    pca_input = PCA(n_components=2).fit_transform(xBatch)
    axes[0].scatter(pca_input[:, 0], pca_input[:, 1], alpha=0.5, s=10)
    axes[0].set_title(f'Input (dim={input_dim})')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    
    # Ground Truth PCA
    pca_gt = PCA(n_components=2).fit_transform(yBatch)
    axes[1].scatter(pca_gt[:, 0], pca_gt[:, 1], alpha=0.5, s=10, c='green')
    axes[1].set_title(f'Ground Truth (dim={output_dim})')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    
    # Prediction PCA (같은 PCA 사용)
    pca = PCA(n_components=2).fit(yBatch)
    pred_pca = pca.transform(yPred)
    gt_pca = pca.transform(yBatch)
    
    axes[2].scatter(gt_pca[:, 0], gt_pca[:, 1], alpha=0.3, s=10, c='green', label='GT')
    axes[2].scatter(pred_pca[:, 0], pred_pca[:, 1], alpha=0.3, s=10, c='red', label='Pred')
    axes[2].set_title('GT vs Prediction')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # MSE 계산
    mse = np.mean((yBatch - yPred) ** 2)
    print(f"MSE: {mse:.6f}")
    
    return mse


def visualize_sequence(model, data, sequence_idx=0, save_path=None):
    """시퀀스 예측 시각화"""
    model.eval()
    device = next(model.parameters()).device
    
    test_sequences = get_test_sequences(data['Sequences'])
    if test_sequences is None or len(test_sequences) == 0:
        print("No sequences available")
        return
    
    input_dim = data['Xshape'][1]
    output_dim = data['Yshape'][1]
    
    # 시퀀스 선택
    seq_indices = test_sequences[sequence_idx % len(test_sequences)]
    
    # 데이터 로드
    xBatch = utility.ReadBatchFromFile(data['InputFile'], seq_indices, input_dim)
    yBatch = utility.ReadBatchFromFile(data['OutputFile'], seq_indices, output_dim)
    
    # 예측
    with torch.no_grad():
        yPred = model(xBatch.to(device))
    
    yBatch = yBatch.cpu().numpy()
    yPred = yPred.cpu().numpy()
    
    # 시각화: 첫 몇 개 output feature의 시간에 따른 변화
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    feature_names = ['RootUpdateX', 'RootUpdateY', 'RootUpdateZ', 
                     'BonePos:root:X', 'BonePos:root:Y', 'BonePos:root:Z']
    
    for i, ax in enumerate(axes.flat):
        if i < output_dim:
            ax.plot(yBatch[:, i], label='GT', color='green', linewidth=2)
            ax.plot(yPred[:, i], label='Pred', color='red', linestyle='--', linewidth=2)
            ax.set_title(feature_names[i] if i < len(feature_names) else f'Feature {i}')
            ax.legend()
            ax.set_xlabel('Frame')
            ax.set_ylabel('Value')
    
    plt.suptitle(f'Sequence {sequence_idx} Prediction')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    
    plt.show()


def compare_models(model1, model2, data, labels=['Model 1', 'Model 2'], num_samples=500):
    """두 모델 비교"""
    device1 = next(model1.parameters()).device
    device2 = next(model2.parameters()).device
    
    model1.eval()
    model2.eval()
    
    input_dim = data['Xshape'][1]
    output_dim = data['Yshape'][1]
    sample_count = data['Xshape'][0]
    
    indices = np.random.choice(sample_count, min(num_samples, sample_count), replace=False)
    
    xBatch = utility.ReadBatchFromFile(data['InputFile'], indices, input_dim)
    yBatch = utility.ReadBatchFromFile(data['OutputFile'], indices, output_dim)
    
    with torch.no_grad():
        yPred1 = model1(xBatch.to(device1)).cpu().numpy()
        yPred2 = model2(xBatch.to(device2)).cpu().numpy()
    
    yBatch = yBatch.cpu().numpy()
    
    # MSE 비교
    mse1 = np.mean((yBatch - yPred1) ** 2)
    mse2 = np.mean((yBatch - yPred2) ** 2)
    
    print(f"{labels[0]} MSE: {mse1:.6f}")
    print(f"{labels[1]} MSE: {mse2:.6f}")
    print(f"Improvement: {(mse1 - mse2) / mse1 * 100:.2f}%")
    
    # PCA 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pca = PCA(n_components=2).fit(yBatch)
    gt_pca = pca.transform(yBatch)
    pred1_pca = pca.transform(yPred1)
    pred2_pca = pca.transform(yPred2)
    
    axes[0].scatter(gt_pca[:, 0], gt_pca[:, 1], alpha=0.5, s=10, c='green')
    axes[0].set_title('Ground Truth')
    
    axes[1].scatter(gt_pca[:, 0], gt_pca[:, 1], alpha=0.3, s=10, c='green', label='GT')
    axes[1].scatter(pred1_pca[:, 0], pred1_pca[:, 1], alpha=0.3, s=10, c='blue', label=labels[0])
    axes[1].set_title(f'{labels[0]} (MSE={mse1:.4f})')
    axes[1].legend()
    
    axes[2].scatter(gt_pca[:, 0], gt_pca[:, 1], alpha=0.3, s=10, c='green', label='GT')
    axes[2].scatter(pred2_pca[:, 0], pred2_pca[:, 1], alpha=0.3, s=10, c='red', label=labels[1])
    axes[2].set_title(f'{labels[1]} (MSE={mse2:.4f})')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize trained model')
    parser.add_argument('--model', type=str, default=None, help='Path to .pt model file')
    parser.add_argument('--dataset', type=str, default='Trackerbodypredictor', help='Dataset name')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples to visualize')
    parser.add_argument('--sequence', type=int, default=0, help='Sequence index to visualize')
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    print(f"  Input: {data['Xshape']}")
    print(f"  Output: {data['Yshape']}")
    
    if args.model:
        print(f"Loading model: {args.model}")
        model = load_model_from_pt(args.model, data)
    else:
        print("Creating untrained model for testing...")
        model = create_model(data)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("\n=== Random Samples Visualization ===")
    visualize_predictions(model, data, num_samples=args.samples)
    
    if data['Sequences'] is not None:
        print("\n=== Sequence Visualization ===")
        visualize_sequence(model, data, sequence_idx=args.sequence)
