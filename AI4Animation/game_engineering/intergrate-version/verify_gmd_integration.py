"""
GMD Integration Verification Script
====================================

Categorical VQ-VAE + GMD + SlowFast 통합 검증
- Synthetic data로 forward/backward pass 테스트
- 원본 Network.py와 비교

사용법:
    cd categorical/PyTorch/Models/CodebookMatching
    python verify_gmd_integration.py
"""

import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter

# Import libraries
import Library.Utility as utility
import Library.Modules as modules

print("="*70)
print("  GMD Integration Verification")
print("="*70)

# =============================================================================
# 1. 원본 Model (Network.py에서 간소화)
# =============================================================================

class OriginalModel(nn.Module):
    """원본 Categorical VQ-VAE (Network.py 기반)"""
    def __init__(self, encoder, estimator, decoder, xNorm, yNorm, C, D):
        super(OriginalModel, self).__init__()
        self.Encoder = encoder
        self.Estimator = estimator
        self.Decoder = decoder
        self.XNorm = xNorm
        self.YNorm = yNorm
        self.C = C
        self.D = D

    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.reshape(-1,1,1,1)
        noise = torch.rand_like(tensor) - 0.5
        samples = scale * noise + 0.5
        return -torch.log(-torch.log(samples + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, scale):
        y = logits + self.sample_gumbel(logits, scale)
        return torch.nn.functional.softmax(y / temperature, dim=-1)
    
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
        if t is not None:
            x_norm = utility.Normalize(x, self.XNorm)
            t_norm = utility.Normalize(t, self.YNorm)
            
            target_logits = self.Encoder(torch.cat((t_norm, x_norm), dim=1))
            target_probs, target = self.sample(target_logits, knn)

            estimate_logits = self.Estimator(x_norm)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            y = self.Decoder(target)
            y_output = utility.Renormalize(y, self.YNorm)

            return (y_output, target_logits, target_probs, target, 
                    estimate_logits, estimate_probs, estimate)
        else:
            x_norm = utility.Normalize(x, self.XNorm)
            estimate_logits = self.Estimator(x_norm)
            estimate_probs, estimate = self.sample(estimate_logits, knn)
            y = self.Decoder(estimate)
            y_output = utility.Renormalize(y, self.YNorm)
            return y_output, estimate


# =============================================================================
# 2. GMD Model Import
# =============================================================================

from Network_Final import Model as GMDModel, GMDPreprocessor, SlowFastMLP

# =============================================================================
# 3. Synthetic Data Generation
# =============================================================================

def generate_synthetic_data(batch_size, num_timesteps=7, features_per_timestep=36, 
                           output_dim=156):
    """
    TrackerBodyPredictor와 유사한 synthetic data 생성
    
    Input: 7 timesteps × 36 features = 252
    Output: 156 features (upper body + root trajectory)
    """
    input_dim = num_timesteps * features_per_timestep  # 252
    
    # Generate input: simulate tracker data
    X = torch.randn(batch_size, input_dim)
    
    # Generate output: simulate body prediction
    Y = torch.randn(batch_size, output_dim)
    
    # Compute normalization stats
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0).clamp(min=1e-5)
    Y_mean = Y.mean(dim=0)
    Y_std = Y.std(dim=0).clamp(min=1e-5)
    
    XNorm = torch.cat([X_mean, X_std], dim=0)
    YNorm = torch.cat([Y_mean, Y_std], dim=0)
    
    return X, Y, XNorm, YNorm


# =============================================================================
# 4. Model Creation
# =============================================================================

def create_models(input_dim, output_dim, XNorm, YNorm, device):
    """원본과 GMD 모델 생성"""
    
    # Config
    C = 128  # codebook channels
    D = 8    # codebook dim
    codebook_size = C * D
    hidden_dim = 512
    dropout = 0.25
    
    # --- Original Model ---
    original_model = OriginalModel(
        encoder=modules.LinearEncoder(
            input_dim + output_dim, hidden_dim, hidden_dim, codebook_size, 0.0
        ),
        estimator=modules.LinearEncoder(
            input_dim, hidden_dim, hidden_dim, codebook_size, dropout
        ),
        decoder=modules.LinearEncoder(
            codebook_size, hidden_dim, hidden_dim, output_dim, 0.0
        ),
        xNorm=Parameter(XNorm, requires_grad=False),
        yNorm=Parameter(YNorm, requires_grad=False),
        C=C, D=D
    ).to(device)
    
    # --- GMD Model ---
    gmd_model = GMDModel(
        encoder=modules.LinearEncoder(
            input_dim + output_dim, hidden_dim, hidden_dim, codebook_size, 0.0
        ),
        estimator=modules.LinearEncoder(
            input_dim, hidden_dim, hidden_dim, codebook_size, dropout
        ),
        decoder=modules.LinearEncoder(
            codebook_size, hidden_dim, hidden_dim, output_dim, 0.0
        ),
        xNorm=Parameter(XNorm.clone(), requires_grad=False),
        yNorm=Parameter(YNorm.clone(), requires_grad=False),
        codebook_channels=C,
        codebook_dim=D,
        use_gmd=True,
        num_timesteps=7,
        features_per_timestep=36,
        slowfast_hidden_dim=hidden_dim,
        dropout=dropout,
        debug_shapes=True  # Shape 로깅 활성화
    ).to(device)
    
    return original_model, gmd_model


# =============================================================================
# 5. Test Functions
# =============================================================================

def test_forward_pass(model, name, X, Y, device):
    """Forward pass 테스트"""
    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"{'='*60}")
    
    model.train()
    
    X_batch = X[:32].to(device)
    Y_batch = Y[:32].to(device)
    knn = torch.ones(1, device=device)
    
    try:
        output = model(X_batch, knn, Y_batch)
        
        if isinstance(output, tuple):
            y_pred = output[0]
            print(f"✅ Forward pass successful!")
            print(f"   Input:  {list(X_batch.shape)}")
            print(f"   Target: {list(Y_batch.shape)}")
            print(f"   Output: {list(y_pred.shape)}")
            return True, output
        else:
            print(f"❌ Unexpected output type: {type(output)}")
            return False, None
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_backward_pass(model, name, X, Y, device):
    """Backward pass (gradient) 테스트"""
    print(f"\n{'='*60}")
    print(f"  Testing Backward: {name}")
    print(f"{'='*60}")
    
    model.train()
    
    X_batch = X[:32].to(device)
    Y_batch = Y[:32].to(device)
    knn = torch.ones(1, device=device)
    
    try:
        output = model(X_batch, knn, Y_batch)
        y_pred = output[0]
        target_probs = output[2]
        estimate_probs = output[5]
        
        # MSE Loss
        loss_fn = nn.MSELoss()
        mse_loss = loss_fn(y_pred, Y_batch)
        
        # Matching Loss
        matching_loss = loss_fn(estimate_probs, target_probs)
        
        # Total Loss
        total_loss = mse_loss + matching_loss
        
        # Backward
        total_loss.backward()
        
        # Check gradients
        grad_count = 0
        for name_p, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
        
        print(f"✅ Backward pass successful!")
        print(f"   MSE Loss:      {mse_loss.item():.4f}")
        print(f"   Matching Loss: {matching_loss.item():.4f}")
        print(f"   Total Loss:    {total_loss.item():.4f}")
        print(f"   Parameters with gradients: {grad_count}")
        return True
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference(model, name, X, device):
    """Inference 테스트"""
    print(f"\n{'='*60}")
    print(f"  Testing Inference: {name}")
    print(f"{'='*60}")
    
    model.eval()
    
    X_batch = X[:8].to(device)
    knn = torch.ones(1, device=device)
    
    try:
        with torch.no_grad():
            output = model(X_batch, knn)
        
        if isinstance(output, tuple):
            y_pred = output[0]
            print(f"✅ Inference successful!")
            print(f"   Input:  {list(X_batch.shape)}")
            print(f"   Output: {list(y_pred.shape)}")
            return True
        else:
            print(f"❌ Unexpected output type: {type(output)}")
            return False
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, name, X, Y, device, num_steps=10):
    """여러 step 학습 테스트"""
    print(f"\n{'='*60}")
    print(f"  Testing Training Steps: {name}")
    print(f"{'='*60}")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    batch_size = 32
    knn = torch.ones(1, device=device)
    
    losses = []
    
    try:
        for step in range(num_steps):
            # Random batch
            idx = np.random.choice(len(X), batch_size, replace=False)
            X_batch = X[idx].to(device)
            Y_batch = Y[idx].to(device)
            
            # Forward
            output = model(X_batch, knn, Y_batch)
            y_pred = output[0]
            target_probs = output[2]
            estimate_probs = output[5]
            
            # Loss
            mse_loss = loss_fn(y_pred, Y_batch)
            matching_loss = loss_fn(estimate_probs, target_probs)
            total_loss = mse_loss + matching_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            
            if step == 0 or step == num_steps - 1:
                print(f"   Step {step+1}/{num_steps}: Loss = {total_loss.item():.4f}")
        
        print(f"✅ Training steps completed!")
        print(f"   Initial Loss: {losses[0]:.4f}")
        print(f"   Final Loss:   {losses[-1]:.4f}")
        print(f"   Improvement:  {(losses[0] - losses[-1]):.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Training steps failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_models(original_model, gmd_model, X, Y, device):
    """두 모델 파라미터 수 비교"""
    print(f"\n{'='*60}")
    print(f"  Model Comparison")
    print(f"{'='*60}")
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    def count_trainable(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    orig_params = count_params(original_model)
    orig_trainable = count_trainable(original_model)
    
    gmd_params = count_params(gmd_model)
    gmd_trainable = count_trainable(gmd_model)
    
    print(f"\n   Original Model:")
    print(f"     Total Parameters:     {orig_params:,}")
    print(f"     Trainable Parameters: {orig_trainable:,}")
    
    print(f"\n   GMD Model:")
    print(f"     Total Parameters:     {gmd_params:,}")
    print(f"     Trainable Parameters: {gmd_trainable:,}")
    print(f"     Added Parameters:     {gmd_params - orig_params:,}")
    
    print(f"\n   Parameter Increase: {(gmd_params / orig_params - 1) * 100:.1f}%")


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Generate data
    print("\n" + "="*70)
    print("  Generating Synthetic Data")
    print("="*70)
    
    batch_size = 1000
    X, Y, XNorm, YNorm = generate_synthetic_data(batch_size)
    
    print(f"   Input X:  {list(X.shape)}")
    print(f"   Output Y: {list(Y.shape)}")
    print(f"   XNorm:    {list(XNorm.shape)}")
    print(f"   YNorm:    {list(YNorm.shape)}")
    
    # Create models
    print("\n" + "="*70)
    print("  Creating Models")
    print("="*70)
    
    original_model, gmd_model = create_models(
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        XNorm=XNorm,
        YNorm=YNorm,
        device=device
    )
    
    # Compare models
    compare_models(original_model, gmd_model, X, Y, device)
    
    # Run tests
    results = {}
    
    # Test Original Model
    print("\n" + "="*70)
    print("  TESTING ORIGINAL MODEL")
    print("="*70)
    
    results['orig_forward'], _ = test_forward_pass(original_model, "Original", X, Y, device)
    results['orig_backward'] = test_backward_pass(original_model, "Original", X, Y, device)
    results['orig_inference'] = test_inference(original_model, "Original", X, device)
    results['orig_training'] = test_training_step(original_model, "Original", X, Y, device)
    
    # Test GMD Model
    print("\n" + "="*70)
    print("  TESTING GMD MODEL")
    print("="*70)
    
    # Shape 로깅을 첫 번째 테스트에서만 보기 위해 활성화
    gmd_model.logger.enabled = True
    gmd_model.logger.logged_once = False
    
    results['gmd_forward'], _ = test_forward_pass(gmd_model, "GMD", X, Y, device)
    
    # 나머지 테스트에서는 로깅 비활성화
    gmd_model.logger.enabled = False
    
    results['gmd_backward'] = test_backward_pass(gmd_model, "GMD", X, Y, device)
    results['gmd_inference'] = test_inference(gmd_model, "GMD", X, device)
    results['gmd_training'] = test_training_step(gmd_model, "GMD", X, Y, device)
    
    # Summary
    print("\n" + "="*70)
    print("  VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("  ✅ ALL TESTS PASSED! GMD Integration is working correctly.")
    else:
        print("  ❌ SOME TESTS FAILED! Please check the errors above.")
    print("="*70)
