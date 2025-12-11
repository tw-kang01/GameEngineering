"""
Categorical VQ-VAE with GMD + SlowFast Enhancement (Full Version)
=================================================================

This implementation integrates:
1. Global Motion Decomposition (GMD) from EgoPoser (ECCV 2024)
   - Spatial Normalization: Head-relative XZ coordinates
   - Temporal Normalization: First-frame relative delta features

2. SlowFast Feature Fusion from EgoPoser (ECCV 2024)
   - Fast Pathway: Recent fine-grained motion (last half of window)
   - Slow Pathway: Long-term context (every 2nd frame)
   - Transformer-based temporal modeling

3. Categorical Codebook Matching from SIGGRAPH 2024
   - Gumbel-Softmax VQ-VAE
   - Teacher (Encoder) - Student (Estimator) architecture
   - Codebook Matching Loss

Author: Integration of EgoPoser + Categorical Codebook Matching
Date: 2024
"""

import sys
sys.path.append("../../../PyTorch")

import Library.Utility as utility
import Library.Plotting as plotting
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import Library.Modules as modules

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import matplotlib.pyplot as plt


# =============================================================================
# GMD Module: Global Motion Decomposition
# =============================================================================

class GMDModule(nn.Module):
    """
    Global Motion Decomposition (GMD) from EgoPoser (ECCV 2024)
    
    Paper Reference: Section 3.2, Equation 2
    "We decompose the global motion into spatial-normalized local features 
    and temporal delta features to reduce drift in autoregressive prediction"
    
    Components:
    1. Spatial Normalization: Makes XZ positions relative to head (keeps global Y)
    2. Temporal Normalization: Computes delta = frame[t] - frame[0] for drift reduction
    """
    def __init__(self, num_timesteps=7, features_per_timestep=36):
        super(GMDModule, self).__init__()
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        
        # Feature layout per timestep (36 features = 3 trackers × 12 features):
        # Head:       Position[0:3], Forward[3:6], Up[6:9], Velocity[9:12]
        # LeftWrist:  Position[12:15], Forward[15:18], Up[18:21], Velocity[21:24]
        # RightWrist: Position[24:27], Forward[27:30], Up[30:33], Velocity[33:36]
        
        # Position offsets for each tracker (within one timestep)
        self.head_pos = 0
        self.lwrist_pos = 12
        self.rwrist_pos = 24
        
        # XZ indices relative to position start (X=0, Y=1, Z=2)
        self.x_idx = 0
        self.z_idx = 2
        
    def spatial_normalization(self, x):
        """
        Make XZ positions relative to head while keeping global Y (vertical)
        
        EgoPoser code reference (lines 75-78):
            head_horizontal_trans = input_tensor.clone()[...,36:38].detach()
            input_tensor[...,36:38] -= head_horizontal_trans
            input_tensor[...,39:41] -= head_horizontal_trans
            input_tensor[...,42:44] -= head_horizontal_trans
        """
        # Get head XZ for all timesteps
        head_x = x[:, :, self.head_pos + self.x_idx].clone()  # (B, T)
        head_z = x[:, :, self.head_pos + self.z_idx].clone()  # (B, T)
        
        # Make head XZ zero (relative to itself)
        x[:, :, self.head_pos + self.x_idx] -= head_x
        x[:, :, self.head_pos + self.z_idx] -= head_z
        
        # Make LeftWrist XZ relative to head
        x[:, :, self.lwrist_pos + self.x_idx] -= head_x
        x[:, :, self.lwrist_pos + self.z_idx] -= head_z
        
        # Make RightWrist XZ relative to head
        x[:, :, self.rwrist_pos + self.x_idx] -= head_x
        x[:, :, self.rwrist_pos + self.z_idx] -= head_z
        
        return x
    
    def temporal_normalization(self, x):
        """
        Compute delta features: frame[t] - frame[0] for each tracker's XZ
        
        EgoPoser code reference (lines 82-88):
            delta_0 = input_tensor[...,36:38] - input_tensor[...,[0],36:38]
            delta_1 = input_tensor[...,39:41] - input_tensor[...,[0],39:41]
            delta_2 = input_tensor[...,42:44] - input_tensor[...,[0],42:44]
            input_tensor = torch.cat([input_tensor, delta_0, delta_1, delta_2], dim=-1)
        """
        # Delta for Head XZ
        delta_head_x = x[:, :, self.head_pos + self.x_idx] - x[:, [0], self.head_pos + self.x_idx]
        delta_head_z = x[:, :, self.head_pos + self.z_idx] - x[:, [0], self.head_pos + self.z_idx]
        
        # Delta for LeftWrist XZ
        delta_lwrist_x = x[:, :, self.lwrist_pos + self.x_idx] - x[:, [0], self.lwrist_pos + self.x_idx]
        delta_lwrist_z = x[:, :, self.lwrist_pos + self.z_idx] - x[:, [0], self.lwrist_pos + self.z_idx]
        
        # Delta for RightWrist XZ
        delta_rwrist_x = x[:, :, self.rwrist_pos + self.x_idx] - x[:, [0], self.rwrist_pos + self.x_idx]
        delta_rwrist_z = x[:, :, self.rwrist_pos + self.z_idx] - x[:, [0], self.rwrist_pos + self.z_idx]
        
        # Stack: (B, T, 6) - matching EgoPoser's order
        deltas = torch.stack([
            delta_head_x, delta_head_z,      # Head XZ delta
            delta_lwrist_x, delta_lwrist_z,  # LeftWrist XZ delta
            delta_rwrist_x, delta_rwrist_z   # RightWrist XZ delta
        ], dim=-1)
        
        return deltas
        
    def forward(self, x):
        """
        Apply GMD: Spatial Normalization + Temporal Normalization
        
        Args:
            x: (B, 252) flat input tensor
        Returns:
            x_gmd: (B, T, F+6) = (B, 7, 42) GMD-processed tensor
        """
        B = x.shape[0]
        
        # Reshape: (B, 252) -> (B, 7, 36)
        x = x.view(B, self.num_timesteps, self.features_per_timestep)
        
        # Apply Spatial Normalization
        x = self.spatial_normalization(x)
        
        # Compute Temporal Deltas
        deltas = self.temporal_normalization(x)  # (B, T, 6)
        
        # Concatenate: (B, T, 36) + (B, T, 6) -> (B, T, 42)
        x_gmd = torch.cat([x, deltas], dim=-1)
        
        return x_gmd


# =============================================================================
# SlowFast Module: Dual-Pathway Feature Fusion
# =============================================================================

class SlowFastTransformer(nn.Module):
    """
    SlowFast Feature Fusion with Transformer from EgoPoser (ECCV 2024)
    
    Paper Reference: Section 3.3
    "We use a SlowFast architecture to capture both fine-grained recent motion 
    and longer-term context simultaneously"
    
    EgoPoser code reference (lines 94-106):
        x_fast = input_tensor[:,-input_tensor.shape[1]//2:,...]  # Last half
        x_slow = input_tensor[:,::2,...]                          # Every 2nd frame
        x_fast = self.linear_embedding(x_fast)
        x_slow = self.linear_embedding(x_slow)
        x = x_fast + x_slow
        x = self.transformer_encoder(x)
        x = x[:, -1]  # Take last timestep output
    
    Architecture:
    - Fast Pathway: High temporal resolution (last half of window)
    - Slow Pathway: Low temporal resolution (every 2nd frame)
    - Both embedded and summed, then processed by Transformer
    """
    def __init__(self, input_dim, embed_dim, num_layers=2, nhead=4, dropout=0.1):
        super(SlowFastTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Linear embeddings for both pathways (same as EgoPoser)
        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        
        # Transformer Encoder (as in EgoPoser)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=False  # EgoPoser uses (T, B, D) format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Apply SlowFast fusion with Transformer
        
        Args:
            x: (B, T, F) GMD-processed tensor, e.g., (B, 7, 42)
        Returns:
            out: (B, embed_dim) fused temporal features
        """
        B, T, F = x.shape
        
        # === SlowFast Pathway Split ===
        # Fast: last half of timesteps (fine-grained recent motion)
        # For T=7: indices 3,4,5,6 (4 frames)
        x_fast = x[:, -(T // 2 + 1):, :]  # Last ~half including middle
        
        # Slow: every 2nd frame (long-term context)
        # For T=7: indices 0,2,4,6 (4 frames)
        x_slow = x[:, ::2, :]
        
        # === Embedding ===
        # Embed both pathways with same embedding layer (as in EgoPoser)
        x_fast_emb = self.linear_embedding(x_fast)  # (B, T_fast, embed_dim)
        x_slow_emb = self.linear_embedding(x_slow)  # (B, T_slow, embed_dim)
        
        # === Fusion: Element-wise sum ===
        # Align temporal dimensions (take min length)
        min_len = min(x_fast_emb.shape[1], x_slow_emb.shape[1])
        x_fused = x_fast_emb[:, :min_len, :] + x_slow_emb[:, :min_len, :]  # (B, min_len, embed_dim)
        
        # === Transformer Processing ===
        # Convert to (T, B, D) for transformer
        x_fused = x_fused.permute(1, 0, 2)  # (T, B, embed_dim)
        
        # Apply transformer encoder
        x_fused = self.transformer_encoder(x_fused)  # (T, B, embed_dim)
        
        # Convert back to (B, T, D)
        x_fused = x_fused.permute(1, 0, 2)  # (B, T, embed_dim)
        
        # Take last timestep output (as in EgoPoser: x[:, -1])
        out = x_fused[:, -1, :]  # (B, embed_dim)
        
        # Apply output normalization
        out = self.output_norm(out)
        
        return out


class SlowFastMLP(nn.Module):
    """
    SlowFast Feature Fusion with MLP (Lightweight version)
    
    Alternative to Transformer-based SlowFast for faster training.
    Uses temporal pooling + MLP instead of attention mechanism.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(SlowFastMLP, self).__init__()
        
        # Separate embeddings for fast and slow pathways
        self.fast_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.slow_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, F) GMD-processed tensor
        Returns:
            out: (B, output_dim) fused features
        """
        B, T, F = x.shape
        
        # Fast pathway: last half, temporal mean pooling
        x_fast = x[:, -(T // 2 + 1):, :].mean(dim=1)  # (B, F)
        
        # Slow pathway: every 2nd frame, temporal mean pooling
        x_slow = x[:, ::2, :].mean(dim=1)  # (B, F)
        
        # Embed
        fast_emb = self.fast_embedding(x_fast)  # (B, output_dim)
        slow_emb = self.slow_embedding(x_slow)  # (B, output_dim)
        
        # Fuse: element-wise sum + processing
        fused = fast_emb + slow_emb
        out = self.fusion(fused)
        
        return out


# =============================================================================
# Main Model: Categorical VQ-VAE with GMD + SlowFast
# =============================================================================

class Model(nn.Module):
    """
    Categorical VQ-VAE with GMD + SlowFast Enhancement
    
    Full Pipeline:
    1. GMD Preprocessing (Spatial + Temporal Normalization)
    2. SlowFast Feature Fusion (Transformer or MLP based)
    3. Categorical VQ-VAE (Gumbel-Softmax + Codebook Matching)
    
    Architecture:
    Input (252) -> GMD (B,7,42) -> SlowFast (B,hidden) -> Projection (B,252)
                                                              ↓
                                           Residual Add with Original Input
                                                              ↓
                                    Normalize -> Encoder/Estimator -> Gumbel-Softmax
                                                              ↓
                                              Codebook -> Decoder -> Output
    """
    def __init__(self, encoder, estimator, decoder, xNorm, yNorm, 
                 codebook_channels, codebook_dim,
                 # GMD + SlowFast settings
                 use_gmd=True,
                 slowfast_type='transformer',  # 'transformer' or 'mlp'
                 num_timesteps=7,
                 features_per_timestep=36,
                 embed_dim=256,
                 num_transformer_layers=2,
                 nhead=4,
                 dropout=0.25):
        super(Model, self).__init__()

        # Original Categorical VQ-VAE components
        self.Encoder = encoder
        self.Estimator = estimator
        self.Decoder = decoder
        self.XNorm = xNorm
        self.YNorm = yNorm
        self.C = codebook_channels
        self.D = codebook_dim
        
        # GMD + SlowFast settings
        self.use_gmd = use_gmd
        self.slowfast_type = slowfast_type
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        
        if self.use_gmd:
            # GMD Module
            self.gmd = GMDModule(num_timesteps, features_per_timestep)
            gmd_output_dim = features_per_timestep + 6  # 36 + 6 = 42
            
            # SlowFast Module
            if slowfast_type == 'transformer':
                self.slowfast = SlowFastTransformer(
                    input_dim=gmd_output_dim,
                    embed_dim=embed_dim,
                    num_layers=num_transformer_layers,
                    nhead=nhead,
                    dropout=dropout
                )
            else:  # 'mlp'
                self.slowfast = SlowFastMLP(
                    input_dim=gmd_output_dim,
                    hidden_dim=embed_dim * 2,
                    output_dim=embed_dim,
                    dropout=dropout
                )
            
            # Projection back to original input dimension
            original_input_dim = num_timesteps * features_per_timestep
            self.projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, original_input_dim)
            )
            
            # Learnable residual weight
            self.residual_weight = nn.Parameter(torch.tensor(0.5))
            
            print("=" * 60)
            print("GMD + SlowFast Configuration:")
            print("=" * 60)
            print(f"  GMD Enabled: {use_gmd}")
            print(f"  SlowFast Type: {slowfast_type}")
            print(f"  Input: {num_timesteps} timesteps × {features_per_timestep} features = {original_input_dim}")
            print(f"  GMD Output: {num_timesteps} × {gmd_output_dim} = {num_timesteps * gmd_output_dim}")
            print(f"  Embed Dim: {embed_dim}")
            if slowfast_type == 'transformer':
                print(f"  Transformer Layers: {num_transformer_layers}")
                print(f"  Attention Heads: {nhead}")
            print("=" * 60)

    # === Gumbel-Softmax Methods (unchanged from original) ===
    
    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.reshape(-1,1,1,1)
        noise = torch.rand_like(tensor) - 0.5
        samples = scale * noise + 0.5
        return -torch.log(-torch.log(samples + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, scale):
        y = logits + self.sample_gumbel(logits, scale)
        return F.softmax(y / temperature, dim=-1)
    
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
    
    # === GMD + SlowFast Processing ===
    
    def apply_gmd_slowfast(self, x):
        """
        Apply GMD preprocessing and SlowFast fusion
        
        Pipeline:
        1. GMD: Spatial + Temporal Normalization -> (B, 7, 42)
        2. SlowFast: Dual-pathway fusion -> (B, embed_dim)
        3. Projection: Back to original dim -> (B, 252)
        """
        # Step 1: GMD
        x_gmd = self.gmd(x)  # (B, 7, 42)
        
        # Step 2: SlowFast
        x_fused = self.slowfast(x_gmd)  # (B, embed_dim)
        
        # Step 3: Projection
        x_proj = self.projection(x_fused)  # (B, 252)
        
        return x_proj
    
    def forward(self, x, knn, t=None):
        """
        Forward pass with GMD + SlowFast preprocessing
        
        Args:
            x: Input (B, 252) - tracker features
            knn: K-nearest neighbor for Gumbel-Softmax sampling
            t: Target (B, output_dim) - only during training
        """
        # === GMD + SlowFast Preprocessing ===
        if self.use_gmd:
            x_original = x.clone()
            x_gmd = self.apply_gmd_slowfast(x)
            
            # Learnable residual connection
            alpha = torch.sigmoid(self.residual_weight)
            x = alpha * x_gmd + (1 - alpha) * x_original
        
        # === Original Categorical VQ-VAE Pipeline ===
        
        # Training mode
        if t is not None:
            x = utility.Normalize(x, self.XNorm)
            t = utility.Normalize(t, self.YNorm)

            # Teacher: Encode target
            target_logits = self.Encoder(torch.cat((t, x), dim=1))
            target_probs, target = self.sample(target_logits, knn)

            # Student: Encode input
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            # Decode
            y = self.Decoder(target)

            return (utility.Renormalize(y, self.YNorm), 
                    target_logits, target_probs, target, 
                    estimate_logits, estimate_probs, estimate)
                
        # Inference mode
        else:
            x = utility.Normalize(x, self.XNorm)
            
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            y = self.Decoder(estimate)

            return utility.Renormalize(y, self.YNorm), estimate


# =============================================================================
# Training Script
# =============================================================================

if __name__ == '__main__':
    # Dataset settings
    name = "LowerBody"
    directory = "../../Datasets/" + name
    id = name + "_GMD_SlowFast_" + utility.GetFileID(__file__)
    load = directory
    save = directory + "/Training_" + id
    utility.MakeDirectory(save)

    # Load data
    XFile = load + "/Input.bin"
    YFile = load + "/Output.bin"
    XShape = utility.LoadTxtAsInt(load + "/InputShape.txt", True)
    YShape = utility.LoadTxtAsInt(load + "/OutputShape.txt", True)

    sample_count = XShape[0]
    input_dim = XShape[1]
    output_dim = YShape[1]

    utility.SetSeed(23456)

    # Training hyperparameters
    epochs = 150
    batch_size = 32
    dropout = 0.25

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    # Network architecture
    encoder_dim = 1024
    estimator_dim = 1024
    decoder_dim = 1024

    codebook_channels = 128
    codebook_dim = 8
    codebook_size = codebook_channels * codebook_dim
    
    # GMD + SlowFast settings
    use_gmd = True
    slowfast_type = 'transformer'  # 'transformer' or 'mlp'
    num_timesteps = 7
    features_per_timestep = 36
    embed_dim = 256
    num_transformer_layers = 2
    nhead = 4
    
    print("\n" + "=" * 60)
    print("Categorical VQ-VAE + GMD + SlowFast Training")
    print("=" * 60)
    print(f"Dataset: {name}")
    print(f"Input Features: {input_dim}")
    print(f"Output Features: {output_dim}")
    print(f"Sample Count: {sample_count}")
    print("=" * 60)

    # Create model
    network = utility.ToDevice(Model(
        encoder=modules.LinearEncoder(input_dim + output_dim, encoder_dim, encoder_dim, codebook_size, dropout),
        estimator=modules.LinearEncoder(input_dim, estimator_dim, estimator_dim, codebook_size, dropout),
        decoder=modules.LinearEncoder(codebook_size, decoder_dim, decoder_dim, output_dim, 0.0),
        
        xNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/InputNormalization.txt", True)), requires_grad=False),
        yNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/OutputNormalization.txt", True)), requires_grad=False),
        
        codebook_channels=codebook_channels,
        codebook_dim=codebook_dim,
        
        # GMD + SlowFast
        use_gmd=use_gmd,
        slowfast_type=slowfast_type,
        num_timesteps=num_timesteps,
        features_per_timestep=features_per_timestep,
        embed_dim=embed_dim,
        num_transformer_layers=num_transformer_layers,
        nhead=nhead,
        dropout=dropout
    ))
        
    # Optimizer and scheduler
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(
        optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count,
        restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True
    )
    loss_function = nn.MSELoss()

    # Plotting setup
    plt.ion()
    _, ax_latent = plt.subplots(1, 5, figsize=(10, 2))
    loss_history = utility.PlottingWindow("Loss History", ax=plt.subplots(figsize=(10, 5)), drawInterval=500, yScale='log')
    
    def Item(value):
        return value.detach().cpu()

    # Test sequences
    Sequences = utility.LoadTxtRaw(load + "/Sequences.txt", False)
    Sequences = np.array(utility.Transpose2DList(Sequences)[0], dtype=np.int64)
    test_sequence_length = 60
    test_sequences = []
    for i in range(int(Sequences[-1])):
        indices = np.where(Sequences == (i + 1))[0]
        intervals = int(np.floor(len(indices) / test_sequence_length))
        if intervals > 0:
            slices = np.array_split(indices, intervals)
            test_sequences += slices
    print(f"Test Sequences: {len(test_sequences)}")

    # Training loop
    I = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(I)
        error = 0.0
        
        for i in range(0, sample_count, batch_size):
            print(f'Progress {round(100 * i / sample_count, 2)}%', end="\r")
            train_indices = I[i:i + batch_size]

            xBatch = utility.ReadBatchFromFile(XFile, train_indices, XShape[1])
            yBatch = utility.ReadBatchFromFile(YFile, train_indices, YShape[1])

            prediction, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = network(
                xBatch, knn=torch.ones(1, device=xBatch.device), t=yBatch
            )

            # Losses
            mse_loss = loss_function(
                utility.Normalize(yBatch, network.YNorm),
                utility.Normalize(prediction, network.YNorm)
            )
            matching_loss = loss_function(target, estimate)
            
            loss = mse_loss + matching_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            error += loss.item()

            loss_history.Add(
                (Item(mse_loss).item(), "MSE Loss"),
                (Item(matching_loss).item(), "Matching Loss")
            )

            # Visualization
            if loss_history.Counter == 0:
                network.eval()

                input_sequences = []
                output_sequences = []
                target_sequences = []
                estimate_sequences = []
                predictions_sequences = []
                
                for s in range(100):
                    idx = random.choice(test_sequences)
                    xBatch = utility.ReadBatchFromFile(XFile, idx, XShape[1])
                    yBatch = utility.ReadBatchFromFile(YFile, idx, YShape[1])
                    prediction, _, _, target, _, _, estimate = network(
                        xBatch, knn=torch.zeros(1, device=xBatch.device), t=yBatch
                    )
                    input_sequences.append(Item(xBatch))
                    output_sequences.append(Item(yBatch))
                    target_sequences.append(Item(target))
                    estimate_sequences.append(Item(estimate))
                    predictions_sequences.append(Item(prediction))
                
                plotting.PCA2DSequence(ax_latent[0], test_sequence_length, input_dim, input_sequences, "Input")
                plotting.PCA2DSequence(ax_latent[1], test_sequence_length, output_dim, output_sequences, "Output")
                plotting.PCA2DSequence(ax_latent[2], test_sequence_length, codebook_size, target_sequences, "Target")
                plotting.PCA2DSequence(ax_latent[3], test_sequence_length, codebook_size, estimate_sequences, "Estimate")
                plotting.PCA2DSequence(ax_latent[4], test_sequence_length, output_dim, predictions_sequences, "Prediction")

                network.train()
                plt.gcf().canvas.draw()
                plt.gcf().canvas.start_event_loop(1e-1)

        print(f'Epoch {epoch + 1}, Loss: {error / (sample_count / batch_size):.6f}')
        loss_history.Print()

        # Save model
        utility.SaveONNX(
            path=save + '/' + id + '_' + str(epoch + 1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim), torch.ones(1)),
            input_names=['X', 'K'],
            output_names=['Y', 'Code'],
            dynamic_axes={'K': {0: 'Size'}}
        )
