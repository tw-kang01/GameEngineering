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


class GMDModule(nn.Module):
    """
    Global Motion Decomposition Module
    Applies Spatial Normalization + Temporal Normalization from EgoPoser (ECCV 2024)
    
    Spatial Normalization: Makes horizontal positions (XZ) relative to head while keeping global vertical (Y)
    Temporal Normalization: Computes delta features (current frame - first frame) for positional drift reduction
    """
    def __init__(self, num_timesteps=7, features_per_timestep=36):
        super(GMDModule, self).__init__()
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep  # 3 trackers × 12 features
        
        # Feature indices within each timestep (12 features per tracker)
        # Layout: [Position(3), Forward(3), Up(3), Velocity(3)] × 3 trackers
        # Head: 0-11, LeftWrist: 12-23, RightWrist: 24-35
        
        # Position indices (X, Y, Z) for each tracker within a timestep
        self.head_pos_offset = 0      # Head position: 0, 1, 2
        self.lwrist_pos_offset = 12   # LeftWrist position: 12, 13, 14  
        self.rwrist_pos_offset = 24   # RightWrist position: 24, 25, 26
        
        # XZ indices (horizontal plane) - X=0, Z=2 relative to position start
        self.x_idx = 0
        self.z_idx = 2
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, num_timesteps * features_per_timestep) = (B, 252)
        Returns:
            x_gmd: GMD processed tensor of shape (B, num_timesteps, features_per_timestep + 6)
                   The +6 is for temporal delta features (Head XZ, LWrist XZ, RWrist XZ deltas)
        """
        B = x.shape[0]
        
        # Reshape to (B, T, F) for easier manipulation
        x = x.view(B, self.num_timesteps, self.features_per_timestep)
        
        # === Spatial Normalization ===
        # Get head XZ position for each timestep as reference
        head_x = x[:, :, self.head_pos_offset + self.x_idx].clone()  # (B, T)
        head_z = x[:, :, self.head_pos_offset + self.z_idx].clone()  # (B, T)
        
        # Make head XZ relative (subtract from itself -> becomes 0)
        x[:, :, self.head_pos_offset + self.x_idx] = x[:, :, self.head_pos_offset + self.x_idx] - head_x
        x[:, :, self.head_pos_offset + self.z_idx] = x[:, :, self.head_pos_offset + self.z_idx] - head_z
        
        # Make LeftWrist XZ relative to head XZ
        x[:, :, self.lwrist_pos_offset + self.x_idx] = x[:, :, self.lwrist_pos_offset + self.x_idx] - head_x
        x[:, :, self.lwrist_pos_offset + self.z_idx] = x[:, :, self.lwrist_pos_offset + self.z_idx] - head_z
        
        # Make RightWrist XZ relative to head XZ
        x[:, :, self.rwrist_pos_offset + self.x_idx] = x[:, :, self.rwrist_pos_offset + self.x_idx] - head_x
        x[:, :, self.rwrist_pos_offset + self.z_idx] = x[:, :, self.rwrist_pos_offset + self.z_idx] - head_z
        
        # === Temporal Normalization ===
        # Compute delta between each frame and the first frame
        # Delta for Head XZ
        delta_head_x = x[:, :, self.head_pos_offset + self.x_idx] - x[:, [0], self.head_pos_offset + self.x_idx]  # (B, T)
        delta_head_z = x[:, :, self.head_pos_offset + self.z_idx] - x[:, [0], self.head_pos_offset + self.z_idx]  # (B, T)
        
        # Delta for LeftWrist XZ
        delta_lwrist_x = x[:, :, self.lwrist_pos_offset + self.x_idx] - x[:, [0], self.lwrist_pos_offset + self.x_idx]
        delta_lwrist_z = x[:, :, self.lwrist_pos_offset + self.z_idx] - x[:, [0], self.lwrist_pos_offset + self.z_idx]
        
        # Delta for RightWrist XZ
        delta_rwrist_x = x[:, :, self.rwrist_pos_offset + self.x_idx] - x[:, [0], self.rwrist_pos_offset + self.x_idx]
        delta_rwrist_z = x[:, :, self.rwrist_pos_offset + self.z_idx] - x[:, [0], self.rwrist_pos_offset + self.z_idx]
        
        # Stack deltas: (B, T, 6)
        deltas = torch.stack([
            delta_head_x, delta_head_z,
            delta_lwrist_x, delta_lwrist_z, 
            delta_rwrist_x, delta_rwrist_z
        ], dim=-1)
        
        # Concatenate original features with deltas
        x_gmd = torch.cat([x, deltas], dim=-1)  # (B, T, F+6) = (B, 7, 42)
        
        return x_gmd


class SlowFastFusion(nn.Module):
    """
    SlowFast Feature Fusion Module from EgoPoser (ECCV 2024)
    
    Fast pathway: Captures fine-grained recent motion (last half of window)
    Slow pathway: Captures longer-term motion context (every 2nd frame)
    Both pathways are embedded and summed for fusion
    """
    def __init__(self, input_dim, hidden_dim, num_timesteps=7):
        super(SlowFastFusion, self).__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        
        # Embedding layers for fast and slow pathways
        self.fast_embedding = nn.Linear(input_dim, hidden_dim)
        self.slow_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, F) where T=num_timesteps, F=features_per_timestep+6
        Returns:
            fused: Fused features of shape (B, hidden_dim)
        """
        B, T, F = x.shape
        
        # Fast pathway: last half of timesteps (recent motion)
        # For 7 timesteps: indices 3, 4, 5, 6 (4 frames)
        fast_start = T // 2
        x_fast = x[:, fast_start:, :]  # (B, T//2, F)
        
        # Slow pathway: every 2nd frame (long-term context)
        # For 7 timesteps: indices 0, 2, 4, 6 (4 frames)
        x_slow = x[:, ::2, :]  # (B, (T+1)//2, F)
        
        # Temporal mean pooling for each pathway
        x_fast_pooled = x_fast.mean(dim=1)  # (B, F)
        x_slow_pooled = x_slow.mean(dim=1)  # (B, F)
        
        # Embed each pathway
        fast_emb = self.fast_embedding(x_fast_pooled)  # (B, hidden_dim)
        slow_emb = self.slow_embedding(x_slow_pooled)  # (B, hidden_dim)
        
        # Element-wise sum fusion (as in EgoPoser)
        fused = fast_emb + slow_emb
        
        # Apply layer normalization
        fused = self.layer_norm(fused)
        
        return fused


class Model(nn.Module):
    """
    Categorical VQ-VAE with GMD + SlowFast Enhancement
    
    Original: Input -> Normalize -> Encoder/Estimator -> Gumbel-Softmax -> Codebook -> Decoder -> Output
    Enhanced: Input -> GMD -> SlowFast -> Normalize -> Encoder/Estimator -> Gumbel-Softmax -> Codebook -> Decoder -> Output
    
    GMD (Global Motion Decomposition): Spatial + Temporal normalization for drift reduction
    SlowFast: Dual-pathway feature fusion for multi-scale temporal modeling
    """
    def __init__(self, encoder, estimator, decoder, xNorm, yNorm, codebook_channels, codebook_dim,
                 use_gmd=True, num_timesteps=7, features_per_timestep=36, gmd_hidden_dim=512):
        super(Model, self).__init__()

        self.Encoder = encoder
        self.Estimator = estimator
        self.Decoder = decoder

        self.XNorm = xNorm
        self.YNorm = yNorm

        self.C = codebook_channels
        self.D = codebook_dim
        
        # GMD + SlowFast settings
        self.use_gmd = use_gmd
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        
        if self.use_gmd:
            # GMD module: adds 6 delta features
            self.gmd = GMDModule(num_timesteps, features_per_timestep)
            gmd_output_dim = features_per_timestep + 6  # 36 + 6 = 42
            
            # SlowFast fusion module
            self.slowfast = SlowFastFusion(gmd_output_dim, gmd_hidden_dim, num_timesteps)
            
            # Projection layer to match original input dimension for Encoder/Estimator
            original_input_dim = num_timesteps * features_per_timestep  # 252
            self.gmd_projection = nn.Linear(gmd_hidden_dim, original_input_dim)
            
            print(f"GMD + SlowFast enabled:")
            print(f"  - Input: {num_timesteps} timesteps × {features_per_timestep} features = {original_input_dim}")
            print(f"  - GMD output: {num_timesteps} × {gmd_output_dim} = {num_timesteps * gmd_output_dim}")
            print(f"  - SlowFast hidden: {gmd_hidden_dim}")
            print(f"  - Projection back to: {original_input_dim}")

    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.reshape(-1,1,1,1)
        noise = torch.rand_like(tensor) - 0.5
        samples = scale * noise + 0.5
        return -torch.log(-torch.log(samples + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, scale):
        y = logits + self.sample_gumbel(logits, scale)
        return F.softmax(y / temperature, dim=-1)
    
    def gumbel_softmax(self, logits, temperature, scale):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
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
    
    def apply_gmd_slowfast(self, x):
        """
        Apply GMD preprocessing and SlowFast fusion
        
        Args:
            x: Raw input tensor (B, 252)
        Returns:
            x_processed: Processed tensor (B, 252) - same shape but with GMD+SlowFast features
        """
        # GMD: Spatial + Temporal Normalization
        x_gmd = self.gmd(x)  # (B, 7, 42)
        
        # SlowFast: Dual-pathway fusion
        x_fused = self.slowfast(x_gmd)  # (B, hidden_dim)
        
        # Project back to original input dimension
        x_processed = self.gmd_projection(x_fused)  # (B, 252)
        
        return x_processed
    
    def forward(self, x, knn, t=None):
        """
        Forward pass with optional GMD + SlowFast preprocessing
        
        Args:
            x: Input tensor (B, input_dim) - tracker features
            knn: K-nearest neighbor samples for Gumbel-Softmax
            t: Target tensor (B, output_dim) - only provided during training
        """
        # === GMD + SlowFast Preprocessing ===
        if self.use_gmd:
            x_original = x.clone()  # Keep original for residual connection
            x = self.apply_gmd_slowfast(x)
            # Residual connection: combine GMD features with original
            x = x + x_original
        
        # === Original VQ-VAE Pipeline ===
        # Training
        if t is not None:
            # Normalize
            x = utility.Normalize(x, self.XNorm)
            t = utility.Normalize(t, self.YNorm)

            # Encode Y (target)
            target_logits = self.Encoder(torch.cat((t, x), dim=1))
            target_probs, target = self.sample(target_logits, knn)

            # Encode X (input)
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            # Decode
            y = self.Decoder(target)

            # Renormalize
            return utility.Renormalize(y, self.YNorm), target_logits, target_probs, target, estimate_logits, estimate_probs, estimate
                
        # Inference
        else:
            # Normalize
            x = utility.Normalize(x, self.XNorm)
            
            # Encode X
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            # Decode
            y = self.Decoder(estimate)

            # Renormalize
            return utility.Renormalize(y, self.YNorm), estimate


if __name__ == '__main__':
    name = "LowerBody"
    directory = "../../Datasets/"+name
    id = name + "_GMD_" + utility.GetFileID(__file__)
    load = directory
    save = directory+"/Training_"+id
    utility.MakeDirectory(save)

    XFile = load + "/Input.bin"
    YFile = load + "/Output.bin"
    XShape = utility.LoadTxtAsInt(load + "/InputShape.txt", True)
    YShape = utility.LoadTxtAsInt(load + "/OutputShape.txt", True)
    Xlabels = load + "/InputLabels.txt"
    Ylabels = load + "/OutputLabels.txt"

    sample_count = XShape[0]
    input_dim = XShape[1]
    output_dim = YShape[1]

    utility.SetSeed(23456)

    epochs = 150
    batch_size = 32
    dropout = 0.25

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    encoder_dim = 1024
    estimator_dim = 1024
    decoder_dim = 1024

    codebook_channels = 128
    codebook_dim = 8
    codebook_size = codebook_channels * codebook_dim
    
    # GMD + SlowFast settings
    use_gmd = True
    num_timesteps = 7  # TrackerBodyPredictor uses 7 timesteps (past 6 + current 1)
    features_per_timestep = 36  # 3 trackers × 12 features per tracker
    gmd_hidden_dim = 512
    
    print("=" * 50)
    print("Categorical VQ-VAE with GMD + SlowFast")
    print("=" * 50)
    print("Input Features:", input_dim)
    print("Output Features:", output_dim)
    print("GMD Enabled:", use_gmd)
    print("Timesteps:", num_timesteps)
    print("Features per Timestep:", features_per_timestep)

    network = utility.ToDevice(Model(
        encoder=modules.LinearEncoder(input_dim + output_dim, encoder_dim, encoder_dim, codebook_size, dropout),

        estimator=modules.LinearEncoder(input_dim, estimator_dim, estimator_dim, codebook_size, dropout),

        decoder=modules.LinearEncoder(codebook_size, decoder_dim, decoder_dim, output_dim, 0.0),

        xNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/InputNormalization.txt", True)), requires_grad=False),
        yNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/OutputNormalization.txt", True)), requires_grad=False),

        codebook_channels=codebook_channels,
        codebook_dim=codebook_dim,
        
        # GMD + SlowFast parameters
        use_gmd=use_gmd,
        num_timesteps=num_timesteps,
        features_per_timestep=features_per_timestep,
        gmd_hidden_dim=gmd_hidden_dim
    ))
        
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
    loss_function = nn.MSELoss()

    # Setup Plotting
    plt.ion()
    _, ax_latent = plt.subplots(1,5, figsize=(10,2))
    loss_history = utility.PlottingWindow("Loss History", ax=plt.subplots(figsize=(10,5)), drawInterval=500, yScale='log')
    def Item(value):
        return value.detach().cpu()

    # Start Generate Test Sequences
    Sequences = utility.LoadTxtRaw(load + "/Sequences.txt", False)
    Sequences = np.array(utility.Transpose2DList(Sequences)[0], dtype=np.int64)
    test_sequence_length = 60
    test_sequences = []
    for i in range(int(Sequences[-1])):
        indices = np.where(Sequences == (i+1))[0]
        intervals = int(np.floor(len(indices) / test_sequence_length))
        if intervals > 0:
            slices = np.array_split(indices, intervals)
            test_sequences += slices
    print("Test Sequences:",len(test_sequences))

    # Training Loop
    I = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(I)
        error = 0.0
        for i in range(0, sample_count, batch_size):
            print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
            train_indices = I[i:i+batch_size]

            xBatch = utility.ReadBatchFromFile(XFile, train_indices, XShape[1])
            yBatch = utility.ReadBatchFromFile(YFile, train_indices, YShape[1])

            prediction, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = network(
                xBatch, 
                knn=torch.ones(1, device=xBatch.device), 
                t=yBatch
            )

            mse_loss = loss_function(utility.Normalize(yBatch, network.YNorm), utility.Normalize(prediction, network.YNorm))
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

            if loss_history.Counter == 0:
                network.eval()

                idx = random.choice(test_sequences)
                xBatch = utility.ReadBatchFromFile(XFile, idx, XShape[1])
                yBatch = utility.ReadBatchFromFile(YFile, idx, YShape[1])

                input_sequences = []
                output_sequences = []
                target_sequences = []
                estimate_sequences = []
                predictions_sequences = []
                for s in range(100):
                    idx = random.choice(test_sequences)
                    xBatch = utility.ReadBatchFromFile(XFile, idx, XShape[1])
                    yBatch = utility.ReadBatchFromFile(YFile, idx, YShape[1])
                    prediction, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = network(xBatch, knn=torch.zeros(1, device=xBatch.device), t=yBatch)
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

        print('Epoch', epoch+1, error/(sample_count/batch_size))
        loss_history.Print()

        utility.SaveONNX(
            path=save + '/' + id + '_' + str(epoch+1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim), torch.ones(1)),
            input_names=['X', 'K'],
            output_names=['Y', 'Code'],
            dynamic_axes={
                'K': {0: 'Size'}
            }
        )
