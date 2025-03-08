import torch
import numpy as np
# Simulate K GPUs (in practice, use torch.distributed with multiple processes)
K = 10  # Number of GPUs
B_small = 16  # Local batch size
B_big = K * B_small  # Global batch size = 64
D = 9770  # Number of parameters

# Placeholder for local gradients on each "GPU" (normally computed by autograd)
# Here, we simulate with random tensors
G_locals = np.load('G_local_filt.npy').reshape(K, D)  # List of K tensors, each [D]
G_locals = [torch.tensor(G_local) for G_local in G_locals]
# Step 1: Compute squared norm of local gradients
local_norm_squared = torch.tensor([torch.norm(G_local)**2 for G_local in G_locals])  # [K]

# Step 2: Average across "GPUs" to get |G_Bsmall|^2
# In real distributed setting: dist.all_reduce(local_norm_squared, op=dist.ReduceOp.SUM)
GBsmall_squared = local_norm_squared.sum() / K  # Scalar

# Step 3: Average local gradients to get G_Bbig
G_big = sum(G_locals) / K  # [D]

# Step 4: Compute |G_Bbig|^2
GBbig_squared = torch.norm(G_big)**2  # Scalar

# Step 5: Compute |G|^2
G2 = (1 / (B_big - B_small)) * (B_big * GBbig_squared - B_small * GBsmall_squared)  # Scalar

# Step 6: Compute S
S = (B_small * B_big / (B_big - B_small)) * (GBbig_squared - GBsmall_squared)  # Scalar

# Step 7: Update moving averages (initialize first step)
if 'G2_ema' not in globals():
    G2_ema = G2
    S_ema = S
else:
    G2_ema = alpha * G2_ema + (1 - alpha) * G2
    S_ema = alpha * S_ema + (1 - alpha) * S

# Step 8: Estimate B_noise
B_noise = S_ema / G2_ema  # Scalar

print(f"|G_Bsmall|^2: {GBsmall_squared.item():.4f}")
print(f"|G_Bbig|^2: {GBbig_squared.item():.4f}")
print(f"|G|^2: {G2.item():.4f}")
print(f"S: {S.item():.4f}")
print(f"B_noise estimate: {B_noise.item():.4f}")