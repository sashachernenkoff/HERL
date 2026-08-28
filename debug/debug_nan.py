"""
Diagnostic script to trace the origin of NaN gradients/activations in the Factored VAE.
Simulates a single forward pass, backward pass, and optimizer step on synthetic genotype 
data matching the real class distribution. Used to identify the interaction between 
sparse multi-hot inputs and Adam optimizer steps that causes activation explosions.
"""
import json, sys, os, torch
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "")))
from vae.model import VAE
from vae.loss import compute_loss

config = json.load(open("vae/configs/wtccc.json"))
config["encoder"]["first_layer_rank"] = 256
config["decoder"]["last_layer_rank"] = 256

device = torch.device("cuda")
input_dim = 378193
vae = VAE(input_dim, config).to(device)

# Fake genotype data matching typical class distribution
x = torch.zeros((16, input_dim))
x[:, :int(input_dim * 0.33)] = 1
x[:, int(input_dim * 0.33):int(input_dim * 0.41)] = 2
x = x.to(device)

# Calculate class weights
x_int = x.long()
class_counts = torch.bincount(x_int.flatten(), minlength=3).float()
class_weights = (class_counts.sum() / (3 * class_counts)).to(device)
print(f"Weights: {class_weights}")

print("\n=== Forward pass (with autocast) ===")
with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    recon, mu, log_var, z = vae(x)
    total, recon_l, kl_l = compute_loss(x, recon, mu, log_var, weight=class_weights, beta=1.0)
    print(f"loss={total.item():.4f} recon={recon_l.item():.4f} kl={kl_l.item():.4f} nan={total.isnan().item()}")

print("\n=== Backward + gradient check ===")
optimizer = torch.optim.AdamW(vae.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=True)
optimizer.zero_grad()

scaler.scale(total).backward()
scaler.unscale_(optimizer)

total_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
print(f"Total grad norm: {total_norm:.4f}")
for name, p in vae.named_parameters():
    if p.grad is not None and p.grad.isnan().any():
        print(f"NaN grad in {name}")

scaler.step(optimizer)
scaler.update()

print("\n=== After 1 optimizer step ===")
with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    recon, mu, log_var, z = vae(x)
    total, recon_l, kl_l = compute_loss(x, recon, mu, log_var, weight=class_weights, beta=1.0)
    print(f"loss={total.item():.4f} nan={total.isnan().item()}")
    print(f"recon: min={recon.min():.4f} max={recon.max():.4f} nan={recon.isnan().any()}")
