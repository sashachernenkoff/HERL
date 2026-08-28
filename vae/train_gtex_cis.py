"""
Train the configurable VAE on a single GTEx gene's cis genotype window.

Reads the gene's PLINK .raw file ({gene_path}/{gene}.raw), estimates an
intrinsic dimension via PCA (recorded to dims_{gene}.json), builds a VAE from a
JSON config, trains it, and writes the latent means (consumed downstream by
grm/gtex_cis.py) and reconstructions.

Example usage:
    python vae/train_gtex_cis.py --config vae/configs/gtex_cis.json \
        --gene_path /path/to/gtex/cis/chr1/GENE --latent 5 \
        --output_dir /path/to/output/GTEX/chr1/GENE/GENE_5
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

try:
    from .model import VAE
except ImportError:  # pragma: no cover - script-style import fallback
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_gtex_cis_data(data_path):
    data = pd.read_csv(data_path, index_col=0, delim_whitespace=True)
    data = data.drop(columns=['IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'])
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(data.mean(), inplace=True)
    return data


def estimate_intrinsic_dimension(data):
    """Number of principal components explaining 95% of variance."""
    pca = PCA()
    pca.fit(data)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumulative >= 0.95) + 1)


def evaluate_r_square(original, reconstructed):
    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    return 1 - (ss_res / ss_tot)


def main(config_path, gene_path, output_dir, latent_override):
    os.makedirs(output_dir, exist_ok=True)
    gene = gene_path.rstrip('/').split('/')[-1]
    data_path = os.path.join(gene_path, f'{gene}.raw')

    with open(config_path) as f:
        config = json.load(f)

    # --- Stage 1: Data Loading ---
    print(f"--- Stage 1: Data Loading ---")
    print(f"Loading data from {data_path}...")
    data = load_gtex_cis_data(data_path)
    print(f"Data loaded successfully. Shape: {data.shape}")

    # --- Stage 2: Intrinsic Dimensionality ---
    print(f"\n--- Stage 2: Intrinsic Dimensionality ---")
    if latent_override is not None:
        config["encoder"]["latent_dim"] = latent_override
        print(f"Latent dimension manually overridden to: {latent_override} (Skipping PCA)")
        with open(os.path.join(output_dir, f'dims_{gene}.json'), 'w') as f:
            json.dump({'input_shape': data.shape[1], 'latent_dim': latent_override, 'pca_skipped': True}, f)
    else:
        print("Calculating intrinsic dimensionality via PCA (95% explained variance)...")
        intrinsic_dim = estimate_intrinsic_dimension(data)
        config["encoder"]["latent_dim"] = intrinsic_dim
        print(f"PCA complete. Intrinsic dimension calculated as: {intrinsic_dim}")
        with open(os.path.join(output_dir, f'dims_{gene}.json'), 'w') as f:
            json.dump({'input_shape': data.shape[1], 'intrinsic_dim': intrinsic_dim}, f)

    # --- Stage 3: VAE Training ---
    print(f"\n--- Stage 3: VAE Training ---")
    data = np.asarray(data)
    train_data, test_data = train_test_split(data, test_size=0.2)
    print(f"Split data into {len(train_data)} training and {len(test_data)} validation samples.")
    print(f"Initializing VAE with input_dim={data.shape[1]}, latent_dim={config['encoder']['latent_dim']}")

    vae = VAE(data.shape[1], config)
    print("Starting training loop...")
    vae.fit(train_data, test_data, config["training"], device)
    
    # --- Stage 4: Outputs and Evaluation ---
    print(f"\n--- Stage 4: Outputs and Evaluation ---")
    print("Saving model weights...")
    vae.save(os.path.join(output_dir, f'{gene}_vae.pt'))

    print("Generating latent representations and reconstructions for full dataset in batches...")
    vae.eval()
    
    from torch.utils.data import DataLoader, TensorDataset
    full_tensor = torch.as_tensor(data, dtype=torch.float32)
    full_loader = DataLoader(TensorDataset(full_tensor), batch_size=config["training"]["batch_size"], shuffle=False)
    
    all_mu = []
    all_reconstructed = []

    with torch.no_grad():
        for (batch,) in full_loader:
            batch = batch.to(device)
            mu, _ = vae.encode(batch)
            reconstructed, _, _, _ = vae(batch)
            all_mu.append(mu.cpu())
            all_reconstructed.append(reconstructed.cpu())
            
    mu_full = torch.cat(all_mu, dim=0).numpy()
    reconstructed_full = torch.cat(all_reconstructed, dim=0).numpy()

    print("Saving representations to CSV...")
    pd.DataFrame(mu_full).to_csv(
        os.path.join(output_dir, f'{gene}_latent_representations.csv'), index=False)
    
    pd.DataFrame(reconstructed_full).to_csv(
        os.path.join(output_dir, f'{gene}_reconstructed_data.csv'), index=False)

    r2 = evaluate_r_square(data, reconstructed_full)
    print(f'Final R-square for {gene}: {r2:.4f}')
    print("Pipeline complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the VAE on a GTEx gene cis window')
    parser.add_argument('--config', type=str, required=True, help='Path to a JSON config (see vae/configs/)')
    parser.add_argument('--gene_path', type=str, required=True,
                        help='Path to the gene directory (contains {gene}.raw)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write outputs to')
    parser.add_argument('--latent', type=int, default=None, help='Override the config encoder.latent_dim')
    args = parser.parse_args()
    main(args.config, args.gene_path, args.output_dir, args.latent)
