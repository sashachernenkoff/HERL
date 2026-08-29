"""
VAE execution script for WTCCC genotype data.

Coordinates the training pipeline for the Categorical VAE. Bootstraps the dataset
via the preprocessing module, instantiates the VAE architecture from JSON configs,
executes model training, and computes evaluations of the final model. Generates
downstream artifacts including extracted latent representations (for GRM heritability
estimation) and full dataset reconstructions.

Example usage:
    python vae/train_wtccc.py --config vae/configs/wtccc.json \
        --disease BD \
        --data_path /work/long_lab/for_Ariel/BD \
        --output_dir /work/long_lab/sasha/heritability/out/WTCCC/BD/BD_128
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

try:
    from .model import VAE
    from .loss import compute_loss
    from .eval import evaluate_r_square, evaluate_r_square_per_snp, compute_reconstruction_error
    from preprocessing.wtccc import load_wtccc_data
except ImportError:  # pragma: no cover - script-style import fallback
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import VAE
    from loss import compute_loss
    from eval import evaluate_r_square, evaluate_r_square_per_snp, compute_reconstruction_error
    from preprocessing.wtccc import load_wtccc_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_vae(model, train_data, test_data, training, device, pruned=False):
    """Executes the VAE training loop and logs metrics."""
    from torch.utils.data import DataLoader, TensorDataset
    import copy
    
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training.get("learning_rate", 1e-4),
        weight_decay=training.get("weight_decay", 0.0)
    )

    batch_size = training["batch_size"]
    train_tensor = torch.as_tensor(np.asarray(train_data), dtype=torch.float32)
    test_tensor = torch.as_tensor(np.asarray(test_data), dtype=torch.float32)

    history = {'loss': [], 'val_loss': [], 'reconstruction_error': [], 'val_reconstruction_error': [], 'r2': [], 'val_r2': []}
    best_val_loss = float('inf')

    best_model_weights = copy.deepcopy(model.state_dict())

    kl_anneal_epochs = training.get("kl_anneal_epochs", 20)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, drop_last=True)

    accumulation_steps = training.get("accumulation_steps", 1)

    for epoch in range(training["epochs"]):
        model.train()
        train_losses, train_errors, train_r2s = [], [], []
        train_recons, train_kls = [], []
        optimizer.zero_grad()
        for idx, (batch,) in enumerate(train_loader):
            batch = batch.to(device)
            
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda'), dtype=torch.bfloat16):
                reconstructed, mu, log_var, _ = model(batch)
                total_loss, recon_loss, kl_loss = compute_loss(batch, reconstructed, mu, log_var)
                loss = total_loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_losses.append(total_loss.item())
            train_recons.append(recon_loss.item())
            train_kls.append(kl_loss.item())
            train_errors.append(compute_reconstruction_error(batch, reconstructed).item())
            train_r2s.append(evaluate_r_square(batch, reconstructed).item())

        model.eval()
        val_losses, val_errors, val_r2s = [], [], []
        val_recons, val_kls = [], []
        with torch.no_grad():
            for (batch,) in test_loader:
                batch = batch.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type=='cuda'), dtype=torch.bfloat16):
                    reconstructed, mu, log_var, _ = model(batch)
                    total_loss, recon_loss, kl_loss = compute_loss(batch, reconstructed, mu, log_var)
                val_losses.append(total_loss.item())
                val_recons.append(recon_loss.item())
                val_kls.append(kl_loss.item())
                val_errors.append(compute_reconstruction_error(batch, reconstructed).item())
                val_r2s.append(evaluate_r_square(batch, reconstructed).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        train_recon = float(np.mean(train_recons)) if train_recons else float('nan')
        val_recon = float(np.mean(val_recons)) if val_recons else float('nan')
        train_kl = float(np.mean(train_kls)) if train_kls else float('nan')
        val_kl = float(np.mean(val_kls)) if val_kls else float('nan')

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['reconstruction_error'].append(float(np.mean(train_errors)) if train_errors else float('nan'))
        history['val_reconstruction_error'].append(float(np.mean(val_errors)) if val_errors else float('nan'))
        history['r2'].append(float(np.mean(train_r2s)) if train_r2s else float('nan'))
        history['val_r2'].append(float(np.mean(val_r2s)) if val_r2s else float('nan'))
        print(f'Epoch {epoch + 1}/{training["epochs"]} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - r2: {history["r2"][-1]:.4f} - val_r2: {history["val_r2"][-1]:.4f}')

        import wandb
        if wandb.run is not None:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_recon_loss": train_recon,
                "val_recon_loss": val_recon,
                "train_kl_loss": train_kl,
                "val_kl_loss": val_kl,
                "train_reconstruction_error": history['reconstruction_error'][-1],
                "val_reconstruction_error": history['val_reconstruction_error'][-1],
                "train_r2": history['r2'][-1],
                "val_r2": history['val_r2'][-1],
                "learning_rate": current_lr
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

    # Keep the latest (final-epoch) weights on the model so the saved model and
    # extracted mu reflect the fully-trained state after KL has converged. The
    # best-val-loss weights are returned separately and saved alongside.
    return history, best_model_weights



def plot_history(history, label, output_dir):
    plt.figure(figsize=(10, 4))
    train_color, val_color = '#35978f', '#c7711a'

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label=f'Training loss {label}', color=train_color)
    plt.plot(history['val_loss'], label=f'Validation loss {label}', color=val_color)
    plt.title(f'Loss {label}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.subplot(1, 2, 2)
    plt.plot(history['reconstruction_error'], label=f'Training reconstruction error {label}', color=train_color)
    plt.plot(history['val_reconstruction_error'], label=f'Validation reconstruction error {label}', color=val_color)
    plt.title(f'Reconstruction error {label}')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction error')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(os.path.join(output_dir, f'plot_{label}.png'))


def main(config_path, disease, data_path, output_dir, pruned=False, batch_size=None, accumulation_steps=1):
    if pruned:
        output_dir = f"{output_dir}_pruned"

    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        config = json.load(f)

    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
    if accumulation_steps is not None:
        config["training"]["accumulation_steps"] = accumulation_steps



    # --- Stage 1: Data Loading ---
    print(f"--- Stage 1: Data Loading ---")
    print(f"Loading raw data prefix from {data_path}...")
    data = load_wtccc_data(data_path, disease, prune=pruned)
    print(f"Data loaded successfully. Shape: {data.shape}")

    # Write dims to JSON 
    with open(os.path.join(output_dir, f'dims_{disease}.json'), 'w') as f:
        json.dump({'input_shape': data.shape[1], 'latent_dim': config["encoder"]["latent_dim"]}, f)

    # --- Stage 3: VAE Training ---
    print(f"\n--- Stage 3: VAE Training ---")
    
    import wandb
    latent_dim = config["encoder"]["latent_dim"]
    run_name = f"{disease}_{latent_dim}_pruned" if pruned else f"{disease}_{latent_dim}"
    wandb.init(
        project="heritability",
        name=run_name,
        dir=os.path.expanduser("~/heritability"),
        config={
            "disease": disease,
            "latent_dim": config["encoder"]["latent_dim"],
            "input_dim": data.shape[1],
            "pruned": pruned,
            "model_config": config
        },
        save_code=True
    )
    
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    print(f"Split data into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples.")
    print(f"Initializing VAE with input_dim={data.shape[1]}, latent_dim={config['encoder']['latent_dim']}")
    
    vae = VAE(data.shape[1], config)
    print("Starting training loop...")
    history, best_weights = train_vae(vae, train_data, val_data, config["training"], device, pruned=pruned)
    
    # --- Stage 4: Outputs and Evaluation ---
    print(f"\n--- Stage 4: Outputs and Evaluation ---")
    print("Saving model weights (latest + best-val-loss)...")
    vae.save(os.path.join(output_dir, f'{disease}_vae.pt'))
    torch.save(best_weights, os.path.join(output_dir, f'{disease}_vae_best.pt'))

    print("Evaluating model on Test Set...")
    vae.eval()
    
    from torch.utils.data import DataLoader, TensorDataset
    test_tensor = torch.as_tensor(test_data, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=config["training"]["batch_size"], shuffle=False)
    
    test_all_recon = []
    test_all_r2_snps = []
    
    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(device)
            mu, _ = vae.encode(batch)
            reconstructed = vae.decode(mu)
            
            probs = torch.softmax(reconstructed, dim=-1)
            expected_dosage = probs[:, :, 1] + 2.0 * probs[:, :, 2]
            
            test_all_recon.append(expected_dosage.cpu())
            test_all_r2_snps.append(evaluate_r_square_per_snp(batch, reconstructed))
            
    test_recon_full = torch.cat(test_all_recon, dim=0).numpy()
    final_test_snp_r2 = float(np.mean(test_all_r2_snps))
    final_test_r2 = evaluate_r_square(test_data, test_recon_full)
    
    print(f'Final Test R-square for {disease}: {final_test_r2:.4f} | Final Test Mean Per-SNP R2: {final_test_snp_r2:.4f}')
    if wandb.run is not None:
        wandb.log({"final_test_r2": final_test_r2, "final_test_snp_r2": final_test_snp_r2})

    print("Generating latent representations and reconstructions for full dataset...")
    full_tensor = torch.as_tensor(data, dtype=torch.float32)
    full_loader = DataLoader(TensorDataset(full_tensor), batch_size=config["training"]["batch_size"], shuffle=False)
    
    all_mu = []
    all_reconstructed = []
    
    with torch.no_grad():
        for (batch,) in full_loader:
            batch = batch.to(device)
            mu, _ = vae.encode(batch)
            
            num_samples = getattr(vae, "num_samples", 1)
            if num_samples > 1:
                z = mu.unsqueeze(1).repeat(1, num_samples, 1).reshape(mu.shape[0], -1)
            else:
                z = mu
                
            reconstructed = vae.decode(z)
            
            probs = torch.softmax(reconstructed, dim=-1)
            expected_dosage = probs[:, :, 1] + 2.0 * probs[:, :, 2]
            all_mu.append(mu.cpu())
            all_reconstructed.append(expected_dosage.cpu())
            
    mu_full = torch.cat(all_mu, dim=0).numpy()
    reconstructed_full = torch.cat(all_reconstructed, dim=0).numpy()

    print("Saving representations to CSV...")
    pd.DataFrame(mu_full).to_csv(
        os.path.join(output_dir, f'{disease}_latent_representations.csv'), index=False)
    
    pd.DataFrame(reconstructed_full).to_csv(
        os.path.join(output_dir, f'{disease}_reconstructed_data.csv'), index=False)
    
    print("Generating history plots...")
    plot_history(history, disease, output_dir)
    print("Pipeline complete.")
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the VAE on WTCCC genotype data')
    parser.add_argument('--config', type=str, required=True, help='Path to a JSON config (see vae/configs/)')
    parser.add_argument('--disease', type=str, required=True, help='Disease name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input genotype CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write outputs to')
    parser.add_argument('--pruned', action='store_true', help='Use LD-pruned data instead of full unpruned SNPs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--accumulation_steps', type=int, default=None, help='Number of gradient accumulation steps')
    args = parser.parse_args()
    main(args.config, args.disease, args.data_path, args.output_dir, args.pruned, args.batch_size, args.accumulation_steps)
