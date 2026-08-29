import torch
import torch.nn.functional as F

def compute_loss(x, reconstructed, mu, log_var):
    """
    Computes the pure VAE Evidence Lower Bound (ELBO) Loss.
    No scaling factors or betas are applied.

    x: (batch, num_snps)
    reconstructed: (batch, num_snps, 3) logits
    mu, log_var: (batch, latent_dim)
    """
    x_target = x.long()
    
    # Categorical CE: Sum over SNPs, Mean over batch
    # reduction='none' returns a loss of shape (batch_size, num_snps)
    ce_loss_per_snp = F.cross_entropy(reconstructed.transpose(1, 2), x_target, reduction='none')
    
    # Sum across the features (SNPs) for each patient, then average across the batch
    recon_loss = ce_loss_per_snp.sum(dim=1).mean(dim=0)
    
    # KL Divergence: Sum over latent space dims, Mean over batch
    kl_loss_per_sample = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1)
    kl_loss = kl_loss_per_sample.mean(dim=0)
    
    # Pure ELBO (No Beta)
    total_loss = recon_loss + kl_loss

    return total_loss, recon_loss, kl_loss
