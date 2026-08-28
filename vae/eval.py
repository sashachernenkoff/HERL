"""
Evaluation metrics for the Categorical VAE.

Provides decoupled evaluation functions for computing Reconstruction Error
(Cross-Entropy), R-squared (variance explained), and Macro F1-Score
(balanced classification accuracy). Designed to handle 3D PyTorch logits
during active training loops.
"""
import torch
import torch.nn.functional as F
import numpy as np

def compute_reconstruction_error(x, reconstructed, weight=None):
    """
    Computes the mean cross-entropy error.
    x: (batch_size, num_snps) original discrete dosages
    reconstructed: (batch_size, num_snps, 3) categorical logits
    """
    x_target = x.long()
    return F.cross_entropy(reconstructed.transpose(1, 2), x_target, weight=weight, reduction='mean')

def evaluate_r_square_per_snp(original, reconstructed):
    """
    Computes the mean R-squared value calculated per-SNP (column-wise).
    """
    from sklearn.metrics import r2_score
    import warnings
    
    if isinstance(original, torch.Tensor):
        if reconstructed.dim() == 3:
            probs = torch.softmax(reconstructed, dim=-1)
            reconstructed = probs[:, :, 1] + 2.0 * probs[:, :, 2]
        
        orig_np = original.float().cpu().numpy()
        recon_np = reconstructed.detach().float().cpu().numpy()
    else:
        orig_np = np.asarray(original)
        recon_np = np.asarray(reconstructed)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Calculates R2 independently for each SNP column, then averages them.
        return r2_score(orig_np, recon_np, multioutput='uniform_average')

def evaluate_r_square(original, reconstructed):
    """
    Computes the R-squared value.
    Converts 3D categorical logits into a continuous expected dosage to measure variance explained.
    """
    if isinstance(original, torch.Tensor):
        if reconstructed.dim() == 3:
            probs = torch.softmax(reconstructed, dim=-1)
            reconstructed = probs[:, :, 1] + 2.0 * probs[:, :, 2]
        
        ss_res = torch.sum((original - reconstructed) ** 2)
        ss_tot = torch.sum((original - torch.mean(original)) ** 2)
        if ss_tot == 0:
            return torch.tensor(0.0, device=original.device)
        return 1 - (ss_res / ss_tot)
    else:
        # NumPy fallback for backward compatibility
        original_arr = np.asarray(original)
        if reconstructed.ndim == 3:
            pass # (NumPy logic kept purely for backward compatibility stub if needed)
        
        ss_res = np.sum((original_arr - reconstructed) ** 2)
        ss_tot = np.sum((original_arr - np.mean(original_arr)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)
