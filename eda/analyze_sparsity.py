"""
Perform Exploratory Data Analysis (EDA) on WTCCC genotype arrays.

Calculates global sparsity (genotype frequencies), Minor Allele Frequency (MAF) 
distributions, and sample-level missingness for both unpruned (raw PLINK binaries) 
and pruned (Numpy) datasets.

Example usage:
    python eda/analyze_sparsity.py --disease BD \
        --output_dir /path/to/output/WTCCC/BD/eda
"""

import argparse
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_unpruned_eda(disease, data_dir, output_dir):
    print(f"\n--- Running UNPRUNED EDA for {disease} ---")
    qc_prefix = os.path.join(data_dir, f"{disease}_qc")
    plink = os.path.expanduser("~/plink")
    
    if not os.path.exists(f"{qc_prefix}.bed"):
        print(f"Error: Could not find unpruned binary {qc_prefix}.bed")
        return
        
    out_prefix = os.path.join(output_dir, f"{disease}_unpruned")
    
    # 1. MAF
    print("Calculating MAF using PLINK...")
    subprocess.run([plink, "--bfile", qc_prefix, "--freq", "--out", out_prefix], check=True, capture_output=True)
    
    frq_df = pd.read_csv(f"{out_prefix}.frq", sep=r'\s+')
    mafs = frq_df['MAF'].values
    
    plt.figure(figsize=(8, 5))
    plt.hist(mafs, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'{disease} UNPRUNED Minor Allele Frequency (MAF) Distribution')
    plt.xlabel('Minor Allele Frequency')
    plt.ylabel('Number of SNPs')
    plt.savefig(f"{out_prefix}_maf_hist.png")
    plt.close()
    
    print(f"Total Unpruned SNPs: {len(mafs):,}")
    print(f"Mean MAF: {np.mean(mafs):.4f} | Median MAF: {np.median(mafs):.4f}")
    
    # 2. Missingness
    print("Calculating missingness using PLINK...")
    subprocess.run([plink, "--bfile", qc_prefix, "--missing", "--out", out_prefix], check=True, capture_output=True)
    
    imiss_df = pd.read_csv(f"{out_prefix}.imiss", sep=r'\s+')
    fmiss = imiss_df['F_MISS'].values
    
    plt.figure(figsize=(8, 5))
    plt.hist(fmiss, bins=50, color='salmon', edgecolor='black')
    plt.title(f'{disease} UNPRUNED Sample Missingness Distribution')
    plt.xlabel('Fraction of Missing SNPs per Individual')
    plt.ylabel('Number of Individuals')
    plt.savefig(f"{out_prefix}_missingness_hist.png")
    plt.close()
    print("Unpruned PLINK EDA complete. Plots saved.")

    # 3. Genotype Frequencies from cached NumPy array
    npy_path = os.path.join(data_dir, f"{disease}_unpruned.npy")
    if not os.path.exists(npy_path):
        print(f"Skipping genotype frequency analysis: {npy_path} not found")
        return

    print("Loading unpruned numpy matrix for genotype frequencies...")
    arr = np.load(npy_path)
    print(f"Shape: {arr.shape} ({arr.shape[0]} individuals, {arr.shape[1]} SNPs)")

    out_prefix = os.path.join(output_dir, f"{disease}_unpruned")

    arr_rounded = np.round(arr)
    unique, counts = np.unique(arr_rounded, return_counts=True)
    total = arr.size

    labels = []
    percents = []
    for u, c in zip(unique, counts):
        pct = c / total * 100
        labels.append(f"Genotype {int(u)}")
        percents.append(pct)
        print(f"Genotype {int(u)}: {c:,} ({pct:.2f}%)")

    plt.figure(figsize=(6, 5))
    plt.bar(labels, percents, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f'{disease} UNPRUNED Global Genotype Proportions')
    plt.ylabel('Percentage of Matrix (%)')
    plt.savefig(f"{out_prefix}_genotype_bar.png")
    plt.close()

    # MAF from the numpy array
    alt_freqs = np.mean(arr, axis=0) / 2.0
    mafs = np.minimum(alt_freqs, 1.0 - alt_freqs)
    print(f"MAF: mean={np.mean(mafs):.4f} | median={np.median(mafs):.4f}")
    print(f"SNPs with MAF < 0.01: {np.sum(mafs < 0.01):,}")
    print(f"SNPs with MAF < 0.05: {np.sum(mafs < 0.05):,}")
    print(f"SNPs with MAF >= 0.05: {np.sum(mafs >= 0.05):,}")
    print("Unpruned genotype frequency analysis complete.")


def run_pruned_eda(disease, data_dir, output_dir):
    print(f"\n--- Running PRUNED EDA for {disease} ---")
    npy_path = os.path.join(data_dir, f"{disease}_pruned.npy")
    
    if not os.path.exists(npy_path):
        print(f"Error: Could not find pruned array {npy_path}")
        return
        
    print("Loading pruned numpy matrix into memory...")
    arr = np.load(npy_path)
    print(f"Shape: {arr.shape} ({arr.shape[0]} individuals, {arr.shape[1]} SNPs)")
    
    out_prefix = os.path.join(output_dir, f"{disease}_pruned")
    
    # 1. Genotype Frequencies (Global Sparsity)
    print("Calculating global genotype frequencies...")
    arr_rounded = np.round(arr)
    unique, counts = np.unique(arr_rounded, return_counts=True)
    total = arr.size
    
    labels = []
    percents = []
    for u, c in zip(unique, counts):
        pct = c / total * 100
        labels.append(f"Genotype {int(u)}")
        percents.append(pct)
        print(f"Genotype {int(u)}: {c:,} ({pct:.2f}%)")
        
    plt.figure(figsize=(6, 5))
    plt.bar(labels, percents, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f'{disease} PRUNED Global Genotype Proportions')
    plt.ylabel('Percentage of Matrix (%)')
    plt.savefig(f"{out_prefix}_genotype_bar.png")
    plt.close()
    
    # 2. MAF
    print("Calculating MAF for pruned SNPs...")
    alt_freqs = np.mean(arr, axis=0) / 2.0
    mafs = np.minimum(alt_freqs, 1.0 - alt_freqs)
    
    plt.figure(figsize=(8, 5))
    plt.hist(mafs, bins=50, color='lightgreen', edgecolor='black')
    plt.title(f'{disease} PRUNED Minor Allele Frequency (MAF) Distribution')
    plt.xlabel('Minor Allele Frequency')
    plt.ylabel('Number of SNPs')
    plt.savefig(f"{out_prefix}_maf_hist.png")
    plt.close()
    
    print(f"Mean MAF: {np.mean(mafs):.4f} | Median MAF: {np.median(mafs):.4f}")
    print("Pruned EDA complete. Plots saved.")


def run_pca_ceiling(disease, data_dir, output_dir, n_components=128):
    """Compute PCA reconstruction R² to establish a linear ceiling for the VAE."""
    from sklearn.decomposition import PCA

    print(f"\n--- PCA-{n_components} Reconstruction Ceiling for {disease} ---")

    for variant, label in [("unpruned", "UNPRUNED"), ("pruned", "PRUNED")]:
        npy_path = os.path.join(data_dir, f"{disease}_{variant}.npy")
        if not os.path.exists(npy_path):
            print(f"Skipping {label}: {npy_path} not found")
            continue

        print(f"\nLoading {label} data...")
        arr = np.load(npy_path)
        print(f"Shape: {arr.shape}")

        # Cap components at min(n_samples, n_features)
        k = min(n_components, arr.shape[0], arr.shape[1])

        print(f"Fitting PCA with {k} components...")
        pca = PCA(n_components=k)
        latent = pca.fit_transform(arr)
        reconstructed = pca.inverse_transform(latent)

        ss_res = np.sum((arr - reconstructed) ** 2)
        ss_tot = np.sum((arr - np.mean(arr)) ** 2)
        r2 = 1 - ss_res / ss_tot

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        print(f"PCA-{k} R²: {r2:.4f}")
        print(f"Cumulative variance explained: {cumvar[-1]:.4f}")
        print(f"  Top  10 PCs: {cumvar[min(9, k-1)]:.4f}")
        if k >= 50:
            print(f"  Top  50 PCs: {cumvar[49]:.4f}")
        if k >= 128:
            print(f"  Top 128 PCs: {cumvar[127]:.4f}")

        # Plot explained variance
        out_prefix = os.path.join(output_dir, f"{disease}_{variant}")
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, k + 1), cumvar, color='#35978f')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title(f'{disease} {label} PCA Cumulative Variance ({k} components)')
        plt.axhline(y=r2, color='gray', linestyle='--', alpha=0.5)
        plt.savefig(f"{out_prefix}_pca_variance.png")
        plt.close()

    print("\nPCA ceiling analysis complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = f"/work/long_lab/sasha/data/WTCCC/{args.disease}"
    
    run_unpruned_eda(args.disease, data_dir, args.output_dir)
    run_pruned_eda(args.disease, data_dir, args.output_dir)
    run_pca_ceiling(args.disease, data_dir, args.output_dir)


if __name__ == "__main__":
    main()
