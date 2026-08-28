"""
Evaluate Linkage Disequilibrium (LD) structures and compare empirical LD blocks against predefined LDetect genomic regions.
"""

import argparse
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_jaccard(emp_df, ld_df):
    """Native pandas Jaccard overlap computation to avoid pybedtools/bedtools dependency."""
    intersection_bp = 0
    
    # Ensure chromosomes match format (strip 'chr')
    emp_df = emp_df.copy()
    ld_df = ld_df.copy()
    emp_df['chr'] = emp_df['chr'].astype(str).str.replace('chr', '')
    ld_df['chr'] = ld_df['chr'].astype(str).str.replace('chr', '')
    
    for chrom in set(emp_df['chr']).union(ld_df['chr']):
        d1 = emp_df[emp_df['chr'] == chrom].copy().assign(key=1)
        d2 = ld_df[ld_df['chr'] == chrom].copy().assign(key=1)
        
        if d1.empty or d2.empty:
            continue
            
        merged = pd.merge(d1, d2, on='key', suffixes=('_1', '_2'))
        overlaps = merged[(merged['start_1'] < merged['stop_2']) & (merged['start_2'] < merged['stop_1'])]
        
        if not overlaps.empty:
            overlap_lengths = np.minimum(overlaps['stop_1'], overlaps['stop_2']) - np.maximum(overlaps['start_1'], overlaps['start_2'])
            intersection_bp += overlap_lengths.sum()
            
    total_bp_1 = (emp_df['stop'] - emp_df['start']).sum()
    total_bp_2 = (ld_df['stop'] - ld_df['start']).sum()
    
    union_bp = total_bp_1 + total_bp_2 - intersection_bp
    return intersection_bp / union_bp if union_bp > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="Evaluate LD structures and compare against LDetect blocks.")
    parser.add_argument("--bfile", required=True, help="PLINK binary file prefix")
    parser.add_argument("--ldetect", required=True, help="Path to predefined LDetect BED file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    plink_bin = os.path.expanduser("~/plink")

    # 1. PLINK 1.9 Execution
    print("Running PLINK for LD Decay...")
    empirical_ld_out = os.path.join(args.out_dir, "empirical_ld")
    cmd_r2 = [
        plink_bin, "--bfile", args.bfile,
        "--r2", "--ld-window-r2", "0.1",
        "--ld-window-kb", "1000", "--ld-window", "99999",
        "--out", empirical_ld_out
    ]
    subprocess.run(cmd_r2, check=True)

    print("Running PLINK for Block Estimation...")
    empirical_blocks_out = os.path.join(args.out_dir, "empirical_blocks")
    cmd_blocks = [
        plink_bin, "--bfile", args.bfile,
        "--blocks", "no-pheno-req", "--blocks-max-kb", "2000",
        "--out", empirical_blocks_out
    ]
    subprocess.run(cmd_blocks, check=True)

    # 2. Data Parsing and Comparison
    print("Parsing blocks and computing statistics...")
    blocks_file = f"{empirical_blocks_out}.blocks.det"
    if not os.path.exists(blocks_file):
        raise FileNotFoundError(f"PLINK blocks output not found at {blocks_file}")
        
    emp_df = pd.read_csv(blocks_file, sep=r'\s+')
    emp_df['length'] = emp_df['BP2'] - emp_df['BP1']
    
    emp_median_length = emp_df['length'].median()
    emp_median_snps = emp_df['NSNPS'].median()
    print(f"Empirical Blocks - Median Length: {emp_median_length} bp, Median SNPs: {emp_median_snps}")

    ld_df = pd.read_csv(args.ldetect, sep='\t', header=None, names=['chr', 'start', 'stop'])
    # LDetect bed files sometimes include a string header row. Safely coerce and drop it.
    ld_df['start'] = pd.to_numeric(ld_df['start'], errors='coerce')
    ld_df['stop'] = pd.to_numeric(ld_df['stop'], errors='coerce')
    ld_df = ld_df.dropna(subset=['start', 'stop']).astype({'start': int, 'stop': int})
    
    ld_df['length'] = ld_df['stop'] - ld_df['start']
    ld_median_length = ld_df['length'].median()
    print(f"LDetect Blocks - Median Length: {ld_median_length} bp")

    print("Computing boundary overlap natively...")
    emp_bed_df = pd.DataFrame({
        'chr': emp_df['CHR'],
        'start': emp_df['BP1'],
        'stop': emp_df['BP2']
    })
    
    jaccard = compute_jaccard(emp_bed_df, ld_df)
    print(f"Jaccard Index (Base-pair intersection fraction): {jaccard:.4f}")

    # 3. Visualization
    print("Generating visualizations...")
    ld_file = f"{empirical_ld_out}.ld"
    if os.path.exists(ld_file):
        ld_r2_df = pd.read_csv(ld_file, sep=r'\s+')
        ld_r2_df['dist'] = (ld_r2_df['BP_B'] - ld_r2_df['BP_A']).abs()
        # Bin physical distance by 10kb
        ld_r2_df['dist_bin'] = (ld_r2_df['dist'] // 10000) * 10000
        
        mean_r2 = ld_r2_df.groupby('dist_bin')['R2'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.plot(mean_r2['dist_bin'], mean_r2['R2'], marker='.', linestyle='-')
        plt.xlabel("Physical Distance (bp)")
        plt.ylabel("Mean R^2")
        plt.title("LD Decay Curve (10 kb bins)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.out_dir, "ld_decay.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"Warning: {ld_file} not found. Skipping LD decay plot.")

    plt.figure(figsize=(10, 6))
    sns.histplot(emp_df['length'], bins=50, color='blue', alpha=0.5, label='Empirical Blocks (PLINK)', stat='density')
    sns.histplot(ld_df['length'], bins=50, color='red', alpha=0.5, label='LDetect Blocks', stat='density')
    plt.xlabel("Block Length (bp)")
    plt.ylabel("Density")
    plt.title("Block Size Distribution")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "block_size_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
