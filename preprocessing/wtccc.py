"""
WTCCC Genotype Data Loader with automated QC and optional LD Pruning.

Executes a PLINK pipeline to filter raw binary genotypes (.bed/.bim/.fam) based
on missingness, Hardy-Weinberg Equilibrium, and Minor Allele Frequency. Optionally 
computes linkage disequilibrium (LD) to prune correlated SNPs. Parses the extracted
`.raw` file using PyArrow into Numpy, mean-imputes missing values, and saves a cached
binary `.npy` matrix (tagged as unpruned or pruned) for subsequent loading.

Example usage:
    from preprocessing.wtccc import load_wtccc_data
    # Unpruned (default)
    data = load_wtccc_data(raw_prefix="/work/long_lab/for_Ariel/BD", disease_name="BD")
    # Pruned
    data = load_wtccc_data(raw_prefix="/work/long_lab/for_Ariel/BD", disease_name="BD", prune=True)
"""

import os
import subprocess
import numpy as np


def load_wtccc_data(raw_prefix, disease_name, prune=False):
    """
    Runs QC (and optionally LD pruning) on raw PLINK files, then loads into Numpy.
    
    Parameters
    ----------
    raw_prefix : str
        Prefix path to raw .bed/.bim/.fam files (e.g. /work/long_lab/for_Ariel/BD)
    disease_name : str
        Name of the disease (used for naming cache files)
    prune : bool
        Whether to perform Linkage Disequilibrium (LD) pruning (default: False)
        
    Returns
    -------
    numpy.ndarray
        Imputed genotype matrix.
    """
    
    # Target directory for all our clean data
    data_dir = f"/work/long_lab/sasha/data/WTCCC/{disease_name}"
    os.makedirs(data_dir, exist_ok=True)
    
    tag = "pruned" if prune else "unpruned"
    final_npy = os.path.join(data_dir, f"{disease_name}_{tag}.npy")
    
    # If we already did all this hard work, just load the Numpy binary instantly!
    if os.path.exists(final_npy):
        print(f"Found cached, {tag} NumPy binary at {final_npy}.")
        print("Loading directly into memory (takes < 1 second)...")
        return np.load(final_npy)

    print(f"No cached data found. Starting full QC pipeline for {disease_name} (Prune={prune})...")
    
    # Path to the PLINK executable we just downloaded
    plink = os.path.expanduser("~/plink")
    
    qc_prefix = os.path.join(data_dir, f"{disease_name}_qc")
    prune_prefix = os.path.join(data_dir, f"{disease_name}_prune")
    raw_output = os.path.join(data_dir, f"{disease_name}_{tag}")
    
    # 1. Quality Control
    print("--- Running PLINK Quality Control ---")
    qc_cmd = [
        plink, "--bfile", raw_prefix,
        "--geno", "0.05",    # Filter SNPs with >5% missingness
        "--mind", "0.05",    # Filter individuals with >5% missingness
        "--maf", "0.01",     # Filter SNPs with Minor Allele Freq < 1%
        "--hwe", "1e-6",     # Filter SNPs failing Hardy-Weinberg Eq
        "--make-bed", "--out", qc_prefix
    ]
    subprocess.run(qc_cmd, check=True)
    
    # 2. Extract SNPs to text format (.raw)
    extract_cmd = [plink, "--bfile", qc_prefix, "--recode", "A", "--out", raw_output]
    
    if prune:
        # LD Pruning calculation
        print("--- Running PLINK LD Pruning ---")
        prune_cmd = [
            plink, "--bfile", qc_prefix,
            "--indep-pairwise", "50", "5", "0.2",
            "--out", prune_prefix
        ]
        subprocess.run(prune_cmd, check=True)
        # Add extract flag to only keep pruned SNPs
        extract_cmd.insert(3, "--extract")
        extract_cmd.insert(4, f"{prune_prefix}.prune.in")
        print("--- Extracting independent SNPs into .raw format ---")
    else:
        print("--- Extracting QC'd (unpruned) SNPs into .raw format ---")
        
    subprocess.run(extract_cmd, check=True)
    
    # 3. Load the .raw file with PyArrow
    print(f"--- Loading {tag} data into Python ---")
    raw_file = f"{raw_output}.raw"
    
    try:
        from pyarrow import csv
        print("Using PyArrow to ingest the .raw file...")
        # .raw uses spaces as delimiters.
        parse_opts = csv.ParseOptions(delimiter=' ')
        read_opts = csv.ReadOptions(block_size=100 * 1024 * 1024)
        table = csv.read_csv(raw_file, parse_options=parse_opts, read_options=read_opts)
        
        # PLINK .raw files have 6 metadata columns at the start:
        # FID, IID, PAT, MAT, SEX, PHENOTYPE. 
        # The SNPs start at column index 6.
        snp_columns = table.column_names[6:]
        
        num_rows = table.num_rows
        num_cols = len(snp_columns)
        print(f"Extracted {num_cols} SNPs for {num_rows} individuals.")
        
        arr = np.empty((num_rows, num_cols), dtype=np.float32)
        
        for i, col_name in enumerate(snp_columns):
            arr[:, i] = table[col_name].to_pandas().to_numpy(dtype=np.float32)
            
    except ImportError:
        import pandas as pd
        print("PyArrow not found. Falling back to Pandas...")
        df = pd.read_csv(raw_file, sep=r'\s+', dtype=np.float32)
        arr = df.iloc[:, 6:].to_numpy()

    # 5. Impute missing values
    print("--- Imputing missing values with column means ---")
    col_means = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_means, inds[1])
    
    # 6. Save the array as an ultra-fast .npy file for future runs
    print(f"--- Saving final matrix to {final_npy} ---")
    np.save(final_npy, arr)
    
    return arr
