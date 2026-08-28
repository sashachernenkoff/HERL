"""
Build phenotype files, GRM, and GCTA binaries from WTCCC VAE latent representations.

Reads the latent means written by vae/train_wtccc.py, computes the genetic
relationship matrix (GRM = (1/M) * E @ E.T), serializes it to GCTA binary
format (.grm.bin, .grm.N.bin, .grm.id), and writes a case/control phenotype
file (.phen). The output directory is ready for gcta.py or a direct GCTA
--reml call.

Example usage:
    python heritability/grm_wtccc.py --disease BD \\
        --data_path /path/to/wtccc/wtccc.geno.BD.csv \\
        --output_dir /path/to/output/WTCCC/BD/BD_200
"""

import argparse
import os

import numpy as np
import pandas as pd


def make_phenotype_file(disease, output_dir):
    """Write a GCTA-format phenotype file for case/control status.

    Individual IDs are read from the original RAW genotype CSV header, because
    the prepped filtered CSVs do not retain the individual IDs.
    """
    print("Writing phenotype file...")
    # The raw files have CHR, LOC, and then the individual IDs as column headers
    raw_path = f"/work/long_lab/GROUP_DATA/Genomes/Human_GWAS/WTCCC_data/wtccc.geno.{disease}.csv"
    col_names = pd.read_csv(raw_path, nrows=0).columns.tolist()
    col_names = col_names[2:]  # drop 'CHR' and 'LOC'

    data = pd.DataFrame(col_names, columns=['FID'])
    data['IID'] = data['FID']
    data['Phenotype'] = data['FID'].apply(lambda x: 1 if x.startswith(disease) else 0)

    os.makedirs(output_dir, exist_ok=True)
    pheno_path = os.path.join(output_dir, f'{disease}.phen')
    data.to_csv(pheno_path, sep='\t', index=False, header=False)
    return pheno_path


def calculate_grm(E):
    """Compute the GRM as (1/M) * E @ E.T where M is the number of latent dimensions."""
    M = E.shape[1]
    return (1 / M) * E @ E.T


def make_grm_and_binaries(disease, pheno_path, output_dir):
    """Load latent means, compute GRM, and serialize directly to GCTA binary format."""
    print("Loading latent representations...")
    mu = pd.read_csv(os.path.join(output_dir, f'{disease}_latent_representations.csv')).to_numpy()
    
    print("Computing GRM...")
    grm = calculate_grm(mu)
    
    phen = pd.read_csv(pheno_path, header=None, sep='\t')

    assert len(grm) == len(phen), (
        f'Mismatch between GRM rows ({len(grm)}) and phen file rows ({len(phen)}).'
    )

    print("Writing GCTA .grm.id file...")
    with open(os.path.join(output_dir, f'{disease}.grm.id'), 'w') as f:
        for iid in phen[1]:
            f.write(f'{iid}\t{iid}\n')

    print("Extracting lower triangle and writing GCTA binaries...")
    i_indices, j_indices = np.tril_indices(grm.shape[0])
    lower_tri_grm = grm[i_indices, j_indices].astype(np.float32)
    
    with open(os.path.join(output_dir, f'{disease}.grm.bin'), 'wb') as f:
        f.write(lower_tri_grm.tobytes())
        
    n_arr = np.ones(len(lower_tri_grm), dtype=np.int32)
    with open(os.path.join(output_dir, f'{disease}.grm.N.bin'), 'wb') as f:
        f.write(n_arr.tobytes())
        
    print("Done!")


def main(disease, data_path, output_dir):
    pheno_path = make_phenotype_file(disease, output_dir)
    make_grm_and_binaries(disease, pheno_path, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build GRM and GCTA binaries from WTCCC VAE latent representations'
    )
    parser.add_argument('--disease', type=str, required=True, help='Disease name (e.g. BD)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the genotype CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write outputs to')
    args = parser.parse_args()
    main(args.disease, args.data_path, args.output_dir)