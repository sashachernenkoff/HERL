"""
Build GRM and GCTA binaries from GTEx gene VAE latent representations.

Reads the latent means written by vae/train_gtex_cis.py for a single gene,
computes the genetic relationship matrix (GRM = (1/M) * E @ E.T), serializes
it to GCTA binary format (.grm.bin, .grm.N.bin, .grm.id). The phenotype file
(.phen) is written by preprocessing/gtex.py and is read from the gene's source
directory. The output directory is ready for gcta.py or a direct GCTA --reml
call.

Example usage:
    python heritability/grm_gtex_cis.py \\
        --gene_path /path/to/gtex/cis/chr1/GENE \\
        --output_dir /path/to/output/GTEX/chr1/GENE/GENE_5
"""

import argparse
import os

import numpy as np
import pandas as pd


def calculate_grm(E):
    """Compute the GRM as (1/M) * E @ E.T where M is the number of latent dimensions."""
    M = E.shape[1]
    return (1 / M) * E @ E.T


def make_grm_and_binaries(gene, gene_path, output_dir):
    """Load latent means, compute GRM, and serialize directly to GCTA binary format."""
    print("Loading latent representations...")
    mu = pd.read_csv(os.path.join(output_dir, f'{gene}_latent_representations.csv')).to_numpy()
    
    print("Computing GRM...")
    grm = calculate_grm(mu)
    
    print("Loading phenotype file for IDs...")
    phen = pd.read_csv(os.path.join(gene_path, f'{gene}.phen'), header=None, sep='\t')

    assert len(grm) == len(phen), (
        f'Mismatch between GRM rows ({len(grm)}) and phen file rows ({len(phen)}).'
    )

    print("Writing GCTA .grm.id file...")
    with open(os.path.join(output_dir, f'{gene}.grm.id'), 'w') as f:
        for iid in phen[1]:
            f.write(f'{iid}\t{iid}\n')

    print("Extracting lower triangle and writing GCTA binaries...")
    # Extract lower triangle efficiently (GCTA expects column-major lower triangle order,
    # but tril_indices returns row-major order. Actually GCTA expects row 1 col 1, row 2 col 1, row 2 col 2...
    # which is exactly row-major lower triangle.
    i_indices, j_indices = np.tril_indices(grm.shape[0])
    lower_tri_grm = grm[i_indices, j_indices].astype(np.float32)
    
    with open(os.path.join(output_dir, f'{gene}.grm.bin'), 'wb') as f:
        f.write(lower_tri_grm.tobytes())
        
    n_arr = np.ones(len(lower_tri_grm), dtype=np.int32)
    with open(os.path.join(output_dir, f'{gene}.grm.N.bin'), 'wb') as f:
        f.write(n_arr.tobytes())
        
    print("Done!")


def main(gene_path, output_dir):
    gene = gene_path.rstrip('/').split('/')[-1]
    make_grm_and_binaries(gene, gene_path, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build GRM and GCTA binaries from GTEx gene VAE latent representations'
    )
    parser.add_argument('--gene_path', type=str, required=True,
                        help='Path to the gene directory (contains {gene}.phen)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write outputs to')
    args = parser.parse_args()
    main(args.gene_path, args.output_dir)