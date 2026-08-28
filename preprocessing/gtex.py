"""
Prepare per-gene cis genotype windows for GTEx heritability analysis.

For each gene in a GTEx TPM expression table, extracts a cis genotype window
(+/- 500 kb around the gene) from the full genotype set using PLINK, restricts
to individuals common to both the expression and genotype data, and writes a
per-gene phenotype file of expression values. Produces one directory per gene
containing PLINK binaries, a recoded .raw genotype file, and a .phen file.

Requires PLINK on the system (pass its path with --plink_path).

Example usage:
    python gtex.py \
        --tpm_file /path/to/gtex/gene_tpm_whole_blood.gct \
        --genotype_file /path/to/gtex/GTEx_filtered.raw \
        --annotation_file /path/to/gtex/Homo_sapiens.GRCh38.112.gtf.gz \
        --output_base_dir /path/to/gtex \
        --plink_path plink
"""

import argparse
import gzip
import os
import subprocess

import pandas as pd


def parse_gtf(gtf_file):
    """Parse the gene annotation (GTF) file directly from the compressed archive."""
    gtf_columns = [
        'chromosome', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attributes'
    ]
    with gzip.open(gtf_file, 'rt') as f:
        gtf_df = pd.read_csv(f, sep='\t', comment='#', names=gtf_columns)

    gtf_genes = gtf_df[gtf_df['feature'] == 'gene'].copy()
    gtf_genes['gene_id'] = gtf_genes['attributes'].str.extract(r'gene_id "([^"]+)"')
    gtf_genes['gene_name'] = gtf_genes['attributes'].str.extract(r'gene_name "([^"]+)"')
    return gtf_genes[['chromosome', 'start', 'end', 'gene_id', 'gene_name']]


def load_expression(tpm_file, gtf_data):
    """Load the TPM table, normalize sample names to the FID convention, and annotate."""
    tpm_df = pd.read_csv(tpm_file, sep='\t', skiprows=2)
    tpm_df.columns = tpm_df.columns.str.strip()
    tpm_df.columns = tpm_df.columns.str.replace(r'^([^-]*-[^-]*)-.*$', r'\1', regex=True)
    tpm_df = tpm_df.rename(columns={'Description': 'gene', 'Name': 'gene_id'})
    tpm_df['gene_id'] = tpm_df['gene_id'].str.split('.').str[0]  # strip Ensembl version
    return pd.merge(gtf_data, tpm_df, how='right', on='gene_id')


def extract_gene_window(row, common_individuals, common_individuals_file,
                        genotype_file, output_base_dir, plink_path):
    """Extract the cis window for one gene and write its PLINK + phenotype files."""
    gene_name = row['gene']
    gene_expression = row[5:]
    gene_chromosome = row['chromosome']

    if '/' in str(gene_name) or pd.isna(gene_chromosome):
        return

    # cis window: +/- 500 kb around the gene
    start_pos = max(0, row['start'] - 500000)
    end_pos = row['end'] + 500000

    gene_dir = os.path.join(output_base_dir, str(gene_chromosome), gene_name)
    os.makedirs(gene_dir, exist_ok=True)

    bfile = genotype_file.split('.')[0]
    window = ['--chr', str(gene_chromosome),
              '--from-kb', str(start_pos // 1000),
              '--to-kb', str(end_pos // 1000),
              '--keep', common_individuals_file]

    # PLINK binaries for the window
    subprocess.run([plink_path, '--bfile', bfile, *window, '--make-bed',
                    '--out', os.path.join(gene_dir, gene_name)])

    # Recoded raw genotype file for the window
    subprocess.run([plink_path, '--bfile', bfile, *window, '--recode', 'A',
                    '--out', os.path.join(gene_dir, gene_name)])

    # Per-gene phenotype file (expression values)
    phenotype_df = pd.DataFrame({'FID': common_individuals, 'IID': common_individuals})
    phenotype_df['Phenotype'] = gene_expression.values
    phenotype_df.to_csv(os.path.join(gene_dir, f'{gene_name}.phen'),
                        sep='\t', index=False, header=False)


def main(tpm_file, genotype_file, annotation_file, output_base_dir, plink_path):
    gtf_data = parse_gtf(annotation_file)
    tpm_df = load_expression(tpm_file, gtf_data)

    # Individuals present in both expression and genotype data
    ids = pd.read_csv(genotype_file, delim_whitespace=True, usecols=['FID'])
    common_individuals = list(set(tpm_df.columns[5:]).intersection(ids['FID']))

    tpm_df = tpm_df[['chromosome', 'start', 'end', 'gene_id', 'gene'] + common_individuals]

    common_individuals_file = os.path.join(output_base_dir, 'common_individuals.txt')
    with open(common_individuals_file, 'w') as f:
        for fid in common_individuals:
            f.write(f'{fid}\t{fid}\n')

    for _, row in tpm_df.iterrows():
        extract_gene_window(row, common_individuals, common_individuals_file,
                            genotype_file, output_base_dir, plink_path)

    print('Processing complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare per-gene cis genotype windows for GTEx')
    parser.add_argument('--tpm_file', type=str, required=True, help='GTEx gene TPM expression table')
    parser.add_argument('--genotype_file', type=str, required=True, help='PLINK .raw genotype file (full set)')
    parser.add_argument('--annotation_file', type=str, required=True, help='Gene annotation GTF (.gz)')
    parser.add_argument('--output_base_dir', type=str, required=True,
                        help='Output directory root (one subdirectory per gene)')
    parser.add_argument('--plink_path', type=str, default='plink', help='Path to the PLINK executable')
    args = parser.parse_args()
    main(args.tpm_file, args.genotype_file, args.annotation_file, args.output_base_dir, args.plink_path)
