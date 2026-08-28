"""
Run GCTA GREML to estimate heritability for each gene directory.

For every gene subdirectory under a base directory, builds the GRM from the
PLINK binaries (--make-grm), estimates heritability by REML (--reml), and
collects V(G)/Vp from the resulting .hsq file into a single CSV. Gene
directories are processed in parallel across worker processes.

Example usage:
    python heritability/gcta.py --base_dir /path/to/gtex/genes --gcta_path gcta64
"""

import argparse
import csv
import os
import subprocess
from multiprocessing import Pool


def run_gcta(gene_dir, gene_name, results, gcta_path):
    try:
        # Build the GRM and binaries
        cmd_grm = f'{gcta_path} --bfile {gene_dir}/{gene_name} --make-grm --out {gene_dir}/{gene_name}_gcta'
        subprocess.run(cmd_grm, shell=True, check=True)

        # Estimate heritability by REML
        cmd_reml = f'{gcta_path} --reml --grm-bin {gene_dir}/{gene_name}_gcta --pheno {gene_dir}/{gene_name}.phen --out {gene_dir}/{gene_name}_gcta_reml --thread-num 10'
        subprocess.run(cmd_reml, shell=True, check=True)

        # Extract V(G)/Vp from the .hsq output
        hsq_file = f'{gene_dir}/{gene_name}_gcta_reml.hsq'
        if os.path.exists(hsq_file):
            with open(hsq_file, 'r') as file:
                for line in file:
                    if line.startswith('V(G)/Vp'):
                        hsq = float(line.split()[1])
                        results.append((gene_dir, hsq))
                        break
    except subprocess.CalledProcessError as e:
        print(f'Error processing {gene_name}: {e}')
        results.append((gene_dir, 'Error'))


def process_genes(task):
    gene_dirs, gcta_path = task
    results = []
    for gene_dir in gene_dirs:
        gene_name = os.path.basename(gene_dir)
        run_gcta(gene_dir, gene_name, results, gcta_path)
    return results


def main(base_dir, gcta_path, num_sections=4):
    all_gene_dirs = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    # Split gene directories into sections for parallel processing
    split_gene_dirs = [all_gene_dirs[i::num_sections] for i in range(num_sections)]

    with Pool(processes=num_sections) as pool:
        results = pool.map(process_genes, [(chunk, gcta_path) for chunk in split_gene_dirs])

    flattened_results = [item for sublist in results for item in sublist]

    output_csv = os.path.join(base_dir, 'h_results.csv')
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['gene dir', 'h'])
        csvwriter.writerows(flattened_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate heritability per gene with GCTA GREML')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Directory containing one subdirectory per gene')
    parser.add_argument('--gcta_path', type=str, default='gcta64',
                        help='Path to the GCTA executable (default: gcta64 on PATH)')
    parser.add_argument('--num_sections', type=int, default=4,
                        help='Number of parallel worker processes')
    args = parser.parse_args()
    main(args.base_dir, args.gcta_path, args.num_sections)
