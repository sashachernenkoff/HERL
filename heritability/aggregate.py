"""
Aggregate GCTA heritability results into a single CSV.

Crawls an output directory tree for .hsq files, extracts V(G)/Vp from each,
pivots the results by latent dimension, and merges them with an existing
results table.

Example usage:
    python heritability/aggregate.py \
        --root_dir /path/to/output/GTEX \
        --results_csv /path/to/all_h_results_gcta.csv \
        --output gtex_gcta_h.csv
"""

import argparse
import os

import pandas as pd


def extract_h(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("V(G)/Vp"):
                return line.split()[1]
    return None


def crawl_and_extract(root_dir):
    data = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".hsq"):
                file_path = os.path.join(dirpath, filename)
                vp_value = extract_h(file_path)
                if vp_value is not None:
                    data.append([dirpath, vp_value])
    return pd.DataFrame(data, columns=['path', 'h'])


def pivot_on_latent(df, results_csv):
    df[['chr/gene', 'latent']] = df['path'].str.extract(r'/GTEX/(\d+/\S+)/\S+_(\S+)')
    df_pivot = df.pivot(index='chr/gene', columns='latent', values='h')
    df_pivot.columns = [f'h_{col}' for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_all = pd.read_csv(results_csv)
    df_all['chr/gene'] = df_all['gene_dir'].str.extract(r'/gtex/(\d+/\S+)')

    return pd.merge(df_all, df_pivot, how='inner', on=['chr/gene'])


def main(root_dir, results_csv, output):
    df = crawl_and_extract(root_dir)
    df = pivot_on_latent(df, results_csv)
    df.to_csv(output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate GCTA heritability (.hsq) results into a CSV')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory to crawl for .hsq files')
    parser.add_argument('--results_csv', type=str, required=True, help='Existing results CSV to merge with')
    parser.add_argument('--output', type=str, default='gtex_gcta_h.csv', help='Output CSV path')
    args = parser.parse_args()
    main(args.root_dir, args.results_csv, args.output)
