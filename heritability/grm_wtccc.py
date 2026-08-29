"""
Build phenotype files, GRM, and GCTA binaries from WTCCC VAE latent representations.

Reads the latent means written by vae/train_wtccc.py -- a CSV whose first column
is the individual IID followed by the latent dimensions -- standardizes each
latent dimension (column) to mean 0 / unit variance, computes the genetic
relationship matrix (GRM = (1/d) * Z @ Z.T, d = number of latent dimensions),
serializes it to GCTA binary format (.grm.bin, .grm.N.bin, .grm.id), and writes
a case/control phenotype file (.phen). The output directory is ready for gcta.py
or a direct GCTA --reml call.

Because the IID travels in the latent CSV, the phenotype and .grm.id join to the
GRM rows by ID (not row position). Case/control status is inferred from the ID
prefix: IIDs starting with the disease name are cases, shared controls (e.g. 58C,
NBS) are controls. In WTCCC the FID and IID are identical.

Example usage:
    python heritability/grm_wtccc.py --disease BD \\
        --output_dir /path/to/output/WTCCC/BD/BD_128
"""

import argparse
import os

import numpy as np
import pandas as pd


def load_latents(output_dir, disease):
    """Load the latent CSV, returning (iid Series, mu array).

    The CSV is written by vae/train_wtccc.py with an 'IID' first column followed
    by the latent dimensions, so individuals stay joined to the GRM rows by ID.
    """
    path = os.path.join(output_dir, f"{disease}_latent_representations.csv")
    df = pd.read_csv(path)
    if "IID" not in df.columns:
        raise ValueError(
            f"{path} has no 'IID' column. Re-run vae/train_wtccc.py to embed IDs."
        )
    iid = df["IID"].astype(str)
    mu = df.drop(columns=["IID"]).to_numpy(dtype=np.float64)
    return iid, mu


def make_phenotype_file(disease, output_dir, iid):
    """Write a GCTA-format .phen (FID, IID, case/control) in latent row order.

    Case/control is inferred from the ID prefix: IIDs starting with the disease
    name are cases (1); shared controls (e.g. 58C, NBS) are 0. WTCCC FID == IID.
    """
    print("Writing phenotype file...")
    data = pd.DataFrame({"FID": iid, "IID": iid})
    data["Phenotype"] = data["IID"].str.startswith(disease).astype(int)

    os.makedirs(output_dir, exist_ok=True)
    pheno_path = os.path.join(output_dir, f"{disease}.phen")
    data.to_csv(pheno_path, sep="\t", index=False, header=False)

    n_case = int(data["Phenotype"].sum())
    print(f"Wrote {pheno_path}: {len(data)} individuals "
          f"({n_case} cases, {len(data) - n_case} controls)")
    return pheno_path


def calculate_grm(E):
    """Compute the latent GRM.

    Each latent dimension (column) is standardized to mean 0 and unit variance,
    then GRM = (1/d) * Z @ Z.T, where d is the number of latent dimensions. This
    yields a positive semi-definite matrix with diagonal ~= 1 -- the direct
    analogue of a standardized-genotype GRM with features = latent dimensions.
    """
    mu_mean = E.mean(axis=0, keepdims=True)
    mu_std = E.std(axis=0, keepdims=True)
    mu_std[mu_std == 0] = 1.0  # guard against zero-variance (collapsed) latent dims
    Z = (E - mu_mean) / mu_std

    d = Z.shape[1]
    return (1.0 / d) * Z @ Z.T


def make_grm_and_binaries(disease, output_dir, iid, mu):
    """Compute the GRM and serialize to GCTA binary format.

    The .grm.id is written from the same IID column, so it stays aligned with
    both the GRM matrix rows and the phenotype file.
    """
    print("Computing GRM...")
    grm = calculate_grm(mu)

    print("Writing GCTA .grm.id file...")
    with open(os.path.join(output_dir, f"{disease}.grm.id"), "w") as f:
        for x in iid:
            f.write(f"{x}\t{x}\n")

    print("Extracting lower triangle and writing GCTA binaries...")
    i_indices, j_indices = np.tril_indices(grm.shape[0])
    lower_tri_grm = grm[i_indices, j_indices].astype(np.float32)

    with open(os.path.join(output_dir, f"{disease}.grm.bin"), "wb") as f:
        f.write(lower_tri_grm.tobytes())

    n_arr = np.ones(len(lower_tri_grm), dtype=np.int32)
    with open(os.path.join(output_dir, f"{disease}.grm.N.bin"), "wb") as f:
        f.write(n_arr.tobytes())

    print("Done!")


def main(disease, output_dir):
    iid, mu = load_latents(output_dir, disease)
    assert len(iid) == len(mu)
    make_phenotype_file(disease, output_dir, iid)
    make_grm_and_binaries(disease, output_dir, iid, mu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build GRM, phenotype, and GCTA binaries from WTCCC VAE latent representations"
    )
    parser.add_argument("--disease", type=str, required=True, help="Disease name (e.g. BD)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory with {disease}_latent_representations.csv; outputs written here")
    args = parser.parse_args()
    main(args.disease, args.output_dir)
