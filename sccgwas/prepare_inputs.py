#!/usr/bin/env python3
"""Build inputs_<section>.pkl from a section h5ad, the ligand-receptor resource and a MAGMA .genes.out."""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scanpy as sc
    import anndata as ad
except ImportError:
    print("ERROR: scanpy and anndata required. Install: pip install scanpy anndata")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from ccgwas_v4core import CCGWASInputs


def read_ct_order(spec):
    """Comma-separated names, or a file with one name per line."""
    if not spec:
        return None
    if os.path.isfile(spec):
        names = [l.strip() for l in open(spec) if l.strip()]
    else:
        names = [x.strip() for x in spec.split(",") if x.strip()]
    if len(set(names)) != len(names):
        raise ValueError("--ct_order lists a cell type twice: %s" % names)
    return names


def load_spatial(path: str, subsample: int | None = None, seed: int = 42, ct_order=None):
    print(f"[load] Spatial: {path}")
    adata = sc.read_h5ad(path)
    print(f"  shape: {adata.shape}, obs cols: {list(adata.obs.columns)[:8]}...")

    if "cell_type" not in adata.obs.columns:
        for alt in ["celltype", "CellType", "cluster", "leiden", "annotation"]:
            if alt in adata.obs.columns:
                print(f"  using '{alt}' as cell_type (rename if wrong)")
                adata.obs["cell_type"] = adata.obs[alt]
                break
        else:
            raise ValueError("No cell_type column found. Required obs columns: cell_type")

    if "spatial" in adata.obsm:
        coord = adata.obsm["spatial"]
    elif {"x", "y"}.issubset(adata.obs.columns):
        coord = adata.obs[["x", "y"]].values
    else:
        raise ValueError("No spatial coordinates found. Required: adata.obsm['spatial'] OR obs['x','y']")
    coord = np.asarray(coord, dtype=np.float32)
    coord_range = coord.max(axis=0) - coord.min(axis=0)
    print(f"  spatial range: x={coord_range[0]:.0f}, y={coord_range[1]:.0f}")
    if coord_range.max() < 100:
        print("  WARNING: spatial range < 100; may not be in um. Check coord units.")

    Y = adata.X
    if hasattr(Y, "toarray"):
        Y = Y.toarray()
    Y = np.asarray(Y, dtype=np.float32)
    if Y.max() < 10 or not np.allclose(Y, Y.astype(int)):
        print("  WARNING: Y may not be raw counts (max<10 or non-integer). raw counts are expected.")

    if subsample is not None and adata.n_obs > subsample:
        print(f"  subsampling {subsample} of {adata.n_obs} cells (seed={seed})")
        rng = np.random.default_rng(seed)
        keep = rng.choice(adata.n_obs, size=subsample, replace=False)
        adata = adata[keep].copy()
        Y = Y[keep]
        coord = coord[keep]

    ct_str = adata.obs["cell_type"].astype(str).values
    if ct_order is not None:
        extra = sorted(set(ct_str) - set(ct_order))
        if extra:
            raise ValueError("cell types not in --ct_order: %s (remove these cells or add the names)" % extra)
        ct_categorical = pd.Categorical(ct_str, categories=ct_order)
    else:
        ct_categorical = pd.Categorical(ct_str)
        print("  WARNING: no --ct_order; codes follow the cell types present in this section. "
              "Give every section of a tissue the same --ct_order.")
    ct_codes = ct_categorical.codes.astype(int)
    print(f"  cell types: {len(ct_categorical.categories)} CTs, "
          f"counts: {pd.Series(ct_codes).value_counts().head(5).to_dict()}")

    gene_names = adata.var_names.values.astype(str)
    return Y, ct_codes, coord, gene_names, ct_categorical.categories.tolist()


def load_magma(path: str, gene_names: np.ndarray, species: str = "human",
               ortholog_map_path: str | None = None,
               gene_id_map_path: str | None = None):
    print(f"[load] MAGMA: {path}")
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    print(f"  shape: {df.shape}, columns: {list(df.columns)}")

    cols = {c.upper(): c for c in df.columns}
    if "ZSTAT" not in cols:
        raise ValueError(f"No ZSTAT column. Found: {list(df.columns)}")
    if "GENE" not in cols:
        raise ValueError(f"No GENE column. Found: {list(df.columns)}")
    if "CHR" not in cols:
        print("  WARN: no CHR column, using fake chromosome=1 for all genes")
        df["CHR"] = 1
        cols["CHR"] = "CHR"

    df = df.rename(columns={cols["GENE"]: "GENE", cols["CHR"]: "CHR",
                            cols["ZSTAT"]: "ZSTAT"})
    has_pos = "START" in cols and "STOP" in cols
    has_nsnps = "NSNPS" in cols
    if has_pos:
        df["GENE_SIZE"] = (df[cols["STOP"]].astype(float)
                           - df[cols["START"]].astype(float)).clip(lower=1)
    if has_nsnps:
        df["NSNPS_"] = df[cols["NSNPS"]].astype(float).clip(lower=1)

    if gene_id_map_path is not None:
        idmap = pd.read_csv(gene_id_map_path, sep=None, engine="python", header=None,
                            dtype=str)
        id2sym = dict(zip(idmap.iloc[:, 0].astype(str), idmap.iloc[:, -1].astype(str)))
        n_before = df["GENE"].isin(id2sym).sum()
        df["GENE"] = df["GENE"].astype(str).map(id2sym).fillna(df["GENE"].astype(str))
        print(f"  applied gene_id_map: {n_before}/{len(df)} MAGMA IDs -> symbols")

    if ortholog_map_path is not None:
        ortho = pd.read_csv(ortholog_map_path, sep="\t")
        print(f"  applying ortholog map: {ortholog_map_path}")
        ortho_map = dict(zip(ortho["source_symbol"], ortho["target_symbol"]))
        df["GENE"] = df["GENE"].map(ortho_map).fillna(df["GENE"])

    gene_to_z = dict(zip(df["GENE"].astype(str), df["ZSTAT"].astype(float)))
    gene_to_chr = dict(zip(df["GENE"].astype(str), df["CHR"]))

    z = np.array([gene_to_z.get(str(g), 0.0) for g in gene_names], dtype=np.float32)
    chrom = np.array([gene_to_chr.get(str(g), 0) for g in gene_names])
    chrom_map = {"X": 23, "Y": 24, "MT": 25, "M": 25}
    chrom = np.array([
        chrom_map.get(str(c).upper(), c) if isinstance(c, str) or c == c else 0
        for c in chrom
    ])
    try:
        chrom = chrom.astype(int)
    except Exception:
        chrom = np.array([int(c) if str(c).isdigit() else 0 for c in chrom])

    matched = (z != 0).sum()
    print(f"  matched {matched}/{len(gene_names)} ({100*matched/len(gene_names):.1f}%) "
          f"of spatial genes to MAGMA z")
    if matched < len(gene_names) * 0.3:
        print("  WARN: low match rate. Check species + gene symbol consistency.")

    magma_feat = None
    if has_pos or has_nsnps:
        gene_to_size = dict(zip(df["GENE"].astype(str), df.get("GENE_SIZE", pd.Series(dtype=float))))
        gene_to_nsnps = dict(zip(df["GENE"].astype(str), df.get("NSNPS_", pd.Series(dtype=float))))
        log_size = np.array([np.log(gene_to_size.get(str(g), np.nan)) if has_pos else np.nan
                             for g in gene_names], dtype=np.float32)
        log_snp = np.array([np.log(gene_to_nsnps.get(str(g), np.nan)) if has_nsnps else np.nan
                            for g in gene_names], dtype=np.float32)
        for arr in (log_size, log_snp):
            m = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
            arr[~np.isfinite(arr)] = m
        magma_feat = {"log_gene_size": log_size, "log_snp_density": log_snp}
        print(f"  derived Dim 8 from MAGMA: "
              f"{'gene_size ' if has_pos else ''}{'snp_density' if has_nsnps else ''}")

    return z.astype(np.float32), chrom.astype(int), magma_feat


def load_lr_db(path: str, gene_names: np.ndarray):
    print(f"[load] LR DB: {path}")
    df = pd.read_csv(path, sep="\t")
    print(f"  shape: {df.shape}, columns: {list(df.columns)}")

    required = {"ligand_symbol", "receptor_symbol", "mode"}
    missing = required - set(df.columns)
    if missing:
        col_map = {
            "ligand": "ligand_symbol", "Ligand": "ligand_symbol", "L": "ligand_symbol",
            "receptor": "receptor_symbol", "Receptor": "receptor_symbol", "R": "receptor_symbol",
            "annotation": "mode", "type": "mode", "interaction_type": "mode",
        }
        df = df.rename(columns=col_map)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required LR DB columns: {missing}")

    mode_map = {
        "cell-cell contact": "juxtacrine", "contact": "juxtacrine",
        "secreted signaling": "paracrine", "secreted signalling": "paracrine",
        "ecm-receptor": "secreted", "ecm": "secreted",
        "juxtacrine": "juxtacrine", "paracrine": "paracrine",
        "secreted": "paracrine",
    }
    unknown = {}

    def map_mode(m):
        first = str(m).split(";")[0].strip().lower()
        if first not in mode_map:
            unknown[first] = unknown.get(first, 0) + 1
        return mode_map.get(first, "paracrine")

    before_counts = df["mode"].astype(str).str.lower().value_counts().to_dict()
    df["mode"] = df["mode"].map(map_mode)
    if unknown:
        print(f"  !! {sum(unknown.values())} row(s) had an unrecognised mode and were "
              f"defaulted to paracrine: {dict(sorted(unknown.items(), key=lambda kv: -kv[1])[:8])}")
        print(f"     Input mode counts were {before_counts}. If a mode disappeared between "
              f"that and the line below, this is why.")

    def split_complex(s):
        return [g.strip().upper() for g in str(s).split("_") if g.strip()]

    expanded = []
    for _, r in df.iterrows():
        for L in split_complex(r["ligand_symbol"]):
            for R in split_complex(r["receptor_symbol"]):
                expanded.append({"ligand_symbol": L, "receptor_symbol": R,
                                 "mode": r["mode"]})
    df = pd.DataFrame(expanded).drop_duplicates().reset_index(drop=True)
    print(f"  after complex-split + mode-map: {len(df)} single-gene LR pairs")
    print(f"  mode counts: {df['mode'].value_counts().to_dict()}")

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    df["ligand_idx"] = df["ligand_symbol"].map(gene_idx)
    df["receptor_idx"] = df["receptor_symbol"].map(gene_idx)
    before = len(df)
    df = df.dropna(subset=["ligand_idx", "receptor_idx"])
    df["ligand_idx"] = df["ligand_idx"].astype(int)
    df["receptor_idx"] = df["receptor_idx"].astype(int)
    after = len(df)
    print(f"  kept {after}/{before} LR pairs with both genes in spatial data")
    print(f"  mode counts: {df['mode'].value_counts().to_dict()}")

    return df[["ligand_idx", "receptor_idx", "mode",
               "ligand_symbol", "receptor_symbol"]].reset_index(drop=True)


def load_gene_features(path: str | None, gene_names: np.ndarray, magma_feat=None):
    n = len(gene_names)
    feat = np.zeros((n, 3), dtype=np.float32)

    if magma_feat is not None:
        if "log_gene_size" in magma_feat:
            feat[:, 0] = magma_feat["log_gene_size"]
        if "log_snp_density" in magma_feat:
            feat[:, 1] = magma_feat["log_snp_density"]
        print(f"[gene features] Dim 8 from MAGMA: log_gene_size + log_snp_density "
              f"(ld_score = 0 constant unless external file given)")

    if path is not None and Path(path).exists():
        print(f"[gene features] merging external file: {path}")
        df = pd.read_csv(path, sep="\t")
        if "gene_symbol" in df.columns:
            df = df.set_index("gene_symbol")
            for i, g in enumerate(gene_names):
                if g in df.index:
                    row = df.loc[g]
                    if "log_gene_size" in df.columns and np.isfinite(row.get("log_gene_size", np.nan)):
                        feat[i, 0] = row["log_gene_size"]
                    if "log_snp_density" in df.columns and np.isfinite(row.get("log_snp_density", np.nan)):
                        feat[i, 1] = row["log_snp_density"]
                    if "ld_score" in df.columns and np.isfinite(row.get("ld_score", np.nan)):
                        feat[i, 2] = row["ld_score"]
            print(f"  merged features for genes present in file")
        else:
            print(f"  WARN: no 'gene_symbol' column; skipping external file")

    for c in (0, 1):
        col = feat[:, c]
        nz = col != 0
        if nz.any():
            feat[~nz, c] = np.median(col[nz])

    if magma_feat is None and (path is None or not Path(path).exists()):
        print(f"[gene features] WARNING: no MAGMA position/NSNPS columns AND no external "
              f"file -> Dim 8 is all-zero (matching on Dim 8 disabled). "
              f"Provide a MAGMA .genes.out with START/STOP/NSNPS for best results.")
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial", required=True, help="path to spatial .h5ad")
    ap.add_argument("--magma", required=True, help="path to MAGMA .genes.out")
    ap.add_argument("--lr_db", required=True, help="path to LR DB TSV")
    ap.add_argument("--gene_features", default=None,
                    help="path to gene features TSV (optional)")
    ap.add_argument("--ortholog_map", default=None,
                    help="path to ortholog mapping TSV (optional, for cross-species)")
    ap.add_argument("--gene_id_map", default=None,
                    help="two columns: MAGMA gene ID <tab> gene symbol")
    ap.add_argument("--output", required=True, help="output .pkl file")
    ap.add_argument("--disease", required=True, help="disease name tag (e.g., AD)")
    ap.add_argument("--species", default="human", choices=["human", "mouse"])
    ap.add_argument("--ct_order", default="",
                    help="cell types in code order (comma list or file); the same for every section")
    ap.add_argument("--no_gene_features", action="store_true",
                    help="do not use gene length and variant count for matching")
    ap.add_argument("--subsample_cells", type=int, default=None,
                    help="randomly subsample to this many cells (for fast first pass)")
    args = ap.parse_args()

    Y, ct_codes, coord, gene_names, ct_categories = load_spatial(
        args.spatial, subsample=args.subsample_cells, ct_order=read_ct_order(args.ct_order),
    )
    print(f"  -> N={len(Y)}, G={Y.shape[1]}, C={len(ct_categories)}")

    z_gwas, chromosome, magma_feat = load_magma(
        args.magma, gene_names, species=args.species,
        ortholog_map_path=args.ortholog_map,
        gene_id_map_path=args.gene_id_map,
    )

    lr_df = load_lr_db(args.lr_db, gene_names)

    if args.no_gene_features:
        magma_feat = None
        print("[gene features] --no_gene_features: gene length and variant count not used for matching")
    gene_features = load_gene_features(args.gene_features, gene_names, magma_feat=magma_feat)

    ligand_set = set(lr_df["ligand_symbol"])
    receptor_set = set(lr_df["receptor_symbol"])
    is_ligand = np.array([g in ligand_set for g in gene_names], dtype=bool)
    is_receptor = np.array([g in receptor_set for g in gene_names], dtype=bool)
    print(f"  ligands: {is_ligand.sum()}, receptors: {is_receptor.sum()}")

    known_LR_set = set()
    for _, row in lr_df.iterrows():
        known_LR_set.add((int(row["ligand_idx"]), int(row["receptor_idx"])))
    print(f"  known LR set size: {len(known_LR_set)}")

    inputs = CCGWASInputs(
        Y=Y, cell_type=ct_codes, coord=coord,
        z_gwas=z_gwas, gene_names=gene_names,
        LR_pairs=lr_df[["ligand_idx", "receptor_idx", "mode"]],
        gene_features=gene_features,
        is_ligand=is_ligand, is_receptor=is_receptor,
        known_LR_set=known_LR_set,
        chromosome=chromosome,
        gene_gene_corr=None,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "inputs": inputs,
            "ct_categories": ct_categories,
            "lr_df_full": lr_df,
            "disease": args.disease,
        }, f)
    print(f"\n[done] Saved CCGWASInputs to {out_path}")
    print(f"  N={inputs.n_cells}, G={inputs.n_genes}, C={inputs.n_cell_types}, "
          f"K_LR={len(inputs.LR_pairs)}, modes={inputs.modes}")


if __name__ == "__main__":
    main()
