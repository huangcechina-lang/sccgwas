#!/usr/bin/env python3
"""Stereo-seq cellbin GEF -> h5ad (requires stereopy)."""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import stereo as st
except ImportError:
    print("ERROR: stereopy not installed. Install via: pip install stereopy")
    sys.exit(1)

try:
    import anndata as ad
    import scanpy as sc
except ImportError:
    print("ERROR: anndata/scanpy required. Install via: pip install scanpy anndata")
    sys.exit(1)


def gef_to_adata(
    gef_path: str,
    sample_id: str | None = None,
    min_counts: int = 100,
    min_genes: int = 50,
) -> ad.AnnData:
    print(f"\n[gef->h5ad] reading {gef_path}")
    sample_id = sample_id or Path(gef_path).stem.replace(".cellbin", "")

    data = st.io.read_gef(
        file_path=gef_path,
        is_sparse=True,
        bin_type="cell_bins",
    )
    print(f"  raw: {data.exp_matrix.shape}")

    data.tl.raw_checkpoint()
    adata = st.io.stereo_to_anndata(
        data,
        flavor="scanpy",
        sample_id=sample_id,
        output=None,
    )

    if adata.X is None:
        raise ValueError(f"Empty expression matrix from {gef_path}")

    if "spatial" not in adata.obsm:
        if {"x", "y"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["x", "y"]].values.astype(np.float32)
        elif {"x_array", "y_array"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["x_array", "y_array"]].values.astype(np.float32)
        else:
            raise ValueError(
                f"No spatial coordinates found. obs columns: {list(adata.obs.columns)}"
            )
    if "x" not in adata.obs.columns:
        adata.obs["x"] = adata.obsm["spatial"][:, 0]
        adata.obs["y"] = adata.obsm["spatial"][:, 1]

    adata.obs["sample"] = sample_id

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    n_after = adata.n_obs
    print(f"  filtered: {n_before} -> {n_after} cells "
          f"(min_counts={min_counts}, min_genes={min_genes})")

    coord = adata.obsm["spatial"]
    print(f"  coord range: x={coord[:,0].max()-coord[:,0].min():.0f}, "
          f"y={coord[:,1].max()-coord[:,1].min():.0f} (units depend on chip)")

    total_counts = adata.X.sum(axis=1)
    if hasattr(total_counts, 'A1'):
        total_counts = total_counts.A1
    total_counts = np.asarray(total_counts).flatten()
    print(f"  counts/cell: median={np.median(total_counts):.0f}, "
          f"max={total_counts.max():.0f}")

    return adata


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", help="path to single .cellbin.gef file")
    grp.add_argument("--input_dir", help="directory containing .cellbin.gef files")

    ap.add_argument("--output", help="output .h5ad path (for --input)")
    ap.add_argument("--output_dir", help="output directory (for --input_dir)")
    ap.add_argument("--sample_id", help="sample tag (for --input)")
    ap.add_argument("--pattern", default="*.cellbin.gef",
                    help="glob pattern for --input_dir")
    ap.add_argument("--min_counts", type=int, default=100)
    ap.add_argument("--min_genes", type=int, default=50)
    ap.add_argument("--merge", help="optional: also save a merged h5ad")
    ap.add_argument("--metadata", default=None,
                    help="optional sample-level metadata TSV with columns: sample_id, condition, ...")
    args = ap.parse_args()

    if args.input:
        if not args.output:
            args.output = args.input.replace(".cellbin.gef", ".h5ad")
        adata = gef_to_adata(args.input, sample_id=args.sample_id,
                              min_counts=args.min_counts, min_genes=args.min_genes)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(args.output)
        print(f"\n[done] wrote {args.output} ({adata.n_obs} cells x {adata.n_vars} genes)")
        return

    out_dir = Path(args.output_dir or "h5ads")
    out_dir.mkdir(parents=True, exist_ok=True)
    gefs = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not gefs:
        print(f"No .gef files matching {args.pattern} in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(gefs)} .gef files")

    metadata = None
    if args.metadata and Path(args.metadata).exists():
        metadata = pd.read_csv(args.metadata, sep="\t").set_index("sample_id")
        print(f"  loaded metadata for {len(metadata)} samples")

    adatas = []
    for gef in gefs:
        stem = Path(gef).stem.replace(".cellbin", "")
        if "_" in stem:
            sample_id = stem.split("_")[-1]
        else:
            sample_id = stem
        try:
            adata = gef_to_adata(gef, sample_id=sample_id,
                                  min_counts=args.min_counts,
                                  min_genes=args.min_genes)
        except Exception as e:
            print(f"  FAILED: {gef}: {e}")
            continue

        if metadata is not None and sample_id in metadata.index:
            for col in metadata.columns:
                adata.obs[col] = metadata.loc[sample_id, col]

        out_path = out_dir / f"{stem}.h5ad"
        adata.write_h5ad(out_path)
        print(f"  -> {out_path}")
        adatas.append(adata)

    print(f"\n[batch done] {len(adatas)} samples converted to {out_dir}")

    if args.merge and adatas:
        print(f"\n[merge] concatenating {len(adatas)} samples...")
        max_extent = max(
            (a.obsm["spatial"][:, 0].max() - a.obsm["spatial"][:, 0].min())
            for a in adatas
        )
        for i, a in enumerate(adatas):
            shift = i * (max_extent + 5000)
            a.obsm["spatial"][:, 0] += shift
            a.obs["x"] = a.obsm["spatial"][:, 0]
        merged = ad.concat(adatas, axis=0, join="inner", merge="same",
                            label="sample", index_unique="-")
        merged_path = Path(args.merge)
        merged.write_h5ad(merged_path)
        print(f"  merged: {merged_path} ({merged.n_obs} cells x {merged.n_vars} genes)")
        print(f"  NOTE: coords are shifted per-sample to avoid spatial overlap")
        print(f"        DO NOT use merged for spatial analysis across samples;")
        print(f"        use per-sample h5ad files for SCCGWAS.")


if __name__ == "__main__":
    main()
