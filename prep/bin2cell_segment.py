#!/usr/bin/env python3
"""Visium HD 2 um bins -> segmented cells with Bin2cell (requires bin2cell, tifffile)."""
import argparse, os, sys, time
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr_dir", required=True)
    ap.add_argument("--he_image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mpp", type=float, default=0.5,
                    help="microns per pixel of the scaled H&E image passed to StarDist")
    ap.add_argument("--prob_thresh", type=float, default=0.01)
    ap.add_argument("--stardist_dir", default="", help="scratch dir for stardist tiffs/npz")
    ap.add_argument("--min_counts", type=int, default=1)
    a = ap.parse_args()

    try:
        import numpy as np, scanpy as sc, bin2cell as b2c
        log(f"bin2cell {getattr(b2c,'__version__','?')} | scanpy {sc.__version__}")
    except Exception as e:
        sys.exit(f"FATAL: bin2cell env not active? {e}\n  conda env create -f env_bin2cell.yml; conda activate bin2cell")
    p2um = os.path.join(a.sr_dir, "binned_outputs", "square_002um")
    if not os.path.isdir(p2um):
        p2um = a.sr_dir if os.path.isdir(os.path.join(a.sr_dir, "spatial")) else p2um
    for need in [os.path.join(p2um, "spatial"), a.he_image]:
        if not os.path.exists(need): sys.exit(f"FATAL missing: {need}")
    stardist_dir = a.stardist_dir or os.path.join(os.path.dirname(a.out), "stardist_tmp")
    os.makedirs(stardist_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    log("1/6 loading 2um bins + H&E")
    adata = b2c.read_visium(p2um, source_image_path=a.he_image)
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_counts=a.min_counts)
    log(f"    {adata.n_obs} bins x {adata.n_vars} genes")

    log("2/6 destripe")
    b2c.destripe(adata)

    log("3/6 StarDist nuclei segmentation on H&E")
    he_tiff = os.path.join(stardist_dir, "he.tiff")
    he_npz = os.path.join(stardist_dir, "he.npz")
    b2c.scaled_he_image(adata, mpp=a.mpp, save_path=he_tiff)
    try:
        import tifffile
        _si = tifffile.imread(he_tiff)
        import numpy as _np
        log(f"    scaled H&E for StarDist: shape={_si.shape} dtype={_si.dtype} "
            f"min={_np.nanmin(_si):.0f} max={_np.nanmax(_si):.0f} NaN={int((~_np.isfinite(_si)).sum())}")
        if _si.dtype != _np.uint8 or not _np.isfinite(_si).all():
            log("    [!!] scaled image is not clean uint8 -> StarDist will find nothing. "
                "Check mpp (use the true microns_per_pixel) / image size.")
    except Exception as _e:
        log(f"    [scaled-image check skipped] {_e}")
    b2c.stardist(image_path=he_tiff, labels_npz_path=he_npz,
                 stardist_model="2D_versatile_he", prob_thresh=a.prob_thresh)
    b2c.insert_labels(adata, labels_npz_path=he_npz, basis="spatial",
                      spatial_key="spatial_cropped_150_buffer", mpp=a.mpp,
                      labels_key="labels_he")
    n_nuc = int((adata.obs["labels_he"] > 0).sum())
    log(f"    nuclei-labelled bins: {n_nuc}")
    if n_nuc == 0:
        sys.exit("FATAL: StarDist found 0 nuclei. Fixes, in order:\n"
                 "  (1) pass the TRUE mpp (scalefactors microns_per_pixel), not a guess;\n"
                 "  (2) raise prob_thresh toward the model default 0.69 (0.01 can over-suppress\n"
                 "      on a huge image where the scaled crop is near-blank);\n"
                 "  (3) run inspect_he_image.py and open the thumbnail to confirm real H&E;\n"
                 "  (4) if the scaled-image check above flagged non-uint8/NaN, the mpp/size made\n"
                 "      the scaled image overflow -> reduce with a correct mpp.")

    log("4/6 expand nuclei to cells")
    b2c.expand_labels(adata, labels_key="labels_he",
                      expanded_labels_key="labels_he_expanded")

    log("5/6 bin_to_cell (sum real counts per cell)")
    cdata = b2c.bin_to_cell(adata, labels_key="labels_he_expanded",
                            spatial_keys=["spatial"])
    log(f"    reconstructed cells: {cdata.n_obs}")

    import numpy as np
    umi = np.asarray(cdata.X.sum(1)).ravel()
    gpc = np.asarray((cdata.X > 0).sum(1)).ravel()
    print(f"\n=== per-CELL depth after Bin2cell ===")
    print(f"  cells={cdata.n_obs}  median UMI/cell={np.median(umi):.0f}  "
          f"median genes/cell={np.median(gpc):.0f}")
    if np.median(umi) < 300:
        print("  [warn] per-cell depth still low; check H&E mpp / prob_thresh, or the sample depth.")

    cdata.write_h5ad(a.out)
    log(f"6/6 wrote {a.out}")
    print(f"\nNEXT: cell-type annotation, then build the CCGWAS pkl:")
    print(f"  # annotate cell types (CellTypist with the matched Chromium reference), then:")
    print(f"  python 01_prepare_inputs.py --spatial {a.out} \\")
    print(f"      --magma <GWAS.genes.out> --lr_db <cellchat_omnipath.tsv> --output <sample>.pkl")


if __name__ == "__main__":
    main()
