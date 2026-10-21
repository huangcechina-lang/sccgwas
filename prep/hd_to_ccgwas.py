#!/usr/bin/env python3
"""Annotated Visium HD cells -> h5ad with coordinates in micrometres for prepare_inputs.py."""
import argparse, json, os, sys, time
import numpy as np
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="cell .h5ad with obs['cell_type'] and obsm['spatial']")
    ap.add_argument("--scalefactors", required=True, help="Visium HD scalefactors_json.json")
    ap.add_argument("--out_h5ad", required=True)
    ap.add_argument("--min_counts", type=int, default=100,
                    help="drop cells below this UMI (Bin2cell can leave tiny fragments)")
    ap.add_argument("--min_cells_per_type", type=int, default=30,
                    help="drop cell types with fewer than this many cells")
    a = ap.parse_args()
    import scanpy as sc

    ad = sc.read_h5ad(a.cells)
    log(f"cells in: {ad.n_obs} x {ad.n_vars}")
    assert "spatial" in ad.obsm, "obsm['spatial'] missing (need Bin2cell output)"
    assert "cell_type" in ad.obs, "obs['cell_type'] missing"

    sf = json.load(open(a.scalefactors))
    mpp = sf.get("microns_per_pixel")
    if mpp is None:
        sys.exit("scalefactors has no 'microns_per_pixel'; cannot convert coords to microns.")
    coord_px = np.asarray(ad.obsm["spatial"], float)
    coord_um = coord_px * mpp
    ad.obsm["spatial"] = coord_um.astype(np.float32)
    rng = coord_um.max(0) - coord_um.min(0)
    log(f"pixels->microns (mpp={mpp:.4f}): tissue extent now {rng[0]/1000:.2f} x {rng[1]/1000:.2f} mm")

    if "counts" in ad.layers:
        ad.X = ad.layers["counts"]
    umi = np.asarray(ad.X.sum(1)).ravel()
    keep = umi >= a.min_counts
    log(f"drop cells <{a.min_counts} UMI: keep {int(keep.sum())}/{ad.n_obs} "
        f"(median UMI/cell = {np.median(umi[keep]):.0f})")
    ad = ad[keep].copy()
    vc = ad.obs["cell_type"].value_counts()
    rare = vc[vc < a.min_cells_per_type].index.tolist()
    if rare:
        log(f"drop rare cell types (<{a.min_cells_per_type} cells): {rare}")
        ad = ad[~ad.obs["cell_type"].isin(rare)].copy()

    try:
        from sklearn.neighbors import NearestNeighbors
        d = NearestNeighbors(n_neighbors=2).fit(ad.obsm["spatial"]).kneighbors(ad.obsm["spatial"])[0][:, 1]
        log(f"nearest-neighbour spacing: median {np.median(d):.1f} um "
            f"(juxtacrine 15um is {'resolvable' if np.median(d) < 15 else 'BELOW spacing -> exclude juxtacrine'})")
    except Exception:
        pass

    print(f"\n=== final ===")
    print(ad.obs["cell_type"].value_counts().to_string())
    os.makedirs(os.path.dirname(a.out_h5ad) or ".", exist_ok=True)
    ad.write_h5ad(a.out_h5ad)
    log(f"wrote {a.out_h5ad} (coords in MICRONS, raw counts in X, obs['cell_type'] set)")
    print("\nNext step (see README, step 1):")
    print(f"  python sccgwas/prepare_inputs.py --spatial {a.out_h5ad} \\")
    print(f"      --magma <trait>.genes.out --lr_db resources/LR_pairs_cellchat_omnipath.tsv --disease <trait> \\")
    print(f"      --output pkl/inputs_{os.path.splitext(os.path.basename(a.out_h5ad))[0].replace('_ready', '')}.pkl")


if __name__ == "__main__":
    main()
