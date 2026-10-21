#!/usr/bin/env python3
"""Per-section channel statistic t = sum_g z_g w_g / ||w|| for every trait and gene-filtering setting."""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

TAB = "\t"


def zsc(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def load_z(path):
    g = pd.read_csv(path, sep=r"\s+")
    g.columns = [c.upper() for c in g.columns]
    ns = "NSNPS" if "NSNPS" in g.columns else "NSNP"
    if "ZSTAT" not in g.columns or "GENE" not in g.columns:
        return None
    glen = (np.abs(pd.to_numeric(g["STOP"], errors="coerce")
                   - pd.to_numeric(g["START"], errors="coerce"))
            if "STOP" in g.columns and "START" in g.columns
            else pd.to_numeric(g[ns], errors="coerce") * 1000.0)
    out = pd.DataFrame({"z": pd.to_numeric(g["ZSTAT"], errors="coerce").values,
                        "nsnps": pd.to_numeric(g[ns], errors="coerce").values,
                        "glen": np.asarray(glen, float)},
                       index=g["GENE"].astype(str))
    return out[~out.index.duplicated()].dropna()


def cell_detection(pkl_dir, code, sample):
    import pickle
    from scipy import sparse as sp
    f = os.path.join(pkl_dir, "inputs_%s.pkl" % sample)
    if not os.path.exists(f):
        return None
    if code:
        sys.path.insert(0, code)
    d = pickle.load(open(f, "rb"))
    inp = d["inputs"]
    inp = inp.item() if isinstance(inp, np.ndarray) else inp
    cts = list(d["ct_categories"])
    ct = np.array(cts, dtype=object)[np.asarray(inp.cell_type).astype(int)]
    Y = inp.Y
    genes = [str(g) for g in inp.gene_names]
    out = {}
    for c in cts:
        m = (ct == c)
        if m.sum() == 0:
            out[c] = pd.Series(0, index=genes)
            continue
        sub = Y[m]
        out[c] = pd.Series(np.asarray((sub > 0).sum(0)).ravel()
                           if sp.issparse(sub) else (np.asarray(sub) > 0).sum(0),
                           index=genes)
    return pd.DataFrame(out).max(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gss_dir", required=True)
    ap.add_argument("--trait_manifest", required=True, help="two columns: label<TAB>path to .genes.out")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--min_channels", default="1,10,20")
    ap.add_argument("--min_cells", default="0")
    ap.add_argument("--pkl_dir", default="")
    ap.add_argument("--code", default="")
    ap.add_argument("--mode", default="paracrine")
    a = ap.parse_args()

    traits = {}
    for ln in open(a.trait_manifest):
        ln = ln.rstrip("\n")
        if not ln.strip():
            continue
        parts = ln.split(TAB)
        if len(parts) != 2:
            sys.exit("[rebuild] manifest lines must be label<TAB>path: %r" % ln)
        lab, p = parts[0].strip(), parts[1].strip()
        if lab in traits:
            sys.exit("[rebuild] duplicate trait label: %s" % lab)
        if not os.path.exists(p):
            sys.exit("[rebuild] %s: file not found: %s" % (lab, p))
        traits[lab] = p
    print("[rebuild] %d traits: %s" % (len(traits), sorted(traits)[:8]))

    zdfs = {}
    for lab, p in traits.items():
        z = load_z(p)
        if z is None:
            print("[rebuild] !! %s: GENE/ZSTAT missing, skipped" % lab)
            continue
        zdfs[lab] = z
    print("[rebuild] z loaded for %d traits" % len(zdfs))

    mchs = [int(x) for x in a.min_channels.split(",")]
    mcls = [int(x) for x in a.min_cells.split(",")] if a.pkl_dir else [0]
    files = sorted(glob.glob(os.path.join(a.gss_dir, "gss_spec_*.tsv")))
    if not files:
        sys.exit("[rebuild] no gss_spec_*.tsv in %s" % a.gss_dir)
    print("[rebuild] %d sections x %d min_channels x %d min_cells settings"
          % (len(files), len(mchs), len(mcls)))

    t0 = time.time()
    summary = []
    for fi, f in enumerate(files, 1):
        smp = re.sub(r"^gss_spec_|\.tsv$", "", os.path.basename(f))
        G = pd.read_csv(f, sep=TAB)
        G = G.rename(columns={G.columns[0]: "gene"}).set_index("gene")
        G.index = G.index.astype(str)
        cols = [c for c in G.columns if str(c).endswith(a.mode)]
        if not cols:
            print("  [%s] no %s channel, skipped" % (smp, a.mode))
            continue
        nzc = (G[cols] > 0).sum(axis=1)
        det = cell_detection(a.pkl_dir, a.code, smp) if a.pkl_dir else None
        if a.pkl_dir and det is None:
            print("  [%s] !! pkl not found; min_cells filter not applied" % smp)

        for mch in mchs:
            for mcl in mcls:
                keep = set(nzc[nzc >= mch].index)
                if det is not None and mcl > 0:
                    keep &= set(det.index[det >= mcl])
                outdir = os.path.join(a.out_root, "mch%d_mcl%d" % (mch, mcl))
                os.makedirs(outdir, exist_ok=True)

                recs = []
                n_kept_ref = 0
                for lab, zdf in zdfs.items():
                    u = [g for g in G.index if g in zdf.index]
                    if len(u) < 50:
                        continue
                    sub = G.loc[u, cols].to_numpy(float).copy()
                    mask = np.array([g in keep for g in u])
                    sub[~mask, :] = 0.0
                    n_kept_ref = max(n_kept_ref, int(mask.sum()))
                    gl = zdf.loc[u, "glen"].to_numpy(float)
                    ns = zdf.loc[u, "nsnps"].to_numpy(float)
                    X = np.column_stack([np.ones(len(u)),
                                         zsc(np.log(np.maximum(gl, 1))),
                                         zsc(np.log(np.maximum(ns, 1))),
                                         zsc(gl), zsc(ns)])
                    XP = np.linalg.pinv(X)
                    mu, sd = sub.mean(axis=0), sub.std(axis=0)
                    Gr = (sub - mu) / np.where(sd > 0, sd, 1.0)
                    Gr = Gr - X @ (XP @ Gr)
                    den = np.sqrt(np.einsum("ij,ij->j", Gr, Gr))
                    zv = zdf.loc[u, "z"].to_numpy(float)
                    zr = zsc(zv - X @ (XP @ zv))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        t = np.where(den > 0, (Gr.T @ zr) / np.where(den > 0, den, 1.0),
                                     np.nan)
                    ok = np.isfinite(t)
                    ens = t[ok].mean() if ok.any() else 0.0
                    tm = t - ens
                    for c, tv, tmv in zip(cols, t, tm):
                        recs.append(dict(trait=lab, channel=c, mode=a.mode,
                                         t=tv, p_vs_zero=np.nan, t_minus_ensemble=tmv,
                                         p_vs_ensemble=float(stats.norm.sf(tmv))))
                if recs:
                    pd.DataFrame(recs).to_csv(
                        os.path.join(outdir, "ch_%s_channels.tsv" % smp),
                        sep=TAB, index=False)
                    summary.append(dict(sample=smp, min_channels=mch, min_cells=mcl,
                                        n_genes_kept=n_kept_ref,
                                        n_traits=len(set(r["trait"] for r in recs))))
        el = time.time() - t0
        print("  [%2d/%d] %-10s done  %.0fs elapsed, about %.0fs left"
              % (fi, len(files), smp, el, el * (len(files) - fi) / fi))

    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(a.out_root, "rebuild_summary.tsv"), sep=TAB, index=False)
    print("\ngenes kept per setting (median over sections):")
    for (mch, mcl), g in S.groupby(["min_channels", "min_cells"]):
        print("  min_ch>=%-3d min_cell>=%-3d  %6.0f genes   -> %s"
              % (mch, mcl, g["n_genes_kept"].median(),
                 os.path.join(a.out_root, "mch%d_mcl%d" % (mch, mcl))))


if __name__ == "__main__":
    main()
