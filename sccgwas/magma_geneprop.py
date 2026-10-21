#!/usr/bin/env python3
"""Channel-level genetic axis: MAGMA gene-property regression on channel specificity, one run per trait."""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

TAB = "\t"


def read_out(path):
    d = pd.read_csv(path, sep=r"\s+")
    d.columns = [c.upper() for c in d.columns]
    return d


def build_map(raw_out, sym_out, trait, tol_frac=0.98):
    A = read_out(raw_out)
    B = read_out(sym_out)
    for D, nm in ((A, "raw-side"), (B, "symbol-side")):
        if "GENE" not in D.columns:
            sys.exit("[gp] %s: no GENE column in the %s file: %s" % (trait, nm, list(D.columns)))
    if len(A) != len(B):
        print("  [%s] !! row counts differ (raw %d, symbol %d); matching by position"
              % (trait, len(A), len(B)))
        key = [c for c in ("CHR", "START", "STOP") if c in A.columns and c in B.columns]
        if len(key) < 3:
            return None, "row counts differ and CHR/START/STOP missing"
        M = A[key + ["GENE"]].merge(B[key + ["GENE"]], on=key,
                                    suffixes=("_raw", "_sym"))
        frac = len(M) / min(len(A), len(B))
        if frac < tol_frac:
            return None, "position match covers only %.1f%%" % (100 * frac)
        return dict(zip(M.GENE_raw.astype(str), M.GENE_sym.astype(str))), \
            "matched by position: %d (%.1f%%)" % (len(M), 100 * frac)

    ok = True
    if {"CHR", "START", "STOP"} <= set(A.columns) & set(B.columns):
        same = ((A.CHR.values == B.CHR.values) &
                (A.START.values == B.START.values) &
                (A.STOP.values == B.STOP.values))
        frac = float(same.mean())
        ok = frac >= tol_frac
        note = "row order, positions agree %.1f%%" % (100 * frac)
        if not ok:
            return None, "row order mismatch (positions agree %.1f%%)" % (100 * frac)
    else:
        note = "row order (no position columns to check)"
    if "ZSTAT" in A.columns and "ZSTAT" in B.columns:
        za, zb = A.ZSTAT.to_numpy(float), B.ZSTAT.to_numpy(float)
        m = np.isfinite(za) & np.isfinite(zb)
        if m.sum() and not np.allclose(za[m], zb[m], atol=1e-4):
            r = float(np.corrcoef(za[m], zb[m])[0, 1])
            if r < 0.999:
                return None, "ZSTAT disagree (r=%.4f): files from different runs" % r
            note += ", ZSTAT r=%.4f" % r
        else:
            note += ", ZSTAT identical"
    return dict(zip(A.GENE.astype(str), B.GENE.astype(str))), note


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gss", required=True, help="averaged GSS (average_gss.py) or one gss_spec_*.tsv")
    ap.add_argument("--raw_dir", required=True, help="directory with <trait>.genes.raw")
    ap.add_argument("--map", required=True, help="report written by collect_genes_raw.py")
    ap.add_argument("--manifest", required=True, help="two columns: label<TAB>symbol .genes.out")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--magma", default="magma", help="MAGMA executable")
    ap.add_argument("--covars", default="detect_breadth,gss_total",
                    help="conditioning covariates; if MAGMA reports collinearity use gss_total alone")
    ap.add_argument("--mode", default="paracrine")
    ap.add_argument("--traits", default="",
                    help="comma-separated; default all traits in --map")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    R = pd.read_csv(a.map, sep=TAB).fillna("")
    R = R[R.status == "OK"]
    man = dict(l.rstrip("\n").split(TAB)[:2] for l in open(a.manifest)
               if l.strip() and TAB in l)
    sel = [x.strip() for x in a.traits.split(",") if x.strip()] or list(R.trait)
    R = R[R.trait.isin(sel)]
    print("[gp] %d traits" % len(R))

    G = pd.read_csv(a.gss, sep=TAB)
    G = G.rename(columns={G.columns[0]: "GENE"}).set_index("GENE")
    G = G[[c for c in G.columns if c.endswith(a.mode)]]
    G.columns = ["ch_%s" % c.split(":")[0].replace("->", "_") for c in G.columns]
    _ch = [c for c in G.columns if c.startswith("ch_")]
    G["detect_breadth"] = (G[_ch] > 0).sum(axis=1)
    G["gss_total"] = np.log1p(G[_ch].sum(axis=1))
    _r = np.corrcoef(G["detect_breadth"], G["gss_total"])[0, 1]
    print("[covar] corr(detect_breadth, gss_total) = %.4f%s" %
          (_r, "   <- near 1: MAGMA may stop on collinearity; use --covars gss_total"
           if abs(_r) > 0.88 else ""))
    print("[gp] GSS %s: %d genes x %d channels (+2 covariates)"
          % (os.path.basename(a.gss), G.shape[0],
             sum(1 for c in G.columns if c.startswith("ch_"))))

    rows = []
    for r in R.itertuples():
        sym_out = man.get(r.trait, "")
        if not sym_out or not os.path.exists(sym_out):
            print("  [%s] symbol .genes.out missing, skipped" % r.trait); continue
        if not r.sibling_out or not os.path.exists(r.sibling_out):
            print("  [%s] no .genes.out next to the .genes.raw, skipped" % r.trait); continue
        m, note = build_map(r.sibling_out, sym_out, r.trait)
        if m is None:
            print("  [%s] ID mapping failed: %s" % (r.trait, note)); continue
        inv = {}
        for k, v in m.items():
            inv.setdefault(v, k)
        gid = G.index.map(lambda s: inv.get(str(s)))
        keep = pd.notna(gid)
        cov = G[keep].copy()
        cov.insert(0, "GENE", pd.Series(gid[keep], index=cov.index).values)
        f = os.path.join(a.out_dir, "covar_%s.txt" % r.trait)
        cov.to_csv(f, sep=" ", index=False)
        print("  [%-16s] %s; GSS %d -> %d genes mapped -> %s"
              % (r.trait, note, G.shape[0], int(keep.sum()), os.path.basename(f)))
        rows.append(dict(trait=r.trait, covar=f,
                         raw=os.path.join(a.raw_dir, "%s.genes.raw" % r.trait),
                         n_gene=int(keep.sum()), note=note))

    if not rows:
        sys.exit("[gp] no trait to run")
    if a.dry:
        print("\n[dry] covariate files written; MAGMA not run")
        return

    chs = [c for c in G.columns if c.startswith("ch_")]
    res = []
    for x in rows:
        pre = os.path.join(a.out_dir, x["trait"])
        cmd = [a.magma, "--gene-results", x["raw"], "--gene-covar", x["covar"],
               "--model", "condition=%s" % a.covars, "direction=greater",
               "--out", pre]
        gsa = pre + ".gsa.out"
        for old in (gsa, pre + ".log"):
            if os.path.exists(old):
                os.remove(old)
        p = subprocess.run(cmd, capture_output=True, text=True)
        log_txt = open(pre + ".log").read() if os.path.exists(pre + ".log") else ""
        if p.returncode != 0 or not os.path.exists(gsa) or "ERROR" in log_txt:
            sys.exit("[gp] %s: MAGMA failed (exit code %d)\n%s"
                     % (x["trait"], p.returncode, (p.stderr or p.stdout or log_txt)[-600:]))
        d = pd.read_csv(gsa, sep=r"\s+", comment="#")
        d.columns = [c.upper() for c in d.columns]
        vc = "VARIABLE" if "VARIABLE" in d.columns else d.columns[0]
        d = d[d[vc].astype(str).str.startswith("ch_")]
        d["trait"] = x["trait"]
        res.append(d)
        n5 = int((d.P < 0.05).sum())
        best = d.nsmallest(1, "P")
        print("  [%-16s] %d/%d channels P<0.05   strongest %s P=%.3g"
              % (x["trait"], n5, len(d), best[vc].iloc[0], best.P.iloc[0]))

    if res:
        D = pd.concat(res)
        f = os.path.join(a.out_dir, "geneprop_all.tsv")
        D.to_csv(f, sep=TAB, index=False)
        print("\nwrote %s" % f)
        print("\n%-18s %8s %8s %10s  %s"
              % ("trait", "P<0.05", "P<0.01", "min P", "strongest channel"))
        vc = "VARIABLE" if "VARIABLE" in D.columns else D.columns[0]
        for t, g in D.groupby("trait"):
            b = g.nsmallest(1, "P")
            print("%-18s %8d %8d %10.3g  %s"
                  % (t, (g.P < 0.05).sum(), (g.P < 0.01).sum(), g.P.min(),
                     b[vc].iloc[0]))


if __name__ == "__main__":
    main()
