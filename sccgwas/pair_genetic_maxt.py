#!/usr/bin/env python3
"""Pair-level genetic axis: studentized pair statistic and maximum-statistic permutation test over the family."""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

import celltypes

TAB = "\t"
K = ["ligand", "receptor", "channel"]
CT = {}


def zsc(v):
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 0 else 1.0)


def chan(c):
    m = re.match(r"\s*(\d+)\s*->\s*(\d+)", str(c))
    if not m:
        return str(c)
    return "%s->%s" % (CT.get(m.group(1), m.group(1)), CT.get(m.group(2), m.group(2)))


def load_z(f, drop):
    d = pd.read_csv(f, sep=r"\s+")
    d = d.rename(columns={c: c.upper() for c in d.columns})
    d["GENE"] = d.GENE.astype(str)
    d = d.drop_duplicates("GENE").set_index("GENE")
    o = pd.DataFrame(index=d.index)
    o["z"] = pd.to_numeric(d.ZSTAT, errors="coerce")
    o["glen"] = (d.STOP - d.START).astype(float)
    o["nsnps"] = d.NSNPS.astype(float)
    o["chr"] = d.CHR.astype(str)
    o["mb"] = (d.START + d.STOP) / 2e6
    o = o.dropna()
    for ch, lo, hi in drop:
        hit = (o.chr == ch) & (o.mb >= lo) & (o.mb <= hi)
        print("[maxT] removed %d genes in %s:%.1f-%.1f Mb" % (int(hit.sum()), ch, lo, hi))
        o = o[~hit]
    return o


def parse_regions(s):
    out = []
    for r in filter(None, (s or "").split(",")):
        m = re.match(r"(?:chr)?(\w+):([\d.]+)-([\d.]+)$", r.strip())
        if not m:
            sys.exit("--drop_regions format: 6:25-34,19:44.9-45.9 (chromosome:start-end, Mb)")
        out.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--percell", required=True, help="per-section table from pair_beyond_niche.py")
    ap.add_argument("--gss_dir", required=True)
    ap.add_argument("--genes_out", required=True)
    ap.add_argument("--ct_names", default="",
                    help="cell-type names in code order (comma list) or a pkl with ct_categories")
    ap.add_argument("--mode", default="paracrine")
    ap.add_argument("--min_sec", type=int, required=True,
                    help="sections in which a pair x channel must be testable to enter the family")
    ap.add_argument("--universe", choices=["key", "all"], default="key",
                    help="genes permuted: key = genes of testable pairs; all = every gene in the GSS")
    ap.add_argument("--strata_file", default="",
                    help="optional gene<TAB>value file; permute within quantile strata of value")
    ap.add_argument("--n_strata", type=int, default=5)
    ap.add_argument("--covars", choices=["log", "log+raw"], default="log",
                    help="log: log gene length and log variant count; log+raw: also the raw values")
    ap.add_argument("--drop_regions", default="6:25-34", help="regions removed before testing (default: extended MHC, GRCh37)")
    ap.add_argument("--locus_mb", type=float, default=1.0)
    ap.add_argument("--n_sim", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("export PYTHONHASHSEED=0 first")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    zdf = load_z(a.genes_out, parse_regions(a.drop_regions))
    d = pd.read_csv(a.percell, sep=TAB)
    d = d[d.direction == "fwd"] if "direction" in d.columns else d
    d = d[d.channel.astype(str).str.endswith(a.mode)]
    d = d[d.ligand.astype(str).isin(zdf.index) & d.receptor.astype(str).isin(zdf.index)]
    d["sample"] = d["sample"].astype(str)
    samples = sorted(d["sample"].unique())
    CT.update(celltypes.resolve(a.ct_names, celltypes.codes_from_columns(d.channel.unique())))

    cnt = d.groupby(K).agg(n=("sample", "nunique")).reset_index()
    keep = cnt[cnt.n >= a.min_sec].reset_index(drop=True)
    KEY = [tuple(x) for x in keep[K].values]
    kidx = {k: i for i, k in enumerate(KEY)}
    present = {s: set(map(tuple, g[K].values)) for s, g in d.groupby("sample")}
    print("[maxT] %d sections; %d pair x channel rows, %d with n>=%d"
          % (len(samples), len(cnt), len(KEY), a.min_sec))

    pre = []
    for f in sorted(glob.glob(os.path.join(a.gss_dir, "gss_spec_*.tsv"))):
        smp = re.sub(r"^gss_spec_|\.tsv$", "", os.path.basename(f))
        if smp not in present:
            continue
        G = pd.read_csv(f, sep=TAB)
        G = G.rename(columns={G.columns[0]: "gene"}).set_index("gene")
        G.index = G.index.astype(str)
        cols = [c for c in G.columns if str(c).endswith(a.mode)]
        u = [g for g in G.index if g in zdf.index]
        pos = {g: i for i, g in enumerate(u)}
        W = G.loc[u, cols].to_numpy(float)
        gl = zdf.loc[u, "glen"].to_numpy(float)
        ns = zdf.loc[u, "nsnps"].to_numpy(float)
        Xc = [np.ones(len(u)), zsc(np.log(np.maximum(gl, 1))), zsc(np.log(np.maximum(ns, 1)))]
        if a.covars == "log+raw":
            Xc += [zsc(gl), zsc(ns)]
        X = np.column_stack(Xc)
        mu, sd = W.mean(axis=0), W.std(axis=0)
        Gr = (W - mu) / np.where(sd > 0, sd, 1.0)
        Gr = Gr - X @ (np.linalg.pinv(X) @ Gr)
        den = np.sqrt(np.einsum("ij,ij->j", Gr, Gr))
        ci = {c: i for i, c in enumerate(cols)}
        li, ri, cj, out_i = [], [], [], []
        for k in present[smp]:
            if k not in kidx:
                continue
            L, R, c = k
            if L in pos and R in pos and c in ci and den[ci[c]] > 0:
                li.append(pos[L]); ri.append(pos[R]); cj.append(ci[c]); out_i.append(kidx[k])
        if not out_i:
            continue
        li, ri, cj = np.array(li), np.array(ri), np.array(cj)
        pre.append(dict(sample=smp, genes=np.array(u),
                        gL=(Gr[li, cj] / den[cj]).astype(np.float32),
                        gR=(Gr[ri, cj] / den[cj]).astype(np.float32),
                        li=li, ri=ri, oi=np.array(out_i)))
        print("  [%s] %d genes, %d rows" % (smp, len(u), len(out_i)), flush=True)
    if not pre:
        sys.exit("[maxT] no usable section")

    ALL = np.array(sorted({g for sl in pre for g in sl["genes"]}))
    gl = zdf.loc[ALL, "glen"].to_numpy(float)
    ns = zdf.loc[ALL, "nsnps"].to_numpy(float)
    Xc = [np.ones(len(ALL)), zsc(np.log(np.maximum(gl, 1))), zsc(np.log(np.maximum(ns, 1)))]
    if a.covars == "log+raw":
        Xc += [zsc(gl), zsc(ns)]
    XA = np.column_stack(Xc)
    z0 = zdf.loc[ALL, "z"].to_numpy(float)
    ZR = zsc(z0 - XA @ (np.linalg.pinv(XA) @ z0))
    for sl in pre:
        sl["gidx"] = np.searchsorted(ALL, sl["genes"])

    in_key = np.zeros(len(ALL), bool)
    kg = {g for k in KEY for g in k[:2]}
    in_key[[i for i, g in enumerate(ALL) if g in kg]] = True
    movable = in_key if a.universe == "key" else np.ones(len(ALL), bool)
    strata = np.zeros(len(ALL), int)
    if a.strata_file:
        sv = pd.read_csv(a.strata_file, sep=TAB, header=None, index_col=0)[1]
        v = np.array([float(sv.get(g, np.nan)) for g in ALL])
        okv = np.isfinite(v) & movable
        q = np.nanquantile(v[okv], np.linspace(0, 1, a.n_strata + 1)[1:-1])
        strata = np.where(np.isfinite(v), np.searchsorted(q, v), -1)
        movable &= strata >= 0
    groups = [np.where(movable & (strata == s))[0] for s in np.unique(strata[movable])]
    print("[maxT] %d genes; %d permuted (universe=%s, %d strata)"
          % (len(ALL), int(movable.sum()), a.universe, len(groups)))

    def agg(zr_all):
        SL = np.zeros(len(KEY)); SR = np.zeros(len(KEY)); N = np.zeros(len(KEY), np.int32)
        for sl in pre:
            zr = zsc(zr_all[sl["gidx"]])
            np.add.at(SL, sl["oi"], sl["gL"] * zr[sl["li"]])
            np.add.at(SR, sl["oi"], sl["gR"] * zr[sl["ri"]])
            np.add.at(N, sl["oi"], 1)
        m = N > 0
        SL[m] /= N[m]; SR[m] /= N[m]; SL[~m] = np.nan; SR[~m] = np.nan
        return SL, SR, N

    oL, oR, N = agg(ZR)
    rng = np.random.default_rng(a.seed)
    ML = np.empty((a.n_sim, len(KEY)), np.float32)
    MR = np.empty((a.n_sim, len(KEY)), np.float32)
    t0 = time.time()
    for b in range(a.n_sim):
        zp = ZR.copy()
        for g in groups:
            zp[g] = ZR[rng.permutation(g)]
        ML[b], MR[b], _ = agg(zp)
        if (b + 1) % max(1, a.n_sim // 10) == 0:
            el = time.time() - t0
            print("  %4d/%d  %.1f min, %.1f min left" % (b + 1, a.n_sim, el / 60,
                  el / 60 * (a.n_sim - b - 1) / (b + 1)), flush=True)

    def stud(o, M):
        P = np.vstack([o[None, :], M])
        mu = np.nanmean(P, axis=0); sd = np.nanstd(P, axis=0)
        sd = np.where(sd > 1e-9, sd, np.nan)
        return (o - mu) / sd, (M - mu) / sd

    t_sum, T_sum = stud(oL + oR, ML + MR)
    tL, TL = stud(oL, ML)
    tR, TR = stud(oR, MR)
    t_min, T_min = np.fmin(tL, tR), np.fmin(TL, TR)

    R = keep.copy()
    R["ch"] = [chan(c) for c in R.channel]
    R["n_used"] = N
    R["t"], R["t_L"], R["t_R"], R["t_min"] = t_sum, tL, tR, t_min
    drv = np.where(np.nan_to_num(tL, nan=-9) >= np.nan_to_num(tR, nan=-9), R.ligand, R.receptor)
    R["driver"] = drv
    R["driver_locus"] = ["%s:%d" % (zdf.chr[g], int(zdf.mb[g] // a.locus_mb)) for g in drv]

    for stat, tobs, Tn in (("sum", t_sum, T_sum), ("min", t_min, T_min)):
        for tag, mask in (("A", np.ones(len(R), bool)),):
            idx = np.where(mask & np.isfinite(tobs))[0]
            col = "p_fwer_%s_%s" % (tag, stat)
            R[col] = np.nan
            if not len(idx):
                continue
            mx = np.nanmax(Tn[:, idx], axis=1)
            R.loc[idx, col] = [(1 + int((mx >= tobs[i]).sum())) / (1 + a.n_sim) for i in idx]
            if tag == "A":
                rc = "p_raw_%s" % stat
                if rc not in R.columns:
                    R[rc] = np.nan
                R.loc[idx, rc] = [
                    (1 + int((Tn[:, i] >= tobs[i]).sum())) / (1 + a.n_sim)
                    for i in idx]
            print("  %s: %d tests; permutation maximum median %.2f, 95%% %.2f, max %.2f"
                  % (stat, len(idx), np.median(mx), np.percentile(mx, 95), mx.max()))

    for stat in ("sum", "min"):
        rc, qc = "p_raw_%s" % stat, "q_bh_%s" % stat
        if rc not in R.columns:
            continue
        R[qc] = np.nan
        v = R[rc].to_numpy(float)
        ok = np.isfinite(v)
        o = np.argsort(v[ok]); m = int(ok.sum())
        if not m:
            continue
        q = np.minimum.accumulate(
            (v[ok][o] * m / np.arange(1, m + 1))[::-1])[::-1]
        tmp = np.full(m, np.nan); tmp[o] = np.minimum(q, 1.0)
        R.loc[ok, qc] = tmp
        print("  %s / BH over rows: %d rows q<0.05, %d rows q<0.10 (%d tests)"
              % (stat, int((R[qc] < 0.05).sum()), int((R[qc] < 0.10).sum()), m))

    _grp = R.groupby(["ligand", "receptor"], sort=False).indices
    for _stat, _tobs, _Tn in (("sum", t_sum, T_sum), ("min", t_min, T_min)):
        _pp = np.full(len(R), np.nan)
        for _k, _idx in _grp.items():
            _idx = np.asarray(_idx)
            _ok = _idx[np.isfinite(_tobs[_idx])]
            if not len(_ok):
                continue
            _o = float(np.max(_tobs[_ok]))
            _nb = np.nanmax(_Tn[:, _ok], axis=1)
            _pp[_idx] = (1 + int((_nb >= _o).sum())) / (1 + a.n_sim)
        R["p_pair_%s" % _stat] = _pp
    print("[maxT] per-pair P written (p_pair_sum, p_pair_min; %d pairs)" % len(_grp))

    R.sort_values("t", ascending=False).to_csv(a.out, sep=TAB, index=False)
    c = ["ligand", "receptor", "ch", "n", "n_used", "t", "t_L", "t_R", "t_min",
         "p_fwer_A_sum", "p_pair_sum", "p_raw_sum", "driver_locus"]
    print("\n15 largest t")
    print(R.nlargest(15, "t")[c].round(4).to_string(index=False))
    print("\nsummary")
    for col in ("p_fwer_A_sum", "p_fwer_A_min", "q_bh_sum", "q_bh_min"):
        if col not in R.columns:
            continue
        s = R[R[col] < (0.10 if col.startswith("q_") else 0.05)]
        print("  %-13s  %3d rows = %2d pairs = %2d driver loci  %s"
              % (col, len(s), s[["ligand", "receptor"]].drop_duplicates().shape[0],
                 s.driver_locus.nunique(),
                 ", ".join(sorted(set(s.ligand + "-" + s.receptor))[:8])))
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
