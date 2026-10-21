#!/usr/bin/env python3
"""Pair-level genetic test conditioned on the selection of the receiving cell type: the hub is
recomputed in every permutation and P values use the rounds that select --target_hub."""
import argparse
import glob
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd

import celltypes

TAB = "\t"
HERE = os.path.dirname(os.path.abspath(__file__))
K = ["ligand", "receptor", "channel"]


def log(m):
    print(m, flush=True)


def zsc_nan(v):
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 0 else 1.0)


def zsc(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def load_z_pair(f, drop):
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
        o = o[~hit]
    return o


def load_z_hub(path):
    g = pd.read_csv(path, sep=r"\s+")
    g.columns = [c.upper() for c in g.columns]
    ns = "NSNPS" if "NSNPS" in g.columns else "NSNP"
    glen = (np.abs(pd.to_numeric(g["STOP"], errors="coerce")
                   - pd.to_numeric(g["START"], errors="coerce"))
            if "STOP" in g.columns and "START" in g.columns
            else pd.to_numeric(g[ns], errors="coerce") * 1000.0)
    out = pd.DataFrame({"z": pd.to_numeric(g["ZSTAT"], errors="coerce").values,
                        "nsnps": pd.to_numeric(g[ns], errors="coerce").values,
                        "glen": np.asarray(glen, float)},
                       index=g["GENE"].astype(str))
    return out[~out.index.duplicated()].dropna()


def parse_regions(s):
    out = []
    for r in filter(None, (s or "").split(",")):
        m = re.match(r"(?:chr)?(\w+):([\d.]+)-([\d.]+)$", r.strip())
        if not m:
            sys.exit("--drop_regions format: 6:25-34")
        out.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return out


def cell_detection(pkl_dir, sample):
    from scipy import sparse as sp
    f = os.path.join(pkl_dir, "inputs_%s.pkl" % sample)
    if not os.path.exists(f):
        return None
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
                           if sp.issparse(sub) else (np.asarray(sub) > 0).sum(0), index=genes)
    return pd.DataFrame(out).max(axis=1)


def bh(v):
    v = np.asarray(v, float)
    q = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    m = int(ok.sum())
    if not m:
        return q
    o = np.argsort(v[ok])
    step = v[ok][o] * m / np.arange(1, m + 1)
    r = np.empty(m)
    r[o] = np.minimum(np.minimum.accumulate(step[::-1])[::-1], 1.0)
    q[ok] = r
    return q


def codes_from_columns(cols, mode):
    out = set()
    for c in cols:
        m = re.match(r"\s*(\d+)\s*->\s*(\d+)(?::(\w+))?", str(c))
        if m and m.group(3) in (None, mode):
            out.update((m.group(1), m.group(2)))
    return sorted(out, key=int)


def build_test(a):
    zdf = load_z_pair(a.genes_out, parse_regions(a.drop_regions))
    d = pd.read_csv(a.percell, sep=TAB)
    d = d[d.direction == "fwd"] if "direction" in d.columns else d
    d = d[d.channel.astype(str).str.endswith(a.mode)]
    d = d[d.ligand.astype(str).isin(zdf.index) & d.receptor.astype(str).isin(zdf.index)]
    d["sample"] = d["sample"].astype(str)
    cnt = d.groupby(K).agg(n=("sample", "nunique")).reset_index()
    if str(a.min_sec) == "auto":
        if not (a.ref_maxt and os.path.exists(a.ref_maxt)):
            sys.exit("[test] --min_sec auto needs --ref_maxt")
        ref = set(map(tuple, pd.read_csv(a.ref_maxt, sep=TAB)[K].astype(str).values))
        rows = {k: set(map(tuple, cnt[cnt.n >= k][K].astype(str).values)) for k in range(1, int(cnt.n.max()) + 1)}
        hit = [k for k, v in rows.items() if v == ref]
        log("[test] rows of %s: %d; rows with n>=k: %s" % (a.ref_maxt, len(ref), ", ".join(
            "%d:%d" % (k, len(v)) for k, v in rows.items())))
        if not hit:
            sys.exit("[test] no --min_sec reproduces the rows of %s" % a.ref_maxt)
        a.min_sec = hit[0]
        log("[test] --min_sec auto -> %d" % a.min_sec)
    a.min_sec = int(a.min_sec)
    keep = cnt[cnt.n >= a.min_sec].reset_index(drop=True)
    KEY = [tuple(x) for x in keep[K].values]
    kidx = {k: i for i, k in enumerate(KEY)}
    present = {s: set(map(tuple, g[K].values)) for s, g in d.groupby("sample")}
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
        Xc = [np.ones(len(u)), zsc_nan(np.log(np.maximum(gl, 1))), zsc_nan(np.log(np.maximum(ns, 1)))]
        if a.covars == "log+raw":
            Xc += [zsc_nan(gl), zsc_nan(ns)]
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
    if not pre:
        sys.exit("[test] no usable section")
    ALL = np.array(sorted({g for sl in pre for g in sl["genes"]}))
    gl = zdf.loc[ALL, "glen"].to_numpy(float)
    ns = zdf.loc[ALL, "nsnps"].to_numpy(float)
    Xc = [np.ones(len(ALL)), zsc_nan(np.log(np.maximum(gl, 1))), zsc_nan(np.log(np.maximum(ns, 1)))]
    if a.covars == "log+raw":
        Xc += [zsc_nan(gl), zsc_nan(ns)]
    XA = np.column_stack(Xc)
    z0 = zdf.loc[ALL, "z"].to_numpy(float)
    resid = z0 - XA @ (np.linalg.pinv(XA) @ z0)
    ZR = zsc_nan(resid)
    for sl in pre:
        sl["gidx"] = np.searchsorted(ALL, sl["genes"])
    in_key = np.zeros(len(ALL), bool)
    kg = {g for k in KEY for g in k[:2]}
    in_key[[i for i, g in enumerate(ALL) if g in kg]] = True
    movable = in_key if a.universe == "key" else np.ones(len(ALL), bool)
    groups = [np.where(movable)[0]]
    log("[test] %d sections, %d pair x channel rows (n>=%d), %d genes, %d permuted"
        % (len(pre), len(KEY), a.min_sec, len(ALL), int(movable.sum())))
    return dict(keep=keep, KEY=KEY, pre=pre, ALL=ALL, ZR=ZR, resid=resid, groups=groups)


def agg(pre, nkey, zr_all):
    SL = np.zeros(nkey); SR = np.zeros(nkey); N = np.zeros(nkey, np.int32)
    for sl in pre:
        zr = zsc_nan(zr_all[sl["gidx"]])
        np.add.at(SL, sl["oi"], sl["gL"] * zr[sl["li"]])
        np.add.at(SR, sl["oi"], sl["gR"] * zr[sl["ri"]])
        np.add.at(N, sl["oi"], 1)
    m = N > 0
    SL[m] /= N[m]; SR[m] /= N[m]; SL[~m] = np.nan; SR[~m] = np.nan
    return SL, SR, N


def build_selection(a, ALL, movable_idx):
    settings = [s.strip() for s in a.settings.split(",") if s.strip()]
    parsed = []
    for s in settings:
        m = re.match(r"mch(\d+)_mcl(\d+)$", s)
        if not m:
            sys.exit("--settings entries look like mch1_mcl0: %s" % s)
        parsed.append((int(m.group(1)), int(m.group(2))))
    zh = load_z_hub(a.sel_genes_out)
    mov_genes = ALL[movable_idx]
    mov_pos = {g: i for i, g in enumerate(mov_genes)}
    files = sorted(glob.glob(os.path.join(a.sel_gss_dir, "gss_spec_*.tsv")))
    if not files:
        sys.exit("[sel] no gss_spec_*.tsv in %s" % a.sel_gss_dir)
    if a.code:
        sys.path.insert(0, a.code)
    secs, chans = [], []
    for f in files:
        smp = re.sub(r"^gss_spec_|\.tsv$", "", os.path.basename(f))
        if smp.upper() == "MEAN":
            continue
        G = pd.read_csv(f, sep=TAB)
        G = G.rename(columns={G.columns[0]: "gene"}).set_index("gene")
        G.index = G.index.astype(str)
        cols = [c for c in G.columns if str(c).endswith(a.mode)]
        if not cols:
            continue
        u = [g for g in G.index if g in zh.index]
        if len(u) < 50:
            continue
        nzc = (G[cols] > 0).sum(axis=1)
        det = cell_detection(a.sel_pkl_dir, smp) if any(m > 0 for _, m in parsed) else None
        if det is None and any(m > 0 for _, m in parsed):
            sys.exit("[sel] pkl for %s not found in %s (needed for min_cells)" % (smp, a.sel_pkl_dir))
        gl = zh.loc[u, "glen"].to_numpy(float)
        ns = zh.loc[u, "nsnps"].to_numpy(float)
        X = np.column_stack([np.ones(len(u)), zsc(np.log(np.maximum(gl, 1))),
                             zsc(np.log(np.maximum(ns, 1))), zsc(gl), zsc(ns)])
        XP = np.linalg.pinv(X)
        Us, sv, _ = np.linalg.svd(X, full_matrices=False)
        Ur = Us[:, sv > sv.max() * max(X.shape) * np.finfo(float).eps]
        zv = zh.loc[u, "z"].to_numpy(float)
        kpos = np.array([i for i, g in enumerate(u) if g in mov_pos], int)
        kmov = np.array([mov_pos[u[i]] for i in kpos], int)
        S = dict(sample=smp, cols=cols, u=u, X=X, XP=XP, zv=zv, n=len(u), kpos=kpos, kmov=kmov,
                 zz0=float(zv @ zv), q0=Ur.T @ zv, UK=Ur[kpos], zK=zv[kpos], C={}, base={})
        for mch, mcl in parsed:
            keep = set(nzc[nzc >= mch].index)
            if det is not None and mcl > 0:
                keep &= set(det.index[det >= mcl])
            sub = G.loc[u, cols].to_numpy(float).copy()
            mask = np.array([g in keep for g in u])
            sub[~mask, :] = 0.0
            mu, sd = sub.mean(axis=0), sub.std(axis=0)
            Gr = (sub - mu) / np.where(sd > 0, sd, 1.0)
            Gr = Gr - X @ (XP @ Gr)
            den = np.sqrt(np.einsum("ij,ij->j", Gr, Gr))
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.where(den > 0, Gr / np.where(den > 0, den, 1.0), np.nan)
            key = "mch%d_mcl%d" % (mch, mcl)
            S["C"][key] = c[kpos]
            S["base"][key] = np.where(den > 0, c.T @ zv if len(u) else 0.0, np.nan)
            S.setdefault("Gr", {})[key] = (Gr, den)
        secs.append(S)
        chans.extend(cols)
        log("  [sel] %-12s %d genes, %d channels, %d movable genes inside" % (smp, len(u), len(cols), len(kpos)))
    chans = sorted(set(chans))
    for S in secs:
        S["cidx"] = np.array([chans.index(c) for c in S["cols"]], int)
    return dict(settings=settings, secs=secs, chans=chans)


def channel_t_direct(S, key, zvec):
    Gr, den = S["Gr"][key]
    zr = zsc(zvec - S["X"] @ (S["XP"] @ zvec))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (Gr.T @ zr) / np.where(den > 0, den, 1.0), np.nan)


def mean_t_chunk(sel, key, D):
    nC = D.shape[0]
    acc = np.zeros((nC, len(sel["chans"]))); cnt = np.zeros(len(sel["chans"]))
    for S in sel["secs"]:
        dK = D[:, S["kmov"]] if len(S["kpos"]) else np.zeros((nC, 0))
        zz = S["zz0"] + 2.0 * (dK @ S["zK"]) + np.einsum("ij,ij->i", dK, dK)
        q = S["q0"][None, :] + dK @ S["UK"]
        var = (zz - np.einsum("ij,ij->i", q, q)) / S["n"]
        sd = np.sqrt(np.maximum(var, 0.0))
        cK = np.nan_to_num(S["C"][key], nan=0.0)
        num = S["base"][key][None, :] + dK @ cK
        t = num / np.where(sd > 0, sd, np.nan)[:, None]
        ok = np.isfinite(S["base"][key])
        acc[:, S["cidx"][ok]] += t[:, ok]
        cnt[S["cidx"][ok]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return acc / np.where(cnt > 0, cnt, np.nan)[None, :]


def prepare_hub(a, sel, zm_paths):
    hubspec = {}
    mats = {k: pd.read_csv(p, sep=TAB, index_col=0) for k, p in zm_paths.items()}
    traits = sorted(set.intersection(*[set(m.index) for m in mats.values()]))
    if a.trait not in traits:
        sys.exit("[hub] %s not in the z-matrices" % a.trait)
    codes = codes_from_columns(set.union(*[set(m.columns) for m in mats.values()]), a.mode)
    for key, Z in mats.items():
        base = pd.read_csv(zm_paths[key].replace("_zmatrix.tsv", "_baseline.tsv"), sep=TAB, index_col=0)
        zcols = list(Z.columns)
        bmean = base["baseline_mean"].reindex(zcols).to_numpy(float)
        bsd = base["baseline_sd"].reindex(zcols).to_numpy(float)
        bsd = np.where(bsd == 0, np.nan, bsd)
        cpos = np.array([sel["chans"].index(c) if c in sel["chans"] else -1 for c in zcols])
        zrow = Z.loc[a.trait].to_numpy(float)
        hubs = []
        for h in codes:
            fi, ri = [], []
            for x in codes:
                if x == h:
                    continue
                f = "%s->%s:%s" % (x, h, a.mode)
                r = "%s->%s:%s" % (h, x, a.mode)
                if f in zcols and r in zcols:
                    i, j = zcols.index(f), zcols.index(r)
                    if np.isfinite(zrow[i]) and np.isfinite(zrow[j]):
                        fi.append(i); ri.append(j)
            if len(fi) < 3:
                continue
            vin, vdf = [], []
            for tr in traits:
                if tr == a.trait:
                    continue
                zr = Z.loc[tr].to_numpy(float)
                ins, outs = [], []
                for x in codes:
                    if x == h:
                        continue
                    f = "%s->%s:%s" % (x, h, a.mode)
                    r = "%s->%s:%s" % (h, x, a.mode)
                    if f in zcols and r in zcols:
                        p, q = zr[zcols.index(f)], zr[zcols.index(r)]
                        if np.isfinite(p) and np.isfinite(q):
                            ins.append(p); outs.append(q)
                if len(ins) >= 3:
                    ins, outs = np.array(ins), np.array(outs)
                    vin.append(ins.mean()); vdf.append((ins - outs).mean())
            vin, vdf = np.array(vin), np.array(vdf)
            ok = len(vin) + 1 >= 5
            hubs.append(dict(code=h, fi=np.array(fi), ri=np.array(ri), ok=ok,
                             mi=vin.mean() if len(vin) else np.nan,
                             si=vin.std(ddof=1) if len(vin) > 1 else np.nan,
                             md=vdf.mean() if len(vdf) else np.nan,
                             sdd=vdf.std(ddof=1) if len(vdf) > 1 else np.nan))
        hubspec[key] = dict(Z=Z, zcols=zcols, bmean=bmean, bsd=bsd, cpos=cpos, zrow=zrow, hubs=hubs)
    return hubspec, traits, codes


def hub_from_mt(hs, mt):
    if (hs["cpos"] < 0).any():
        Zad = np.full((mt.shape[0], len(hs["zcols"])), np.nan)
        okc = hs["cpos"] >= 0
        Zad[:, okc] = (mt[:, hs["cpos"][okc]] - hs["bmean"][okc]) / hs["bsd"][okc]
    else:
        Zad = (mt[:, hs["cpos"]] - hs["bmean"]) / hs["bsd"]
    S = np.full((mt.shape[0], len(hs["hubs"])), np.nan)
    for j, h in enumerate(hs["hubs"]):
        if not h["ok"]:
            continue
        a_in = Zad[:, h["fi"]].mean(axis=1)
        a_df = (Zad[:, h["fi"]] - Zad[:, h["ri"]]).mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            zi = (a_in - h["mi"]) / h["si"] if h["si"] > 0 else np.full_like(a_in, np.nan)
            zd = (a_df - h["md"]) / h["sdd"] if h["sdd"] > 0 else np.full_like(a_df, np.nan)
        S[:, j] = np.fmin(zi, zd)
    best = np.full(mt.shape[0], -1, int)
    okr = np.isfinite(S).any(axis=1)
    best[okr] = np.nanargmax(np.where(np.isfinite(S[okr]), S[okr], -np.inf), axis=1)
    codes = np.array([h["code"] for h in hs["hubs"]] + ["NA"], dtype=object)
    return Zad, S, codes[best]


def find_zmatrix(a, sel, mt_obs):
    s0 = sel["settings"][0]
    if a.sel_zmatrix != "auto":
        cands = [a.sel_zmatrix]
    else:
        hits = glob.glob(os.path.join(a.search_root, "**", "*%s_zmatrix.tsv" % s0), recursive=True)
        cands = sorted({h[: -len("%s_zmatrix.tsv" % s0)] + "{D}_zmatrix.tsv" for h in hits})
    good = []
    for tmpl in cands:
        paths = {s: tmpl.replace("{D}", s) for s in sel["settings"]}
        miss = [p for p in paths.values() if not os.path.exists(p)
                or not os.path.exists(p.replace("_zmatrix.tsv", "_baseline.tsv"))]
        if miss:
            log("  [zmatrix] %s: missing %d file(s)" % (tmpl, len(miss)))
            continue
        worst = 0.0
        try:
            for s, p in paths.items():
                Z = pd.read_csv(p, sep=TAB, index_col=0)
                if a.trait not in Z.index:
                    raise KeyError(a.trait)
                base = pd.read_csv(p.replace("_zmatrix.tsv", "_baseline.tsv"), sep=TAB, index_col=0)
                zc = list(Z.columns)
                cp = [sel["chans"].index(c) if c in sel["chans"] else -1 for c in zc]
                if min(cp) < 0:
                    raise KeyError("channel set differs")
                bm = base["baseline_mean"].reindex(zc).to_numpy(float)
                bs = base["baseline_sd"].reindex(zc).to_numpy(float)
                bs = np.where(bs == 0, np.nan, bs)
                rec = (mt_obs[s][np.array(cp)] - bm) / bs
                fil = Z.loc[a.trait].to_numpy(float)
                if not np.array_equal(np.isfinite(rec), np.isfinite(fil)):
                    raise ValueError("NaN pattern differs")
                f = np.isfinite(rec)
                worst = max(worst, float(np.max(np.abs(rec[f] - fil[f]))) if f.any() else 0.0)
        except Exception as e:
            log("  [zmatrix] %s: not this tissue/run (%s)" % (tmpl, e))
            continue
        log("  [zmatrix] %s: %s row max |diff| %.2e  %s" % (tmpl, a.trait, worst,
            "MATCH" if worst < 1e-6 else "no match"))
        if worst < 1e-6:
            good.append(tmpl)
    if not good:
        sys.exit("[zmatrix] no z-matrix set reproduces the %s row; give --sel_zmatrix explicitly" % a.trait)
    if len(good) > 1:
        log("  [zmatrix] %d sets reproduce the row; using the first: %s" % (len(good), good[0]))
    return {s: good[0].replace("{D}", s) for s in sel["settings"]}


def check_filtered(a, sel, tobs):
    if a.sel_filtered == "none":
        return
    roots = []
    if a.sel_filtered == "auto":
        s0 = sel["settings"][0]
        smp = sel["secs"][0]["sample"]
        for h in glob.glob(os.path.join(a.search_root, "**", s0, "ch_%s_channels.tsv" % smp), recursive=True):
            roots.append(os.path.dirname(os.path.dirname(h)))
    else:
        roots = [a.sel_filtered]
    for root in sorted(set(roots)):
        worst, n = 0.0, 0
        try:
            for s in sel["settings"]:
                for S in sel["secs"]:
                    f = os.path.join(root, s, "ch_%s_channels.tsv" % S["sample"])
                    d = pd.read_csv(f, sep=TAB)
                    d = d[d.trait == a.trait].set_index("channel")["t"]
                    v = d.reindex(S["cols"]).to_numpy(float)
                    r = tobs[(s, S["sample"])]
                    m = np.isfinite(v) & np.isfinite(r)
                    if not np.array_equal(np.isfinite(v), np.isfinite(r)):
                        raise ValueError("NaN pattern")
                    if m.any():
                        worst = max(worst, float(np.max(np.abs(v[m] - r[m]))))
                    n += 1
        except Exception as e:
            log("  [filtered] %s: no match (%s)" % (root, e))
            continue
        log("  [filtered] %s: %d section x setting tables, per-section t max |diff| %.2e" % (root, n, worst))


def check_cells(a, zm_paths, S_obs, hubspec):
    d0 = os.path.dirname(list(zm_paths.values())[0])
    for f in sorted(glob.glob(os.path.join(d0, "*_cells.tsv"))):
        try:
            c = pd.read_csv(f, sep=TAB)
            if not {"filter", "trait", "hub_code", "S"} <= set(c.columns):
                continue
            c = c[c.trait == a.trait]
            worst, n = 0.0, 0
            for key, hs in hubspec.items():
                x = c[c["filter"] == key].set_index(c[c["filter"] == key].hub_code.astype(str))["S"]
                if x.empty:
                    continue
                for j, h in enumerate(hs["hubs"]):
                    if h["code"] in x.index and np.isfinite(S_obs[key][0, j]) and np.isfinite(x[h["code"]]):
                        worst = max(worst, abs(float(x[h["code"]]) - float(S_obs[key][0, j]))); n += 1
            if n:
                log("  [cells] %s: %d hub statistics compared, max |diff| %.2e" % (f, n, worst))
        except Exception as e:
            log("  [cells] %s: skipped (%s)" % (f, e))


def check_final(a, zm_paths, lab, want=None):
    d0 = os.path.dirname(list(zm_paths.values())[0])
    for f in sorted(glob.glob(os.path.join(d0, "*_final.tsv"))):
        try:
            c = pd.read_csv(f, sep=TAB)
            if not {"filter", "trait", "hub", "p_cont"} <= set(c.columns):
                continue
            c = c[c.trait == a.trait]
            log("  [final] %s: %s" % (f, "  ".join("%s %s p_cont %.4g" % (r["filter"], r["hub"], r["p_cont"])
                                                   for _, r in c.iterrows())))
            if want is not None:
                bad = [r["filter"] for _, r in c.iterrows()
                       if r["filter"] in want and str(r["hub"]) != want[r["filter"]]]
                if bad:
                    log("  [final] WARNING: hub name differs from the observed hub in %s (check --ct_names)" % bad)
        except Exception as e:
            log("  [final] %s: skipped (%s)" % (f, e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--percell", required=True)
    ap.add_argument("--gss_dir", required=True)
    ap.add_argument("--genes_out", required=True)
    ap.add_argument("--mode", default="paracrine")
    ap.add_argument("--min_sec", required=True,
                    help="value given to pair_genetic_maxt.py, or auto (from --ref_maxt)")
    ap.add_argument("--universe", choices=["key", "all"], default="key")
    ap.add_argument("--covars", choices=["log", "log+raw"], default="log")
    ap.add_argument("--drop_regions", default="6:25-34")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_ref", type=int, default=20000,
                    help="rounds of the pair_genetic_maxt.py run being conditioned (its --n_sim)")
    ap.add_argument("--n_total", type=int, default=100000)
    ap.add_argument("--ref_maxt", default="",
                    help="pair_genetic_maxt.py output for the same inputs; its rows and p_fwer_A_sum are checked")
    ap.add_argument("--sel_gss_dir", required=True, help="GSS of the tissue in which the hub was selected")
    ap.add_argument("--sel_pkl_dir", required=True)
    ap.add_argument("--sel_genes_out", default="", help="default: --genes_out")
    ap.add_argument("--code", default=HERE, help="directory with ccgwas_v4core.py (to unpickle)")
    ap.add_argument("--sel_zmatrix", default="auto", help="template with {D}, e.g. out/x/hub_{D}_zmatrix.tsv")
    ap.add_argument("--sel_filtered", default="auto", help="channel_t_matrix out_root, 'auto' or 'none'")
    ap.add_argument("--search_root", default=".", help="searched for z-matrices and channel tables in 'auto' mode")
    ap.add_argument("--settings", default="mch1_mcl0,mch10_mcl0,mch20_mcl0,mch1_mcl10,mch10_mcl10,mch20_mcl10")
    ap.add_argument("--trait", required=True)
    ap.add_argument("--target_hub", required=True, help="code or name of the selected receiving cell type")
    ap.add_argument("--ct_names", default="auto",
                    help="auto (first pkl in --sel_pkl_dir), a pkl, or names in code order")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--n_check", type=int, default=3)
    ap.add_argument("--out_prefix", required=True)
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("export PYTHONHASHSEED=0 first")
    if a.n_total < a.n_ref:
        sys.exit("--n_total must be >= --n_ref")
    a.sel_genes_out = a.sel_genes_out or a.genes_out
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    if a.ct_names == "auto":
        if a.code:
            sys.path.insert(0, a.code)
        pk = sorted(glob.glob(os.path.join(a.sel_pkl_dir, "inputs_*.pkl")))
        if not pk:
            sys.exit("--ct_names auto: no inputs_*.pkl in %s" % a.sel_pkl_dir)
        names = [str(x) for x in pickle.load(open(pk[0], "rb"))["ct_categories"]]
    else:
        CT = celltypes.resolve(a.ct_names)
        names = [CT[str(i)] for i in range(len(CT))]
    lab = lambda c: names[int(c)] if c.isdigit() and int(c) < len(names) else c
    if not a.target_hub.isdigit():
        hit = [i for i, x in enumerate(names) if x.lower() == a.target_hub.lower()] or \
              [i for i, x in enumerate(names) if x.lower().startswith(a.target_hub.lower())]
        if len(hit) != 1:
            sys.exit("--target_hub %s matches %d of %s" % (a.target_hub, len(hit), names))
        a.target_hub = str(hit[0])
    log("cell types (code order): %s; target hub %s = %s" % (
        ", ".join("%d %s" % (i, x) for i, x in enumerate(names)), a.target_hub, lab(a.target_hub)))
    t_start = time.time()

    log("== test side")
    T = build_test(a)
    keep, KEY, pre, ZR, resid, groups = T["keep"], T["KEY"], T["pre"], T["ZR"], T["resid"], T["groups"]
    movable_idx = groups[0]
    nkey = len(KEY)

    log("== selection side")
    sel = build_selection(a, T["ALL"], movable_idx)
    mt_obs, tobs_sec = {}, {}
    D0 = np.zeros((1, len(movable_idx)))
    for s in sel["settings"]:
        mt_obs[s] = mean_t_chunk(sel, s, D0)[0]
        for S in sel["secs"]:
            tobs_sec[(s, S["sample"])] = channel_t_direct(S, s, S["zv"])
    worst = 0.0
    for s in sel["settings"]:
        acc = np.zeros(len(sel["chans"])); cnt = np.zeros(len(sel["chans"]))
        for S in sel["secs"]:
            t = tobs_sec[(s, S["sample"])]
            ok = np.isfinite(t)
            acc[S["cidx"][ok]] += t[ok]; cnt[S["cidx"][ok]] += 1
        ref = acc / np.where(cnt > 0, cnt, np.nan)
        m = np.isfinite(ref)
        if not m.any():
            sys.exit("[sel] setting %s: no channel statistic in any section (no gene passes the filter); "
                     "leave it out of --settings, as hub_iut.py does" % s)
        worst = max(worst, float(np.max(np.abs(ref[m] - mt_obs[s][m]))))
    log("  fast vs direct channel statistic (observed): max |diff| %.2e" % worst)
    if worst > 1e-8:
        sys.exit("[sel] fast formula does not reproduce the direct computation")
    check_filtered(a, sel, tobs_sec)
    zm_paths = find_zmatrix(a, sel, mt_obs)
    hubspec, traits, codes = prepare_hub(a, sel, zm_paths)
    if a.target_hub not in codes:
        sys.exit("[hub] target hub %s not among codes %s" % (a.target_hub, codes))
    S_obs, hub_obs = {}, {}
    for s in sel["settings"]:
        _, Sv, hb = hub_from_mt(hubspec[s], mt_obs[s][None, :])
        S_obs[s], hub_obs[s] = Sv, hb[0]
        log("  %-12s observed hub %s (%s)   S: %s" % (s, hb[0], lab(hb[0]), "  ".join(
            "%s %.3f" % (lab(h["code"]), Sv[0, j]) for j, h in enumerate(hubspec[s]["hubs"]))))
    check_cells(a, zm_paths, S_obs, hubspec)
    check_final(a, zm_paths, lab, {s: lab(hub_obs[s]) for s in hub_obs})
    nset = len(sel["settings"])
    rules = {"all_settings": lambda H: (H == a.target_hub).all(axis=1),
             "first_setting": lambda H: H[:, 0] == a.target_hub,
             "majority": lambda H: (H == a.target_hub).sum(axis=1) > nset / 2}
    Hobs = np.array([[hub_obs[s] for s in sel["settings"]]], dtype=object)
    active = {k: f for k, f in rules.items() if bool(f(Hobs)[0])}
    for k in rules:
        log("  rule %-14s observed: %s" % (k, "holds" if k in active else "does NOT hold -> not used"))
    if not active:
        sys.exit("[hub] observed hub is not %s under any rule" % a.target_hub)

    log("== permutations (%d; first %d = reference run)" % (a.n_total, a.n_ref))
    oL, oR, N = agg(pre, nkey, ZR)
    o = oL + oR
    rng = np.random.default_rng(a.seed)
    g = movable_idx
    rg = resid[g]
    hcode = {c: i for i, c in enumerate(codes)}
    ti = hcode[a.target_hub]
    H = np.full((a.n_total, nset), -1, np.int8)
    famax = np.empty(a.n_total)
    Sref = np.empty((a.n_ref, nkey), np.float32)
    rule_int = {"all_settings": lambda Hb: (Hb == ti).all(axis=1),
                "first_setting": lambda Hb: Hb[:, 0] == ti,
                "majority": lambda Hb: (Hb == ti).sum(axis=1) > nset / 2}
    use = ["ref", "all"] + [k for k in rule_int if k in active]
    st = {}

    def hubs_for(Pbuf, b0, b1, check):
        Dl = resid[Pbuf] - rg[None, :]
        for k, s in enumerate(sel["settings"]):
            _, _, hb = hub_from_mt(hubspec[s], mean_t_chunk(sel, s, Dl))
            H[b0:b1, k] = [hcode.get(x, -1) for x in hb]
        if check:
            for i in range(min(a.n_check, b1 - b0)):
                w = 0.0
                for s in sel["settings"]:
                    acc = np.zeros(len(sel["chans"])); cnt = np.zeros(len(sel["chans"]))
                    for S in sel["secs"]:
                        zb = S["zv"].copy()
                        zb[S["kpos"]] += Dl[i, S["kmov"]]
                        t = channel_t_direct(S, s, zb)
                        ok = np.isfinite(t)
                        acc[S["cidx"][ok]] += t[ok]; cnt[S["cidx"][ok]] += 1
                    ref = acc / np.where(cnt > 0, cnt, np.nan)
                    fast = mean_t_chunk(sel, s, Dl[i:i + 1])[0]
                    m = np.isfinite(ref)
                    w = max(w, float(np.max(np.abs(ref[m] - fast[m]))))
                log("  check permutation %d: fast vs direct channel statistic max |diff| %.2e" % (b0 + i, w))
                if w > 1e-8:
                    sys.exit("[sel] fast formula failed on a permutation")

    def count(Sblk, b0, b1):
        Tn = (Sblk.astype(np.float64) - st["mu"]) / st["sd"]
        famax[b0:b1] = Tn[:, st["idx"]].max(axis=1)
        ge = np.maximum.reduceat(Tn[:, st["order"]], st["starts"], axis=1) >= st["pobs"][None, :]
        Hb = H[b0:b1]
        bi = np.arange(b0, b1)
        for k in use:
            m = (bi < a.n_ref) if k == "ref" else (np.ones(b1 - b0, bool) if k == "all" else rule_int[k](Hb))
            st["cnt"][k] += ge[m].sum(axis=0)
            st["nA"][k] += int(m.sum())

    t0 = time.time()
    buf = np.empty((a.chunk, len(g)), np.int64)
    Sbuf = np.empty((a.chunk, nkey), np.float32)
    b0 = 0
    for b in range(a.n_total):
        zp = ZR.copy()
        p = rng.permutation(g)
        zp[g] = ZR[p]
        SL, SR, _ = agg(pre, nkey, zp)
        s32 = SL.astype(np.float32) + SR.astype(np.float32)
        if b < a.n_ref:
            Sref[b] = s32
        else:
            Sbuf[b - b0] = s32
        buf[b - b0] = p
        last = b == a.n_total - 1
        if b - b0 + 1 == a.chunk or b == a.n_ref - 1 or last:
            hubs_for(buf[: b - b0 + 1], b0, b + 1, b0 == 0 and a.n_check > 0)
            if b == a.n_ref - 1:
                P = np.vstack([o[None, :], Sref])
                mu = np.nanmean(P, axis=0); sd = np.nanstd(P, axis=0)
                del P
                sd = np.where(sd > 1e-9, sd, np.nan)
                t_ref = (o - mu) / sd
                idx = np.where(np.isfinite(t_ref))[0]
                gi = keep.iloc[idx].groupby(["ligand", "receptor"], sort=False).indices
                pkeys = list(gi)
                order = idx[np.concatenate([np.asarray(gi[k]) for k in pkeys])]
                starts = np.cumsum([0] + [len(gi[k]) for k in pkeys])[:-1]
                pobs = np.maximum.reduceat(t_ref[order], starts)
                st.update(mu=mu, sd=sd, idx=idx, order=order, starts=starts, pobs=pobs,
                          cnt={k: np.zeros(len(pkeys)) for k in use}, nA={k: 0 for k in use})
                for c0 in range(0, a.n_ref, 2000):
                    c1 = min(c0 + 2000, a.n_ref)
                    count(Sref[c0:c1], c0, c1)
                del Sref
            elif b >= a.n_ref:
                count(Sbuf[: b - b0 + 1], b0, b + 1)
            b0 = b + 1
        if (b + 1) % max(1, a.n_total // 20) == 0:
            el = time.time() - t0
            log("  %7d/%d  %.1f min, %.1f min left" % (b + 1, a.n_total, el / 60,
                                                       el / 60 * (a.n_total - b - 1) / (b + 1)))

    def pv(null, obs):
        s = np.sort(null)
        return (1 + (len(s) - np.searchsorted(s, obs, side="left"))) / (1 + len(s))

    bi = np.arange(a.n_total)
    masks = {"ref": bi < a.n_ref, "all": np.ones(a.n_total, bool)}
    for k in use[2:]:
        masks[k] = rule_int[k](H)
    t_ref, idx, pobs = (o - st["mu"]) / st["sd"], st["idx"], st["pobs"]

    log("== reproduction of the reference run (first %d permutations)" % a.n_ref)
    R = keep.copy()
    R["ch"] = ["%s->%s" % tuple(lab(x) for x in re.match(r"(\d+)->(\d+)", c).groups()) for c in R.channel]
    R["t"] = t_ref
    for k, m in masks.items():
        R["p_fwer_%s" % k] = np.nan
        R.loc[R.index[idx], "p_fwer_%s" % k] = pv(famax[m], t_ref[idx])
    if a.ref_maxt and os.path.exists(a.ref_maxt):
        PM = pd.read_csv(a.ref_maxt, sep=TAB)
        M = R.merge(PM[K + ["t", "p_fwer_A_sum"]], on=K, how="outer", suffixes=("", "_given"), indicator=True)
        both = M[M._merge == "both"]
        dt = np.nanmax(np.abs(both.t - both.t_given)) if len(both) else np.nan
        dp = np.nanmax(np.abs(both.p_fwer_ref - both.p_fwer_A_sum)) if len(both) else np.nan
        allrows = bool((M._merge == "both").all())
        verdict = ("REPRODUCED" if allrows and dt < 1e-8 and dp < 1e-12 else
                   "REPRODUCED up to ties (one permutation)" if allrows and dt < 1e-8 and dp <= 1.0001 / (1 + a.n_ref)
                   else "NOT REPRODUCED: check arguments before reading the selective P values")
        log("  rows: shared %d, only here %d, only in --ref_maxt %d" % (
            len(both), int((M._merge == "left_only").sum()), int((M._merge == "right_only").sum())))
        log("  max |t - t_ref| %.2e   max |p_fwer - p_fwer_ref| %.2e   %s" % (dt, dp, verdict))
    else:
        log("  no --ref_maxt given; reproduction not checked")

    pkeys_df = keep.iloc[st["order"]][["ligand", "receptor"]].iloc[st["starts"]].reset_index(drop=True)
    pair_tab = pkeys_df.copy()
    pair_tab["t_pair"] = pobs
    for k, m in masks.items():
        pp = (1 + st["cnt"][k]) / (1 + st["nA"][k])
        pair_tab["p_fwer_%s" % k] = pv(famax[m], pobs)
        pair_tab["p_pair_%s" % k] = pp
        pair_tab["q_pair_%s" % k] = bh(pp)
    pair_tab = pair_tab.sort_values("t_pair", ascending=False)
    R.sort_values("t", ascending=False).to_csv(a.out_prefix + "_rows.tsv", sep=TAB, index=False)
    pair_tab.to_csv(a.out_prefix + "_pairs.tsv", sep=TAB, index=False)
    np.savez_compressed(a.out_prefix + "_perm.npz", famax=famax, hubs=H, hub_codes=np.array(codes),
                        settings=np.array(sel["settings"]), n_ref=a.n_ref)

    log("== selection (P_FWER uses the studentization of the reference run throughout)")
    c95 = np.quantile(famax[masks["all"]], 0.95)
    log("  family maximum, all %d permutations: median %.3f, 95th percentile %.3f"
        % (a.n_total, np.median(famax), c95))
    for k in use[2:]:
        m = masks[k]
        nA = int(m.sum())
        e = float((famax[m] >= c95).mean())
        se = np.sqrt(e * (1 - e) / max(nA, 1))
        log("  rule %-14s |A| = %d (%.3f)   family maximum in A: median %.3f, 95th %.3f;"
            "  in not A: 95th %.3f" % (k, nA, nA / a.n_total, np.median(famax[m]),
                                      np.percentile(famax[m], 95), np.percentile(famax[~m], 95)))
        log("      FWER of the unadjusted procedure conditional on A: %.4f (95%% CI %.4f-%.4f; nominal 0.05)"
            % (e, e - 1.96 * se, e + 1.96 * se))
    hc = pd.Series(H[:, 0]).map(lambda i: lab(codes[i]) if i >= 0 else "NA").value_counts()
    log("  hub in %s across permutations: %s" % (sel["settings"][0], ", ".join("%s %d" % kv for kv in hc.items())))

    log("== pairs (ref = reference run; all = %d permutations; other columns conditional on A)" % a.n_total)
    cols = ["ligand", "receptor", "t_pair"] + ["p_fwer_%s" % k for k in masks] + ["q_pair_%s" % k for k in masks]
    top = pair_tab[(pair_tab[["p_fwer_%s" % k for k in masks]].min(axis=1) < 0.2)
                   | (pair_tab[["q_pair_%s" % k for k in masks]].min(axis=1) < 0.2)]
    if top.empty:
        top = pair_tab.head(5)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        log(top[cols].to_string(index=False, float_format=lambda x: "%.4g" % x))
    log("  |A| per column: " + ", ".join("%s %d" % (k, st["nA"][k]) for k in use))
    log("\nwrote %s_{rows,pairs}.tsv and %s_perm.npz  (%.1f min)" % (a.out_prefix, a.out_prefix,
                                                                    (time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
