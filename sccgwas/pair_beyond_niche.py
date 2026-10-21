#!/usr/bin/env python3
"""Beyond-niche axis per section: S = (W - mean) / sd under permutation within cell-type x niche blocks."""
import argparse
import glob
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse as sp

import celltypes

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = "\t"
RADIUS = {"juxtacrine": 15.0, "paracrine": 80.0, "secreted": 200.0}


def zsc(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def load_z(path):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gss_dir", required=True, help="directory with gss_spec_<section>.tsv")
    ap.add_argument("--pkl_dir", required=True, help="directory with inputs_<section>.pkl")
    ap.add_argument("--genes_out", required=True, help="MAGMA .genes.out (gene symbols)")
    ap.add_argument("--out", required=True, help="output per-section table")
    ap.add_argument("--hub", default="", help="receiving cell type (code or name)")
    ap.add_argument("--channels", default="",
                    help="explicit directed channels, e.g. '0->4,1->4' or 'Ast->Mic'; overrides --hub")
    ap.add_argument("--ct_names", default="",
                    help="cell-type names in code order (comma list) or a pkl with ct_categories")
    ap.add_argument("--trait", default="")
    ap.add_argument("--mode", default="paracrine")
    ap.add_argument("--n_perm_z", type=int, default=5000)
    ap.add_argument("--n_perm_e", type=int, default=200)
    ap.add_argument("--K_niche", type=int, default=10)
    ap.add_argument("--k_spatial", type=int, default=30)
    ap.add_argument("--max_cell_pairs", type=int, default=5000)
    ap.add_argument("--code", default=HERE, help="directory with ccgwas_v4core.py")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.hub and not a.channels:
        sys.exit("give --hub or --channels")
    CT = celltypes.resolve(a.ct_names, celltypes.codes_from_gss(a.gss_dir, a.mode))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    sys.path.insert(0, a.code)
    from niches import build_niches

    zdf = load_z(a.genes_out)
    H = celltypes.hub_code(a.hub, CT) if a.hub else ""
    if a.channels:
        _inv = {v: k for k, v in CT.items()}
        def _code(x):
            x = x.strip()
            return _inv.get(x, x)
        _ch = []
        for c in a.channels.split(","):
            c = c.strip()
            if not c:
                continue
            s_, r_ = c.split("->")
            s_, r_ = _code(s_), _code(r_)
            if s_ == r_:
                continue
            _ch.append("%s->%s:%s" % (s_, r_, a.mode))
        fwd = sorted(set(_ch))
        rev = sorted({"%s->%s:%s" % (c.split(":")[0].split("->")[1],
                                     c.split(":")[0].split("->")[0], a.mode)
                      for c in fwd})
        print("%d directed channels" % len(fwd))
    else:
        fwd = ["%s->%s:%s" % (x, H, a.mode) for x in sorted(CT) if x != H]
        rev = ["%s->%s:%s" % (H, x, a.mode) for x in sorted(CT) if x != H]
    print("hub %s, %d channel pairs, trait %s" % (CT.get(H, H), len(fwd), a.trait))

    gss_files = sorted(glob.glob(os.path.join(a.gss_dir, "gss_spec_*.tsv")))
    rng = np.random.default_rng(a.seed)
    recs = []
    t_all = time.time()

    VOCAB = None
    for fi, gf in enumerate(gss_files, 1):
        smp = re.sub(r"^gss_spec_|\.tsv$", "", os.path.basename(gf))
        pkl = os.path.join(a.pkl_dir, "inputs_%s.pkl" % smp)
        if not os.path.exists(pkl):
            print("  [%s] no pkl, skipped" % smp)
            continue
        G = pd.read_csv(gf, sep=TAB)
        G = G.rename(columns={G.columns[0]: "gene"}).set_index("gene")
        G.index = G.index.astype(str)
        cols = [c for c in G.columns if str(c).endswith(a.mode)]
        need = [c for c in fwd + rev if c in cols]
        if len(need) < 2:
            continue

        u = [g for g in G.index if g in zdf.index]
        pos = {g: i for i, g in enumerate(u)}
        Wm = G.loc[u, cols].to_numpy(float)
        gl = zdf.loc[u, "glen"].to_numpy(float)
        ns = zdf.loc[u, "nsnps"].to_numpy(float)
        X = np.column_stack([np.ones(len(u)), zsc(np.log(np.maximum(gl, 1))),
                             zsc(np.log(np.maximum(ns, 1))), zsc(gl), zsc(ns)])
        XP = np.linalg.pinv(X)
        mu, sd = Wm.mean(axis=0), Wm.std(axis=0)
        Gr = (Wm - mu) / np.where(sd > 0, sd, 1.0)
        Gr = Gr - X @ (XP @ Gr)
        den = np.sqrt(np.einsum("ij,ij->j", Gr, Gr))
        ci = {c: i for i, c in enumerate(cols)}
        z0 = zdf.loc[u, "z"].to_numpy(float)

        d = pickle.load(open(pkl, "rb"))
        inp = d["inputs"]
        inp = inp.item() if isinstance(inp, np.ndarray) else inp
        cts = list(d["ct_categories"])
        VOCAB = celltypes.same_vocab(cts, VOCAB, pkl)
        ct_raw = np.asarray(inp.cell_type)
        ct_code = (ct_raw.astype(int) if np.issubdtype(ct_raw.dtype, np.number)
                   else np.array([cts.index(str(x)) for x in ct_raw]))
        coord = np.asarray(inp.coord, np.float32)
        Y = inp.Y
        gi = {str(g): i for i, g in enumerate(inp.gene_names)}
        niche = build_niches(coord, ct_code, len(cts), k_spatial=a.k_spatial,
                             K=a.K_niche, seed=a.seed)
        block = ct_code.astype(np.int64) * (a.K_niche + 1) + niche.astype(np.int64)
        ublk = {b: np.where(block == b)[0] for b in np.unique(block)}
        ecol = {}

        def expr_col(g):
            j = gi.get(g)
            if j is None:
                return None
            if j not in ecol:
                ecol[j] = (np.asarray(Y[:, j].todense()).ravel() if sp.issparse(Y)
                           else np.asarray(Y)[:, j]).astype(np.float32)
            return ecol[j]

        from sklearn.neighbors import NearestNeighbors
        pair_idx = {}
        for c in need:
            head, _, md = c.partition(":")
            sc, rc = head.split("->")
            si = np.where(ct_code == int(sc))[0]
            ri = np.where(ct_code == int(rc))[0]
            if not len(si) or not len(ri):
                continue
            nn = NearestNeighbors(radius=RADIUS[md]).fit(coord[ri])
            nb = nn.radius_neighbors(coord[si], return_distance=False)
            S_, R_ = [], []
            for x, nbx in enumerate(nb):
                if len(nbx):
                    S_.append(np.full(len(nbx), si[x]))
                    R_.append(ri[nbx])
            S_ = np.concatenate(S_) if S_ else np.array([], int)
            R_ = np.concatenate(R_) if R_ else np.array([], int)
            if sc == rc and len(S_):
                k = S_ != R_
                S_, R_ = S_[k], R_[k]
            if a.max_cell_pairs and len(S_) > a.max_cell_pairs:
                s2 = rng.choice(len(S_), a.max_cell_pairs, replace=False)
                S_, R_ = S_[s2], R_[s2]
            pair_idx[c] = (S_, R_)

        lr = d["lr_df_full"]
        lr = lr[lr["mode"] == a.mode]
        pairs_of = {}
        for c in need:
            if c not in pair_idx:
                continue
            j = ci[c]
            nz = {u[i] for i in np.where(Wm[:, j] > 0)[0]}
            names, li, ri_ = [], [], []
            for _, r in lr.iterrows():
                L, R = str(r["ligand_symbol"]), str(r["receptor_symbol"])
                if L in nz and R in nz and L in gi and R in gi:
                    names.append((L, R))
                    li.append(pos[L])
                    ri_.append(pos[R])
            pairs_of[c] = (np.array(li, int), np.array(ri_, int), names)

        def contrib(zv):
            zr = zsc(zv - X @ (XP @ zv))
            out = {}
            for c in pairs_of:
                j = ci[c]
                li, ri_, _ = pairs_of[c]
                out[c] = ((Gr[li, j] * zr[li] + Gr[ri_, j] * zr[ri_]) / den[j]
                          if len(li) and den[j] > 0 else np.array([]))
            return out

        obs = contrib(z0)
        cnt = {c: np.zeros(len(obs[c]), int) for c in obs}

        pair_key = {}
        for c in pairs_of:
            if c not in fwd:
                continue
            head = c.split(":")[0]
            sc_, rc_ = head.split("->")
            crev = "%s->%s:%s" % (rc_, sc_, a.mode)
            if crev not in pairs_of:
                continue
            nf = {nm: k for k, nm in enumerate(pairs_of[c][2])}
            nr = {nm: k for k, nm in enumerate(pairs_of[crev][2])}
            for nm in set(nf) & set(nr):
                pair_key[(nm[0], nm[1], CT.get(sc_, sc_), CT.get(rc_, rc_))] = (c, crev, nf[nm], nr[nm])

        def dvec(cb):
            return {k: float(cb[cf][i] - cb[cr][j])
                    for k, (cf, cr, i, j) in pair_key.items()
                    if len(cb[cf]) > i and len(cb[cr]) > j}

        d_obs = dvec(obs)
        d_cnt = {k: 0 for k in d_obs}
        t0 = time.time()
        for _ in range(a.n_perm_z):
            pm = contrib(z0[rng.permutation(len(u))])
            for c in cnt:
                if len(cnt[c]):
                    cnt[c] += (pm[c] >= obs[c])
            dp = dvec(pm)
            for k, v in d_obs.items():
                if dp.get(k, -np.inf) >= v:
                    d_cnt[k] += 1
        pz = {c: (1 + cnt[c]) / (1 + a.n_perm_z) for c in cnt}
        pdz = {k: (1 + d_cnt[k]) / (1 + a.n_perm_z) for k in d_obs}
        t_a = time.time() - t0

        t0 = time.time()
        pe, Se = {}, {}
        for c in pairs_of:
            _, _, names = pairs_of[c]
            S_, R_ = pair_idx[c]
            if not len(S_) or not names:
                pe[c] = np.full(len(names), np.nan)
                Se[c] = np.full(len(names), np.nan)
                continue
            Lm = np.stack([expr_col(L) for L, _ in names])
            Rm = np.stack([expr_col(R) for _, R in names])
            w_real = (Lm[:, S_] * Rm[:, R_]).mean(axis=1)
            acc = np.empty((a.n_perm_e, len(names)), np.float32)
            for b in range(a.n_perm_e):
                perm = np.arange(len(ct_code))
                for idxs in ublk.values():
                    if len(idxs) > 1:
                        perm[idxs] = rng.permutation(idxs)
                acc[b] = (Lm[:, perm[S_]] * Rm[:, perm[R_]]).mean(axis=1)
            m, s = acc.mean(axis=0), acc.std(axis=0)
            Se[c] = (w_real - m) / np.where(s > 0, s, np.nan)
            pe[c] = (1 + (acc >= w_real).sum(axis=0)) / (1 + a.n_perm_e)
        t_c = time.time() - t0

        for c in pairs_of:
            _, _, names = pairs_of[c]
            head = c.split(":")[0]
            sc, rc = head.split("->")
            for k, (L, R) in enumerate(names):
                _s, _r = CT.get(sc, sc), CT.get(rc, rc)
                kk = (L, R, _s, _r) if (L, R, _s, _r) in d_obs \
                    else (L, R, _r, _s)
                recs.append(dict(sample=smp, channel=c,
                                 sender=CT.get(sc, sc), receiver=CT.get(rc, rc),
                                 ligand=L, receptor=R,
                                 contrib=float(obs[c][k]), p_z=float(pz[c][k]),
                                 S_nullE=float(Se[c][k]), p_nullE=float(pe[c][k]),
                                 d_fwd_minus_rev=float(d_obs.get(kk, np.nan)),
                                 p_dir_zperm=float(pdz.get(kk, np.nan)),
                                 direction="fwd" if c in fwd else "rev"))
        print("  [%2d/%d] %-10s %d channels %d pairs  z-perm %.0fs  niche-perm %.0fs  total %.0f min"
              % (fi, len(gss_files), smp, len(pairs_of),
                 sum(len(pairs_of[c][2]) for c in pairs_of), t_a, t_c,
                 (time.time() - t_all) / 60), flush=True)


    if not recs:
        sys.exit("no result")
    D = pd.DataFrame(recs)
    D.to_csv(a.out, sep=TAB, index=False)
    print("wrote %s (%d rows, %d sections, %d channels)"
          % (a.out, len(D), D["sample"].nunique(), D["channel"].nunique()))


if __name__ == "__main__":
    main()
