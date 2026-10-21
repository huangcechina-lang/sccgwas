#!/usr/bin/env python3
"""Control ligand-receptor pairs for the beyond-niche threshold (<out>_nullE.tsv, <out>_ztypeI.tsv)."""
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


def strat_bins(v, n_bin=10):
    v = np.asarray(v, float)
    q = np.quantile(v[np.isfinite(v)], np.linspace(0, 1, n_bin + 1))
    q[0] -= 1e-9
    q[-1] += 1e-9
    return np.clip(np.digitize(v, q) - 1, 0, n_bin - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gss_dir", required=True)
    ap.add_argument("--pkl_dir", required=True)
    ap.add_argument("--genes_out", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--hub", required=True, help="receiving cell type (code or name)")
    ap.add_argument("--ct_names", default="",
                    help="cell-type names in code order (comma list) or a pkl with ct_categories")
    ap.add_argument("--mode", default="paracrine")
    ap.add_argument("--n_match", type=int, default=1000, help="control pairs per channel and section")
    ap.add_argument("--n_perm_e", type=int, default=200, help="within-block permutations per control pair")
    ap.add_argument("--n_fake_traits", type=int, default=50, help="permuted traits for the z type-I check")
    ap.add_argument("--n_perm_z", type=int, default=1000, help="gene-label permutations per permuted trait")
    ap.add_argument("--K_niche", type=int, default=10)
    ap.add_argument("--k_spatial", type=int, default=30)
    ap.add_argument("--max_cell_pairs", type=int, default=5000)
    ap.add_argument("--samples", default="", help="comma-separated sections; default all")
    ap.add_argument("--code", default=HERE, help="directory with ccgwas_v4core.py")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    CT = celltypes.resolve(a.ct_names, celltypes.codes_from_gss(a.gss_dir, a.mode))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    sys.path.insert(0, a.code)
    from niches import build_niches
    from sklearn.neighbors import NearestNeighbors

    zdf = load_z(a.genes_out)
    H = celltypes.hub_code(a.hub, CT)
    fwd = ["%s->%s:%s" % (x, H, a.mode) for x in sorted(CT) if x != H]
    print("hub %s, %d channels" % (CT.get(H, H), len(fwd)))
    gss_files = sorted(glob.glob(os.path.join(a.gss_dir, "gss_spec_*.tsv")))
    if a.samples:
        want = {x.strip() for x in a.samples.split(",")}
        gss_files = [f for f in gss_files
                     if re.sub(r"^gss_spec_|\.tsv$", "", os.path.basename(f)) in want]
    rng = np.random.default_rng(a.seed)
    rec_e, rec_z = [], []
    t_all = time.time()

    VOCAB = None
    for fi, gf in enumerate(gss_files, 1):
        smp = re.sub(r"^gss_spec_|\.tsv$", "", os.path.basename(gf))
        pkl = os.path.join(a.pkl_dir, "inputs_%s.pkl" % smp)
        if not os.path.exists(pkl):
            continue
        G = pd.read_csv(gf, sep=TAB)
        G = G.rename(columns={G.columns[0]: "gene"}).set_index("gene")
        G.index = G.index.astype(str)
        cols = [c for c in G.columns if str(c).endswith(a.mode)]
        need = [c for c in fwd if c in cols]
        if not need:
            continue

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
        genes_pkl = [str(g) for g in inp.gene_names]
        gi = {g: i for i, g in enumerate(genes_pkl)}
        niche = build_niches(coord, ct_code, len(cts), k_spatial=a.k_spatial,
                             K=a.K_niche, seed=a.seed)
        block = ct_code.astype(np.int64) * (a.K_niche + 1) + niche.astype(np.int64)
        ublk = [np.where(block == b)[0] for b in np.unique(block)]

        det_all = (np.asarray((Y > 0).sum(0)).ravel() if sp.issparse(Y)
                   else (np.asarray(Y) > 0).sum(0)) / Y.shape[0]
        ecol = {}

        def expr_col(g):
            j = gi.get(g)
            if j is None:
                return None
            if j not in ecol:
                ecol[j] = (np.asarray(Y[:, j].todense()).ravel() if sp.issparse(Y)
                           else np.asarray(Y)[:, j]).astype(np.float32)
            return ecol[j]

        lr = d["lr_df_full"]
        lr = lr[lr["mode"] == a.mode]
        lig_pool = sorted({str(x) for x in lr["ligand_symbol"] if str(x) in gi})
        rec_pool = sorted({str(x) for x in lr["receptor_symbol"] if str(x) in gi})
        real_set = {(str(r["ligand_symbol"]), str(r["receptor_symbol"]))
                    for _, r in lr.iterrows()}
        lb = strat_bins([det_all[gi[g]] for g in lig_pool])
        rb = strat_bins([det_all[gi[g]] for g in rec_pool])
        lig_by = {b: [g for g, bb in zip(lig_pool, lb) if bb == b] for b in set(lb)}
        rec_by = {b: [g for g, bb in zip(rec_pool, rb) if bb == b] for b in set(rb)}

        t0 = time.time()
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
            if not len(S_):
                continue

            picks = []
            tries = 0
            while len(picks) < a.n_match and tries < a.n_match * 20:
                tries += 1
                b1 = rng.choice(sorted(lig_by))
                b2 = rng.choice(sorted(rec_by))
                if not lig_by[b1] or not rec_by[b2]:
                    continue
                L = lig_by[b1][rng.integers(len(lig_by[b1]))]
                R = rec_by[b2][rng.integers(len(rec_by[b2]))]
                if (L, R) in real_set or L == R:
                    continue
                picks.append((L, R))
            if not picks:
                continue
            Lm = np.stack([expr_col(L) for L, _ in picks])
            Rm = np.stack([expr_col(R) for _, R in picks])
            w_real = (Lm[:, S_] * Rm[:, R_]).mean(axis=1)
            acc = np.empty((a.n_perm_e, len(picks)), np.float32)
            for b in range(a.n_perm_e):
                perm = np.arange(len(ct_code))
                for idxs in ublk:
                    if len(idxs) > 1:
                        perm[idxs] = rng.permutation(idxs)
                acc[b] = (Lm[:, perm[S_]] * Rm[:, perm[R_]]).mean(axis=1)
            m, s = acc.mean(axis=0), acc.std(axis=0)
            Sv = (w_real - m) / np.where(s > 0, s, np.nan)
            pv = (1 + (acc >= w_real).sum(axis=0)) / (1 + a.n_perm_e)
            for k, (L, R) in enumerate(picks):
                rec_e.append(dict(sample=smp, channel=c, ligand=L, receptor=R,
                                  S=float(Sv[k]), p_raw=float(pv[k])))
        t_c = time.time() - t0

        t0 = time.time()
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

        pidx = {}
        for c in need:
            j = ci[c]
            nz = {u[i] for i in np.where(Wm[:, j] > 0)[0]}
            li, ri_ = [], []
            for _, r in lr.iterrows():
                L, R = str(r["ligand_symbol"]), str(r["receptor_symbol"])
                if L in nz and R in nz:
                    li.append(pos[L])
                    ri_.append(pos[R])
            pidx[c] = (np.array(li, int), np.array(ri_, int))

        def contrib(zv):
            zr = zsc(zv - X @ (XP @ zv))
            return {c: ((Gr[pidx[c][0], ci[c]] * zr[pidx[c][0]]
                         + Gr[pidx[c][1], ci[c]] * zr[pidx[c][1]]) / den[ci[c]]
                        if len(pidx[c][0]) and den[ci[c]] > 0 else np.array([]))
                    for c in need}

        for ft in range(a.n_fake_traits):
            zfake = z0[rng.permutation(len(u))]
            obs = contrib(zfake)
            cnt = {c: np.zeros(len(obs[c]), int) for c in obs}
            for _ in range(a.n_perm_z):
                pm = contrib(zfake[rng.permutation(len(u))])
                for c in cnt:
                    if len(cnt[c]):
                        cnt[c] += (pm[c] >= obs[c])
            for c in cnt:
                if len(cnt[c]):
                    p = (1 + cnt[c]) / (1 + a.n_perm_z)
                    rec_z.append(dict(sample=smp, channel=c, fake=ft,
                                      frac_lt_05=float((p < 0.05).mean()),
                                      frac_lt_01=float((p < 0.01).mean()),
                                      p_median=float(np.median(p)), n_pair=len(p)))
        t_a = time.time() - t0
        print("  [%2d/%d] %-10s  controls %.0fs  z type-I %.0fs  total %.0f min"
              % (fi, len(gss_files), smp, t_c, t_a, (time.time() - t_all) / 60), flush=True)

    if rec_e:
        E = pd.DataFrame(rec_e)
        E.to_csv(a.out + "_nullE.tsv", sep=TAB, index=False)
        print("\n95th percentile of S among control pairs:")
        for c, g in E.groupby("channel"):
            v = g["S"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v):
                print("  %-22s n=%5d  %.3f" % (c, len(v), np.percentile(v, 95)))
        print("wrote %s_nullE.tsv" % a.out)
    if rec_z:
        Z = pd.DataFrame(rec_z)
        Z.to_csv(a.out + "_ztypeI.tsv", sep=TAB, index=False)
        print("z type-I at 0.05: %.4f; wrote %s_ztypeI.tsv" % (Z["frac_lt_05"].mean(), a.out))


if __name__ == "__main__":
    main()
