#!/usr/bin/env python3
"""Pair-level communication axis: Z = (W - median) / (1.4826 MAD) against matched control pairs,
T = median of Z over sections, p_fwer from the maximum over the pairs of each channel."""
import argparse
import glob
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import celltypes

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = "\t"


def parse_channels(spec, cands, CT):
    if not spec:
        return list(cands)
    inv = {v: k for k, v in CT.items()}

    def code(tok):
        tok = tok.strip()
        if tok.isdigit():
            return int(tok)
        if tok in inv:
            return inv[tok]
        sys.exit("[np1] unknown cell type '%s'; known %s" % (tok, sorted(inv)))

    s = spec.strip()
    if s.upper().startswith("ALL_TO_HUB:"):
        h = code(s.split(":", 1)[1])
        keep = [c for c in cands if c[1] == h]
    elif s.upper().startswith("ALL_FROM:"):
        q = code(s.split(":", 1)[1])
        keep = [c for c in cands if c[0] == q]
    else:
        want = set()
        for tok in s.split(","):
            if not tok.strip():
                continue
            base = tok.split(":")[0]
            aa, bb = base.split("->")
            want.add((code(aa), code(bb)))
        keep = [c for c in cands if (c[0], c[1]) in want]
    if not keep:
        sys.exit("[np1] --channels '%s' matched no channel" % spec)
    return keep


def loo_z(w_obs, W_ctrl):
    B, n = W_ctrl.shape
    med_all = np.median(W_ctrl, axis=0)
    mad_all = np.median(np.abs(W_ctrl - med_all), axis=0) * 1.4826
    ok = mad_all > 1e-12
    Z_obs = np.full(n, np.nan)
    Z_obs[ok] = (w_obs[ok] - med_all[ok]) / mad_all[ok]
    Z_null = np.full((B, n), np.nan)
    if ok.any():
        Z_null[:, ok] = (W_ctrl[:, ok] - med_all[ok]) / mad_all[ok]
    return Z_obs, Z_null, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl_dir", required=True)
    ap.add_argument("--code", default=HERE, help="directory with ccgwas_v4core.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", default="",
                    help="ALL_TO_HUB:<code or name>, ALL_FROM:<...>, or a list such as 'Ast->Mic,End->Mic'")
    ap.add_argument("--mode", default="paracrine")
    ap.add_argument("--only_mode", action="store_true", default=True,
                    help="preprocess only the signalling mode given by --mode (default; saves memory)")
    ap.add_argument("--all_modes", dest="only_mode", action="store_false")
    ap.add_argument("--max_cell_pairs", type=int, default=5000)
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--samples", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    from ccgwas_v4core import (CCGWASConfig, preprocess_inputs,
                               build_channel_cell_pairs,
                               build_channel_specific_control_pool,
                               compute_W_batch)
    try:
        from ccgwas_v4core import MIN_EFFECTIVE_CONTROLS_LOWCONF as MINEFF
    except ImportError:
        MINEFF = 5

    pkls = sorted(glob.glob(os.path.join(a.pkl_dir, "inputs_*.pkl")))
    if a.samples:
        want = {x.strip() for x in a.samples.split(",")}
        pkls = [p for p in pkls
                if re.sub(r"^inputs_|\.pkl$", "", os.path.basename(p)) in want]
    if not pkls:
        sys.exit("[np1] no inputs_*.pkl in %s" % a.pkl_dir)
    print("[np1] %d sections, mode=%s, B=%d" % (len(pkls), a.mode, a.n_null))

    config = CCGWASConfig(random_seed=a.seed)
    for f in ("n_pseudo_db_reps", "n_null_draws", "B"):
        if hasattr(config, f):
            setattr(config, f, a.n_null)

    store, CT, VOCAB = {}, {}, None
    t_all = time.time()
    for si, pkl in enumerate(pkls, 1):
        smp = re.sub(r"^inputs_|\.pkl$", "", os.path.basename(pkl))
        print("\n[%2d/%d] %s" % (si, len(pkls), smp), flush=True)
        d = pickle.load(open(pkl, "rb"))
        inputs = d["inputs"]
        inputs = inputs.item() if isinstance(inputs, np.ndarray) else inputs
        VOCAB = celltypes.same_vocab(d["ct_categories"], VOCAB, pkl)
        if not CT:
            CT = {i: str(c) for i, c in enumerate(d["ct_categories"])}
            print("  cell types: %s" % ", ".join("%d=%s" % kv for kv in CT.items()))
        gene_names = [str(g) for g in inputs.gene_names]

        t0 = time.time()
        if a.only_mode:
            _rad = dict(getattr(config, "signaling_radii", {}) or {})
            if a.mode in _rad and len(_rad) > 1:
                config.signaling_radii = {a.mode: _rad[a.mode]}
                print("  mode %s only (of %s)"
                      % (a.mode, ",".join(_rad)))
            elif a.mode not in _rad:
                print("  !! %s not in config.signaling_radii (%s); all modes kept"
                      % (a.mode, ",".join(_rad)))
        preproc = preprocess_inputs(inputs, config)
        print("  preprocess %.0fs" % (time.time() - t0), flush=True)
        cp = build_channel_cell_pairs(inputs, config)
        cands = [c for c in cp if c[2] == a.mode]
        keep = parse_channels(a.channels, cands, CT)
        print("  channels %d -> %d" % (len(cands), len(keep)))

        rng0 = np.random.default_rng(a.seed)
        cell_pairs = {}
        for c in keep:
            i_, j_ = cp[c]
            if a.max_cell_pairs and len(i_) > a.max_cell_pairs:
                s_ = rng0.choice(len(i_), a.max_cell_pairs, replace=False)
                i_, j_ = i_[s_], j_[s_]
            cell_pairs[c] = (i_, j_)

        t0 = time.time()
        pool = build_channel_specific_control_pool(inputs, preproc, config, cell_pairs)
        print("  control pool %.0fs" % (time.time() - t0), flush=True)
        ctrl = pool.control_pairs
        expr = preproc.expr_norm_test

        real_by_mode = {}
        for _, row in inputs.LR_pairs.iterrows():
            real_by_mode.setdefault(row["mode"], []).append(
                (int(row["ligand_idx"]), int(row["receptor_idx"])))

        for c in keep:
            if c not in ctrl:
                continue
            real_pairs = real_by_mode.get(c[2], [])
            assign = ctrl[c]
            valid = [k for k, cs in enumerate(assign) if len(cs) >= MINEFF]
            if len(valid) < 5:
                continue
            i_arr, j_arr = cell_pairs[c]
            Ei, Ej = expr[i_arr], expr[j_arr]
            L = np.array([real_pairs[k][0] for k in valid])
            R = np.array([real_pairs[k][1] for k in valid])
            W_obs = compute_W_batch(L, R, expr, i_arr, j_arr, Ei, Ej)

            B = a.n_null
            n = len(valid)
            W_null = np.zeros((B, n), dtype=np.float32)
            arrs = [np.asarray(assign[k]) for k in valid]
            for b in range(B):
                rb = np.random.default_rng((a.seed, b))
                picks = rb.integers(0, np.array([len(x) for x in arrs]))
                sl = np.array([arrs[kk][picks[kk], 0] for kk in range(n)])
                sr = np.array([arrs[kk][picks[kk], 1] for kk in range(n)])
                W_null[b] = compute_W_batch(sl, sr, expr, i_arr, j_arr, Ei, Ej)
            Zo, Zn, ok = loo_z(W_obs, W_null)
            store[(smp, c)] = dict(
                pairs=[(int(real_pairs[k][0]), int(real_pairs[k][1])) for k in valid],
                Zo=Zo, Zn=Zn, ok=ok, genes=gene_names)
            print("    %s  %d pairs, %d testable, %.0fs"
                  % ("%d->%d" % c[:2], n, int(ok.sum()), time.time() - t0), flush=True)
            del Ei, Ej, W_null

    if not store:
        sys.exit("[np1] no result")

    print("\n[np1] aggregation over sections and maxT")
    chans = sorted({c for _, c in store})
    rows = []
    for c in chans:
        keys = [(s, cc) for (s, cc) in store if cc == c]
        if not keys:
            continue
        nm = {}
        for k in keys:
            g = store[k]["genes"]
            nm[k] = [(str(g[Li]), str(g[Ri])) for (Li, Ri) in store[k]["pairs"]]
        sets = [set(nm[k]) for k in keys]
        common = sorted(set.intersection(*sets))
        if not common:
            n_each = ", ".join("%s:%d" % (k[0], len(nm[k])) for k in keys)
            print("  %s: no pair common to all sections, skipped (%s)"
                  % ("%d->%d" % c[:2], n_each))
            continue
        idxs = []
        for k in keys:
            pmap = {}
            for i, key in enumerate(nm[k]):
                pmap.setdefault(key, i)
            idxs.append(np.array([pmap[p] for p in common]))

        Zo = np.stack([store[k]["Zo"][idxs[j]] for j, k in enumerate(keys)])
        Zn = np.stack([store[k]["Zn"][:, idxs[j]] for j, k in enumerate(keys)])
        ok = np.all(np.stack([store[k]["ok"][idxs[j]]
                              for j, k in enumerate(keys)]), axis=0)
        T_obs = np.nanmedian(Zo, axis=0)
        T_null = np.nanmedian(Zn, axis=0)
        B = T_null.shape[0]
        Tn_ok = np.where(ok[None, :], T_null, -np.inf)
        M = np.nanmax(Tn_ok, axis=1) if ok.any() else np.full(B, -np.inf)
        del Zo, Zn, Tn_ok

        for ii, (Lname, Rname) in enumerate(common):
            if not ok[ii]:
                rows.append(dict(channel="%d->%d:%s" % c, sender=CT.get(c[0]),
                                 receiver=CT.get(c[1]),
                                 ligand=Lname, receptor=Rname,
                                 T=np.nan, p_fwer=np.nan, p_raw=np.nan,
                                 n_slice=len(keys), status="NOT_TESTABLE"))
                continue
            p_f = (1 + int((M >= T_obs[ii]).sum())) / (B + 1)
            p_r = (1 + int((T_null[:, ii] >= T_obs[ii]).sum())) / (B + 1)
            rows.append(dict(channel="%d->%d:%s" % c, sender=CT.get(c[0]),
                             receiver=CT.get(c[1]),
                             ligand=Lname, receptor=Rname,
                             T=float(T_obs[ii]), p_fwer=p_f, p_raw=p_r,
                             n_slice=len(keys), status="OK"))
        nsig = sum(1 for r in rows[-len(common):] if r["status"] == "OK"
                   and r["p_fwer"] < 0.05)
        print("  %-12s %d pairs (%d testable), %d with p_fwer < 0.05"
              % ("%d->%d" % c[:2], len(common), int(ok.sum()), nsig))

    R = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    R.to_csv(a.out, sep=TAB, index=False)
    print("\nwrote %s (%d rows)" % (a.out, len(R)))
    if not len(R) or "status" not in R.columns:
        print("[np1] no testable pair; check --channels and --samples")
        return
    S = R[R.status == "OK"]
    if len(S):
        print("\npairs with p_fwer < 0.05")
        top = S[S.p_fwer < 0.05].nsmallest(20, "p_fwer")
        if len(top):
            print(top[["channel", "ligand", "receptor", "T", "p_fwer", "p_raw"]]
                  .to_string(index=False))
        else:
            print("  none; strongest five:")
            print(S.nsmallest(5, "p_fwer")[["channel", "ligand", "receptor",
                                            "T", "p_fwer", "p_raw"]]
                  .to_string(index=False))
    print("\n%.0f min" % ((time.time() - t_all) / 60))


def _peak_rss():
    try:
        import resource
        k = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if k > 1 << 40:
            k //= 1024
        print("[peak] maxrss %d KB = %.1f GB" % (k, k / 1024 / 1024), flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        _peak_rss()
