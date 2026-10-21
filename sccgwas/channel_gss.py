#!/usr/bin/env python3
"""Channel specificity scores (GSS) for one section, normalized within mode by each gene's mean over channels."""
import argparse, os, pickle, sys
import numpy as np
from scipy import sparse as sp
import pandas as pd
from sklearn.neighbors import NearestNeighbors

MODE_RADIUS = {"juxtacrine": 15.0, "paracrine": 80.0, "secreted": 200.0}
CHUNK = 40_000


def directed_pairs(coord, cell_type_str, c_s, c_r, radius, max_pairs, seed):
    si = np.where(cell_type_str == c_s)[0]
    ri = np.where(cell_type_str == c_r)[0]
    if len(si) == 0 or len(ri) == 0:
        return np.array([], int), np.array([], int)
    nn = NearestNeighbors(radius=radius).fit(coord[ri])
    nbrs = nn.radius_neighbors(coord[si], return_distance=False)
    S, R = [], []
    for a, nb in enumerate(nbrs):
        if len(nb):
            S.append(np.full(len(nb), si[a])); R.append(ri[nb])
    if not S:
        return np.array([], int), np.array([], int)
    S = np.concatenate(S); R = np.concatenate(R)
    if c_s == c_r:
        m = S != R; S, R = S[m], R[m]
    if len(S) > max_pairs:
        idx = np.random.default_rng(seed).choice(len(S), max_pairs, replace=False)
        S, R = S[idx], R[idx]
    return S, R


def channel_W(expr, S, R, Lg, Rg):
    EL = expr[:, Lg]; ER = expr[:, Rg]
    P = len(S); acc = np.zeros(EL.shape[1], np.float64)
    for st in range(0, P, CHUNK):
        sl = slice(st, st + CHUNK)
        acc += (EL[S[sl]] * ER[R[sl]]).sum(0)
    return acc / max(P, 1)


def eff_n_genes(gss):
    w = gss[gss > 0]
    if w.size == 0:
        return 0.0
    s = w.sum()
    return float(s * s / np.sum(w * w)) if s > 0 else 0.0


def normalize_channels(A, modes, how, eps_frac):
    A = np.asarray(A, np.float64)
    out = np.zeros_like(A)
    rep = []
    for md in sorted(set(modes)):
        cidx = np.array([i for i, m in enumerate(modes) if m == md])
        nonempty = cidx[[A[:, i].any() for i in cidx]]
        if len(nonempty) < 2:
            for i in cidx:
                out[:, i] = A[:, i]
            rep.append(dict(mode=md, n_channels=len(cidx), n_nonempty=len(nonempty),
                            eps=np.nan, n_genes_shrunk=0,
                            note="<2 non-empty channels -> passed through unnormalised"))
            continue
        ref = A[:, nonempty]
        rmean = ref.mean(axis=1)
        pos = rmean[rmean > 0]
        eps = eps_frac * float(np.median(pos)) if pos.size else 0.0
        if how == "specificity":
            out[:, nonempty] = ref / (rmean[:, None] + eps)
        elif how == "zscore":
            rsd = ref.std(axis=1)
            keep = rsd > 0
            z = np.zeros_like(ref)
            z[keep] = (ref[keep] - rmean[keep, None]) / rsd[keep, None]
            out[:, nonempty] = z
        else:
            raise SystemExit(f"unknown --channel_norm {how}")
        rep.append(dict(mode=md, n_channels=len(cidx), n_nonempty=len(nonempty),
                        eps=eps, n_genes_shrunk=int(np.sum((rmean > 0) & (rmean < eps))),
                        note=""))
    return out, rep


def separability(A, labels, modes, tag, min_informative_genes=50):
    rows = []
    key = {}
    for i, lab in enumerate(labels):
        head = lab.split("|")[0]
        if "->" in head and ":" in head:
            cs_cr, md = head.rsplit(":", 1)
            cs, cr = cs_cr.split("->", 1)
            key[(cs, cr, md)] = i
    for md in sorted(set(modes)):
        cidx = [i for i, m in enumerate(modes) if m == md and A[:, i].any()]
        if len(cidx) < 2:
            continue
        M = A[:, cidx]
        keepg = M.std(axis=1) > 0
        n_inf = int(keepg.sum())
        degenerate = n_inf < min_informative_genes
        C = np.corrcoef(M[keepg].T) if n_inf > 2 else np.full((len(cidx),) * 2, np.nan)
        off = C[np.triu_indices(len(cidx), 1)]
        off = off[np.isfinite(off)]
        fr = []
        for (cs, cr, m_), i in key.items():
            if m_ != md or cs == cr:
                continue
            j = key.get((cr, cs, m_))
            if j is None or i not in cidx or j not in cidx:
                continue
            if i < j:
                fr.append(C[cidx.index(i), cidx.index(j)])
        fr = np.array([x for x in fr if np.isfinite(x)])
        rows.append(dict(tag=tag, mode=md, n_channels=len(cidx),
                         n_informative_genes=n_inf, degenerate=degenerate,
                         r_median=float(np.median(off)) if off.size else np.nan,
                         r_p90=float(np.percentile(off, 90)) if off.size else np.nan,
                         r_max=float(off.max()) if off.size else np.nan,
                         n_fwd_rev=len(fr),
                         fwd_rev_r_median=float(np.median(fr)) if fr.size else np.nan,
                         fwd_rev_r_max=float(fr.max()) if fr.size else np.nan))
    return rows


def channel_gss(expr, coord, cell_type_str, lig, rec, mode, genes, c_s, c_r, md,
                max_pairs, seed):
    pm = np.where(mode == md)[0]
    Lg, Rg = lig[pm], rec[pm]
    S, R = directed_pairs(coord, cell_type_str, c_s, c_r, MODE_RADIUS[md], max_pairs, seed)
    gss = np.zeros(len(genes), np.float64)
    if len(S) == 0:
        return gss, 0, 0, 0, 0.0
    W = channel_W(expr, S, R, Lg, Rg)
    np.add.at(gss, Lg, W)
    np.add.at(gss, Rg, W)
    return gss, len(S), int((W > 0).sum()), int((gss > 0).sum()), eff_n_genes(gss)


def channel_pair_gss(expr, coord, cell_type_str, lig, rec, mode, genes, ligsym, recsym,
                     c_s, c_r, md, max_pairs, seed):
    pm = np.where(mode == md)[0]
    Lg, Rg = lig[pm], rec[pm]
    S, R = directed_pairs(coord, cell_type_str, c_s, c_r, MODE_RADIUS[md], max_pairs, seed)
    out = {}
    if len(S) == 0:
        return out
    W = channel_W(expr, S, R, Lg, Rg)
    for k in np.where(W > 0)[0]:
        g = np.zeros(len(genes), np.float64)
        g[Lg[k]] += W[k]; g[Rg[k]] += W[k]
        out[f"{c_s}->{c_r}:{md}|{ligsym[pm[k]]}-{recsym[pm[k]]}"] = g
    return out


def load_expr(inp, normalize):
    Y = inp.Y
    if sp.issparse(Y):
        Y = np.asarray(Y.todense(), np.float32)
    else:
        Y = np.asarray(Y, np.float32)
    if normalize == "raw":
        return Y
    if normalize == "library":
        tot = Y.sum(1, keepdims=True)
        med = np.median(tot[tot > 0]) if np.any(tot > 0) else 1.0
        sf = np.where(tot > 0, tot / med, 1.0)
        return np.log1p(Y / sf).astype(np.float32)
    raise SystemExit(f"--normalize must be raw|library, got {normalize}")


def resolve_lr(inp):
    lr = inp.LR_pairs
    if isinstance(lr, pd.DataFrame) and {"ligand_idx", "receptor_idx", "mode"} <= set(lr.columns):
        lig = lr["ligand_idx"].to_numpy(np.int64)
        rec = lr["receptor_idx"].to_numpy(np.int64)
        mode = lr["mode"].astype(str).to_numpy()
    else:
        A = np.asarray(lr, dtype=object)
        lig = A[:, 0].astype(np.int64); rec = A[:, 1].astype(np.int64); mode = A[:, 2].astype(str)
    return lig, rec, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="CCGWAS inputs_<sample>.pkl")
    ap.add_argument("--channels", required=True,
                    help="TSV with c_s,c_r,mode (make_channels.py)")
    ap.add_argument("--out", required=True, help="genes x channels GSS tsv")
    ap.add_argument("--normalize", default="raw", choices=["raw", "library"],
                    help="expression used for W: raw counts (default) or log1p of size-factor-scaled counts")
    ap.add_argument("--channel_norm", default="specificity",
                    choices=["none", "specificity", "zscore"],
                    help="specificity (default): W / (mean over channels + eps); none; zscore")
    ap.add_argument("--spec_eps_frac", type=float, default=0.05,
                    help="specificity pseudocount, fraction of the median non-zero row mean")
    ap.add_argument("--min_informative_genes", type=int, default=50,
                    help="a mode with fewer genes varying across channels is marked DEGENERATE")
    ap.add_argument("--separability_max_r", type=float, default=0.30,
                    help="advisory: flag modes whose between-channel correlation exceeds this")
    ap.add_argument("--min_genes", type=int, default=10,
                    help="channels with fewer non-zero genes (or n_eff) are flagged in the log")
    ap.add_argument("--per_pair", action="store_true",
                    help="emit one annotation per active LR pair (2-gene) instead of per channel")
    ap.add_argument("--max_pairs", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code", default="", help="directory with ccgwas_v4core.py if not alongside this script")
    a = ap.parse_args()

    if a.code:
        sys.path.insert(0, a.code)
    d = pickle.load(open(a.inputs, "rb"))
    inp = d["inputs"]; inp = inp.item() if isinstance(inp, np.ndarray) else inp
    expr = load_expr(inp, a.normalize)
    coord = np.asarray(inp.coord, np.float32)
    cell_type_str = np.asarray(inp.cell_type).astype(str)
    genes = np.array([str(g) for g in inp.gene_names])
    lig, rec, mode = resolve_lr(inp)

    if a.per_pair:
        lr_df = d.get("lr_df_full", None)
        if lr_df is not None and {"ligand_symbol", "receptor_symbol"} <= set(lr_df.columns):
            ligsym = lr_df["ligand_symbol"].astype(str).to_numpy()
            recsym = lr_df["receptor_symbol"].astype(str).to_numpy()
        else:
            ligsym = genes[lig]; recsym = genes[rec]

    valid_cts = set(np.unique(cell_type_str).tolist())
    ch = pd.read_csv(a.channels, sep="\t", dtype={"c_s": str, "c_r": str, "mode": str})
    ch = ch[ch["robust"] == True] if "robust" in ch else ch
    req = set(ch["c_s"]).union(set(ch["c_r"]))
    missing = sorted(req - valid_cts)
    if missing:
        raise SystemExit(
            f"[channel_gss] channel table references cell types not in the pkl: {missing}\n"
            f"  pkl cell types = {sorted(valid_cts)}\n"
            f"  -> regenerate the channel table from this pkl (make_channels.py).")
    names = d.get("ct_categories")
    if names is not None and {"sender", "receiver"} <= set(ch.columns):
        names = [str(x) for x in names]
        wrong = [(c, n) for c, n in list(zip(ch["c_s"], ch["sender"])) + list(zip(ch["c_r"], ch["receiver"]))
                 if isinstance(n, str) and n and (int(c) >= len(names) or names[int(c)] != n)]
        if wrong:
            raise SystemExit(
                f"[channel_gss] cell-type codes of {a.inputs} do not match the channel table: "
                f"{sorted(set(wrong))[:6]}\n  pkl vocabulary = {names}\n"
                f"  -> rebuild every section with the same prepare_inputs.py --ct_order.")
    bad_mode = sorted(set(ch["mode"]) - set(MODE_RADIUS))
    if bad_mode:
        raise SystemExit(f"[channel_gss] unknown mode(s) {bad_mode}; expected {list(MODE_RADIUS)}")

    print(f"[channel_gss] {a.inputs}")
    print(f"  cells={len(cell_type_str)} genes={len(genes)} LR_pairs={len(lig)} "
          f"cell_types={sorted(valid_cts)} normalize={a.normalize}")

    cols = {}
    col_mode = []
    n_below = 0
    for _, row in ch.iterrows():
        c_s, c_r, md = row["c_s"], row["c_r"], row["mode"]
        if a.per_pair:
            new = channel_pair_gss(expr, coord, cell_type_str, lig, rec, mode,
                                   genes, ligsym, recsym, c_s, c_r, md, a.max_pairs, a.seed)
            cols.update(new)
            col_mode.extend([md] * len(new))
            continue
        label = f"{c_s}->{c_r}:{md}"
        gss, npairs, nactive, ng, neff = channel_gss(expr, coord, cell_type_str, lig, rec,
                                        mode, genes, c_s, c_r, md, a.max_pairs, a.seed)
        cols[label] = gss
        col_mode.append(md)
        usable = (ng >= a.min_genes) and (neff >= a.min_genes)
        if npairs > 0 and not usable:
            n_below += 1
        if npairs == 0:
            flag = "  << empty (0 cell-pairs)"
        elif ng < a.min_genes:
            flag = f"  << nonzero_genes<{a.min_genes} (unstable SE)"
        elif neff < a.min_genes:
            flag = f"  << n_eff<{a.min_genes}: GSS mass concentrated (effectively few-gene; use matched-perm)"
        else:
            flag = ""
        print(f"  {label:34} pairs={npairs:7d} active_LR={nactive:4d} "
              f"nonzero_genes={ng:5d} n_eff={neff:6.1f} sum={gss.sum():.4g}{flag}")

    out = pd.DataFrame(cols, index=genes); out.index.name = "gene"
    labels = list(out.columns)
    A = out.to_numpy(np.float64)

    sep = separability(A, labels, col_mode, "before", a.min_informative_genes)
    if a.channel_norm != "none":
        if a.per_pair:
            raise SystemExit(
                "[channel_gss] --channel_norm is not defined for --per_pair. A per-pair "
                "annotation has exactly 2 non-zero genes, so dividing by a cross-annotation "
                "row mean is not the same operation and the 2-gene jackknife is unstable "
                "regardless. Use --channel_norm none with --per_pair.")
        A, nrep = normalize_channels(A, col_mode, a.channel_norm, a.spec_eps_frac)
        out = pd.DataFrame(A, index=genes, columns=labels); out.index.name = "gene"
        print(f"\n[channel_norm={a.channel_norm}] applied within mode, "
              f"non-empty channels only")
        for r in nrep:
            print(f"  mode={r['mode']:11s} channels={r['n_channels']:3d} "
                  f"non-empty={r['n_nonempty']:3d} eps={r['eps']:.4g} "
                  f"genes_shrunk_by_eps={r['n_genes_shrunk']}{'  ' + r['note'] if r['note'] else ''}")
        print(f"  post-normalisation per-channel n_eff (Herfindahl on the positive part):")
        for i, lab in enumerate(labels):
            v = A[:, i]
            vp = np.where(v > 0, v, 0.0)
            if not vp.any():
                continue
            print(f"    {lab:34} nonzero_genes={int((vp>0).sum()):5d} "
                  f"n_eff={eff_n_genes(vp):6.1f}")
        sep += separability(A, labels, col_mode, "after", a.min_informative_genes)

    if sep:
        print("\n[separability] between-channel correlation of the EMITTED annotation "
              "(z-free; Layer -1 gate)")
        print(f"  {'stage':7s} {'mode':11s} {'n':>4s} {'genes':>7s} {'r_med':>7s} "
              f"{'r_p90':>7s} {'r_max':>7s} {'fwd/rev':>8s} {'fr_med':>7s} {'fr_max':>7s}")
        for r in sep:
            mark = "  << DEGENERATE: too few informative genes; r is an artefact of n, ignore" \
                   if r["degenerate"] else ""
            print(f"  {r['tag']:7s} {r['mode']:11s} {r['n_channels']:4d} "
                  f"{r['n_informative_genes']:7d} {r['r_median']:7.3f} {r['r_p90']:7.3f} "
                  f"{r['r_max']:7.3f} {r['n_fwd_rev']:8d} {r['fwd_rev_r_median']:7.3f} "
                  f"{r['fwd_rev_r_max']:7.3f}{mark}")
        final = [r for r in sep
                 if r["tag"] == ("after" if a.channel_norm != "none" else "before")
                 and not r["degenerate"]]
        bad = [r for r in final
               if (np.isfinite(r["r_p90"]) and r["r_p90"] > a.separability_max_r)
               or (np.isfinite(r["fwd_rev_r_max"]) and r["fwd_rev_r_max"] > a.separability_max_r)]
        if bad:
            wholesale = [r for r in bad if r["r_median"] > a.separability_max_r]
            tail_only = [r for r in bad if r not in wholesale]
            if wholesale:
                print(f"  !! GATE (wholesale): median r > {a.separability_max_r} in mode(s) "
                      f"{', '.join(r['mode'] for r in wholesale)}. The annotation does not "
                      f"carry channel identity at all -- the competitive regression will "
                      f"return near-identical z for every channel and a pre-registered "
                      f"single-channel hypothesis is not testable on it. Use "
                      f"--channel_norm specificity, or move the directed claim to the "
                      f"communication/niche axes, which act on the per-LR-pair W and do "
                      f"resolve direction.")
            if tail_only:
                print(f"  !! GATE (tail): median r is fine but r_p90 or the worst "
                      f"forward/reverse pair exceeds {a.separability_max_r} in mode(s) "
                      f"{', '.join(r['mode'] for r in tail_only)}. Most channels in these "
                      f"modes ARE separable; a minority are still near-copies. This does "
                      f"not invalidate the annotation, but any Layer-1 channel drawn from "
                      f"these modes must be checked individually against its own reverse "
                      f"before it is pre-registered -- the mode-level median does not "
                      f"protect a specific pair.")
        else:
            print(f"  GATE PASSED: r_p90 and max forward/reverse r both <= "
                  f"{a.separability_max_r} in every non-degenerate mode; channels are "
                  f"distinguishable, so a channel-level hypothesis is testable here.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, sep="\t")
    kind = "per-pair (2-gene) annotations" if a.per_pair else "channels"
    print(f"\nwrote {a.out}: {out.shape[0]} genes x {out.shape[1]} {kind} "
          f"(z-free GSS, channel_norm={a.channel_norm})")
    if not a.per_pair and n_below:
        print(f"NOTE: {n_below} nonempty channels fail the calibration floor "
              f"(nonzero_genes<{a.min_genes} OR n_eff<{a.min_genes}: GSS mass too concentrated) "
              f"-> for those, use the matched-permutation genetic test (or drop them) at the "
              f"reporting stage; the analytic jackknife SE is unstable there.")


if __name__ == "__main__":
    main()
