#!/usr/bin/env python3
"""Channel-level communication axis: T = sum_k z_k W_k / ||W|| against matched control pairs."""
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

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = "\t"


def load_z(path):
    g = pd.read_csv(path, sep=r"\s+")
    g.columns = [c.upper() for c in g.columns]
    if "ZSTAT" not in g.columns or "GENE" not in g.columns:
        return None
    s = pd.Series(pd.to_numeric(g["ZSTAT"], errors="coerce").values,
                  index=g["GENE"].astype(str))
    return s[~s.index.duplicated()].dropna()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl_dir", required=True)
    ap.add_argument("--code", default=HERE, help="directory with ccgwas_v4core.py")
    ap.add_argument("--trait_manifest", required=True, help="two columns: label<TAB>path to .genes.out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="paracrine")
    ap.add_argument("--max_cell_pairs", type=int, default=5000)
    ap.add_argument("--n_null", type=int, default=0,
                    help="pseudo-database draws; 0 = CCGWASConfig default")
    ap.add_argument("--samples", default="", help="comma-separated sections; default all")
    ap.add_argument("--channels", default="",
                    help="ALL_TO_HUB:<code or name>, ALL_FROM:<...>, or a list such as '0->4,1->4'; default all")
    ap.add_argument("--traits", default="",
                    help="comma-separated subset of manifest labels; default all")
    ap.add_argument("--ct_names", default="",
                    help="cell-type names in code order (comma list) or a pkl; default: ct_categories of the pkl")
    ap.add_argument("--only_mode", action="store_true", default=True,
                    help="preprocess only the signalling mode given by --mode (default; saves memory)")
    ap.add_argument("--all_modes", dest="only_mode", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    sys.path.insert(0, a.code)
    from ccgwas_v4core import (CCGWASConfig, preprocess_inputs,
                               build_channel_cell_pairs,
                               build_channel_specific_control_pool,
                               compute_channel_test_dual_null)
    try:
        from ccgwas_v4core import MIN_EFFECTIVE_CONTROLS_LOWCONF as MINEFF
    except ImportError:
        MINEFF = 5
        print("[null1] MIN_EFFECTIVE_CONTROLS_LOWCONF not in core; using 5")

    traits = {}
    for ln in open(a.trait_manifest):
        ln = ln.rstrip("\n")
        if not ln.strip():
            continue
        lab, p = ln.split(TAB)
        z = load_z(p.strip())
        if z is not None:
            traits[lab.strip()] = z
    if not traits:
        sys.exit("[null1] no usable trait in the manifest")
    if a.traits:
        sel = [x.strip() for x in a.traits.split(",") if x.strip()]
        miss = [x for x in sel if x not in traits]
        if miss:
            raise SystemExit("[null1] not in the manifest: %s" % miss)
        traits = {k: v for k, v in traits.items() if k in sel}
        print("[null1] trait subset: %s" % sel)
    print("[null1] %d traits" % len(traits))

    pkls = sorted(glob.glob(os.path.join(a.pkl_dir, "inputs_*.pkl")))
    if a.samples:
        want = {x.strip() for x in a.samples.split(",")}
        pkls = [p for p in pkls
                if re.sub(r"^inputs_|\.pkl$", "", os.path.basename(p)) in want]
    if not pkls:
        sys.exit("[null1] no matching inputs_*.pkl in %s" % a.pkl_dir)
    print("[null1] %d sections, mode=%s, n_null=%d" % (len(pkls), a.mode, a.n_null))

    config = CCGWASConfig(random_seed=a.seed)
    if a.only_mode:
        r = dict(getattr(config, "signaling_radii", {}) or {})
        if a.mode in r and len(r) > 1:
            config.signaling_radii = {a.mode: r[a.mode]}
    if a.n_null:
        cands = [f for f in ("n_pseudo_db_reps", "n_null_draws", "B", "n_boot",
                             "n_control_draws", "n_resample")
                 if hasattr(config, f)]
        if not cands:
            sys.exit("[null1] no draw-count field found in CCGWASConfig")
        for f in cands:
            setattr(config, f, a.n_null)
        print("[null1] config.%s = %d" % ("/".join(cands), a.n_null))
    print("[null1] config: %s"
          % {k: v for k, v in vars(config).items()
             if not k.startswith("_") and isinstance(v, (int, float, str, bool))})
    rows, dropped = [], []
    CT, VOCAB = {}, None
    t_all = time.time()
    for si, pkl in enumerate(pkls, 1):
        smp = re.sub(r"^inputs_|\.pkl$", "", os.path.basename(pkl))
        print("\n[%2d/%d] %s" % (si, len(pkls), smp), flush=True)
        d = pickle.load(open(pkl, "rb"))
        inputs = d["inputs"]
        inputs = inputs.item() if isinstance(inputs, np.ndarray) else inputs
        gene_names = [str(g) for g in inputs.gene_names]
        VOCAB = celltypes.same_vocab(d["ct_categories"], VOCAB, pkl)
        if not CT:
            CT = {int(k): v for k, v in celltypes.resolve(
                a.ct_names or ",".join(map(str, d["ct_categories"]))).items()}

        t0 = time.time()
        preproc = preprocess_inputs(inputs, config)
        print("  preprocess %.0fs" % (time.time() - t0))
        cp = build_channel_cell_pairs(inputs, config)
        cands = [k for k in cp if k[2] == a.mode and len(cp[k][0]) > 0]
        rng = np.random.default_rng(a.seed)
        if a.channels:
            spec = a.channels.strip()
            inv = {v: k for k, v in CT.items()} if isinstance(CT, dict) else {}
            def _code(tok):
                tok = tok.strip()
                if tok.isdigit():
                    return int(tok)
                if tok in inv:
                    return inv[tok]
                raise SystemExit("[null1] unknown cell type '%s'; known %s"
                                 % (tok, sorted(inv) if inv else list(CT)))
            if spec.upper().startswith("ALL_TO_HUB:"):
                hub = _code(spec.split(":", 1)[1])
                want = None
                keep = [c for c in cands if c[1] == hub]
                desc = "ALL_TO_HUB:%s" % CT.get(hub, hub)
            elif spec.upper().startswith("ALL_FROM:"):
                snd = _code(spec.split(":", 1)[1])
                keep = [c for c in cands if c[0] == snd]
                desc = "ALL_FROM:%s" % CT.get(snd, snd)
            else:
                want = set()
                for tok in spec.split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    base = tok.split(":")[0]
                    if "->" not in base:
                        raise SystemExit("[null1] channel must be 'A->B', got '%s'" % tok)
                    aa, bb = base.split("->")
                    want.add((_code(aa), _code(bb)))
                keep = [c for c in cands if (c[0], c[1]) in want]
                desc = "%d listed" % len(want)
            if not keep:
                raise SystemExit("[null1] --channels '%s' matched no channel; "
                                 "available e.g. %s"
                                 % (spec, ["%d->%d:%s" % c for c in list(cands)[:5]]))
            print("  channels %s: %d -> %d" % (desc, len(cands), len(keep)))
            cands = keep
        
        cell_pairs = {}
        for c in cands:
            i_, j_ = cp[c]
            if a.max_cell_pairs and len(i_) > a.max_cell_pairs:
                s_ = rng.choice(len(i_), a.max_cell_pairs, replace=False)
                i_, j_ = i_[s_], j_[s_]
            cell_pairs[c] = (i_, j_)
        print("  %d %s channels with cell pairs" % (len(cell_pairs), a.mode))

        t0 = time.time()
        pool = build_channel_specific_control_pool(inputs, preproc, config, cell_pairs)
        ctrl = pool.control_pairs
        ok = [c for c in cell_pairs
              if c in ctrl and sum(len(x) >= MINEFF for x in ctrl[c]) >= 5]
        bad = [c for c in cell_pairs if c not in ok]
        print("  control pool %.0fs; %d/%d channels with enough controls"
              % (time.time() - t0, len(ok), len(cell_pairs)))
        for c in bad:
            dropped.append(dict(sample=smp, channel="%d->%d:%s" % c,
                                reason="too few effective controls"))
        if not ok:
            print("  !! no usable channel in this section")
            continue
        cp_ok = {c: cell_pairs[c] for c in ok}

        aligned = {}
        for lab, zs in traits.items():
            aligned[lab] = zs.reindex(gene_names).fillna(0.0).to_numpy(np.float32)
        n_cov = {lab: int((v != 0).sum()) for lab, v in aligned.items()}
        print("  z aligned: median %d/%d genes with non-zero z"
              % (int(np.median(list(n_cov.values()))), len(gene_names)))

        for ti, (lab, zv) in enumerate(sorted(aligned.items()), 1):
            t0 = time.time()
            res = compute_channel_test_dual_null(
                inputs, preproc, config, cp_ok, pool,
                z_gwas_override=zv)
            for c in cp_ok:
                rows.append(dict(
                    sample=smp, trait=lab,
                    channel="%d->%d:%s" % c,
                    sender=CT.get(c[0], c[0]), receiver=CT.get(c[1], c[1]),
                    mode=c[2],
                    T=float(res.channel_T.get(c, np.nan)),
                    p_null1=float(res.channel_p_null1.get(c, np.nan)),
                    p_null2=float(res.channel_p_null2.get(c, np.nan)),
                    status=str(res.channel_status.get(c, "")),
                    n_cell_pairs=len(cp_ok[c][0]),
                    n_genes_with_z=n_cov[lab]))
            if ti == 1 or ti % 10 == 0:
                print("    [%2d/%d] %-14s %.0fs" % (ti, len(aligned), lab,
                                                    time.time() - t0), flush=True)
        el = time.time() - t_all
        print("  section done, %.0f min elapsed, about %.0f min left"
              % (el / 60, el / 60 * (len(pkls) - si) / si))

    if not rows:
        sys.exit("[null1] no result")
    D = pd.DataFrame(rows)
    D.to_csv(a.out, sep=TAB, index=False)
    if dropped:
        pd.DataFrame(dropped).to_csv(a.out.replace(".tsv", "_dropped.tsv"),
                                     sep=TAB, index=False)

    print("\n%d sections x %d traits x %d channels = %d rows"
          % (D["sample"].nunique(), D["trait"].nunique(), D["channel"].nunique(), len(D)))
    if dropped:
        print("%d section x channel combinations not testable (see *_dropped.tsv)" % len(dropped))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
