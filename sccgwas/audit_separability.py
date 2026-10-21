#!/usr/bin/env python3
"""Forward-reverse specificity correlation of every ordered cell-type pair, per section."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("[fatal] export PYTHONHASHSEED=0 before running")


def eff_n(v: np.ndarray) -> float:
    p = v[v > 0]
    s = p.sum()
    return float(s * s / (p * p).sum()) if p.size and s > 0 else 0.0


def parse_label(lab: str):
    head = lab.split("|")[0]
    if "->" not in head or ":" not in head:
        return None
    cs_cr, md = head.rsplit(":", 1)
    if "->" not in cs_cr:
        return None
    cs, cr = cs_cr.split("->", 1)
    return cs, cr, md


def load_ct_names(pkl_path: Path | None, code_dir: Path | None = None):
    if pkl_path is None or not pkl_path.exists():
        return {}
    if code_dir is not None and str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
    except Exception as e:
        print(f"[warn] could not read {pkl_path} ({type(e).__name__}: {e}); "
              f"cell-type labels stay numeric. Pass --code <dir with ccgwas_v4core.py>.")
        return {}
    cats = d.get("ct_categories")
    if cats is None:
        cats = getattr(d.get("inputs"), "ct_categories", None)
    return {str(i): str(c) for i, c in enumerate(cats)} if cats is not None else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gss_dir", required=True, help="dir holding gss_spec_<sample>.tsv")
    ap.add_argument("--pkl", default=None, help="any pkl from the same tissue, for ct names")
    ap.add_argument("--code", default=os.path.dirname(os.path.abspath(__file__)),
                    help="dir containing ccgwas_v4core.py (needed to unpickle)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--modes", default="paracrine",
                    help="comma-separated modes to audit")
    ap.add_argument("--min_genes", type=int, default=10,
                    help="calibration floor, must match channel_gss.py --min_genes")
    ap.add_argument("--max_r", type=float, default=0.30,
                    help="fwd/rev separability threshold, must match "
                         "channel_gss.py --separability_max_r")
    ap.add_argument("--min_sig", type=int, default=5,
                    help="samples a channel must be available in to enter the pool")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    gss_dir, outdir = Path(a.gss_dir), Path(a.outdir)
    modes_want = [m.strip() for m in a.modes.split(",") if m.strip()]
    files = sorted(gss_dir.glob("gss_spec_*.tsv"))
    if not files:
        sys.exit(f"[fatal] no gss_spec_*.tsv under {gss_dir}")
    if outdir.exists() and any(outdir.glob("*.tsv")) and not a.force:
        sys.exit(f"[fatal] {outdir} already holds results; use --force or move it aside")
    outdir.mkdir(parents=True, exist_ok=True)

    ct = load_ct_names(Path(a.pkl) if a.pkl else None,
                       Path(a.code) if a.code else None)
    nm = (lambda c: ct.get(c, c))
    print(f"[audit] {len(files)} samples, modes={modes_want}, floor={a.min_genes}, "
          f"threshold r>{a.max_r}")
    print(f"[audit] cell types: {ct if ct else 'NOT RESOLVED (labels stay numeric)'}")

    pair_rows, avail_rows = [], []

    for f in files:
        sample = f.stem.replace("gss_spec_", "")
        A = pd.read_csv(f, sep="\t", index_col=0)
        meta = {c: parse_label(c) for c in A.columns}
        meta = {c: v for c, v in meta.items() if v is not None}

        for c, (cs, cr, md) in meta.items():
            v = A[c].to_numpy(np.float64)
            ng = int((v > 0).sum())
            ne = eff_n(v)
            avail_rows.append({"sample": sample, "mode": md,
                               "sender": nm(cs), "receiver": nm(cr), "channel": c,
                               "nonzero_genes": ng, "n_eff": round(ne, 2),
                               "nonempty": bool(np.any(v != 0)),
                               "pass_floor": bool(ng >= a.min_genes and ne >= a.min_genes)})

        floor_ok = {r["channel"] for r in avail_rows
                    if r["sample"] == sample and r["pass_floor"]}

        for md in modes_want:
            cols_all = [c for c, (_, _, m) in meta.items()
                        if m == md and np.any(A[c].to_numpy() != 0)]
            cols_flr = [c for c in cols_all if c in floor_ok]
            if len(cols_all) < 2:
                continue

            def corr_matrix(cols):
                M = A[cols].to_numpy(np.float64)
                keep = M.std(axis=1) > 0
                if int(keep.sum()) <= 2:
                    return None, int(keep.sum())
                return pd.DataFrame(np.corrcoef(M[keep].T), index=cols, columns=cols), \
                       int(keep.sum())

            C_all, ng_all = corr_matrix(cols_all)
            C_flr, ng_flr = (corr_matrix(cols_flr) if len(cols_flr) >= 2 else (None, 0))

            seen = set()
            for c, (cs, cr, m) in meta.items():
                if m != md or cs == cr or c not in cols_all:
                    continue
                rev = f"{cr}->{cs}:{md}"
                if rev not in cols_all or (cr, cs) in seen:
                    continue
                seen.add((cs, cr))
                vi, vj = A[c].to_numpy(np.float64), A[rev].to_numpy(np.float64)
                pair_rows.append({
                    "mode": md, "sender": nm(cs), "receiver": nm(cr), "sample": sample,
                    "r_gate": (float(C_all.loc[c, rev]) if C_all is not None else np.nan),
                    "r_floor": (float(C_flr.loc[c, rev])
                                if (C_flr is not None and c in C_flr.index
                                    and rev in C_flr.index) else np.nan),
                    "both_pass_floor": bool(c in floor_ok and rev in floor_ok),
                    "n_genes_union": int(((vi > 0) | (vj > 0)).sum()),
                    "n_genes_mode_keep": ng_all,
                })
        print(f"  {sample:14s} channels={len(meta):3d} floor_ok={len(floor_ok):3d}")

    pairs = pd.DataFrame(pair_rows)
    avail = pd.DataFrame(avail_rows)
    pairs.to_csv(outdir / "pair_fwd_rev.tsv", sep="\t", index=False)
    avail.to_csv(outdir / "availability.tsv", sep="\t", index=False)

    def agg(g):
        r = g["r_floor"].where(g["both_pass_floor"], np.nan).dropna()
        rg = g["r_gate"].dropna()
        return pd.Series({
            "n_samples_seen": len(g),
            "n_samples_both_floor": int(g["both_pass_floor"].sum()),
            "r_gate_med": rg.median() if len(rg) else np.nan,
            "r_gate_max": rg.max() if len(rg) else np.nan,
            "n_exceed_gate": int((rg > a.max_r).sum()),
            "r_floor_med": r.median() if len(r) else np.nan,
            "r_floor_max": r.max() if len(r) else np.nan,
            "n_exceed_floor": int((r > a.max_r).sum()),
        })

    summ = (pairs.groupby(["mode", "sender", "receiver"], sort=True)
                 .apply(agg, include_groups=False).reset_index()
                 .sort_values("r_floor_max", ascending=False))
    summ["verdict"] = np.where(
        summ["n_samples_both_floor"] < a.min_sig, "NOT_TESTABLE",
        np.where(summ["n_exceed_floor"] == 0, "SEPARABLE",
                 np.where(summ["n_exceed_floor"] >= summ["n_samples_both_floor"] / 2,
                          "NEAR_COPY", "UNSTABLE")))
    summ.to_csv(outdir / "pair_summary.tsv", sep="\t", index=False)

    pool = (avail[avail["mode"].isin(modes_want)]
            .groupby(["mode", "sender", "receiver"], sort=True)
            .agg(n_samples_available=("pass_floor", "sum"),
                 n_samples_nonempty=("nonempty", "sum"),
                 med_nonzero_genes=("nonzero_genes", "median"),
                 med_n_eff=("n_eff", "median")).reset_index())
    pool["in_pool"] = pool["n_samples_available"] >= a.min_sig
    pool.sort_values("n_samples_available", ascending=False).to_csv(
        outdir / "min_sig_pool.tsv", sep="\t", index=False)

    print("\n" + "=" * 78)
    print(f"AVAILABILITY  (min_sig >= {a.min_sig})")
    print("=" * 78)
    for md in modes_want:
        p = pool[pool["mode"] == md]
        print(f"  {md}: {int(p['in_pool'].sum())}/{len(p)} channels available in "
              f">= {a.min_sig} of {len(files)} samples")

    print("\n" + "=" * 78)
    print(f"FORWARD/REVERSE SEPARABILITY  (threshold r > {a.max_r})")
    print("=" * 78)
    for md in modes_want:
        s = summ[summ["mode"] == md]
        t = s[s["verdict"] != "NOT_TESTABLE"]
        print(f"  {md}: {len(t)} testable pairs of {len(s)}")
        for v in ("SEPARABLE", "UNSTABLE", "NEAR_COPY"):
            n = int((t["verdict"] == v).sum())
            print(f"      {v:12s} {n:3d}")
        bad = t[t["verdict"].isin(("NEAR_COPY", "UNSTABLE"))]
        if len(bad):
            print(f"      pairs to exclude from any directional claim:")
            for _, r in bad.iterrows():
                print(f"        {r['sender']:>14s} <-> {r['receiver']:<14s} "
                      f"r_floor med={r['r_floor_med']:.3f} max={r['r_floor_max']:.3f}  "
                      f"exceeds in {int(r['n_exceed_floor'])}/"
                      f"{int(r['n_samples_both_floor'])}  [{r['verdict']}]")
        if len(t):
            print(f"      this tissue: r_floor median over testable pairs = "
                  f"{t['r_floor_med'].median():.3f}, worst = {t['r_floor_max'].max():.3f}")

    print(f"\n  -> {outdir}/pair_summary.tsv  (sort by r_floor_max)")
    print(f"  -> {outdir}/min_sig_pool.tsv   (in_pool=True are testable)")


if __name__ == "__main__":
    main()
