#!/usr/bin/env python3
"""Standardize channel statistics against control traits (input to hub_iut.py)."""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

TAB = "\t"


def load(mdir, mode, alpha):
    fs = sorted(glob.glob(os.path.join(mdir, "ch_*_channels.tsv")))
    if not fs:
        sys.exit("[hub] no ch_*_channels.tsv in %s" % mdir)
    d = pd.concat([pd.read_csv(f, sep=TAB).assign(
        sample=re.sub(r"^ch_|_channels\.tsv$", "", os.path.basename(f)))
        for f in fs], ignore_index=True)
    for c in ("trait", "channel", "t", "p_vs_ensemble"):
        if c not in d.columns:
            sys.exit("[hub] column %s missing; found %s" % (c, list(d.columns)))
    if mode:
        d = d[d["channel"].astype(str).str.endswith(mode)]
    g = (d.groupby(["trait", "channel"])
           .agg(mean_t=("t", "mean"), med_t=("t", "median"),
                n_unit=("t", "size"),
                n_sig=("p_vs_ensemble", lambda x: int((x < alpha).sum())))
           .reset_index())
    return d, g


def standardise(g, controls, stat="mean_t"):
    piv = g.pivot(index="trait", columns="channel", values=stat)
    ctl = [c for c in controls if c in piv.index]
    if len(ctl) < 3:
        sys.exit("[hub] fewer than three control traits (%d)" % len(ctl))
    Z = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=float)
    for tr in piv.index:
        pool = [c for c in ctl if c != tr]
        mu = piv.loc[pool].mean(axis=0)
        sd = piv.loc[pool].std(axis=0, ddof=1)
        Z.loc[tr] = (piv.loc[tr] - mu) / sd.replace(0, np.nan)
    base_all = pd.DataFrame({"baseline_mean": piv.loc[ctl].mean(axis=0),
                             "baseline_sd": piv.loc[ctl].std(axis=0, ddof=1)})
    return Z, base_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix_dir", required=True, help="one setting directory from channel_t_matrix.py")
    ap.add_argument("--controls", required=True, help="comma-separated control traits")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--target_trait", default="", help="removed from --controls if listed there")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--mode", default="paracrine")
    ap.add_argument("--stat", default="mean_t", choices=["mean_t", "med_t", "n_sig"])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)

    controls = [c.strip() for c in a.controls.split(",") if c.strip()]
    if a.target_trait in controls:
        controls = [c for c in controls if c != a.target_trait]

    d, g = load(a.matrix_dir, a.mode, a.alpha)
    print("[hub] %d sections, %d traits, %d channels"
          % (d["sample"].nunique(), g["trait"].nunique(), g["channel"].nunique()))
    Z, base = standardise(g, controls, a.stat)
    Z.to_csv(a.out_prefix + "_zmatrix.tsv", sep=TAB)
    base.to_csv(a.out_prefix + "_baseline.tsv", sep=TAB)
    print("wrote %s_zmatrix.tsv" % a.out_prefix)


if __name__ == "__main__":
    main()
