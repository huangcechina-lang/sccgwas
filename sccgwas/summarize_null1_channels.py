#!/usr/bin/env python3
"""Channel-level communication axis across sections: one-sided Wilcoxon test of centred trait ranks."""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

TAB = "\t"


def centred(d, ch, t, unit):
    dev = []
    for s, g in d[d.channel == ch].groupby("sample"):
        if t in set(g.trait) and len(g) >= 3:
            r = g.p_null1.rank(method="average")
            dev.append((unit(s), float(r[g.trait == t].iloc[0] - r.mean())))
    if not dev:
        return np.array([])
    v = pd.DataFrame(dev, columns=["u", "dev"]).groupby("u", sort=False).dev.mean()
    return v.to_numpy()


def wil(v, min_n):
    if len(v) < min_n:
        return np.nan
    return float(stats.wilcoxon(v, alternative="less").pvalue)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--null1", required=True, help="output of null1_channels.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trait", default="", help="target trait to report")
    ap.add_argument("--channels", default="", help="comma-separated channel labels; default all")
    ap.add_argument("--drop_traits", default="", help="comma-separated traits left out of the panel")
    ap.add_argument("--donor_map", default="", help="two columns: section<TAB>donor")
    ap.add_argument("--donor_regex", default="",
                    help="derive the donor by deleting this pattern from the section name, e.g. '\\.\\d+$'")
    ap.add_argument("--min_n", type=int, default=5, help="minimum sections (or donors)")
    a = ap.parse_args()

    d = pd.read_csv(a.null1, sep=TAB)
    drop = {x.strip() for x in a.drop_traits.split(",") if x.strip()}
    d = d[~d.trait.astype(str).isin(drop)]
    d["sample"] = d["sample"].astype(str)
    if a.donor_map:
        m = dict(l.rstrip("\n").split(TAB)[:2] for l in open(a.donor_map) if TAB in l)
        unit = lambda s: m.get(s, s)
    elif a.donor_regex:
        unit = lambda s: re.sub(a.donor_regex, "", s)
    else:
        unit = lambda s: s
    chans = [c.strip() for c in a.channels.split(",") if c.strip()] or sorted(d.channel.unique())

    rows = []
    for ch in chans:
        x = d[d.channel == ch]
        if x.empty:
            continue
        med = x.groupby("trait").p_null1.median()
        for t in med.index:
            v = centred(d, ch, t, unit)
            rows.append(dict(channel=ch, trait=t, n_units=len(v), p_median=float(med[t]),
                             p_wilcoxon=wil(v, a.min_n)))
    if not rows:
        sys.exit("no channel found")
    R = pd.DataFrame(rows)
    R["rank_median"] = R.groupby("channel").p_median.rank(method="min")
    R["rank_wilcoxon"] = R.groupby("channel").p_wilcoxon.rank(method="min")
    R["n_traits"] = R.groupby("channel").p_wilcoxon.transform("count")
    best = R.dropna(subset=["p_wilcoxon"]).sort_values("p_wilcoxon").groupby("trait").first().reset_index()
    best["rank_best"] = best.p_wilcoxon.rank(method="min")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    R.to_csv(a.out, sep=TAB, index=False)
    best.to_csv(a.out.replace(".tsv", "") + "_best_channel.tsv", sep=TAB, index=False)

    if a.trait:
        q = R[R.trait == a.trait].sort_values("p_wilcoxon")
        print("%s by channel:" % a.trait)
        print(q[["channel", "n_units", "p_median", "rank_median", "p_wilcoxon", "rank_wilcoxon",
                 "n_traits"]].to_string(index=False))
        b = best[best.trait == a.trait]
        if len(b):
            print("best channel %s: P = %.4g, rank %d of %d traits"
                  % (b.channel.iloc[0], b.p_wilcoxon.iloc[0], int(b.rank_best.iloc[0]), len(best)))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
