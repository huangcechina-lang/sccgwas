#!/usr/bin/env python3
"""Per-channel beyond-niche threshold: 95th percentile of S among control pairs;
degenerate channels use the distribution pooled over the other channels."""
import argparse
import os

import numpy as np
import pandas as pd

TAB = "\t"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_n", type=int, default=50,
                    help="minimum defined S values for a channel-specific threshold")
    ap.add_argument("--frac_ratio", type=float, default=0.60)
    ap.add_argument("--thr_ratio", type=float, default=0.60)
    ap.add_argument("--sd_mult", type=float, default=0.80,
                    help="degenerate if p95 < median + sd_mult x MAD")
    ap.add_argument("--global_only", action="store_true",
                    help="use the pooled threshold for every channel")
    a = ap.parse_args()

    E = pd.read_csv(a.inp, sep=TAB)
    if "S" not in E.columns or "channel" not in E.columns:
        raise SystemExit("need columns S and channel; found %s" % list(E.columns))
    E["S"] = pd.to_numeric(E.S, errors="coerce").replace([np.inf, -np.inf], np.nan)
    print("[thr] %s: %d rows, %d channels, S defined in %.1f%%"
          % (os.path.basename(a.inp), len(E), E.channel.nunique(),
             100 * E.S.notna().mean()))

    st = []
    for c, g in E.groupby("channel"):
        v = g.S.dropna()
        if len(v) < a.min_n:
            st.append(dict(channel=c, n=len(g), n_ok=len(v), frac=np.nan,
                           med=np.nan, sd=np.nan, p95=np.nan, verdict="TOO_FEW"))
            continue
        st.append(dict(channel=c, n=len(g), n_ok=len(v), frac=len(v) / len(g),
                       med=float(v.median()), sd=float(v.std()),
                       mad=float(np.median(np.abs(v - v.median())) * 1.4826),
                       p95=float(np.percentile(v, 95)),
                       p99=float(np.percentile(v, 99)), verdict=""))
    S = pd.DataFrame(st)
    good = S[S.verdict == ""]
    if not len(good):
        raise SystemExit("no channel reaches min_n=%d" % a.min_n)
    ref_frac = float(good.frac.median())
    ref_p95 = float(good.p95.median())
    print("  reference: median defined fraction %.1f%%, median p95 %.3f" % (100 * ref_frac, ref_p95))

    for i, r in S.iterrows():
        if r.verdict == "TOO_FEW":
            continue
        why = []
        if r.frac < a.frac_ratio * ref_frac:
            why.append("defined %.1f%% < %.0f%% x %.1f%%"
                       % (100 * r.frac, 100 * a.frac_ratio, 100 * ref_frac))
        if r.p95 < a.thr_ratio * ref_p95:
            why.append("p95 %.3f < %.0f%% x %.3f"
                       % (r.p95, 100 * a.thr_ratio, ref_p95))
        if r.mad > 1e-9 and r.p95 < r.med + a.sd_mult * r.mad:
            why.append("p95 %.3f < median + %.1f MAD = %.3f"
                       % (r.p95, a.sd_mult, r.med + a.sd_mult * r.mad))
        S.loc[i, "verdict"] = "DEGENERATE" if why else "OK"
        S.loc[i, "why"] = "; ".join(why)

    print("\n%-20s %7s %7s %8s %8s %8s %8s %8s  %s"
          % ("channel", "n", "n_ok", "def%", "median", "sd", "MAD", "p95", "verdict"))
    for r in S.sort_values("p95", ascending=False).itertuples():
        print("%-20s %7d %7d %7.1f%% %8.3f %8.3f %8.3f %8.3f  %s"
              % (r.channel, r.n, r.n_ok,
                 100 * r.frac if r.frac == r.frac else np.nan,
                 r.med, r.sd, getattr(r, "mad", np.nan), r.p95, r.verdict))
        if r.verdict == "DEGENERATE":
            print("%22s%s" % ("", getattr(r, "why", "")))

    bad = set(S[S.verdict != "OK"].channel)
    pool = E[~E.channel.isin(bad)].S.dropna()
    gthr = float(np.percentile(pool, 95))
    print("\n  pooled over good channels: %d values, p95 = %.4f" % (len(pool), gthr))

    if a.global_only:
        out = E[~E.channel.isin(bad)].copy(); out["channel"] = "__pooled__"
        print("  --global_only: pooled threshold %.4f for every channel" % gthr)
    else:
        keep = E[~E.channel.isin(bad)]
        rep = []
        for c in sorted(bad):
            n = int((E.channel == c).sum())
            samp = pool.sample(min(n, len(pool)),
                               replace=n > len(pool), random_state=0)
            rep.append(pd.DataFrame(dict(channel=c, S=samp.to_numpy())))
            print("  %s: %d rows replaced by the pooled distribution (p95 %.4f)" % (c, n, gthr))
        out = pd.concat([keep] + rep, ignore_index=True) if rep else keep
    keepc = [c for c in ("sample", "channel", "ligand", "receptor", "S", "p_raw")
             if c in out.columns]
    out = out[keepc]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, sep=TAB, index=False)

    print("\n  thresholds used downstream:")
    for c, g in out.groupby("channel"):
        v = g.S.dropna()
        if len(v) >= a.min_n:
            print("       %-22s %.3f" % (c, np.percentile(v, 95)))
    print("       %-22s %.3f" % ("pooled", np.percentile(out.S.dropna(), 95)))
    print("\nwrote %s (%d rows)" % (a.out, len(out)))


if __name__ == "__main__":
    main()
