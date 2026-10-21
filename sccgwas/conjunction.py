#!/usr/bin/env python3
"""Three axes per pair x channel and their conjunction.

genetic        p_fwer_A_sum < 0.05 per row (--level fwer) or BH q < 0.10 per pair (--level candidate)
beyond-niche   S above the channel threshold in at least m sections (binomial rule)
communication  p_fwer < 0.05, family-wise within the channel
--selective uses the selection-adjusted values of selective_maxt.py."""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

TAB = "\t"
K = ["ligand", "receptor", "channel"]
ADAPTORS = {"TYROBP", "FCER1G", "CD3E", "CD3D", "CD3G", "CD247",
            "LAT", "LAT2", "NCSTN", "APH1A", "PSENEN", "IL2RG"}


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


def add_qpair(x):
    for st in ("sum", "min"):
        rc = "p_raw_%s" % st
        if rc not in x.columns:
            continue
        pp = "p_pair_%s" % st
        if pp in x.columns:
            per = x.groupby(["ligand", "receptor"])[pp].first().astype(float)
        else:
            per = x.groupby(["ligand", "receptor"])[rc].min()
        mp = dict(zip(per.index, bh(per.to_numpy(float))))
        x["q_pair_%s" % st] = [mp.get((a, b), np.nan) for a, b in zip(x.ligand, x.receptor)]
    return x


def load_sthr(f):
    d = pd.read_csv(f, sep=TAB)
    S = pd.to_numeric(d["S"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    thr = {}
    if "channel" in d.columns:
        for c, g in d.assign(S=S).groupby("channel"):
            v = g["S"].dropna()
            if len(v) >= 50:
                thr[c] = float(np.percentile(v, 95))
    thr["__global__"] = float(np.percentile(S.dropna(), 95))
    return thr


def min_m(n, p0, target):
    n = int(max(n, 1))
    return int(min(range(1, n + 1), key=lambda m: abs(float(binom.sf(m - 1, n, p0)) - target)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maxt", required=True, help="pair_genetic_maxt.py output")
    ap.add_argument("--percell", required=True, help="pair_beyond_niche.py output")
    ap.add_argument("--sthr", required=True, help="beyond_niche_threshold.py output")
    ap.add_argument("--null1", default="", help="null1_pairs.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="fwer", choices=["fwer", "candidate"])
    ap.add_argument("--stat", default="sum", choices=["sum", "min"])
    ap.add_argument("--min_c", type=int, default=0, help="beyond-niche requirement; 0 = binomial rule")
    ap.add_argument("--c_null", type=float, default=0.02)
    ap.add_argument("--donor_map", default="", help="two columns: section<TAB>donor")
    ap.add_argument("--donor_regex", default="", help="donor = section name with this pattern deleted")
    ap.add_argument("--lr_resource", default="", help="resource TSV (pair, ligand, receptor, ...) to trace complexes")
    ap.add_argument("--selective", default="",
                    help="--out_prefix of selective_maxt.py; its _rows.tsv / _pairs.tsv replace the genetic axis")
    ap.add_argument("--event", default="all_settings", choices=["all_settings", "first_setting", "majority"],
                    help="selection event used with --selective")
    a = ap.parse_args()

    X = add_qpair(pd.read_csv(a.maxt, sep=TAB))
    col = ("p_fwer_A_%s" % a.stat) if a.level == "fwer" else ("q_pair_%s" % a.stat)
    lim = 0.05 if a.level == "fwer" else 0.10
    if a.selective:
        if a.stat != "sum":
            sys.exit("--selective is available for --stat sum only")
        if a.level == "fwer":
            Sx = pd.read_csv(a.selective + "_rows.tsv", sep=TAB)
            col = "p_fwer_%s" % a.event
            X = X.merge(Sx[K + [col]], on=K, how="left")
        else:
            Sx = pd.read_csv(a.selective + "_pairs.tsv", sep=TAB)
            col = "q_pair_%s" % a.event
            X = X.merge(Sx[["ligand", "receptor", col]], on=["ligand", "receptor"], how="left")
        if col not in X.columns or X[col].isna().all():
            sys.exit("%s: event %s not reported (it does not hold for the observed data)" % (a.selective, a.event))
        print("genetic axis: selection-adjusted %s from %s" % (col, a.selective))
    if col not in X.columns:
        sys.exit("column %s not in %s" % (col, a.maxt))
    X["genetic"] = X[col] < lim
    X["genetic_scope"] = "row" if a.level == "fwer" else "pair"
    if {"t_L", "t_R"} <= set(X.columns):
        X["both_end"] = (X.t_L > 2.0) & (X.t_R > 2.0)

    thr = load_sthr(a.sthr)
    d = pd.read_csv(a.percell, sep=TAB)
    if "direction" in d.columns:
        d = d[d.direction == "fwd"]
    d["sample"] = d["sample"].astype(str)
    d["ok"] = d.S_nullE > d.channel.map(lambda c: thr.get(c, thr["__global__"]))
    NI = d.groupby(K).agg(n_sec=("p_nullE", "size"), c_sec=("ok", "sum")).reset_index()
    nsec = int(NI.n_sec.max())
    m = a.min_c or min_m(nsec, 0.05, a.c_null)
    NI["m_sec"] = m
    NI["niche"] = NI.c_sec >= m
    print("beyond-niche: %d sections, requirement %d (null level %.4f)"
          % (nsec, m, float(binom.sf(m - 1, nsec, 0.05))))

    if a.donor_map or a.donor_regex:
        if a.donor_map:
            mp = dict(l.rstrip("\n").split(TAB)[:2] for l in open(a.donor_map) if TAB in l)
            d["donor"] = d["sample"].map(lambda s: mp.get(s, s))
        else:
            d["donor"] = d["sample"].map(lambda s: re.sub(a.donor_regex, "", s))
        k = int(d.groupby("donor")["sample"].nunique().max())
        p_d = 1 - 0.95 ** k
        dd = d.groupby(K + ["donor"]).ok.any().reset_index()
        ND = dd.groupby(K).agg(n_donor=("donor", "size"), c_donor=("ok", "sum")).reset_index()
        nd = int(ND.n_donor.max())
        md = min_m(nd, p_d, a.c_null)
        ND["m_donor"] = md
        ND["niche_donor"] = ND.c_donor >= md
        NI = NI.merge(ND, on=K, how="left")
        print("donor level: %d donors, null rate %.4f, requirement %d (null level %.4f)"
              % (nd, p_d, md, float(binom.sf(md - 1, nd, p_d))))

    R = X.merge(NI, on=K, how="left")
    R["niche"] = R["niche"].fillna(False).astype(bool)
    if a.null1:
        N = pd.read_csv(a.null1, sep=TAB)
        if "status" in N.columns:
            N = N[N.status == "OK"]
        R = R.merge(N[K + ["T", "p_fwer"]].rename(columns={"T": "T_null1", "p_fwer": "p_fwer_null1"}),
                    on=K, how="left")
        R["communication"] = R.p_fwer_null1 < 0.05
    else:
        R["communication"] = False
    R["conj_channel"] = R.genetic & R.niche & R.communication
    pg = set(zip(R.ligand[R.genetic], R.receptor[R.genetic]))
    pc = set(zip(NI.ligand[NI.niche], NI.receptor[NI.niche]))
    pn = set(zip(N.ligand[N.p_fwer < 0.05], N.receptor[N.p_fwer < 0.05])) if a.null1 else set()
    R["conj_pair"] = [(l, r) in pg & pc & pn for l, r in zip(R.ligand, R.receptor)]

    if a.lr_resource:
        O = pd.read_csv(a.lr_resource, sep=TAB)
        idx = {}
        for r in O.itertuples():
            ls, rs = str(r.ligand).split("_"), str(r.receptor).split("_")
            for L in ls:
                for Rr in rs:
                    idx.setdefault((L, Rr), []).append((len(ls) * len(rs), str(r.pair)))
        R["record"] = [max(idx[(l, r)])[1] if (l, r) in idx else ""
                       for l, r in zip(R.ligand.astype(str), R.receptor.astype(str))]
        R["receptor_is_adaptor"] = R.receptor.astype(str).isin(ADAPTORS)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    R.to_csv(a.out, sep=TAB, index=False)
    P = R.drop_duplicates(["ligand", "receptor"])
    print("family: %d pairs, %d pair x channel rows" % (len(P), len(R)))
    for c in ("genetic", "niche", "communication", "conj_channel", "conj_pair"):
        s = R[R[c]][["ligand", "receptor"]].drop_duplicates()
        print("  %-14s %4d pairs  %s" % (c, len(s), ", ".join(s.ligand + "-" + s.receptor)[:200]))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
