#!/usr/bin/env python3
"""Hub statistic S = min(z_in, z_diff) per receiving cell type and the rank of each trait among traits."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import celltypes

TAB = "\t"
CT = {}


def bh(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if not ok.any():
        return q
    v = p[ok]
    n = len(v)
    o = np.argsort(v)
    r = v[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(r, 1.0)
    q[ok] = out
    return q


def cell_stats(Z, trait, hub, mode):
    ins, outs = [], []
    for x in sorted(CT.keys()):
        if x == hub:
            continue
        f = "%s->%s:%s" % (x, hub, mode)
        r = "%s->%s:%s" % (hub, x, mode)
        if f in Z.columns and r in Z.columns:
            a, b = Z.loc[trait, f], Z.loc[trait, r]
            if np.isfinite(a) and np.isfinite(b):
                ins.append(float(a))
                outs.append(float(b))
    if len(ins) < 3:
        return None
    sc = "%s->%s:%s" % (hub, hub, mode)
    self_z = (float(Z.loc[trait, sc])
              if sc in Z.columns and np.isfinite(Z.loc[trait, sc]) else np.nan)
    ins, outs = np.array(ins), np.array(outs)
    d = ins - outs
    return dict(in_mean=float(ins.mean()), out_mean=float(outs.mean()),
                diff=float(d.mean()), self_z=self_z,
                n_pos_in=int((ins > 0).sum()), n_pos_diff=int((d > 0).sum()))


def loo_z(vals):
    v = vals.dropna()
    if len(v) < 5:
        return pd.Series(np.nan, index=vals.index)
    out = {}
    for tr in v.index:
        rest = v.drop(index=tr)
        sd = rest.std(ddof=1)
        out[tr] = (v[tr] - rest.mean()) / sd if sd > 0 else np.nan
    return pd.Series(out).reindex(vals.index)


def loo_p(vals):
    v = vals.dropna()
    if len(v) < 5:
        return pd.Series(np.nan, index=vals.index)
    out = {}
    for tr in v.index:
        null = v.drop(index=tr).to_numpy(float)
        out[tr] = (1 + int((null >= v[tr]).sum())) / (1 + len(null))
    return pd.Series(out).reindex(vals.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zmatrix", nargs="+", required=True, help="path=label, one per setting")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--ct_names", default="",
                    help="cell-type names in code order (comma list) or a pkl with ct_categories")
    ap.add_argument("--mode", default="paracrine")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)

    mats = {}
    for spec in a.zmatrix:
        if "=" not in spec:
            sys.exit("[hub] --zmatrix entries are path=label: %s" % spec)
        p, lab = spec.rsplit("=", 1)
        if not os.path.isfile(p):
            sys.exit("[hub] not found: %s" % p)
        mats[lab] = pd.read_csv(p, sep=TAB, index_col=0)
        print("[hub] %-16s %d traits x %d channels" % (lab, *mats[lab].shape))
    labs = list(mats)
    traits = sorted(set.intersection(*[set(m.index) for m in mats.values()]))
    print("[hub] %d settings, %d common traits" % (len(labs), len(traits)))
    CT.update(celltypes.resolve(a.ct_names, celltypes.codes_from_columns(
        set.union(*[set(m.columns) for m in mats.values()]), a.mode)))

    rows = []
    for lab, Z in mats.items():
        for tr in traits:
            for h in sorted(CT.keys()):
                s = cell_stats(Z, tr, h, a.mode)
                if s:
                    s.update(filter=lab, trait=tr, hub=CT[h], hub_code=h)
                    rows.append(s)
    if not rows:
        sys.exit("[hub] no result")
    F = pd.DataFrame(rows)

    for c in ("z_in", "z_diff", "p_in", "p_diff"):
        F[c] = np.nan
    for (lab, h), g in F.groupby(["filter", "hub_code"]):
        s = g.set_index("trait")
        F.loc[g.index, "z_in"] = loo_z(s["in_mean"]).reindex(s.index).values
        F.loc[g.index, "z_diff"] = loo_z(s["diff"]).reindex(s.index).values
        F.loc[g.index, "p_in"] = loo_p(s["in_mean"]).reindex(s.index).values
        F.loc[g.index, "p_diff"] = loo_p(s["diff"]).reindex(s.index).values
    F["S"] = F[["z_in", "z_diff"]].min(axis=1)
    F["p_iut_discrete"] = F[["p_in", "p_diff"]].max(axis=1)
    F.to_csv(a.out_prefix + "_cells.tsv", sep=TAB, index=False)

    res = []
    for lab in labs:
        sub = F[F["filter"] == lab]
        best = (sub.sort_values("S", ascending=False)
                   .groupby("trait", as_index=False).first())
        bestd = (sub.sort_values(["p_iut_discrete", "S"], ascending=[True, False])
                    .groupby("trait", as_index=False).first())
        dmap = bestd.set_index("trait")
        for _, r in best.iterrows():
            null = best[best["trait"] != r["trait"]]["S"].to_numpy(float)
            null = null[np.isfinite(null)]
            if not len(null):
                continue
            k = int((null >= r["S"]).sum())
            dr = dmap.loc[r["trait"]]
            nulld = bestd[bestd["trait"] != r["trait"]]["p_iut_discrete"].to_numpy(float)
            nulld = nulld[np.isfinite(nulld)]
            kd = int((nulld <= dr["p_iut_discrete"]).sum())
            res.append(dict(
                filter=lab, trait=r["trait"], hub=r["hub"],
                in_mean=r["in_mean"], out_mean=r["out_mean"], diff=r["diff"],
                self_z=r["self_z"], n_pos_in=r["n_pos_in"], n_pos_diff=r["n_pos_diff"],
                z_in=r["z_in"], z_diff=r["z_diff"], S=r["S"],
                n_null=len(null), p_cont=(1 + k) / (1 + len(null)),
                min_attainable=1.0 / (1 + len(null)),
                hub_discrete=dr["hub"], p_iut_discrete=dr["p_iut_discrete"],
                p_discrete=(1 + kd) / (1 + len(nulld))))
    R = pd.DataFrame(res)
    for col, q in [("p_cont", "q_cont"), ("p_discrete", "q_discrete")]:
        R[q] = np.nan
        for lab in labs:
            m = R["filter"] == lab
            R.loc[m, q] = bh(R.loc[m, col].to_numpy())
    R.to_csv(a.out_prefix + "_final.tsv", sep=TAB, index=False)

    piv = R.pivot(index="trait", columns="filter", values="hub")
    sv = R.pivot(index="trait", columns="filter", values="S")
    pc = R.pivot(index="trait", columns="filter", values="p_cont")
    st = []
    for tr in piv.index:
        hs = piv.loc[tr].dropna().tolist()
        if not hs:
            continue
        top = max(set(hs), key=hs.count)
        st.append(dict(trait=tr, hub_mode=top, n_consistent=hs.count(top),
                       n_filters=len(hs), S_min=float(sv.loc[tr].min()),
                       S_max=float(sv.loc[tr].max()), p_min=float(pc.loc[tr].min())))
    S = pd.DataFrame(st).sort_values(["p_min", "S_min"], ascending=[True, False])
    S.to_csv(a.out_prefix + "_stability.tsv", sep=TAB, index=False)

    for lab in labs:
        sub = R[R["filter"] == lab].sort_values("S", ascending=False)
        print("\n%s" % lab)
        print(sub.head(a.top)[["trait", "hub", "z_in", "z_diff", "S", "p_cont"]]
              .round(4).to_string(index=False))
    print("\nwrote %s_{cells,final,stability}.tsv" % a.out_prefix)


if __name__ == "__main__":
    main()
