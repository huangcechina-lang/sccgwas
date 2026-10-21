#!/usr/bin/env python3
"""Average channel specificity over sections (write the output outside the GSS directory)."""
import argparse
import glob
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gss_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="paracrine")
    a = ap.parse_args()
    fs = [f for f in sorted(glob.glob(os.path.join(a.gss_dir, "gss_spec_*.tsv"))) if "MEAN" not in f]
    if not fs:
        raise SystemExit("no gss_spec_*.tsv in %s" % a.gss_dir)
    if os.path.abspath(os.path.dirname(a.out)) == os.path.abspath(a.gss_dir):
        raise SystemExit("--out must not be inside --gss_dir")
    acc, n = None, 0
    for f in fs:
        d = pd.read_csv(f, sep="\t")
        d = d.rename(columns={d.columns[0]: "gene"}).set_index("gene")
        d = d[[c for c in d.columns if c.endswith(a.mode)]]
        acc = d if acc is None else acc.add(d, fill_value=0)
        n += 1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    (acc / n).rename_axis("gene").to_csv(a.out, sep="\t")
    print("averaged %d sections -> %d genes x %d channels: %s" % (n, acc.shape[0], acc.shape[1], a.out))


if __name__ == "__main__":
    main()
