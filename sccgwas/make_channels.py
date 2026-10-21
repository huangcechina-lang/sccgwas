#!/usr/bin/env python3
"""Channel table (c_s, c_r, mode, sender, receiver) for channel_gss.py."""
import argparse
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True, help="any inputs_<section>.pkl of the tissue")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="juxtacrine,paracrine,secreted")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    d = pickle.load(open(a.inputs, "rb"))
    inp = d["inputs"]
    inp = inp.item() if isinstance(inp, np.ndarray) else inp
    codes = sorted(set(np.asarray(inp.cell_type).astype(str)), key=lambda x: (len(x), x))
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    names = d.get("ct_categories")
    nm = (lambda c: str(names[int(c)])) if names is not None else (lambda c: "")
    rows = [dict(c_s=s, c_r=r, mode=m, sender=nm(s), receiver=nm(r))
            for s, r in itertools.product(codes, codes) for m in modes]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
    print("%d cell types (%s), %d channels -> %s"
          % (len(codes), ", ".join("%s=%s" % (c, names[int(c)]) for c in codes) if names is not None
             else ", ".join(codes), len(rows), a.out))


if __name__ == "__main__":
    main()
