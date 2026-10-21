#!/usr/bin/env python3
"""Convert the expression matrix of inputs_*.pkl from dense to CSR sparse (values unchanged)."""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
from scipy import sparse as sp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--code", default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory with ccgwas_v4core.py")
    ap.add_argument("--pattern", default="inputs_*.pkl")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep_backup", action="store_true")
    ap.add_argument("--n_check", type=int, default=200,
                    help="random entries compared before writing")
    a = ap.parse_args()

    sys.path.insert(0, a.code)
    files = sorted(glob.glob(os.path.join(a.in_dir, a.pattern)))
    if not files:
        sys.exit("[sparse] no %s in %s" % (a.pattern, a.in_dir))
    print("[sparse] %d files, %s" % (len(files), "apply" if a.apply else "dry run"))

    rng = np.random.default_rng(0)
    tot_before = tot_after = 0
    for f in files:
        t0 = time.time()
        sz0 = os.path.getsize(f)
        tot_before += sz0
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        inp = d["inputs"] if isinstance(d, dict) else d
        Y = inp.Y
        if sp.issparse(Y):
            print("  %-26s already sparse (%s)" % (os.path.basename(f), Y.format))
            tot_after += sz0
            continue

        Y = np.asarray(Y)
        nnz = int((Y != 0).sum())
        frac = nnz / Y.size
        Ys = sp.csr_matrix(Y)

        ok = (Ys.shape == Y.shape) and (Ys.nnz == nnz)
        if ok and a.n_check:
            ii = rng.integers(0, Y.shape[0], a.n_check)
            jj = rng.integers(0, Y.shape[1], a.n_check)
            ok = bool(np.allclose(np.asarray(Ys[ii, jj]).ravel(), Y[ii, jj]))
        if ok and nnz:
            nzi, nzj = Y.nonzero()
            k = rng.choice(len(nzi), min(a.n_check, len(nzi)), replace=False)
            ok = bool(np.allclose(np.asarray(Ys[nzi[k], nzj[k]]).ravel(),
                                  Y[nzi[k], nzj[k]]))
        if not ok:
            print("  %-26s !! check failed, not written" % os.path.basename(f))
            continue

        est = (Ys.data.nbytes + Ys.indices.nbytes + Ys.indptr.nbytes)
        print("  %-26s %6.1f GB  non-zero %.4f%%  ->  about %6.1f MB  (%.0fx)  %.0fs"
              % (os.path.basename(f), sz0 / 1e9, 100 * frac, est / 1e6,
                 sz0 / max(est, 1), time.time() - t0))

        if a.apply:
            if a.keep_backup:
                os.replace(f, f + ".dense.bak")
            inp.Y = Ys
            tmp = f + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(d, fh, protocol=4)
            os.replace(tmp, f)
            tot_after += os.path.getsize(f)
        else:
            tot_after += est

    print("\n  total %.1f GB -> %.1f GB (%.0fx)"
          % (tot_before / 1e9, tot_after / 1e9, tot_before / max(tot_after, 1)))
    if not a.apply:
        print("  rerun with --apply to write")


if __name__ == "__main__":
    main()
