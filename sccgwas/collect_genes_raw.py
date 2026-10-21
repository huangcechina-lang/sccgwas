#!/usr/bin/env python3
"""Locate the MAGMA .genes.raw of each trait and write the report used by magma_geneprop.py."""
import argparse
import os
import re
import shutil
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="two columns: label<TAB>symbol .genes.out")
    ap.add_argument("--search", nargs="+", required=True, help="directories searched for *.genes.raw")
    ap.add_argument("--out_dir", required=True, help="where <label>.genes.raw is placed")
    ap.add_argument("--report", default="", help="report TSV for magma_geneprop.py --map")
    ap.add_argument("--link", action="store_true",
                    help="symlink instead of copy")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    want = {}
    for ln in open(a.manifest):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 2 or not p[0].strip():
            continue
        lab, path = p[0].strip(), p[1].strip()
        m = re.search(r"(GCST\d+)", os.path.basename(path)) or \
            re.search(r"(GCST\d+)", path)
        acc = m.group(1) if m else None
        want[lab] = dict(out=path, acc=acc, base=os.path.basename(path))
    print("[raw] %d traits in the manifest, %d with a GCST accession"
          % (len(want), sum(1 for v in want.values() if v["acc"])))
    noacc = [k for k, v in want.items() if not v["acc"]]
    if noacc:
        print("      without accession (matched by label): %s" % noacc)

    found = defaultdict(list)
    n_scan = 0
    for root_dir in a.search:
        root_dir = os.path.expanduser(root_dir)
        if not os.path.isdir(root_dir):
            print("      !! not a directory: %s" % root_dir); continue
        for dp, _, fns in os.walk(root_dir):
            for fn in fns:
                if not fn.endswith(".genes.raw"):
                    continue
                n_scan += 1
                full = os.path.join(dp, fn)
                m = re.search(r"(GCST\d+)", fn) or re.search(r"(GCST\d+)", dp)
                key = m.group(1) if m else fn.replace(".genes.raw", "").lower()
                found[key].append(full)
    print("[raw] %d .genes.raw files, %d keys" % (n_scan, len(found)))

    rows, ok, miss = [], 0, []
    for lab, v in sorted(want.items()):
        cands = []
        if v["acc"] and v["acc"] in found:
            cands = found[v["acc"]]
        else:
            kw = lab.lower()
            for k, fs in found.items():
                if kw in k.lower():
                    cands += fs
        if not cands:
            miss.append(lab)
            rows.append(dict(trait=lab, acc=v["acc"] or "", raw="", n_cand=0,
                             status="MISSING")); continue
        cands = sorted(cands, key=lambda f: -os.path.getsize(f))
        src = cands[0]
        dst = os.path.join(a.out_dir, "%s.genes.raw" % lab)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        if a.link:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        sib = src.replace(".genes.raw", ".genes.out")
        rows.append(dict(trait=lab, acc=v["acc"] or "", raw=src,
                         n_cand=len(cands), size_mb=round(os.path.getsize(src)/1e6, 1),
                         sibling_out=sib if os.path.exists(sib) else "",
                         status="OK"))
        ok += 1

    print("\n[raw] found %d / %d" % (ok, len(want)))
    if miss:
        print("      missing %d: %s" % (len(miss), miss))
    print("\n%-18s %-14s %8s %8s  %s" % ("trait", "GCST", "n_cand", "MB", "source"))
    for r in rows:
        if r["status"] != "OK":
            print("%-18s %-14s %8s %8s  -- not found" % (r["trait"], r["acc"], 0, "-"))
            continue
        print("%-18s %-14s %8d %8.1f  %s"
              % (r["trait"], r["acc"], r["n_cand"], r["size_mb"],
                 r["raw"].replace(os.path.expanduser("~"), "~")))

    if a.report:
        import pandas as pd
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(a.report, sep="\t", index=False)
        print("\nwrote %s" % a.report)

    if ok:
        f0 = [r for r in rows if r["status"] == "OK"][0]
        print("\n[raw] gene ID type in %s:" % f0["trait"])
        with open(os.path.join(a.out_dir, "%s.genes.raw" % f0["trait"])) as fh:
            for i, ln in enumerate(fh):
                if ln.startswith("#"):
                    continue
                gid = ln.split()[0]
                kind = ("ENSG" if gid.startswith("ENSG")
                        else ("Entrez" if gid.isdigit() else "symbol or other"))
                print("      first gene ID = %s  ->  %s" % (gid, kind))
                break


if __name__ == "__main__":
    main()
