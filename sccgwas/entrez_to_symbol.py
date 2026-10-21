#!/usr/bin/env python3
"""Convert MAGMA .genes.out gene IDs (Entrez) to gene symbols using the MAGMA gene location file."""
import argparse, glob, os, shutil


def load_gene_loc(path):
    m, n_lines = {}, 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            n_lines += 1
            if len(p) < 2:
                continue
            sym = p[-1]
            if sym.isdigit():
                continue
            m[p[0]] = sym
    if not m:
        raise ValueError(
            f"{path}: no entrez->symbol pairs parsed ({n_lines} lines read).\n"
            "  A MAGMA gene.loc looks like:  79501  1  69091  70008  +  OR4F5\n"
            "  If yours has no symbol column, use the NCBI##.#.gene.loc from MAGMA's\n"
            "  auxiliary files, or supply a 2-column 'entrez<TAB>symbol' map.")
    return m


def read_header_and_rows(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    hi = next((i for i, l in enumerate(lines) if l.strip() and not l.startswith("#")), None)
    if hi is None:
        return None, None, None
    header = lines[hi].split()
    up = [h.upper() for h in header]
    if "GENE" not in up:
        raise ValueError(f"no GENE column in header: {header}")
    return header, up, lines[hi + 1:]


def convert(in_path, out_path, mapping, collision="keep_max_nsnps"):
    header, up, body = read_header_and_rows(in_path)
    if header is None:
        return dict(status="EMPTY")
    gi = up.index("GENE")
    nsnps_i = up.index("NSNPS") if "NSNPS" in up else (up.index("NSNP") if "NSNP" in up else None)

    first = next((l.split()[gi] for l in body if l.strip() and not l.startswith("#")), None)
    if first is not None and not first.isdigit():
        if os.path.realpath(in_path) != os.path.realpath(out_path):
            shutil.copy2(in_path, out_path)
        return dict(status="ALREADY_SYMBOL")

    kept, n_in, n_unmapped, collisions = {}, 0, 0, []
    for l in body:
        if not l.strip() or l.startswith("#"):
            continue
        p = l.split()
        if gi >= len(p):
            continue
        n_in += 1
        sym = mapping.get(p[gi])
        if sym is None:
            n_unmapped += 1
            continue
        nsnps = 0.0
        if nsnps_i is not None and nsnps_i < len(p):
            try:
                nsnps = float(p[nsnps_i])
            except ValueError:
                nsnps = 0.0
        if sym in kept:
            collisions.append((sym, kept[sym][2] if kept[sym] else "?", p[gi]))
            if collision == "drop":
                kept[sym] = None
                continue
            if kept[sym] is None:
                continue
            if nsnps > kept[sym][1]:
                q = list(p); q[gi] = sym
                kept[sym] = (q, nsnps, p[gi])
            continue
        q = list(p); q[gi] = sym
        kept[sym] = (q, nsnps, p[gi])

    rows = [v[0] for v in kept.values() if v is not None]
    with open(out_path, "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(r) + "\n")
    return dict(status="OK", n_in=n_in, n_out=len(rows), n_unmapped=n_unmapped,
                n_collisions=len(collisions), collisions=collisions[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_loc", required=True)
    ap.add_argument("--in", dest="inp", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--in_dir", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--collision", choices=["keep_max_nsnps", "drop"], default="keep_max_nsnps")
    a = ap.parse_args()

    mapping = load_gene_loc(a.gene_loc)
    print(f"[gene.loc] {a.gene_loc}: {len(mapping)} entrez->symbol pairs\n")

    jobs = []
    if a.inp:
        if not a.out:
            ap.error("--in requires --out")
        jobs.append((a.inp, a.out))
    elif a.in_dir:
        if not a.out_dir:
            ap.error("--in_dir requires --out_dir")
        os.makedirs(a.out_dir, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(a.in_dir, "*genes*.out"))):
            jobs.append((p, os.path.join(a.out_dir, os.path.basename(p))))
    else:
        ap.error("give --in/--out or --in_dir/--out_dir")

    print(f"{'file':40} {'status':14} {'in':>7} {'out':>7} {'unmap':>7} {'collide':>8}  rate")
    print("-" * 100)
    bad, n_conv, n_pass = [], 0, 0
    for src, dst in jobs:
        base = os.path.basename(src)
        try:
            r = convert(src, dst, mapping, a.collision)
        except Exception as e:
            print(f"{base:40} ERROR: {e}")
            bad.append(src); continue
        if r["status"] == "ALREADY_SYMBOL":
            n_pass += 1
            print(f"{base:40} {'PASSTHROUGH':14} {'-':>7} {'-':>7} {'-':>7} {'-':>8}  (already symbol)")
            continue
        if r["status"] != "OK":
            print(f"{base:40} {r['status']:14}")
            bad.append(src); continue
        n_conv += 1
        rate = r["n_out"] / max(r["n_in"], 1) * 100
        flag = "" if rate > 90 else "  <-- LOW: wrong gene.loc build?"
        print(f"{base:40} {'CONVERTED':14} {r['n_in']:7d} {r['n_out']:7d} "
              f"{r['n_unmapped']:7d} {r['n_collisions']:8d}  {rate:.1f}%{flag}")
        for sym, e1, e2 in r["collisions"][:3]:
            print(f"{'':42}collision {sym}: entrez {e1} vs {e2}")

    print("\n" + "=" * 72)
    print(f"converted: {n_conv}   passed through (already symbol): {n_pass}   failed: {len(bad)}")
    print(f"out_dir now holds {n_conv + n_pass} panel files -- point every run at it.")
    print("\nBEFORE launching the panel, verify ONE file against the GSS gene universe:")
    print("  the number of aligned genes should be close to the GSS gene count.")
    if bad:
        print(f"\n!! {len(bad)} file(s) failed -- do not run those:")
        for b in bad:
            print(f"   {b}")


if __name__ == "__main__":
    main()
