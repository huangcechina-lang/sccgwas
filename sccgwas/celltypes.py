"""Cell-type code -> name mapping shared by the pipeline scripts."""
import glob
import os
import pickle
import re


def codes_from_columns(cols, mode=None):
    out = set()
    for c in cols:
        m = re.match(r"\s*(\d+)\s*->\s*(\d+)(?::(\w+))?", str(c))
        if m and (mode is None or m.group(3) in (None, mode)):
            out.update((m.group(1), m.group(2)))
    return sorted(out, key=int)


def codes_from_gss(gss_dir, mode=None):
    fs = sorted(glob.glob(os.path.join(gss_dir, "gss_spec_*.tsv")))
    if not fs:
        return []
    with open(fs[0]) as f:
        return codes_from_columns(f.readline().rstrip("\n").split("\t")[1:], mode)


def resolve(ct_names="", codes=()):
    if ct_names:
        if os.path.isfile(ct_names):
            with open(ct_names, "rb") as f:
                names = list(pickle.load(f)["ct_categories"])
        else:
            names = [x.strip() for x in ct_names.split(",") if x.strip()]
        if len(names) < 2:
            raise SystemExit("--ct_names gives fewer than two cell types: %s" % names)
        return {str(i): str(n) for i, n in enumerate(names)}
    return {str(c): str(c) for c in codes}


def same_vocab(cats, ref, where):
    """Return the vocabulary; stop if it differs from the one seen first (ref)."""
    cats = [str(x) for x in cats]
    if ref is not None and cats != ref:
        raise SystemExit(
            "%s: cell-type vocabulary %s differs from %s seen in an earlier section.\n"
            "Integer cell-type codes would refer to different cell types across sections. "
            "Rebuild every section with the same prepare_inputs.py --ct_order." % (where, cats, ref))
    return cats


def hub_code(hub, CT):
    if hub in CT:
        return hub
    inv = {v: k for k, v in CT.items()}
    if hub in inv:
        return inv[hub]
    raise SystemExit("unknown hub '%s'; known: %s" % (hub, ", ".join("%s=%s" % kv for kv in CT.items())))
