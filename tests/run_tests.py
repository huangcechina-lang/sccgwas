#!/usr/bin/env python3
"""Regression tests on synthetic inputs: PYTHONHASHSEED=0 python tests/run_tests.py"""
import os
import pickle
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(os.path.dirname(HERE), "sccgwas")
LR = os.path.join(os.path.dirname(HERE), "resources", "LR_pairs_cellchat_omnipath.tsv")
sys.path.insert(0, S)
TAB = "\t"


def run(args, cwd):
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable] + args, cwd=cwd, env=env, capture_output=True, text=True)


def write_genes_out(path, genes, seed=0):
    rng = np.random.default_rng(seed)
    st = rng.integers(1_000_000, 100_000_000, len(genes))
    pd.DataFrame(dict(GENE=genes, CHR=rng.integers(1, 23, len(genes)), START=st,
                      STOP=st + rng.integers(1_000, 100_000, len(genes)),
                      NSNPS=rng.integers(5, 300, len(genes)), NPARAM=5, N=10000,
                      ZSTAT=rng.standard_normal(len(genes)), P=0.5)).to_csv(path, sep=" ", index=False)


def test_cell_type_vocabulary(tmp):
    """Sections with different cell-type sets keep the same codes when --ct_order is given."""
    import anndata as ad
    import celltypes
    lr = pd.read_csv(LR, sep=TAB)
    lr = lr[~lr.ligand.str.contains("_") & ~lr.receptor.str.contains("_")].head(30)
    genes = sorted(set(lr.ligand) | set(lr.receptor))
    write_genes_out(os.path.join(tmp, "z.genes.out"), genes)
    rng = np.random.default_rng(1)
    for name, types in (("s1", ["Ast", "End", "Mic"]), ("s2", ["Ast", "Mic"])):
        n = 90
        ct = [types[i % len(types)] for i in range(n)]
        a = ad.AnnData(X=rng.poisson(1.0, (n, len(genes))).astype(np.float32),
                       obs=pd.DataFrame({"cell_type": ct}, index=["c%d" % i for i in range(n)]),
                       var=pd.DataFrame(index=genes))
        a.obsm["spatial"] = rng.uniform(0, 500, (n, 2))
        a.write_h5ad(os.path.join(tmp, name + ".h5ad"))
    cats = {}
    for order in ("", "Ast,End,Mic"):
        for name in ("s1", "s2"):
            out = os.path.join(tmp, "inputs_%s%s.pkl" % (name, "_o" if order else ""))
            args = [os.path.join(S, "prepare_inputs.py"), "--spatial", name + ".h5ad", "--magma", "z.genes.out",
                    "--lr_db", LR, "--disease", "T", "--output", out]
            if order:
                args += ["--ct_order", order]
            p = run(args, tmp)
            assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
            cats[(name, bool(order))] = pickle.load(open(out, "rb"))["ct_categories"]
    assert cats[("s1", False)] != cats[("s2", False)], "without --ct_order the vocabularies should differ here"
    assert cats[("s1", True)] == cats[("s2", True)] == ["Ast", "End", "Mic"]
    try:
        celltypes.same_vocab(cats[("s2", False)], [str(x) for x in cats[("s1", False)]], "s2")
        raise AssertionError("same_vocab accepted different vocabularies")
    except SystemExit:
        pass
    p = run([os.path.join(S, "prepare_inputs.py"), "--spatial", "s1.h5ad", "--magma", "z.genes.out",
             "--lr_db", LR, "--disease", "T", "--output", "bad.pkl", "--ct_order", "Ast,Mic"], tmp)
    assert p.returncode != 0 and "not in --ct_order" in (p.stdout + p.stderr)


def test_magma_failure_is_fatal(tmp):
    """A failing MAGMA run stops the script and a stale .gsa.out is not reused."""
    genes = ["G%d" % i for i in range(40)]
    write_genes_out(os.path.join(tmp, "AD.genes.out"), genes)
    rng = np.random.default_rng(2)
    pd.DataFrame(rng.random((40, 2)), index=genes,
                 columns=["0->1:paracrine", "1->0:paracrine"]).rename_axis("gene").to_csv(
        os.path.join(tmp, "gss.tsv"), sep=TAB)
    os.makedirs(os.path.join(tmp, "raw"), exist_ok=True)
    pd.DataFrame([dict(trait="AD", status="OK", sibling_out=os.path.join(tmp, "AD.genes.out"))]).to_csv(
        os.path.join(tmp, "report.tsv"), sep=TAB, index=False)
    open(os.path.join(tmp, "manifest.tsv"), "w").write("AD\t%s\n" % os.path.join(tmp, "AD.genes.out"))
    os.makedirs(os.path.join(tmp, "gp"), exist_ok=True)
    stale = os.path.join(tmp, "gp", "AD.gsa.out")
    open(stale, "w").write("VARIABLE TYPE NGENES BETA BETA_STD SE P\nch_0_1 COVAR 40 1 1 0.1 1e-6\n")
    fake = os.path.join(tmp, "magma_fails.sh")
    open(fake, "w").write("#!/bin/sh\nexit 1\n")
    os.chmod(fake, 0o755)
    p = run([os.path.join(S, "magma_geneprop.py"), "--gss", "gss.tsv", "--raw_dir", "raw", "--map", "report.tsv",
             "--manifest", "manifest.tsv", "--out_dir", "gp", "--magma", fake], tmp)
    assert p.returncode != 0, "a failing MAGMA run must stop the script"
    assert not os.path.exists(stale), "the stale .gsa.out must be removed before the run"
    assert not os.path.exists(os.path.join(tmp, "gp", "geneprop_all.tsv"))


def test_global_only_drops_degenerate_channels(tmp):
    rng = np.random.default_rng(3)
    rows = []
    for c in ("0->1:paracrine", "2->1:paracrine"):
        rows += [dict(sample="s", channel=c, ligand="L", receptor="R", S=v) for v in rng.normal(0, 1, 400)]
    rows += [dict(sample="s", channel="3->1:paracrine", ligand="L", receptor="R", S=0.0) for _ in range(400)]
    pd.DataFrame(rows).to_csv(os.path.join(tmp, "nullE.tsv"), sep=TAB, index=False)
    p = run([os.path.join(S, "beyond_niche_threshold.py"), "--in", "nullE.tsv", "--out", "thr.tsv", "--global_only"], tmp)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    out = pd.read_csv(os.path.join(tmp, "thr.tsv"), sep=TAB)
    assert len(out) == 800, "rows of the degenerate channel must not enter the pooled distribution"
    good = pd.DataFrame(rows[:800]).S
    assert abs(np.percentile(out.S, 95) - np.percentile(good, 95)) < 1e-9


def test_conjunction_scope_and_selection(tmp):
    """Candidate level copies the pair-level genetic result to every channel and says so;
    --selective replaces the genetic axis with selection-adjusted values."""
    K = ["ligand", "receptor", "channel"]
    A, B = "0->2:paracrine", "1->2:paracrine"
    pd.DataFrame([dict(ligand="L", receptor="R", channel=A, t=5.0, t_L=3, t_R=3, p_raw_sum=1e-4,
                       p_pair_sum=1e-4, p_fwer_A_sum=0.01),
                  dict(ligand="L", receptor="R", channel=B, t=0.1, t_L=3, t_R=3, p_raw_sum=0.9,
                       p_pair_sum=1e-4, p_fwer_A_sum=0.99)]).to_csv(os.path.join(tmp, "maxt.tsv"), sep=TAB, index=False)
    per = [dict(sample="s%d" % i, ligand="L", receptor="R", channel=c, S_nullE=(5.0 if c == B else -1.0),
                p_nullE=0.01) for i in range(5) for c in (A, B)]
    pd.DataFrame(per).to_csv(os.path.join(tmp, "percell.tsv"), sep=TAB, index=False)
    pd.DataFrame(dict(channel=[A] * 60 + [B] * 60, S=np.linspace(-2, 2, 120))).to_csv(
        os.path.join(tmp, "sthr.tsv"), sep=TAB, index=False)
    pd.DataFrame([dict(channel=c, ligand="L", receptor="R", T=9.0, p_fwer=0.001, status="OK") for c in (A, B)]).to_csv(
        os.path.join(tmp, "null1.tsv"), sep=TAB, index=False)
    base = [os.path.join(S, "conjunction.py"), "--maxt", "maxt.tsv", "--percell", "percell.tsv",
            "--sthr", "sthr.tsv", "--null1", "null1.tsv", "--min_c", "2"]
    p = run(base + ["--out", "fwer.tsv"], tmp)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    F = pd.read_csv(os.path.join(tmp, "fwer.tsv"), sep=TAB).set_index("channel")
    assert not F.loc[B, "conj_channel"] and (F.genetic_scope == "row").all()
    p = run(base + ["--out", "cand.tsv", "--level", "candidate"], tmp)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    C = pd.read_csv(os.path.join(tmp, "cand.tsv"), sep=TAB).set_index("channel")
    assert C.loc[B, "conj_channel"] and (C.genetic_scope == "pair").all()
    pd.DataFrame([dict(ligand="L", receptor="R", channel=A, p_fwer_all_settings=0.2),
                  dict(ligand="L", receptor="R", channel=B, p_fwer_all_settings=0.9)]).to_csv(
        os.path.join(tmp, "sel_rows.tsv"), sep=TAB, index=False)
    pd.DataFrame([dict(ligand="L", receptor="R", q_pair_all_settings=0.3)]).to_csv(
        os.path.join(tmp, "sel_pairs.tsv"), sep=TAB, index=False)
    p = run(base + ["--out", "sel.tsv", "--selective", "sel"], tmp)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    G = pd.read_csv(os.path.join(tmp, "sel.tsv"), sep=TAB)
    assert not G.genetic.any(), "selection-adjusted P_FWER 0.2 must not pass"


def test_min_sec_is_required(tmp):
    p = run([os.path.join(S, "pair_genetic_maxt.py"), "--percell", "x", "--gss_dir", "x", "--genes_out", "x",
             "--out", "x"], tmp)
    assert p.returncode == 2 and "--min_sec" in p.stderr


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                t(tmp)
                print("PASS  %s" % t.__name__)
            except Exception:
                failed += 1
                print("FAIL  %s\n%s" % (t.__name__, traceback.format_exc()))
    print("%d of %d tests passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
