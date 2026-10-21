# SCCGWAS

SCCGWAS tests whether the genetic signal of a trait is enriched in directed cell–cell
communication: in signals sent by one cell type and received by another, read from a
cell-resolved spatial transcriptome. It works on directed channels (sender → receiver cell type)
and on individual ligand–receptor pairs within them, and evaluates each on three axes:

* **genetic**: is trait association concentrated in the genes of the channel or pair?
  Channel level: MAGMA gene-property regression. Pair level: gene-label permutation test with a
  maximum statistic over the tested pairs.
* **communication**: is the ligand–receptor co-expression between neighbouring cells higher than
  for control pairs matched on expression level and spatial autocorrelation?
* **beyond-niche**: does the co-expression exceed what the local cell-type composition predicts?
  Cells are permuted within cell-type × spatial-niche blocks.

A pair that passes all three axes on the same channel is the strongest result.

## Installation

```bash
git clone <repository URL>
cd sccgwas
pip install -r requirements.txt
```

Python 3.10 with the package versions in `requirements.txt` has been tested. MAGMA v1.10, its
reference panel (e.g. `g1000_eur`) and a gene-location file (e.g. NCBI37.3) are needed for the
gene-level statistics; they are available from the MAGMA website.

Run the scripts by path (`python sccgwas/<script>.py`) and set the hash seed first, because
random seeds depend on string hashing:

```bash
export PYTHONHASHSEED=0
python tests/run_tests.py        # quick check on synthetic data
```

## Input data

* **Spatial data**: one `.h5ad` per section with raw counts in `X`, cell types in
  `obs['cell_type']` and coordinates in micrometres in `obsm['spatial']`. Each observation must be
  one segmented cell; spot-level data with several cells per spot are not supported.
  `prep/` has converters for Stereo-seq cellbin GEF files and Visium HD (via Bin2cell).
* **GWAS**: for each trait, the MAGMA gene analysis output (`.genes.out`, converted to gene symbols
  with `entrez_to_symbol.py`, and `.genes.raw`).
* **Trait list**: a file without header, one line per trait: `label<TAB>path/to/label.genes.out`.
  Include a few traits unrelated to the tissue: they serve as controls for the hub statistic.
* **Ligand–receptor resource**: `resources/LR_pairs_cellchat_omnipath.tsv` (CellChat and
  OmniPath). Any table with ligand, receptor and annotation columns can be used instead.

## Workflow

Set the variables for your data:

```bash
S=sccgwas
LR=resources/LR_pairs_cellchat_omnipath.tsv
Z=magma_symbol/<trait>.genes.out     # trait of interest
T=<trait>                            # its label in traits.tsv
ORDER=<all cell types, comma-separated>
CT=pkl/inputs_<any section>.pkl
```

**1. Prepare each section**

```bash
for h in h5ad/*.h5ad; do
  python $S/prepare_inputs.py --spatial $h --magma $Z --lr_db $LR --disease $T --ct_order $ORDER \
      --output pkl/inputs_$(basename $h .h5ad).pkl
done
```

Use the same `--ct_order` for every section, so that each cell type has the same code everywhere.

**2. Channel specificity**

```bash
python $S/make_channels.py --inputs $CT --out channels.tsv
for p in pkl/inputs_*.pkl; do
  s=$(basename $p .pkl); s=${s#inputs_}
  python $S/channel_gss.py --inputs $p --channels channels.tsv --out gss/gss_spec_$s.tsv
done
python $S/audit_separability.py --gss_dir gss --pkl $CT --outdir audit
```

`audit/` lists cell-type pairs whose forward and reverse specificity profiles are highly
correlated (above 0.30). Such pairs cannot be resolved by direction; leave them out of the hub
analysis, or merge labels that separate poorly and rebuild the inputs.

**3. Channel level: which channels and which receiving cell type?**

```bash
python $S/average_gss.py --gss_dir gss --out gss_mean/gss_mean.tsv
python $S/collect_genes_raw.py --manifest traits.tsv --search magma --out_dir raw --report raw/report.tsv
python $S/magma_geneprop.py --gss gss_mean/gss_mean.tsv --raw_dir raw --map raw/report.tsv \
    --manifest traits.tsv --out_dir geneprop --magma /path/to/magma

python $S/channel_t_matrix.py --gss_dir gss --trait_manifest traits.tsv --pkl_dir pkl \
    --min_channels 1,10,20 --min_cells 0,10 --out_root tmat
for d in tmat/mch*_mcl*; do
  python $S/hub_zmatrix.py --matrix_dir $d --controls <control traits, comma-separated> \
      --out_prefix hub/$(basename $d)
done
python $S/hub_iut.py --ct_names $CT --out_prefix hub/iut \
    --zmatrix $(for f in hub/*_zmatrix.tsv; do echo $f=$(basename $f _zmatrix.tsv); done)
```

`geneprop/geneprop_all.tsv` gives the genetic association of every channel with every trait.
`hub/iut_final.tsv` gives, for each trait, the receiving cell type (hub) whose incoming channels
carry the signal, across six gene-filtering settings. Choose the receiving cell type `HUB` for
step 4 from this table, then run the channel-level communication axis on the channels into it:

```bash
python $S/null1_channels.py --pkl_dir pkl --trait_manifest traits.tsv --channels ALL_TO_HUB:$HUB \
    --out null1_hub.tsv
python $S/summarize_null1_channels.py --null1 null1_hub.tsv --trait $T --out null1_hub_summary.tsv
```

**4. Pair level: which ligand–receptor pairs?**

```bash
python $S/pair_beyond_niche.py --gss_dir gss --pkl_dir pkl --genes_out $Z --trait $T --hub $HUB \
    --ct_names $CT --out pair_percell.tsv
python $S/beyond_niche_controls.py --gss_dir gss --pkl_dir pkl --genes_out $Z --hub $HUB \
    --ct_names $CT --out calib
python $S/beyond_niche_threshold.py --in calib_nullE.tsv --out s_thr.tsv
python $S/pair_genetic_maxt.py --percell pair_percell.tsv --gss_dir gss --genes_out $Z \
    --ct_names $CT --min_sec <N> --out maxt.tsv
python $S/null1_pairs.py --pkl_dir pkl --channels ALL_TO_HUB:$HUB --out null1_pairs.tsv
python $S/conjunction.py --maxt maxt.tsv --percell pair_percell.tsv --sthr s_thr.tsv \
    --null1 null1_pairs.tsv --lr_resource $LR --out conjunction.tsv
```

**5. Account for the choice of receiving cell type**

When the receiving cell type was chosen in step 3 with the same GWAS, the pair test of step 4 is
conditional on that choice. `selective_maxt.py` repeats the permutations of step 4 and, in every
round, also repeats the hub selection; P values are then computed over the rounds that select the
same hub.

```bash
python $S/selective_maxt.py --percell pair_percell.tsv --gss_dir gss --genes_out $Z \
    --sel_gss_dir gss --sel_pkl_dir pkl --sel_zmatrix 'hub/{D}_zmatrix.tsv' --sel_filtered tmat \
    --trait $T --target_hub $HUB --ct_names $CT --min_sec <N> --ref_maxt maxt.tsv \
    --out_prefix selective/$T
python $S/conjunction.py --maxt maxt.tsv --percell pair_percell.tsv --sthr s_thr.tsv \
    --null1 null1_pairs.tsv --selective selective/$T --out conjunction_selective.tsv
```

Use the same `--min_sec` as in step 4. `--settings` must list the gene-filtering settings used in
step 3 (leave out a setting in which no gene passes the filter). If the hub was chosen in another
dataset, give that dataset as `--sel_gss_dir`, `--sel_pkl_dir` and `--sel_zmatrix`.

Every script lists its options with `--help`.

## Choosing parameters

| option | what it controls | guidance |
|---|---|---|
| `prepare_inputs.py --ct_order` | cell-type codes | the same list for every section of a tissue |
| `pair_genetic_maxt.py --min_sec` | which pair × channel combinations are tested: both genes must have positive specificity in at least this many sections | larger values test fewer, more consistently detected pairs and lighten the multiple-testing burden; with few sections, use all of them |
| `conjunction.py --min_c` | sections in which the beyond-niche axis must pass | default 0 picks the number whose Binomial(n, 0.05) upper tail is closest to `--c_null` (0.02) |
| `--mode` | signalling distance | `paracrine` (80 µm) by default; `juxtacrine` (15 µm) and `secreted` (200 µm) are available |
| `pair_genetic_maxt.py --n_sim` | gene-label permutations (default 20,000) | the smallest P value is 1/(n_sim + 1); increase it when several pairs reach that floor |
| `null1_pairs.py --n_null` | control redraws for the communication axis (default 1,000) | smallest P value 1/(n_null + 1) |
| `beyond_niche_controls.py --n_match` | control pairs per channel and section (default 1,000) | lower it for very large datasets |
| `prepare_inputs.py --no_gene_features` | leaves gene length and variant count out of the matched controls | matches on expression and spatial features only |
| `--donor_map` / `--donor_regex` | sections from the same donor | adds donor-level counts in `conjunction.py` and `summarize_null1_channels.py` |

## Reading the results

| file | content |
|---|---|
| `geneprop/geneprop_all.tsv` | channel-level genetic axis for every trait |
| `hub/iut_final.tsv` | receiving cell type per trait and its rank among the traits |
| `null1_hub_summary.tsv` | channel-level communication axis across sections |
| `maxt.tsv` | pair-level genetic axis: statistic `t`, ligand- and receptor-side components `t_L`, `t_R`, uncorrected P, family-wise P `p_fwer_A_sum` |
| `null1_pairs.tsv` | pair-level communication axis: statistic `T` and `p_fwer` |
| `conjunction.tsv` | the three axes for every pair × channel; `conj_channel` = all three on the same channel |
| `selective/<trait>_pairs.tsv`, `_rows.tsv` | pair-level genetic axis conditional on the hub selection |

* `p_fwer_A_sum` controls the family-wise error rate over all tested pair × channel combinations.
  `t_L` and `t_R` above 2 indicate support at both the ligand and the receptor rather than at one
  gene.
* `p_fwer` of the communication axis controls the family-wise error rate within each channel.
* `conjunction.py --level candidate` uses a Benjamini–Hochberg q < 0.10 over pairs instead of the
  family-wise threshold. This is a property of the pair and is copied to its channels
  (`genetic_scope = pair`).

## Citation

*Directed ligand–receptor channels are enriched for disease-associated genetic signals*
(in preparation; the reference will be added on publication).

## Licence

MIT (`LICENSE`). The ligand–receptor resource in `resources/` combines CellChat and OmniPath
records and keeps their licences; see `resources/NOTICE.md`.
