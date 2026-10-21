"""SCCGWAS core: channels, communication weights, matched controls and the Null-1 statistic."""
from __future__ import annotations
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set, Callable, Literal
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


DEFAULT_SIGNALING_RADII = {
    "juxtacrine": 15.0,
    "paracrine":  80.0,
    "secreted":  200.0,
}


MIN_EFFECTIVE_CONTROLS_OK = 50


MIN_EFFECTIVE_CONTROLS_LOWCONF = 20


MAX_DROP_RATE_PER_CHANNEL = 0.30


MIN_CELLS_PER_CT_FOR_MORAN = 50


@dataclass
class CCGWASConfig:
    n_matched_controls: int = 100
    knn_pool_size: int = 500
    n_null_a_reps: int = 100
    n_null_c_reps: int = 100
    n_pseudo_db_reps: int = 1000
    n_fdr_null_seeds: int = 50
    n_fdr_bootstrap: int = 200
    signaling_radii: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SIGNALING_RADII)
    )
    random_seed: int = 42
    verbose: bool = True
    use_lambda_gene_corr: bool = False
    r_hat_floor: float = -0.4
    r_hat_ceil: float = 0.95
    min_cells_per_channel: int = 50
    studentize_channel_stat: bool = True

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.random_seed + salt)


@dataclass
class CCGWASInputs:
    Y: np.ndarray
    cell_type: np.ndarray
    coord: np.ndarray
    z_gwas: np.ndarray
    gene_names: np.ndarray
    LR_pairs: pd.DataFrame
    gene_features: np.ndarray
    is_ligand: np.ndarray
    is_receptor: np.ndarray
    known_LR_set: Set[Tuple[int, int]]
    chromosome: np.ndarray
    gene_gene_corr: Optional[np.ndarray] = None

    @property
    def n_cells(self) -> int:
        return self.Y.shape[0]

    @property
    def n_genes(self) -> int:
        return self.Y.shape[1]

    @property
    def n_cell_types(self) -> int:
        return int(self.cell_type.max() + 1)

    @property
    def modes(self) -> List[str]:
        return sorted(self.LR_pairs["mode"].unique().tolist())


@dataclass
class PreprocResult:
    Y_train: np.ndarray
    Y_test: np.ndarray
    expr_norm_test: np.ndarray
    per_ct_mean: np.ndarray
    per_ct_morans_i: Dict[str, np.ndarray]
    gene_log_mean: np.ndarray
    gene_log_var: np.ndarray
    cell_indices_by_ct: List[np.ndarray]


@dataclass
class MatchedControlPool:
    control_pairs: Dict[Tuple[int, int, str], List[List[Tuple[int, int]]]]
    diagnostics: pd.DataFrame


@dataclass
class ChannelTestResult:
    channel_T: Dict[Tuple[int, int, str], float]
    channel_p_null1: Dict[Tuple[int, int, str], float]
    channel_p_null2: Dict[Tuple[int, int, str], float]
    channel_p_acat: Dict[Tuple[int, int], float]
    channel_status: Dict[Tuple[int, int, str], str]


def poisson_count_split(
    Y: np.ndarray, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Y_dense = Y.toarray() if sparse.issparse(Y) else np.asarray(Y)
    Y_train = rng.binomial(Y_dense.astype(np.int64), 0.5).astype(np.float32)
    Y_test = (Y_dense.astype(np.float32) - Y_train).astype(np.float32)
    return Y_train, Y_test


def normalize_size_factor(Y: np.ndarray) -> np.ndarray:
    totals = Y.sum(axis=1)
    med = float(np.median(totals[totals > 0])) if (totals > 0).any() else 1.0
    sf = np.where(totals > 0, totals / med, 1.0)
    return Y / sf[:, None]


def standardize_per_gene(
    Y_norm: np.ndarray,
    log_mean: np.ndarray,
    log_var: np.ndarray,
    pseudocount: float = 1.0,
) -> np.ndarray:
    log_y = np.log1p(Y_norm)
    sigma = np.sqrt(np.maximum(log_var, 1e-6))
    return (log_y - log_mean[None, :]) / sigma[None, :]


def compute_per_ct_mean(
    expr_norm: np.ndarray, cell_type: np.ndarray, n_ct: int
) -> np.ndarray:
    G = expr_norm.shape[1]
    out = np.zeros((n_ct, G), dtype=np.float32)
    for c in range(n_ct):
        idx = np.where(cell_type == c)[0]
        if len(idx) == 0:
            continue
        out[c] = expr_norm[idx].mean(axis=0)
    return out


def compute_morans_i_per_ct(
    expr_norm: np.ndarray,
    coord: np.ndarray,
    cell_type: np.ndarray,
    n_ct: int,
    radius: float,
    verbose: bool = False,
) -> np.ndarray:
    G = expr_norm.shape[1]
    morans = np.zeros((n_ct, G), dtype=np.float32)
    for c in range(n_ct):
        cells_c = np.where(cell_type == c)[0]
        if len(cells_c) < MIN_CELLS_PER_CT_FOR_MORAN:
            continue
        coord_c = coord[cells_c]
        tree = cKDTree(coord_c)
        pairs = tree.query_pairs(r=radius, output_type='ndarray')
        if len(pairs) == 0:
            continue
        i_idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
        j_idx = np.concatenate([pairs[:, 1], pairs[:, 0]])
        W_total = float(len(i_idx))
        N_c = len(cells_c)
        expr_c = expr_norm[cells_c]
        means_c = expr_c.mean(axis=0)
        diffs_c = expr_c - means_c[None, :]
        var_c = (diffs_c ** 2).sum(axis=0)
        var_c = np.maximum(var_c, 1e-6)
        numer = np.empty(G, dtype=np.float32)
        _cap_bytes = 1 << 30
        _blk = int(max(64, min(G, _cap_bytes // (max(len(i_idx), 1) * 4))))
        for _s in range(0, G, _blk):
            _cols = np.arange(_s, min(_s + _blk, G))
            numer[_s:_s + len(_cols)] = np.einsum(
                "ng,ng->g",
                diffs_c[np.ix_(i_idx, _cols)],
                diffs_c[np.ix_(j_idx, _cols)],
            )
        morans[c] = (N_c / W_total) * numer / var_c
        if verbose:
            print(f"  Moran's I CT={c}: {len(cells_c)} cells, {len(pairs)} pairs, "
                  f"mean |I|={np.abs(morans[c]).mean():.3f}")
    return morans


def preprocess_inputs(
    inputs: CCGWASInputs,
    config: CCGWASConfig,
) -> PreprocResult:
    t0 = time.time()
    if config.verbose:
        print("[preprocess] Count splitting...")
    Y_train, Y_test = poisson_count_split(inputs.Y, seed=config.random_seed)
    Y_train_norm = normalize_size_factor(Y_train)
    Y_test_norm = normalize_size_factor(Y_test)
    log_train = np.log1p(Y_train_norm)
    log_mean = log_train.mean(axis=0)
    log_var = log_train.var(axis=0)
    log_mean = log_mean.astype(np.float32)
    log_var = np.maximum(log_var.astype(np.float32), 1e-4)

    if config.verbose:
        print("[preprocess] Standardizing test expression...")
    expr_norm_test = standardize_per_gene(Y_test_norm, log_mean, log_var)

    if config.verbose:
        print("[preprocess] Per-CT mean...")
    per_ct_mean = compute_per_ct_mean(
        expr_norm_test, inputs.cell_type, inputs.n_cell_types
    )

    cell_indices_by_ct = [
        np.where(inputs.cell_type == c)[0] for c in range(inputs.n_cell_types)
    ]

    if config.verbose:
        print("[preprocess] Moran's I per mode...")
    per_ct_morans_i = {}
    for mode, radius in config.signaling_radii.items():
        if config.verbose:
            print(f"  mode={mode}, r={radius}um")
        morans = compute_morans_i_per_ct(
            expr_norm_test, inputs.coord, inputs.cell_type,
            inputs.n_cell_types, radius, verbose=False,
        )
        per_ct_morans_i[mode] = morans

    if config.verbose:
        print(f"[preprocess] Done in {time.time() - t0:.1f}s.")

    return PreprocResult(
        Y_train=Y_train, Y_test=Y_test,
        expr_norm_test=expr_norm_test,
        per_ct_mean=per_ct_mean,
        per_ct_morans_i=per_ct_morans_i,
        gene_log_mean=log_mean,
        gene_log_var=log_var,
        cell_indices_by_ct=cell_indices_by_ct,
    )


def compute_pair_z_batch(
    pair_lig: np.ndarray, pair_rec: np.ndarray,
    z_gwas: np.ndarray,
    chromosome: np.ndarray,
    gene_gene_corr: Optional[np.ndarray],
    config: CCGWASConfig,
) -> np.ndarray:
    z_L = z_gwas[pair_lig]
    z_R = z_gwas[pair_rec]
    if gene_gene_corr is not None:
        r_hat = gene_gene_corr[pair_lig, pair_rec]
    else:
        same_chr = chromosome[pair_lig] == chromosome[pair_rec]
        r_hat = np.where(same_chr, 0.05, 0.0)
    r_hat = np.clip(r_hat, config.r_hat_floor, config.r_hat_ceil)
    denom = np.sqrt(2.0 * (1.0 + r_hat))
    return (z_L + z_R) / denom


def get_spatial_neighbors_in_channel(
    cells_s: np.ndarray, cells_r: np.ndarray,
    coord: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(cells_s) == 0 or len(cells_r) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    coord_s = coord[cells_s]
    coord_r = coord[cells_r]
    tree_r = cKDTree(coord_r)
    nbrs = tree_r.query_ball_point(coord_s, r=radius)
    i_list, j_list = [], []
    for si, jlist in enumerate(nbrs):
        if not jlist:
            continue
        global_i = cells_s[si]
        for jj in jlist:
            global_j = cells_r[jj]
            if global_i == global_j:
                continue
            i_list.append(global_i)
            j_list.append(global_j)
    return np.array(i_list, dtype=np.int64), np.array(j_list, dtype=np.int64)


def compute_W_batch(
    L_indices: np.ndarray, R_indices: np.ndarray,
    expr_norm: np.ndarray,
    i_arr: np.ndarray, j_arr: np.ndarray,
    Ei: Optional[np.ndarray] = None, Ej: Optional[np.ndarray] = None,
) -> np.ndarray:
    if len(i_arr) == 0:
        return np.zeros(len(L_indices), dtype=np.float32)
    if Ei is None:
        Ei = expr_norm[i_arr]
    if Ej is None:
        Ej = expr_norm[j_arr]
    L_expr = Ei[:, L_indices]
    R_expr = Ej[:, R_indices]
    return np.mean(L_expr * R_expr, axis=0).astype(np.float32)


def build_channel_cell_pairs(
    inputs: CCGWASInputs,
    config: CCGWASConfig,
) -> Dict[Tuple[int, int, str], Tuple[np.ndarray, np.ndarray]]:
    if config.verbose:
        print("[cellpairs] Building cell-pair index for all channels x modes...")
    out = {}
    cells_by_ct = [
        np.where(inputs.cell_type == c)[0] for c in range(inputs.n_cell_types)
    ]
    for c_s in range(inputs.n_cell_types):
        for c_r in range(inputs.n_cell_types):
            for mode, radius in config.signaling_radii.items():
                i_arr, j_arr = get_spatial_neighbors_in_channel(
                    cells_by_ct[c_s], cells_by_ct[c_r],
                    inputs.coord, radius,
                )
                if len(i_arr) < config.min_cells_per_channel:
                    continue
                out[(c_s, c_r, mode)] = (i_arr, j_arr)
        if config.verbose:
            print(f"  CT_sender={c_s}: built channels for {sum(1 for k in out if k[0]==c_s)} (c_r, mode) combos")
    return out


def build_gene_channel_features(
    inputs: CCGWASInputs, preproc: PreprocResult,
) -> np.ndarray:
    G = inputs.n_genes
    feat = np.zeros((G, 5), dtype=np.float32)
    feat[:, 0] = preproc.gene_log_mean
    feat[:, 1] = np.log1p(preproc.gene_log_var)
    feat[:, 2] = inputs.gene_features[:, 0]
    feat[:, 3] = inputs.gene_features[:, 1]
    feat[:, 4] = inputs.gene_features[:, 2]
    return feat


def _standardize_columns(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (X - mu) / sd


def build_channel_specific_control_pool(
    inputs: CCGWASInputs,
    preproc: PreprocResult,
    config: CCGWASConfig,
    cell_pairs: Dict[Tuple[int, int, str], Tuple[np.ndarray, np.ndarray]],
    skip_dim_4_5: bool = False,
) -> MatchedControlPool:
    if config.verbose:
        print("[matching] Building channel-specific control pools...")
    t0 = time.time()
    rng = config.rng(salt=1)

    global_feat = build_gene_channel_features(inputs, preproc)

    ligand_pool_idx = np.where(inputs.is_ligand)[0]
    receptor_pool_idx = np.where(inputs.is_receptor)[0]

    known = inputs.known_LR_set

    control_pairs = {}
    diag_rows = []

    mode_to_lr = {}
    for k, row in inputs.LR_pairs.iterrows():
        mode_to_lr.setdefault(row["mode"], []).append(
            (int(row["ligand_idx"]), int(row["receptor_idx"]))
        )

    for channel_key, (i_arr, j_arr) in cell_pairs.items():
        c_s, c_r, mode = channel_key
        real_lr_in_mode = mode_to_lr.get(mode, [])
        if len(real_lr_in_mode) == 0:
            continue

        morans_mode = preproc.per_ct_morans_i[mode]
        per_ct_mean = preproc.per_ct_mean

        if skip_dim_4_5:
            lig_feat = np.column_stack([
                global_feat[ligand_pool_idx],
                morans_mode[c_s, ligand_pool_idx][:, None],
            ])
            rec_feat = np.column_stack([
                global_feat[receptor_pool_idx],
                morans_mode[c_r, receptor_pool_idx][:, None],
            ])
        else:
            lig_feat = np.column_stack([
                global_feat[ligand_pool_idx],
                per_ct_mean[c_s, ligand_pool_idx][:, None],
                morans_mode[c_s, ligand_pool_idx][:, None],
            ])
            rec_feat = np.column_stack([
                global_feat[receptor_pool_idx],
                per_ct_mean[c_r, receptor_pool_idx][:, None],
                morans_mode[c_r, receptor_pool_idx][:, None],
            ])

        lig_feat_std = _standardize_columns(lig_feat)
        rec_feat_std = _standardize_columns(rec_feat)

        knn_lig = NearestNeighbors(
            n_neighbors=min(config.knn_pool_size, len(ligand_pool_idx)),
            metric="euclidean",
        ).fit(lig_feat_std)
        knn_rec = NearestNeighbors(
            n_neighbors=min(config.knn_pool_size, len(receptor_pool_idx)),
            metric="euclidean",
        ).fit(rec_feat_std)

        channel_assignments = []

        for k_idx, (L, R) in enumerate(real_lr_in_mode):
            if skip_dim_4_5:
                real_lig_feat = np.array([[
                    global_feat[L, 0], global_feat[L, 1], global_feat[L, 2],
                    global_feat[L, 3], global_feat[L, 4],
                    morans_mode[c_s, L],
                ]])
                real_rec_feat = np.array([[
                    global_feat[R, 0], global_feat[R, 1], global_feat[R, 2],
                    global_feat[R, 3], global_feat[R, 4],
                    morans_mode[c_r, R],
                ]])
            else:
                real_lig_feat = np.array([[
                    global_feat[L, 0], global_feat[L, 1], global_feat[L, 2],
                    global_feat[L, 3], global_feat[L, 4],
                    per_ct_mean[c_s, L], morans_mode[c_s, L],
                ]])
                real_rec_feat = np.array([[
                    global_feat[R, 0], global_feat[R, 1], global_feat[R, 2],
                    global_feat[R, 3], global_feat[R, 4],
                    per_ct_mean[c_r, R], morans_mode[c_r, R],
                ]])
            mu_l = lig_feat.mean(axis=0); sd_l = np.maximum(lig_feat.std(axis=0), 1e-8)
            mu_r = rec_feat.mean(axis=0); sd_r = np.maximum(rec_feat.std(axis=0), 1e-8)
            real_lig_feat_std = (real_lig_feat - mu_l) / sd_l
            real_rec_feat_std = (real_rec_feat - mu_r) / sd_r

            _, lig_nn = knn_lig.kneighbors(real_lig_feat_std)
            _, rec_nn = knn_rec.kneighbors(real_rec_feat_std)
            lig_candidates = ligand_pool_idx[lig_nn[0]]
            rec_candidates = receptor_pool_idx[rec_nn[0]]

            lig_candidates = lig_candidates[lig_candidates != L]
            rec_candidates = rec_candidates[rec_candidates != R]

            controls = []
            attempts = 0
            max_attempts = config.n_matched_controls * 20
            while len(controls) < config.n_matched_controls and attempts < max_attempts:
                attempts += 1
                if len(lig_candidates) == 0 or len(rec_candidates) == 0:
                    break
                l_idx = lig_candidates[rng.integers(0, len(lig_candidates))]
                r_idx = rec_candidates[rng.integers(0, len(rec_candidates))]
                if (l_idx, r_idx) in known:
                    continue
                if l_idx == r_idx:
                    continue
                controls.append((int(l_idx), int(r_idx)))
            channel_assignments.append(controls)

            n_eff = len(controls)
            if n_eff >= MIN_EFFECTIVE_CONTROLS_OK:
                status = "OK"
            elif n_eff >= MIN_EFFECTIVE_CONTROLS_LOWCONF:
                status = "LowConf"
            else:
                status = "Drop"

            if controls:
                ctrl_arr = np.array(controls)
                ctrl_lig_idx = ctrl_arr[:, 0]
                ctrl_rec_idx = ctrl_arr[:, 1]
                lig_global = global_feat[ctrl_lig_idx]
                rec_global = global_feat[ctrl_rec_idx]
                lig_ct_mean = per_ct_mean[c_s, ctrl_lig_idx]
                rec_ct_mean = per_ct_mean[c_r, ctrl_rec_idx]
                lig_moran = morans_mode[c_s, ctrl_lig_idx]
                rec_moran = morans_mode[c_r, ctrl_rec_idx]
                real_lig_global = global_feat[L]
                real_rec_global = global_feat[R]
                pool_lig_sd = np.maximum(global_feat[ligand_pool_idx].std(axis=0), 1e-6)
                pool_rec_sd = np.maximum(global_feat[receptor_pool_idx].std(axis=0), 1e-6)
                smd_lig_global = np.abs(lig_global.mean(axis=0) - real_lig_global) / pool_lig_sd
                smd_rec_global = np.abs(rec_global.mean(axis=0) - real_rec_global) / pool_rec_sd
                smd_lig_ct = np.abs(lig_ct_mean.mean() - per_ct_mean[c_s, L]) / max(
                    per_ct_mean[c_s, ligand_pool_idx].std(), 1e-6
                )
                smd_rec_ct = np.abs(rec_ct_mean.mean() - per_ct_mean[c_r, R]) / max(
                    per_ct_mean[c_r, receptor_pool_idx].std(), 1e-6
                )
                smd_lig_mor = np.abs(lig_moran.mean() - morans_mode[c_s, L]) / max(
                    morans_mode[c_s, ligand_pool_idx].std(), 1e-6
                )
                smd_rec_mor = np.abs(rec_moran.mean() - morans_mode[c_r, R]) / max(
                    morans_mode[c_r, receptor_pool_idx].std(), 1e-6
                )
                unique_pairs = len(set(controls))
                reuse_rate = 1.0 - unique_pairs / len(controls) if controls else 0.0
            else:
                smd_lig_global = np.zeros(5); smd_rec_global = np.zeros(5)
                smd_lig_ct = smd_rec_ct = smd_lig_mor = smd_rec_mor = 0.0
                reuse_rate = 0.0

            diag_rows.append({
                "c_s": c_s, "c_r": c_r, "mode": mode,
                "L_idx": L, "R_idx": R, "k_idx": k_idx,
                "n_effective_controls": n_eff,
                "status": status,
                "reuse_rate": reuse_rate,
                "SMD_lig_logmean": smd_lig_global[0],
                "SMD_lig_logvar":  smd_lig_global[1],
                "SMD_lig_size":    smd_lig_global[2],
                "SMD_lig_snp":     smd_lig_global[3],
                "SMD_lig_ld":      smd_lig_global[4],
                "SMD_rec_logmean": smd_rec_global[0],
                "SMD_rec_logvar":  smd_rec_global[1],
                "SMD_rec_size":    smd_rec_global[2],
                "SMD_rec_snp":     smd_rec_global[3],
                "SMD_rec_ld":      smd_rec_global[4],
                "SMD_lig_ctmean":  smd_lig_ct,
                "SMD_rec_ctmean":  smd_rec_ct,
                "SMD_lig_moran":   smd_lig_mor,
                "SMD_rec_moran":   smd_rec_mor,
            })

        control_pairs[channel_key] = channel_assignments

    diag_df = pd.DataFrame(diag_rows)
    if config.verbose:
        print(f"[matching] Done in {time.time() - t0:.1f}s. "
              f"{len(control_pairs)} channels, {len(diag_df)} LRxchannel rows.")
        drop_rate = (diag_df["status"] == "Drop").mean() if len(diag_df) else 0
        print(f"[matching] Drop rate: {drop_rate*100:.1f}%, "
              f"LowConf rate: {(diag_df['status'] == 'LowConf').mean()*100:.1f}%")
    return MatchedControlPool(control_pairs=control_pairs, diagnostics=diag_df)


def acat_combine(pvals: np.ndarray) -> float:
    pvals = np.asarray(pvals)
    pvals = pvals[np.isfinite(pvals)]
    pvals = np.clip(pvals, 1e-15, 1 - 1e-15)
    if len(pvals) == 0:
        return np.nan
    if len(pvals) == 1:
        return float(pvals[0])
    t_stat = np.mean(np.tan((0.5 - pvals) * np.pi))
    p_combined = 0.5 - np.arctan(t_stat) / np.pi
    return float(p_combined)


def compute_channel_test_dual_null(
    inputs: CCGWASInputs,
    preproc: PreprocResult,
    config: CCGWASConfig,
    cell_pairs: Dict[Tuple[int, int, str], Tuple[np.ndarray, np.ndarray]],
    matching: MatchedControlPool,
    z_gwas_override: Optional[np.ndarray] = None,
    return_T_only: bool = False,
) -> ChannelTestResult:
    z_gwas = z_gwas_override if z_gwas_override is not None else inputs.z_gwas
    expr_norm = preproc.expr_norm_test

    real_by_mode = {}
    for k, row in inputs.LR_pairs.iterrows():
        L = int(row["ligand_idx"]); R = int(row["receptor_idx"])
        m = row["mode"]
        real_by_mode.setdefault(m, []).append((L, R))

    real_pair_z_by_mode = {}
    for m, pairs in real_by_mode.items():
        L_arr = np.array([p[0] for p in pairs])
        R_arr = np.array([p[1] for p in pairs])
        real_pair_z_by_mode[m] = compute_pair_z_batch(
            L_arr, R_arr, z_gwas, inputs.chromosome, inputs.gene_gene_corr, config,
        )

    channel_T = {}
    channel_p1 = {}
    channel_p2 = {}
    channel_status = {}

    for channel_key, (i_arr, j_arr) in cell_pairs.items():
        c_s, c_r, mode = channel_key
        real_pairs = real_by_mode.get(mode, [])
        if not real_pairs:
            continue
        if channel_key not in matching.control_pairs:
            continue
        ctrl_assignments = matching.control_pairs[channel_key]
        valid_k = []
        for k_idx, ctrls in enumerate(ctrl_assignments):
            if len(ctrls) >= MIN_EFFECTIVE_CONTROLS_LOWCONF:
                valid_k.append(k_idx)
        if len(valid_k) < 5:
            channel_status[channel_key] = "DropChannel"
            continue
        drop_rate = 1 - len(valid_k) / len(real_pairs)
        if drop_rate > MAX_DROP_RATE_PER_CHANNEL:
            channel_status[channel_key] = "DropChannel"
            continue
        if drop_rate > 0:
            channel_status[channel_key] = "LowConf"
        else:
            channel_status[channel_key] = "OK"

        L_real = np.array([real_pairs[k][0] for k in valid_k])
        R_real = np.array([real_pairs[k][1] for k in valid_k])
        z_pair_real = real_pair_z_by_mode[mode][valid_k]
        Ei_cache = expr_norm[i_arr]
        Ej_cache = expr_norm[j_arr]
        W_real = compute_W_batch(L_real, R_real, expr_norm, i_arr, j_arr,
                                 Ei_cache, Ej_cache)
        if config.studentize_channel_stat:
            T_real = float(np.sum(z_pair_real * W_real) / (np.linalg.norm(W_real) + 1e-12))
        else:
            T_real = float(np.sum(z_pair_real * W_real))
        channel_T[channel_key] = T_real
        if return_T_only:
            del Ei_cache, Ej_cache
            continue

        B = config.n_pseudo_db_reps
        rng = np.random.default_rng(
            config.random_seed + hash(channel_key) % 2**31
        )
        per_k_ctrl_arrays = [
            np.array(ctrl_assignments[k]) for k in valid_k
        ]
        T_null1 = np.zeros(B, dtype=np.float32)
        T_null2 = np.zeros(B, dtype=np.float32)
        for b in range(B):
            sampled_lig = np.zeros(len(valid_k), dtype=np.int64)
            sampled_rec = np.zeros(len(valid_k), dtype=np.int64)
            for kk, ctrl_arr in enumerate(per_k_ctrl_arrays):
                pick = rng.integers(0, len(ctrl_arr))
                sampled_lig[kk] = ctrl_arr[pick, 0]
                sampled_rec[kk] = ctrl_arr[pick, 1]
            W_ctrl = compute_W_batch(
                sampled_lig, sampled_rec, expr_norm, i_arr, j_arr,
                Ei_cache, Ej_cache
            )
            w_ctrl_norm = np.linalg.norm(W_ctrl) + 1e-12
            if config.studentize_channel_stat:
                T_null1[b] = np.sum(z_pair_real * W_ctrl) / w_ctrl_norm
            else:
                T_null1[b] = np.sum(z_pair_real * W_ctrl)
            z_pair_ctrl = compute_pair_z_batch(
                sampled_lig, sampled_rec, z_gwas,
                inputs.chromosome, inputs.gene_gene_corr, config,
            )
            if config.studentize_channel_stat:
                T_null2[b] = np.sum(z_pair_ctrl * W_ctrl) / w_ctrl_norm
            else:
                T_null2[b] = np.sum(z_pair_ctrl * W_ctrl)

        p1 = (1 + np.sum(T_null1 >= T_real)) / (1 + B)
        p2 = (1 + np.sum(T_null2 >= T_real)) / (1 + B)
        channel_p1[channel_key] = float(p1)
        channel_p2[channel_key] = float(p2)

    channel_p_acat = {}
    cs_cr_seen = set()
    for (c_s, c_r, mode) in channel_T:
        cs_cr_seen.add((c_s, c_r))
    for (c_s, c_r) in cs_cr_seen:
        p_list = []
        for mode in inputs.modes:
            key = (c_s, c_r, mode)
            if key in channel_p1:
                p_list.append(channel_p1[key])
        if p_list:
            channel_p_acat[(c_s, c_r)] = acat_combine(np.array(p_list))

    return ChannelTestResult(
        channel_T=channel_T,
        channel_p_null1=channel_p1,
        channel_p_null2=channel_p2,
        channel_p_acat=channel_p_acat,
        channel_status=channel_status,
    )
