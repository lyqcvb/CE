"""
step09_batch_pipeline.py
========================

批量处理脚本：遍历 HC/MCI/AD 组、频段和被试，从 EEG .set 文件计算
LOFC/HOFC 矩阵以及网络指标，并将结果写入 output/。

用法
----
    # 处理全部数据
    uv run python src/step09_batch_pipeline.py

    # 仅处理指定组和频段
    uv run python src/step09_batch_pipeline.py --group HC --band "1-4(Delta)"

    # 仅生成 FC 矩阵，不计算网络指标
    uv run python src/step09_batch_pipeline.py --fc-only

输出目录
--------
  output/fc_matrices/{fc_type}/{group}/{band}/{subject}.csv
  output/core_nodes/{fc_type}/{group}/{band}/{subject}_core_nodes.csv
  output/target_attack/{fc_type}/{group}/{band}/{subject}.csv
  output/mean/{group}/{band}/
  output/all_scalar_metrics_summary.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 抑制常量通道（方差为 0）导致的 Pearson 相关 RuntimeWarning。
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(__file__))
import brain_metrics as bm
from pipeline_config import (
    FC_STABILITY_CV_EPS,
    BAND_LABELS,
    FC_STABILITY_EPOCHS,
    FC_STABILITY_MIN_SAMPLES,
    FC_TYPES_LOWER,
    GROUP_DIRS,
    HUB_TOP_RATIO,
    RANDOM_ATTACK_REPEATS,
    RAW_DIR,
    SPARSITY_LEVELS,
    STABLE_CORE_MIN_DENSITIES,
    THRESHOLD_DEFAULT,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_FC = BASE_DIR / "output" / "fc_matrices"
OUT_CORE = BASE_DIR / "output" / "core_nodes"
OUT_CORE_DENSITY = BASE_DIR / "output" / "core_nodes_density"
OUT_ATTACK = BASE_DIR / "output" / "target_attack"
OUT_ATTACK_CONTROL = BASE_DIR / "output" / "target_attack_controls"
OUT_ATTACK_DENSITY = BASE_DIR / "output" / "target_attack_density"
OUT_CE_ABLATION_ATTACK = BASE_DIR / "output" / "ce_ablation_attack"
OUT_FC_STABILITY = BASE_DIR / "output" / "fc_stability"
OUT_LAYER_SIMILARITY = BASE_DIR / "output" / "fc_layer_similarity"
OUT_MEAN = BASE_DIR / "output" / "mean"
OUT_QC = BASE_DIR / "output" / "network_qc"
SUMMARY_CSV = BASE_DIR / "output" / "all_scalar_metrics_summary.csv"

GROUPS = GROUP_DIRS
FC_TYPES = FC_TYPES_LOWER

def save_csv(arr: np.ndarray, path: Path, header: bool = False) -> None:
    """将 ndarray 保存为 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(arr).to_csv(path, index=False, header=header)


def list_subjects(raw_group_band_dir: Path) -> list[str]:
    """返回原始组别/频段目录下的 .set 文件名。"""
    if not raw_group_band_dir.exists():
        return []
    return sorted(f.name for f in raw_group_band_dir.iterdir() if f.suffix == ".set")


def _filter_current_summary_scope(
    df: pd.DataFrame,
    allowed_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop rows outside the configured bands and trim to the current schema."""
    current_bands = set(BAND_LABELS.values())
    filtered = df[df["Band"].isin(current_bands)].copy() if "Band" in df.columns else df.copy()
    if allowed_columns is None:
        return filtered
    keep_columns = [column for column in allowed_columns if column in filtered.columns]
    return filtered.loc[:, keep_columns]


def _fc_distribution_metrics(matrix: np.ndarray, directed: bool) -> dict[str, float]:
    mat = np.asarray(matrix, dtype=np.float64)
    if directed:
        mask = ~np.eye(mat.shape[0], dtype=bool)
    else:
        mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    values = mat[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "FCMean": np.nan,
            "FCStd": np.nan,
            "FCQ05": np.nan,
            "FCQ25": np.nan,
            "FCQ50": np.nan,
            "FCQ75": np.nan,
            "FCQ95": np.nan,
        }
    return {
        "FCMean": float(np.mean(values)),
        "FCStd": float(np.std(values)),
        "FCQ05": float(np.quantile(values, 0.05)),
        "FCQ25": float(np.quantile(values, 0.25)),
        "FCQ50": float(np.quantile(values, 0.50)),
        "FCQ75": float(np.quantile(values, 0.75)),
        "FCQ95": float(np.quantile(values, 0.95)),
    }


def _normalized_auc(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 2:
        return np.nan
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    span = x_arr[-1] - x_arr[0]
    if span <= 0.0:
        return np.nan
    return float(np.trapezoid(y_arr, x_arr) / span)


def _curve_auc(curve: np.ndarray) -> float:
    values = np.asarray(curve, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    x = np.arange(values.size, dtype=np.float64)
    return float(np.trapezoid(values, x) / (x[-1] - x[0]))


def _offdiag_values(matrix: np.ndarray, directed: bool) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if directed:
        mask = ~np.eye(mat.shape[0], dtype=bool)
    else:
        mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    return mat[mask]


def _vector_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    x = a[valid]
    y = b[valid]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def _matrix_corr(matrix_a: np.ndarray, matrix_b: np.ndarray, directed: bool) -> float:
    return _vector_corr(
        _offdiag_values(matrix_a, directed=directed),
        _offdiag_values(matrix_b, directed=directed),
    )


def _write_subject_qc(
    subject_id: str,
    group: str,
    band_label: str,
    band_dir: str,
    fc_name: str,
    matrix: np.ndarray,
    directed: bool,
    core_zero_mean_std: bool,
    attack_initial_eglob: float,
) -> None:
    rows = []

    fixed_graph = bm.binarize(matrix, THRESHOLD_DEFAULT)
    fixed_qc = bm.graph_qc_metrics(fixed_graph, directed=directed)
    rows.append(
        {
            "SubjectID": subject_id,
            "Group": group,
            "Band": band_label,
            "FC_Type": fc_name.upper(),
            "ThresholdMode": "absolute",
            "ThresholdValue": THRESHOLD_DEFAULT,
            **_fc_distribution_metrics(matrix, directed=directed),
            **fixed_qc,
            "CoreZero_MeanStd": int(core_zero_mean_std),
            "AttackInitialEglob": attack_initial_eglob,
        }
    )

    for density in SPARSITY_LEVELS:
        sparse_graph = bm.proportional_threshold(matrix, density=density, directed=directed)
        rows.append(
            {
                "SubjectID": subject_id,
                "Group": group,
                "Band": band_label,
                "FC_Type": fc_name.upper(),
                "ThresholdMode": "density",
                "ThresholdValue": density,
                **_fc_distribution_metrics(matrix, directed=directed),
                **bm.graph_qc_metrics(sparse_graph, directed=directed),
                "CoreZero_MeanStd": int(core_zero_mean_std),
                "AttackInitialEglob": attack_initial_eglob,
            }
        )

    out_path = OUT_QC / fc_name / group / band_dir / f"{Path(subject_id).stem}_qc.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def _compute_layer_similarity(
    subject_id: str,
    group: str,
    band_label: str,
    band_dir: str,
    lofc: np.ndarray,
    hofc: np.ndarray,
) -> dict[str, float]:
    lofc_vec = _offdiag_values(lofc, directed=False)
    hofc_vec = _offdiag_values(hofc, directed=False)

    result = {
        "Corr_LOFC_HOFC": _vector_corr(lofc_vec, hofc_vec),
    }
    result["LayerDecoupling"] = (
        1.0 - result["Corr_LOFC_HOFC"]
        if np.isfinite(result["Corr_LOFC_HOFC"])
        else np.nan
    )
    row = {
        "SubjectID": subject_id,
        "Group": group,
        "Band": band_label,
        **result,
    }
    out_path = OUT_LAYER_SIMILARITY / group / band_dir / f"{Path(subject_id).stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def _compute_fc_stability(
    eeg: np.ndarray,
    subject_id: str,
    group: str,
    band_label: str,
    band_dir: str,
    epoch_count: int,
    min_samples: int,
) -> dict[str, float]:
    if epoch_count <= 1 or eeg.shape[1] < epoch_count * min_samples:
        return {}

    chunks = np.array_split(eeg, epoch_count, axis=1)
    fc_by_type: dict[str, list[np.ndarray]] = {"LOFC": [], "HOFC": []}
    for chunk in chunks:
        if chunk.shape[1] < min_samples:
            continue
        lofc_epoch = bm.compute_lofc(chunk)
        hofc_epoch = bm.compute_hofc(lofc_epoch)
        fc_by_type["LOFC"].append(lofc_epoch)
        fc_by_type["HOFC"].append(hofc_epoch)

    rows: list[dict[str, object]] = []
    scalar: dict[str, float] = {}
    for fc_type, matrices in fc_by_type.items():
        directed = False
        if len(matrices) < 2:
            continue

        pair_corrs = [
            _matrix_corr(matrices[i], matrices[j], directed=directed)
            for i in range(len(matrices) - 1)
            for j in range(i + 1, len(matrices))
        ]
        stack = np.vstack([_offdiag_values(mat, directed=directed) for mat in matrices])
        edge_mean = np.nanmean(stack, axis=0)
        edge_sd = np.nanstd(stack, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            edge_cv = np.where(
                np.abs(edge_mean) >= FC_STABILITY_CV_EPS,
                edge_sd / np.abs(edge_mean),
                np.nan,
            )
        finite_cv = edge_cv[np.isfinite(edge_cv)]
        stable_ratio = float(np.mean(finite_cv <= 0.20)) if finite_cv.size else np.nan

        row = {
            "SubjectID": subject_id,
            "Group": group,
            "Band": band_label,
            "FC_Type": fc_type,
            "EpochCount": len(matrices),
            "MatrixCorr_Mean": float(np.nanmean(pair_corrs)),
            "MatrixCorr_SD": float(np.nanstd(pair_corrs, ddof=1)) if len(pair_corrs) > 1 else 0.0,
            "EdgeCV_Mean": float(np.nanmean(finite_cv)) if finite_cv.size else np.nan,
            "StableEdgeRatio": stable_ratio,
        }
        rows.append(row)
        scalar[f"FCStability_MatrixCorrMean_{fc_type}"] = row["MatrixCorr_Mean"]
        scalar[f"FCStability_EdgeCVMean_{fc_type}"] = row["EdgeCV_Mean"]
        scalar[f"FCStability_StableEdgeRatio_{fc_type}"] = row["StableEdgeRatio"]

    if rows:
        out_path = OUT_FC_STABILITY / group / band_dir / f"{Path(subject_id).stem}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return scalar


def _compute_density_sensitivity(
    subject_id: str,
    group: str,
    band_label: str,
    band_dir: str,
    fc_name: str,
    matrix: np.ndarray,
    directed: bool,
    seed: int,
) -> dict[str, float]:
    node_count = matrix.shape[0]
    ce_by_density: list[np.ndarray] = []
    score_by_density: list[np.ndarray] = []
    hub_by_density: list[np.ndarray] = []
    attack_aucs: list[float] = []

    core_rows: list[dict[str, object]] = []
    attack_rows: list[dict[str, object]] = []

    for density_idx, density in enumerate(SPARSITY_LEVELS):
        graph = bm.proportional_threshold(matrix, density=density, directed=directed)
        components = bm.compute_ce_components(graph, directed=directed, weighted=False)
        is_hub, ce, threshold = bm.identify_hub_nodes(
            graph,
            directed=directed,
            threshold_method="top_ratio",
            top_ratio=HUB_TOP_RATIO,
        )
        score = components["S"]
        entropy = components["H"]
        traditional_hubs = bm.identify_traditional_hubs(
            graph,
            directed=directed,
            top_ratio=HUB_TOP_RATIO,
        )
        attack_curve = bm.compute_target_attack(graph, directed=directed)
        attack_auc = _curve_auc(attack_curve)

        ce_by_density.append(ce)
        score_by_density.append(score)
        hub_by_density.append(is_hub)
        attack_aucs.append(attack_auc)

        for node_idx in range(node_count):
            core_rows.append(
                {
                    "SubjectID": subject_id,
                    "Group": group,
                    "Band": band_label,
                    "FC_Type": fc_name.upper(),
                    "Density": density,
                    "NodeIndex": node_idx,
                    "CE": float(ce[node_idx]),
                    "S_score": float(score[node_idx]),
                    "H_entropy": float(entropy[node_idx]),
                    "IsCoreNode": int(is_hub[node_idx]),
                    "IsDegreeHub": int(traditional_hubs["Degree"][node_idx]),
                    "IsBetweennessHub": int(traditional_hubs["Betweenness"][node_idx]),
                    "IsClosenessHub": int(traditional_hubs["Closeness"][node_idx]),
                    "TopRatioThreshold": threshold,
                }
            )

        attack_rows.append(
            {
                "SubjectID": subject_id,
                "Group": group,
                "Band": band_label,
                "FC_Type": fc_name.upper(),
                "Density": density,
                "AttackAUC": attack_auc,
                "EarlyDrop_1": 1.0 - attack_curve[1] if attack_curve.size > 1 else np.nan,
                "EarlyDrop_2": 1.0 - attack_curve[2] if attack_curve.size > 2 else np.nan,
                "EarlyDrop_3": 1.0 - attack_curve[3] if attack_curve.size > 3 else np.nan,
                "MaxDrop": float(1.0 - np.nanmin(attack_curve)) if attack_curve.size else np.nan,
            }
        )

    ce_matrix = np.vstack(ce_by_density)
    score_matrix = np.vstack(score_by_density)
    hub_matrix = np.vstack(hub_by_density).astype(int)
    core_frequency = hub_matrix.mean(axis=0)
    stable_core = core_frequency >= (STABLE_CORE_MIN_DENSITIES / len(SPARSITY_LEVELS))
    ce_auc_by_node = np.array(
        [_normalized_auc(SPARSITY_LEVELS, ce_matrix[:, idx]) for idx in range(node_count)]
    )
    score_auc_by_node = np.array(
        [_normalized_auc(SPARSITY_LEVELS, score_matrix[:, idx]) for idx in range(node_count)]
    )

    for row in core_rows:
        node_idx = int(row["NodeIndex"])
        row["CoreFrequency"] = float(core_frequency[node_idx])
        row["CoreStability"] = float(core_frequency[node_idx])
        row["StableCoreNode"] = int(stable_core[node_idx])
        row["CE_auc"] = float(ce_auc_by_node[node_idx])
        row["S_auc"] = float(score_auc_by_node[node_idx])

    out_core = OUT_CORE_DENSITY / fc_name / group / band_dir / f"{Path(subject_id).stem}.csv"
    out_core.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(core_rows).to_csv(out_core, index=False, encoding="utf-8-sig")

    out_attack = OUT_ATTACK_DENSITY / fc_name / group / band_dir / f"{Path(subject_id).stem}.csv"
    out_attack.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(attack_rows).to_csv(out_attack, index=False, encoding="utf-8-sig")

    return {
        f"CE_auc_mean_{fc_name.upper()}": float(np.nanmean(ce_auc_by_node)),
        f"CE_auc_max_{fc_name.upper()}": float(np.nanmax(ce_auc_by_node)),
        f"CoreNode_stability_mean_{fc_name.upper()}": float(np.nanmean(core_frequency)),
        f"StableCoreNode_count_{fc_name.upper()}": float(np.sum(stable_core)),
        f"TargetAttack_auc_across_density_{fc_name.upper()}": _normalized_auc(
            SPARSITY_LEVELS,
            attack_aucs,
        ),
    }


def process_subject(
    set_path: Path,
    group: str,
    band_dir: str,
    band_label: str,
    fc_only: bool = False,
    seed: int = 42,
    random_attack_repeats: int = RANDOM_ATTACK_REPEATS,
    fc_stability_epochs: int = FC_STABILITY_EPOCHS,
) -> dict | None:
    """
    处理单个被试，计算 FC 矩阵，并可选计算网络指标和曲线指标。

    Parameters
    ----------
    set_path:
        .set 文件完整路径。
    group:
        "HC" / "MCI" / "AD"。
    band_dir:
        频段目录名，例如 "12-30(Beta)"。
    band_label:
        CSV 中使用的频段标签，例如 "Beta"。
    fc_only:
        为 True 时仅保存 FC 矩阵，跳过图指标和靶向攻击。

    Returns
    -------
    dict | None
        标量指标行；出错时返回 None。
    """
    subj_name = set_path.stem

    try:
        eeg = bm.read_eeg_set(str(set_path))
        lofc = bm.compute_lofc(eeg)
        signed_lofc = bm.compute_signed_lofc(eeg)
        hofc = bm.compute_hofc(lofc)

        for fc_name, mat in [("lofc", lofc), ("hofc", hofc)]:
            out_path = OUT_FC / fc_name / group / band_dir / f"{subj_name}.csv"
            save_csv(mat, out_path)
        save_csv(signed_lofc, OUT_FC / "lofc_signed" / group / band_dir / f"{subj_name}.csv")

        if fc_only:
            return {"Group": group, "Band": band_label, "SubjectID": set_path.name}

        scalar = bm.compute_all_scalar_metrics(lofc, hofc)
        scalar_row = {
            "Group": group,
            "Band": band_label,
            "SubjectID": set_path.name,
            **scalar,
        }
        scalar_row.update(
            _compute_layer_similarity(
                subject_id=set_path.name,
                group=group,
                band_label=band_label,
                band_dir=band_dir,
                lofc=lofc,
                hofc=hofc,
            )
        )
        scalar_row.update(
            _compute_fc_stability(
                eeg=eeg,
                subject_id=set_path.name,
                group=group,
                band_label=band_label,
                band_dir=band_dir,
                epoch_count=fc_stability_epochs,
                min_samples=FC_STABILITY_MIN_SAMPLES,
            )
        )

        for fc_name, mat, is_dir in [
            ("lofc", lofc, False),
            ("hofc", hofc, False),
        ]:
            graph = bm.binarize(mat, THRESHOLD_DEFAULT)
            is_hub_top, ce, top_threshold = bm.identify_hub_nodes(
                graph,
                directed=is_dir,
                threshold_method="top_ratio",
                top_ratio=HUB_TOP_RATIO,
            )
            is_hub_mean_std, _, mean_std_threshold = bm.identify_hub_nodes(
                graph,
                directed=is_dir,
                threshold_method="mean_std",
                top_ratio=HUB_TOP_RATIO,
            )
            _, centrality_score = bm.compute_ce(graph, directed=is_dir)
            is_weighted_hub, weighted_ce, weighted_threshold = bm.identify_weighted_hub_nodes(
                mat,
                directed=is_dir,
                top_ratio=HUB_TOP_RATIO,
            )
            _, weighted_score = bm.compute_weighted_ce(mat, directed=is_dir)
            union = np.logical_or(is_hub_top, is_weighted_hub).sum()
            overlap = (
                float(np.logical_and(is_hub_top, is_weighted_hub).sum() / union)
                if union
                else np.nan
            )
            traditional_hubs = bm.identify_traditional_hubs(
                graph,
                directed=is_dir,
                top_ratio=HUB_TOP_RATIO,
            )
            core_rows = [
                {
                    "NodeIndex": i,
                    "IsCoreNode": int(is_hub_top[i]),
                    "IsCoreNode_BinaryCE": int(is_hub_top[i]),
                    "IsCoreNode_WeightedCE": int(is_weighted_hub[i]),
                    "IsCoreNodeTopRatio": int(is_hub_top[i]),
                    "IsCoreNodeMeanStd": int(is_hub_mean_std[i]),
                    "CE": float(ce[i]),
                    "CentralityScore": float(centrality_score[i]),
                    "WeightedCE": float(weighted_ce[i]),
                    "WeightedCentralityScore": float(weighted_score[i]),
                    "WeightedCEThreshold": weighted_threshold,
                    "BinaryWeightedCoreOverlap": overlap,
                    "TopRatioThreshold": top_threshold,
                    "MeanStdThreshold": mean_std_threshold,
                    "IsDegreeHubTopRatio": int(traditional_hubs["Degree"][i]),
                    "IsBetweennessHubTopRatio": int(traditional_hubs["Betweenness"][i]),
                    "IsClosenessHubTopRatio": int(traditional_hubs["Closeness"][i]),
                }
                for i in range(len(is_hub_top))
            ]
            out_path = OUT_CORE / fc_name / group / band_dir / f"{subj_name}_core_nodes.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(core_rows).to_csv(out_path, index=False)
            scalar_row[f"CoreOverlap_BinaryCE_WeightedCE_{fc_name.upper()}"] = overlap

        graph_lofc = bm.binarize(lofc, THRESHOLD_DEFAULT)
        graph_hofc = bm.binarize(hofc, THRESHOLD_DEFAULT)

        attack_curves_by_fc = {
            "lofc": bm.compute_attack_curves(
                graph_lofc,
                directed=False,
                random_repeats=random_attack_repeats,
                seed=seed,
            ),
            "hofc": bm.compute_attack_curves(
                graph_hofc,
                directed=False,
                random_repeats=random_attack_repeats,
                seed=seed,
            ),
        }

        for fc_name, curves in attack_curves_by_fc.items():
            attack_curve = curves["ce"]
            out_path = OUT_ATTACK / fc_name / group / band_dir / f"{subj_name}.csv"
            save_csv(attack_curve.reshape(-1, 1), out_path)
            for strategy, curve in curves.items():
                out_path = (
                    OUT_ATTACK_CONTROL
                    / strategy
                    / fc_name
                    / group
                    / band_dir
                    / f"{subj_name}.csv"
                )
                save_csv(curve.reshape(-1, 1), out_path)

        for fc_name, graph, is_dir in [
            ("lofc", graph_lofc, False),
            ("hofc", graph_hofc, False),
        ]:
            ablation_curves = bm.compute_centrality_ablation_attacks(
                graph,
                directed=is_dir,
            )
            for method, curve in ablation_curves.items():
                out_path = (
                    OUT_CE_ABLATION_ATTACK
                    / method
                    / fc_name
                    / group
                    / band_dir
                    / f"{subj_name}.csv"
                )
                save_csv(curve.reshape(-1, 1), out_path)
                scalar_row[f"CEAblationAttackAUC_{method}_{fc_name.upper()}"] = _curve_auc(curve)

        for fc_name, mat, graph, is_dir in [
            ("lofc", lofc, graph_lofc, False),
            ("hofc", hofc, graph_hofc, False),
        ]:
            is_hub_mean_std, _, _ = bm.identify_hub_nodes(
                graph,
                directed=is_dir,
                threshold_method="mean_std",
                top_ratio=HUB_TOP_RATIO,
            )
            _write_subject_qc(
                subject_id=set_path.name,
                group=group,
                band_label=band_label,
                band_dir=band_dir,
                fc_name=fc_name,
                matrix=mat,
                directed=is_dir,
                core_zero_mean_std=not bool(is_hub_mean_std.any()),
                attack_initial_eglob=bm.global_efficiency_bin(graph),
            )

        for fc_name, mat, is_dir in [
            ("lofc", lofc, False),
            ("hofc", hofc, False),
        ]:
            scalar_row.update(
                _compute_density_sensitivity(
                    subject_id=set_path.name,
                    group=group,
                    band_label=band_label,
                    band_dir=band_dir,
                    fc_name=fc_name,
                    matrix=mat,
                    directed=is_dir,
                    seed=seed,
                )
            )

        return scalar_row

    except Exception as exc:
        print(f"  [!] 错误 {subj_name}: {exc}")
        traceback.print_exc()
        return None


def compute_group_mean(group: str, band_dir: str) -> None:
    """计算并保存指定组别/频段的组均值 FC 矩阵。"""
    for fc_name in FC_TYPES:
        fc_dir = OUT_FC / fc_name / group / band_dir
        if not fc_dir.exists():
            continue

        mats = []
        for csv_file in sorted(fc_dir.glob("*.csv")):
            try:
                matrix = pd.read_csv(csv_file, header=None).values
                if matrix.shape == (16, 16):
                    mats.append(matrix)
            except Exception:
                pass

        if not mats:
            continue

        mean_mat = np.mean(mats, axis=0)
        out_path = OUT_MEAN / group / band_dir / f"{group}_{band_dir}_{fc_name}.csv"
        save_csv(mean_mat, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量计算 EEG 脑网络 FC 矩阵和指标。")
    parser.add_argument(
        "--group",
        choices=list(GROUPS.keys()),
        default=None,
        help="仅处理指定组；默认处理全部组。",
    )
    parser.add_argument(
        "--band",
        default=None,
        help="仅处理指定频段目录名，例如 '12-30(Beta)'。",
    )
    parser.add_argument(
        "--fc-only",
        action="store_true",
        help="仅保存 FC 矩阵，跳过图指标和靶向攻击。",
    )
    parser.add_argument(
        "--random-attack-repeats",
        type=int,
        default=RANDOM_ATTACK_REPEATS,
        help="random attack 重复次数，默认 1000。",
    )
    parser.add_argument(
        "--fc-stability-epochs",
        type=int,
        default=FC_STABILITY_EPOCHS,
        help="FC 稳定性分析的连续 epoch 数，默认 5；设为 0 可跳过。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已经存在 LOFC 输出的被试。",
    )
    args = parser.parse_args()

    if args.group and args.group not in GROUPS:
        parser.error(f"--group must be one of: {', '.join(GROUPS)}")
    if args.band and args.band not in BAND_LABELS:
        parser.error(f"--band must be one of: {', '.join(BAND_LABELS)}")

    groups_to_run = {args.group: GROUPS[args.group]} if args.group else GROUPS
    bands_to_run = {args.band: BAND_LABELS[args.band]} if args.band else BAND_LABELS

    all_scalar_rows: list[dict] = []
    total_subjects = 0
    total_done = 0
    total_errors = 0

    for group, group_dir in groups_to_run.items():
        for band_dir, band_label in bands_to_run.items():
            raw_dir = RAW_DIR / group_dir / band_dir
            subjects = list_subjects(raw_dir)
            if not subjects:
                print(f"[跳过] 目录不存在或没有 .set 文件: {raw_dir}")
                continue

            print(f"\n{'=' * 60}")
            print(f"处理 {group} / {band_label} ({len(subjects)} 个被试)")
            print(f"{'=' * 60}")
            total_subjects += len(subjects)

            for i, subj_file in enumerate(subjects, 1):
                set_path = raw_dir / subj_file
                subj_name = set_path.stem

                if args.skip_existing:
                    existing = OUT_FC / "lofc" / group / band_dir / f"{subj_name}.csv"
                    if existing.exists():
                        print(f"  [{i:3d}/{len(subjects)}] 跳过（已存在） {subj_name}")
                        continue

                t0 = time.time()
                print(f"  [{i:3d}/{len(subjects)}] {subj_name} ... ", end="", flush=True)

                row = process_subject(
                    set_path=set_path,
                    group=group,
                    band_dir=band_dir,
                    band_label=band_label,
                    fc_only=args.fc_only,
                    seed=args.seed,
                    random_attack_repeats=args.random_attack_repeats,
                    fc_stability_epochs=args.fc_stability_epochs,
                )

                elapsed = time.time() - t0
                if row is not None:
                    total_done += 1
                    all_scalar_rows.append(row)
                    print(f"完成 ({elapsed:.1f}s)")
                else:
                    total_errors += 1
                    print("失败")

            print(f"  计算 {group}/{band_dir} 组均值矩阵...", end=" ", flush=True)
            compute_group_mean(group, band_dir)
            print("完成")

    if all_scalar_rows and not args.fc_only:
        df_new = _filter_current_summary_scope(pd.DataFrame(all_scalar_rows))
        current_columns = df_new.columns.tolist()
        if SUMMARY_CSV.exists():
            df_old = _filter_current_summary_scope(pd.read_csv(SUMMARY_CSV), current_columns)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(
                subset=["Group", "Band", "SubjectID"],
                keep="last",
            )
        else:
            df_combined = df_new

        df_combined = _filter_current_summary_scope(df_combined, current_columns)
        df_combined.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
        print(f"\n已更新汇总指标: {SUMMARY_CSV} ({len(df_combined)} 行)")

    print(f"\n{'=' * 60}")
    print("批量处理完成")
    print(f"  总被试: {total_subjects}")
    print(f"  成功:   {total_done}")
    print(f"  失败:   {total_errors}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
