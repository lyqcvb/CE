"""
step11_statistical_analysis.py
======================
Statistical analysis entrypoint for scalar metrics, rich-club individual
means, and target attack stability curves.

The analysis uses a three-group omnibus test before post-hoc pairwise tests:

1. one-way ANOVA when all groups pass Shapiro-Wilk normality and Levene
   variance homogeneity checks;
2. Kruskal-Wallis otherwise;
3. post-hoc pairwise tests only when the omnibus test is significant.

Post-hoc normal comparisons use Welch's t-test. Non-normal comparisons use
Mann-Whitney U. Raw p-values are retained and Benjamini-Hochberg FDR-adjusted
p-values are added for omnibus and post-hoc rows separately.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from pipeline_config import (
    BAND_LABELS as CONFIG_BAND_MAPPING,
    BANDS as CONFIG_BANDS,
    FC_TYPES_LOWER as CONFIG_FC_TYPES_LOWER,
    FC_TYPES_UPPER as CONFIG_FC_TYPES_UPPER,
    GROUP_NAMES as CONFIG_GROUPS,
    OUTPUT_DIR as CONFIG_OUTPUT_DIR,
    RESULTS_DIR as CONFIG_RESULTS_DIR,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = CONFIG_OUTPUT_DIR
RESULTS_DIR = CONFIG_RESULTS_DIR

FC_TYPES_LOWER = CONFIG_FC_TYPES_LOWER
FC_TYPES_UPPER = CONFIG_FC_TYPES_UPPER

BAND_MAPPING = CONFIG_BAND_MAPPING
BANDS = CONFIG_BANDS
GROUPS = CONFIG_GROUPS
COMPARISONS = [("HC", "MCI"), ("HC", "AD"), ("MCI", "AD")]

ALPHA = 0.05
P_ADJUST_METHOD = "fdr_bh"
BOOTSTRAP_ITER = 1000
BOOTSTRAP_SEED = 42


def _ensure_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(exist_ok=True)


def _sig_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _as_numeric_series(values: object) -> pd.Series:
    series = pd.Series(values)
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return series.astype(float)


def _normality_test(values: object) -> tuple[float, bool]:
    values = _as_numeric_series(values)
    if len(values) < 3 or values.nunique(dropna=True) < 2:
        return float("nan"), False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_value = stats.shapiro(values.to_numpy(dtype=float))

    p_value = float(p_value)
    return p_value, bool(np.isfinite(p_value) and p_value > ALPHA)


def _fdr_bh(p_values: object) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not valid.any():
        return adjusted

    valid_p = p_values[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    m = len(ranked)
    adjusted_ranked = ranked * m / np.arange(1, m + 1, dtype=float)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    valid_indices = np.where(valid)[0]
    adjusted[valid_indices[order]] = adjusted_ranked
    return adjusted


def _apply_p_adjustment(
    df: pd.DataFrame,
    *,
    level: str,
    scope: str,
    p_col: str = "p_raw",
) -> None:
    mask = df["Test_Level"].eq(level) & pd.to_numeric(df[p_col], errors="coerce").notna()
    if not mask.any():
        return

    if "Correction_Family" in df.columns:
        for family, family_index in df.loc[mask].groupby("Correction_Family").groups.items():
            df.loc[family_index, "p_adj"] = _fdr_bh(
                df.loc[family_index, p_col].to_numpy(dtype=float)
            )
            df.loc[family_index, "Correction_Method"] = P_ADJUST_METHOD
            df.loc[family_index, "Correction_Scope"] = f"{scope}: {family}"
        return

    df.loc[mask, "p_adj"] = _fdr_bh(df.loc[mask, p_col].to_numpy(dtype=float))
    df.loc[mask, "Correction_Method"] = P_ADJUST_METHOD
    df.loc[mask, "Correction_Scope"] = scope


def _finalize_results(
    records: list[dict[str, object]],
    output_file: Path,
    *,
    omnibus_scope: str,
    posthoc_scope: str,
) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        return df

    df["p_raw"] = pd.to_numeric(df["p_raw"], errors="coerce")
    df["p_value"] = df["p_raw"]
    df["p_adj"] = np.nan
    df["Correction_Method"] = ""
    df["Correction_Scope"] = ""

    _apply_p_adjustment(df, level="Omnibus", scope=omnibus_scope)
    _apply_p_adjustment(df, level="Posthoc", scope=posthoc_scope)

    df["Significance_raw"] = df["p_raw"].map(_sig_label)
    df["Significance_adj"] = df["p_adj"].map(_sig_label)
    df["Significant_adj"] = df["p_adj"].lt(ALPHA)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return df


def _group_summary(group_data: dict[str, pd.Series]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for group in GROUPS:
        values = _as_numeric_series(group_data.get(group, pd.Series(dtype=float)))
        summary[f"{group}_n"] = int(len(values))
        summary[f"{group}_Mean"] = float(values.mean()) if len(values) else np.nan
        summary[f"{group}_SD"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        summary[f"{group}_Median"] = float(values.median()) if len(values) else np.nan
        summary[f"{group}_IQR"] = (
            float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan
        )
    return summary


def _levene_test(group_data: dict[str, pd.Series], groups: list[str]) -> tuple[float, bool]:
    arrays = [group_data[group].to_numpy(dtype=float) for group in groups]
    if any(len(values) < 2 for values in arrays):
        return float("nan"), False
    if len(np.unique(np.concatenate(arrays))) < 2:
        return float("nan"), False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_value = stats.levene(*arrays, center="median")

    p_value = float(p_value)
    return p_value, bool(np.isfinite(p_value) and p_value > ALPHA)


def _anova_eta_squared(group_data: dict[str, pd.Series], groups: list[str]) -> float:
    arrays = [group_data[group].to_numpy(dtype=float) for group in groups]
    all_values = np.concatenate(arrays)
    grand_mean = float(np.mean(all_values))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    if ss_total <= 0.0:
        return float("nan")

    ss_between = sum(len(values) * (float(np.mean(values)) - grand_mean) ** 2 for values in arrays)
    return float(ss_between / ss_total)


def _kruskal_epsilon_squared(statistic: float, group_data: dict[str, pd.Series], groups: list[str]) -> float:
    n_total = sum(len(group_data[group]) for group in groups)
    k = len(groups)
    if n_total <= k:
        return float("nan")
    epsilon = (statistic - k + 1.0) / (n_total - k)
    return float(np.clip(epsilon, 0.0, 1.0))


def _omnibus_test(group_data: dict[str, pd.Series], groups: list[str]) -> dict[str, object]:
    normality = {group: _normality_test(group_data[group]) for group in groups}
    all_normal = all(is_normal for _, is_normal in normality.values())
    levene_p, equal_variance = _levene_test(group_data, groups)
    arrays = [group_data[group].to_numpy(dtype=float) for group in groups]

    if all_normal and equal_variance:
        test_type = "One-way ANOVA"
        effect_type = "Eta squared"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, p_value = stats.f_oneway(*arrays)
        except ValueError:
            statistic, p_value = np.nan, np.nan
        statistic = float(statistic)
        p_value = float(p_value)
        effect_size = _anova_eta_squared(group_data, groups)
    else:
        test_type = "Kruskal-Wallis"
        effect_type = "Epsilon squared"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, p_value = stats.kruskal(*arrays)
        except ValueError:
            statistic, p_value = np.nan, np.nan
        statistic = float(statistic)
        p_value = float(p_value)
        if not np.isfinite(p_value):
            statistic = float("nan")
            p_value = float("nan")
        effect_size = _kruskal_epsilon_squared(statistic, group_data, groups)

    return {
        "Test_Level": "Omnibus",
        "Test_Type": test_type,
        "Test_Statistic": statistic,
        "p_raw": p_value,
        "Effect_Size": effect_size,
        "Effect_Size_Type": effect_type,
        "CI_95_Lower": np.nan,
        "CI_95_Upper": np.nan,
        "Levene_p": levene_p,
        "Equal_Variance": equal_variance,
        "All_Groups_Normal": all_normal,
        **{f"{group}_Normal_p": normality[group][0] for group in groups},
        **{f"{group}_Normal": normality[group][1] for group in groups},
    }


def _hedges_g(data1: object, data2: object) -> float:
    x = _as_numeric_series(data1).to_numpy(dtype=float)
    y = _as_numeric_series(data2).to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    n1, n2 = len(x), len(y)
    df = n1 + n2 - 2
    if df <= 0:
        return float("nan")

    var1 = np.var(x, ddof=1)
    var2 = np.var(y, ddof=1)
    pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
    if pooled <= 0.0:
        return float("nan")

    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(correction * (np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def _cliffs_delta(data1: object, data2: object) -> float:
    x = _as_numeric_series(data1).to_numpy(dtype=float)
    y = _as_numeric_series(data2).to_numpy(dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    comparisons = x[:, None] - y[None, :]
    greater = np.sum(comparisons > 0)
    less = np.sum(comparisons < 0)
    return float((greater - less) / (len(x) * len(y)))


def _bootstrap_ci(
    data1: object,
    data2: object,
    effect_fn: Callable[[object, object], float],
    *,
    n_boot: int = BOOTSTRAP_ITER,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    x = _as_numeric_series(data1).to_numpy(dtype=float)
    y = _as_numeric_series(data2).to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        x_sample = x[rng.integers(0, len(x), len(x))]
        y_sample = y[rng.integers(0, len(y), len(y))]
        estimate = effect_fn(x_sample, y_sample)
        if np.isfinite(estimate):
            estimates.append(float(estimate))

    if len(estimates) < 10:
        return float("nan"), float("nan")
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def _pairwise_test(
    data1: pd.Series,
    data2: pd.Series,
    *,
    group1_normal: bool,
    group2_normal: bool,
) -> dict[str, object]:
    if group1_normal and group2_normal:
        test_type = "Welch's t-test"
        statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False, nan_policy="omit")
        effect_type = "Hedges' g"
        effect_size = _hedges_g(data1, data2)
        ci_lower, ci_upper = _bootstrap_ci(data1, data2, _hedges_g)
    else:
        test_type = "Mann-Whitney U test"
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        effect_type = "Cliff's delta"
        effect_size = _cliffs_delta(data1, data2)
        ci_lower, ci_upper = _bootstrap_ci(data1, data2, _cliffs_delta)

    return {
        "Test_Level": "Posthoc",
        "Test_Type": test_type,
        "Test_Statistic": float(statistic),
        "p_raw": float(p_value),
        "Effect_Size": effect_size,
        "Effect_Size_Type": effect_type,
        "CI_95_Lower": ci_lower,
        "CI_95_Upper": ci_upper,
        "Levene_p": np.nan,
        "Equal_Variance": np.nan,
        "All_Groups_Normal": np.nan,
    }


def _analyze_endpoint(
    *,
    context: dict[str, object],
    group_data: dict[str, object],
    groups: list[str] = GROUPS,
    comparisons: list[tuple[str, str]] = COMPARISONS,
) -> list[dict[str, object]]:
    cleaned = {
        group: _as_numeric_series(group_data.get(group, pd.Series(dtype=float)))
        for group in groups
    }
    available_groups = [group for group in groups if len(cleaned[group]) >= 2]
    if len(available_groups) < 3:
        return []

    records: list[dict[str, object]] = []
    summary = _group_summary(cleaned)
    omnibus = _omnibus_test(cleaned, available_groups)
    records.append({**context, **summary, **omnibus, "Group1": "", "Group2": ""})

    if not np.isfinite(float(omnibus["p_raw"])) or float(omnibus["p_raw"]) >= ALPHA:
        return records

    for group1, group2 in comparisons:
        if group1 not in available_groups or group2 not in available_groups:
            continue

        data1 = cleaned[group1]
        data2 = cleaned[group2]
        if len(data1) < 2 or len(data2) < 2:
            continue

        pair = _pairwise_test(
            data1,
            data2,
            group1_normal=bool(omnibus.get(f"{group1}_Normal", False)),
            group2_normal=bool(omnibus.get(f"{group2}_Normal", False)),
        )
        pair.update(
            {
                "Group1": group1,
                "Group2": group2,
                "Group1_n": int(len(data1)),
                "Group2_n": int(len(data2)),
                "Group1_Mean": float(data1.mean()),
                "Group1_SD": float(data1.std(ddof=1)) if len(data1) > 1 else np.nan,
                "Group2_Mean": float(data2.mean()),
                "Group2_SD": float(data2.std(ddof=1)) if len(data2) > 1 else np.nan,
                "Group1_Normal_p": omnibus.get(f"{group1}_Normal_p", np.nan),
                "Group2_Normal_p": omnibus.get(f"{group2}_Normal_p", np.nan),
                "Omnibus_Test_Type": omnibus["Test_Type"],
                "Omnibus_p_raw": omnibus["p_raw"],
            }
        )
        records.append({**context, **summary, **pair})

    return records


def run_scalar_metric_tests(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Run scalar metric omnibus and post-hoc tests."""
    _ensure_results_dir(results_dir)

    input_file = output_dir / "all_scalar_metrics_summary.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_file}")

    df = pd.read_csv(input_file)
    metric_columns = [
        col for col in df.columns if col not in ["Group", "Band", "SubjectID"]
    ]
    legacy_to_current_prefix = {
        "Eglob_": "Eglob_bin_",
        "Cluster_": "Clustering_bin_",
    }
    metric_columns = [
        col
        for col in metric_columns
        if not any(
            col.startswith(old_prefix)
            and f"{new_prefix}{col.removeprefix(old_prefix)}" in df.columns
            for old_prefix, new_prefix in legacy_to_current_prefix.items()
        )
    ]

    print("=" * 100)
    print("标量指标统计检验: omnibus + post-hoc + FDR")
    print("=" * 100)

    def correction_family(metric_name: str) -> str:
        parts = metric_name.split("_")
        fc_type = parts[-1] if parts else "UNKNOWN"
        if metric_name.startswith(("Corr_", "LayerDecoupling", "HighOrderDecoupling")):
            return "layer_similarity"
        if metric_name.startswith("CEAblationAttackAUC"):
            return "ce_ablation_attack"
        if metric_name.startswith(("CE_auc", "S_auc", "CoreNode_stability", "StableCoreNode")):
            return f"core_stability | {fc_type}"
        if metric_name.startswith("FCStability"):
            return f"fc_stability | {fc_type}"
        if metric_name.startswith(("RichClub_auc", "TargetAttack_auc_across_density")):
            return f"density_sensitivity | {fc_type}"
        if metric_name.startswith(("RichClubMean_", "RichClubMax", "RichClubPeakK", "RichClubEnhancedKCount")):
            return f"richclub_features | {fc_type}"
        if metric_name.startswith(("Phi_w_", "WeightedRichClub")):
            return f"weighted_richclub_features | {fc_type}"
        if metric_name.startswith("CoreOverlap_BinaryCE_WeightedCE"):
            return f"weighted_ce_overlap | {fc_type}"
        if metric_name.startswith(("Eglob_bin_density", "Clustering_bin_density")):
            return f"Scalar density curve | {fc_type}"
        if metric_name.startswith(("Eglob_bin_auc", "Clustering_bin_auc")):
            return f"Scalar density AUC | {fc_type}"
        if metric_name.startswith(("Eglob_weighted", "Clustering_weighted")):
            return f"Scalar weighted topology | {fc_type}"
        if metric_name.startswith(("Eglob_bin", "Clustering_bin", "MeanConn")):
            return f"Scalar primary | {fc_type}"
        if "thr035" in metric_name or metric_name.startswith("Density_actual"):
            return f"Scalar QC | {fc_type}"
        return f"Scalar exploratory | {fc_type}"

    records: list[dict[str, object]] = []
    bands = df["Band"].dropna().unique().tolist()
    for band in bands:
        band_data = df[df["Band"] == band]
        for metric in metric_columns:
            group_data = {
                group: pd.to_numeric(
                    band_data.loc[band_data["Group"] == group, metric],
                    errors="coerce",
                )
                for group in GROUPS
            }
            records.extend(
                _analyze_endpoint(
                    context={
                        "Analysis": "Scalar",
                        "Band": band,
                        "Metric": metric,
                        "Correction_Family": correction_family(metric),
                    },
                    group_data=group_data,
                )
            )

    output_file = results_dir / "显著性分析.csv"
    results_df = _finalize_results(
        records,
        output_file,
        omnibus_scope="all scalar omnibus tests",
        posthoc_scope="all scalar post-hoc tests after significant omnibus",
    )

    _print_result_summary(results_df, output_file)
    return results_df


def _ols_rss(design: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    rank = int(np.linalg.matrix_rank(design))
    df_resid = int(len(y) - rank)
    rss = float(np.sum(residual**2))
    return rss, rank, df_resid


def _two_way_design(
    groups: pd.Series,
    bands: pd.Series,
    *,
    include_group: bool,
    include_band: bool,
    include_interaction: bool,
) -> np.ndarray:
    columns = [np.ones(len(groups), dtype=np.float64)]
    group_levels = sorted(pd.Series(groups).dropna().unique().tolist())
    band_levels = sorted(pd.Series(bands).dropna().unique().tolist())
    group_dummies: list[np.ndarray] = []
    band_dummies: list[np.ndarray] = []

    if include_group:
        for level in group_levels[1:]:
            col = (groups.to_numpy() == level).astype(np.float64)
            group_dummies.append(col)
            columns.append(col)

    if include_band:
        for level in band_levels[1:]:
            col = (bands.to_numpy() == level).astype(np.float64)
            band_dummies.append(col)
            columns.append(col)

    if include_interaction:
        if not group_dummies:
            for level in group_levels[1:]:
                group_dummies.append((groups.to_numpy() == level).astype(np.float64))
        if not band_dummies:
            for level in band_levels[1:]:
                band_dummies.append((bands.to_numpy() == level).astype(np.float64))
        for group_col in group_dummies:
            for band_col in band_dummies:
                columns.append(group_col * band_col)

    return np.column_stack(columns)


def _nested_f_test(
    reduced_design: np.ndarray,
    full_design: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    rss_reduced, rank_reduced, _ = _ols_rss(reduced_design, y)
    rss_full, rank_full, df_full = _ols_rss(full_design, y)
    df_num = rank_full - rank_reduced
    if df_num <= 0 or df_full <= 0 or rss_full < 0:
        return {"F": np.nan, "p_raw": np.nan, "df_num": df_num, "df_den": df_full, "EtaSq_partial": np.nan}
    ms_num = (rss_reduced - rss_full) / df_num
    ms_den = rss_full / df_full
    if ms_den <= 0.0:
        f_value = np.inf if ms_num > 0.0 else np.nan
    else:
        f_value = ms_num / ms_den
    p_value = float(stats.f.sf(f_value, df_num, df_full)) if np.isfinite(f_value) else 0.0
    partial_eta = (rss_reduced - rss_full) / (rss_reduced - rss_full + rss_full)
    return {
        "F": float(f_value),
        "p_raw": p_value,
        "df_num": float(df_num),
        "df_den": float(df_full),
        "EtaSq_partial": float(np.clip(partial_eta, 0.0, 1.0)),
    }


def run_two_way_anova_metrics(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Run Metric ~ Group * Band fixed-effect ANOVA for scalar summary metrics."""
    _ensure_results_dir(results_dir)
    input_file = output_dir / "all_scalar_metrics_summary.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_file}")

    df = pd.read_csv(input_file)
    metric_columns = [
        col for col in df.columns if col not in ["Group", "Band", "SubjectID"]
    ]
    records: list[dict[str, object]] = []
    for metric in metric_columns:
        data = df[["Group", "Band", metric]].copy()
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data = data.dropna()
        if data["Group"].nunique() < 2 or data["Band"].nunique() < 2 or len(data) < 8:
            continue

        y = data[metric].to_numpy(dtype=np.float64)
        groups = data["Group"].astype(str)
        bands = data["Band"].astype(str)
        full = _two_way_design(
            groups,
            bands,
            include_group=True,
            include_band=True,
            include_interaction=True,
        )
        additive = _two_way_design(
            groups,
            bands,
            include_group=True,
            include_band=True,
            include_interaction=False,
        )
        group_only = _two_way_design(
            groups,
            bands,
            include_group=True,
            include_band=False,
            include_interaction=False,
        )
        band_only = _two_way_design(
            groups,
            bands,
            include_group=False,
            include_band=True,
            include_interaction=False,
        )
        intercept_only = np.ones((len(y), 1), dtype=np.float64)

        tests = {
            "Group": _nested_f_test(band_only, additive, y),
            "Band": _nested_f_test(group_only, additive, y),
            "Group:Band": _nested_f_test(additive, full, y),
            "Model_Additive": _nested_f_test(intercept_only, additive, y),
            "Model_Full": _nested_f_test(intercept_only, full, y),
        }
        for effect, stats_row in tests.items():
            records.append(
                {
                    "Metric": metric,
                    "Effect": effect,
                    "N": len(data),
                    "GroupLevels": data["Group"].nunique(),
                    "BandLevels": data["Band"].nunique(),
                    **stats_row,
                }
            )

    results_df = pd.DataFrame(records)
    if not results_df.empty:
        results_df["p_adj"] = np.nan
        for effect, index in results_df.groupby("Effect").groups.items():
            results_df.loc[index, "p_adj"] = _fdr_bh(
                results_df.loc[index, "p_raw"].to_numpy(dtype=float)
            )
        results_df["Significance_raw"] = results_df["p_raw"].map(_sig_label)
        results_df["Significance_adj"] = results_df["p_adj"].map(_sig_label)
        results_df["Correction_Method"] = P_ADJUST_METHOD
        results_df["Correction_Scope"] = "two-way ANOVA metrics, separately by Effect"

    output_file = results_dir / "two_way_anova_metrics.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    _print_result_summary(results_df, output_file)
    return results_df


def _subject_design(subjects: pd.Series) -> np.ndarray:
    levels = sorted(pd.Series(subjects).dropna().astype(str).unique().tolist())
    columns = [np.ones(len(subjects), dtype=np.float64)]
    values = subjects.astype(str).to_numpy()
    for level in levels[1:]:
        columns.append((values == level).astype(np.float64))
    return np.column_stack(columns)


def _repeated_band_design(
    subjects: pd.Series,
    groups: pd.Series,
    bands: pd.Series,
    *,
    include_band: bool,
    include_interaction: bool,
) -> np.ndarray:
    columns = [_subject_design(subjects)]
    band_levels = sorted(pd.Series(bands).dropna().astype(str).unique().tolist())
    group_levels = sorted(pd.Series(groups).dropna().astype(str).unique().tolist())
    band_values = bands.astype(str).to_numpy()
    group_values = groups.astype(str).to_numpy()

    band_dummies: list[np.ndarray] = []
    for level in band_levels[1:]:
        col = (band_values == level).astype(np.float64)
        band_dummies.append(col)
        if include_band:
            columns.append(col[:, None])

    if include_interaction:
        for group_level in group_levels[1:]:
            group_col = (group_values == group_level).astype(np.float64)
            for band_col in band_dummies:
                columns.append((group_col * band_col)[:, None])

    return np.column_stack(columns)


def _metric_family(metric_name: str) -> str:
    parts = metric_name.split("_")
    fc_type = parts[-1] if parts else "UNKNOWN"
    if metric_name.startswith(("Corr_", "LayerDecoupling", "HighOrderDecoupling")):
        return "layer_similarity"
    if metric_name.startswith("CEAblationAttackAUC"):
        return "ce_ablation_attack"
    if metric_name.startswith(("CE_auc", "S_auc", "CoreNode_stability", "StableCoreNode")):
        return f"core_stability | {fc_type}"
    if metric_name.startswith("FCStability"):
        return f"fc_stability | {fc_type}"
    if metric_name.startswith(("RichClub_auc", "TargetAttack_auc_across_density")):
        return f"density_sensitivity | {fc_type}"
    if metric_name.startswith(("RichClubMean_", "RichClubMax", "RichClubPeakK", "RichClubEnhancedKCount")):
        return f"richclub_features | {fc_type}"
    if metric_name.startswith(("Phi_w_", "WeightedRichClub")):
        return f"weighted_richclub_features | {fc_type}"
    if metric_name.startswith("CoreOverlap_BinaryCE_WeightedCE"):
        return f"weighted_ce_overlap | {fc_type}"
    if metric_name.startswith(("Eglob_bin_auc", "Clustering_bin_auc")):
        return f"Density AUC | {fc_type}"
    if metric_name.startswith(("Eglob_weighted", "Clustering_weighted")):
        return f"Weighted topology | {fc_type}"
    if metric_name.startswith(("CE_auc", "CoreNode_stability", "StableCoreNode")):
        return f"CE stability | {fc_type}"
    if metric_name.startswith(("AttackAUC", "CE_vs_", "EarlyDrop", "MaxDrop")):
        return f"Attack features | {fc_type}"
    if metric_name.startswith(("Corr_", "LayerDecoupling", "HighOrderDecoupling")):
        return "FC layer similarity"
    return f"Exploratory | {fc_type}"


def run_mixed_effects_metrics(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """
    Run repeated-measures fallback for Metric ~ Group * Band + subject intercept.

    This avoids treating Band rows as independent by blocking on SubjectID. The
    Group main effect is tested on subject-level means; Band and Group:Band are
    tested with nested F tests in a subject-fixed-intercept model.
    """
    _ensure_results_dir(results_dir)
    input_file = output_dir / "all_scalar_metrics_summary.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_file}")

    df = pd.read_csv(input_file)
    id_cols = ["Group", "Band", "SubjectID"]
    metric_columns = [col for col in df.columns if col not in id_cols]
    records: list[dict[str, object]] = []
    model_name = "Subject fixed-intercept repeated-measures OLS fallback"

    for metric in metric_columns:
        data = df[id_cols + [metric]].copy()
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data = data.dropna()
        if (
            data["SubjectID"].nunique() < 3
            or data["Group"].nunique() < 2
            or data["Band"].nunique() < 2
        ):
            continue

        y = data[metric].to_numpy(dtype=np.float64)
        subject_only = _repeated_band_design(
            data["SubjectID"],
            data["Group"],
            data["Band"],
            include_band=False,
            include_interaction=False,
        )
        subject_band = _repeated_band_design(
            data["SubjectID"],
            data["Group"],
            data["Band"],
            include_band=True,
            include_interaction=False,
        )
        subject_band_interaction = _repeated_band_design(
            data["SubjectID"],
            data["Group"],
            data["Band"],
            include_band=True,
            include_interaction=True,
        )

        for effect, test in [
            ("Band_within_subject", _nested_f_test(subject_only, subject_band, y)),
            (
                "Group:Band_within_subject",
                _nested_f_test(subject_band, subject_band_interaction, y),
            ),
        ]:
            records.append(
                {
                    "Metric": metric,
                    "Model_Type": "repeated_measures_fallback",
                    "Model": model_name,
                    "Effect": effect,
                    "N": len(data),
                    "SubjectN": data["SubjectID"].nunique(),
                    "Correction_Family": _metric_family(metric),
                    **test,
                }
            )

        subject_means = (
            data.groupby(["SubjectID", "Group"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "SubjectMean"})
        )
        group_data = {
            group: pd.to_numeric(
                subject_means.loc[subject_means["Group"] == group, "SubjectMean"],
                errors="coerce",
            )
            for group in GROUPS
        }
        available = {group: _as_numeric_series(values) for group, values in group_data.items()}
        available_groups = [group for group, values in available.items() if len(values) >= 2]
        if len(available_groups) >= 2:
            try:
                arrays = [available[group].to_numpy(dtype=float) for group in available_groups]
                statistic, p_value = stats.f_oneway(*arrays)
            except Exception:
                statistic, p_value = np.nan, np.nan
            records.append(
                {
                    "Metric": metric,
                    "Model_Type": "repeated_measures_fallback",
                    "Model": "Subject mean between-group one-way ANOVA",
                    "Effect": "Group_between_subject_mean",
                    "N": len(subject_means),
                    "SubjectN": subject_means["SubjectID"].nunique(),
                    "Correction_Family": _metric_family(metric),
                    "F": float(statistic),
                    "p_raw": float(p_value),
                    "df_num": float(len(available_groups) - 1),
                    "df_den": float(len(subject_means) - len(available_groups)),
                    "EtaSq_partial": np.nan,
                }
            )

    results_df = pd.DataFrame(records)
    if not results_df.empty:
        results_df["p_adj"] = np.nan
        for family, index in results_df.groupby(["Correction_Family", "Effect"]).groups.items():
            results_df.loc[index, "p_adj"] = _fdr_bh(
                results_df.loc[index, "p_raw"].to_numpy(dtype=float)
            )
        results_df["Significance_raw"] = results_df["p_raw"].map(_sig_label)
        results_df["Significance_adj"] = results_df["p_adj"].map(_sig_label)
        results_df["Correction_Method"] = P_ADJUST_METHOD
        results_df["Correction_Scope"] = (
            "repeated-measures metrics, separately by Correction_Family and Effect"
        )

    output_file = results_dir / "mixed_effects_metrics.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    _print_result_summary(results_df, output_file)
    return results_df


def _load_richclub_individual_means(input_file: Path) -> pd.DataFrame:
    """Read either multi-index or flat rich-club individual-mean tables."""
    try:
        multi_df = pd.read_csv(input_file, header=[0, 1])
        if isinstance(multi_df.columns, pd.MultiIndex):
            normalized: dict[str, pd.Series] = {}
            for band, group in multi_df.columns:
                band_str = str(band).strip()
                group_str = str(group).strip()
                if band_str in BANDS and group_str in GROUPS:
                    normalized[f"{band_str}_{group_str}"] = pd.to_numeric(
                        multi_df[(band, group)], errors="coerce"
                    )
            if normalized:
                return pd.DataFrame(normalized)
    except Exception:
        pass

    flat_df = pd.read_csv(input_file)
    normalized_flat: dict[str, pd.Series] = {}
    for col in flat_df.columns:
        col_str = str(col)
        col_compact = col_str.lower().replace(" ", "")
        for band in BANDS:
            for group in GROUPS:
                key = f"{band}_{group}"
                if key in normalized_flat:
                    continue
                if band.lower() in col_compact and group.lower() in col_compact:
                    normalized_flat[key] = pd.to_numeric(flat_df[col], errors="coerce")

    return pd.DataFrame(normalized_flat)


def run_richclub_tests(
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_UPPER,
    bands: list[str] = BANDS,
    groups: list[str] = GROUPS,
) -> pd.DataFrame:
    """Run binary normalized rich-club individual-mean tests."""
    _ensure_results_dir(results_dir)

    print("=" * 100)
    print("二值归一化 Rich-club 个体均值统计检验: omnibus + post-hoc + FDR")
    print("=" * 100)

    records: list[dict[str, object]] = []
    for fc_type in fc_types:
        input_file = results_dir / f"richclub_individual_means_{fc_type}.csv"
        if not input_file.exists():
            print(f"警告: 文件 {input_file} 不存在，跳过")
            continue

        df = _load_richclub_individual_means(input_file)
        if df.empty:
            print(f"警告: 文件 {input_file} 未解析出有效列，跳过")
            continue

        for band in bands:
            group_data = {
                group: pd.to_numeric(df.get(f"{band}_{group}"), errors="coerce")
                if f"{band}_{group}" in df.columns
                else pd.Series(dtype=float)
                for group in groups
            }
            records.extend(
                _analyze_endpoint(
                    context={
                        "Analysis": "RichClub_IndividualMean",
                        "FC_Type": fc_type,
                        "Band": band,
                        "Metric": "BinaryNormalizedRichClubMean",
                    },
                    group_data=group_data,
                    groups=groups,
                )
            )

    output_file = results_dir / "richclub_显著性分析.csv"
    results_df = _finalize_results(
        records,
        output_file,
        omnibus_scope="all rich-club omnibus tests",
        posthoc_scope="all rich-club post-hoc tests after significant omnibus",
    )

    _print_result_summary(results_df, output_file)
    return results_df


def _read_attack_curve(csv_file: Path) -> np.ndarray:
    values = pd.read_csv(csv_file, header=None).to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    return values.astype(float)


def attack_auc(curve: object) -> float:
    curve = np.asarray(curve, dtype=float)
    curve = curve[np.isfinite(curve)]
    if len(curve) < 2:
        return float("nan")

    x = np.arange(len(curve), dtype=float)
    area_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(area_fn(curve, x) / (x[-1] - x[0]))


def _load_attack_curves(
    root: Path,
    band_folder: str,
    *,
    groups: list[str],
) -> dict[str, list[np.ndarray]]:
    curves_by_group: dict[str, list[np.ndarray]] = {}
    lengths: dict[str, set[int]] = {}

    for group in groups:
        band_path = root / group / band_folder
        if not band_path.exists():
            print(f"警告: {band_path} 不存在，跳过")
            continue

        csv_files = sorted(band_path.glob("*.csv"))
        if not csv_files:
            print(f"警告: {band_path} 中没有CSV文件")
            continue

        curves: list[np.ndarray] = []
        for csv_file in csv_files:
            try:
                curve = _read_attack_curve(csv_file)
            except Exception as exc:
                print(f"读取文件 {csv_file} 时出错: {exc}")
                continue
            if curve.size:
                curves.append(curve)

        if curves:
            curves_by_group[group] = curves
            lengths[group] = {len(curve) for curve in curves}

    flat_lengths = {length for group_lengths in lengths.values() for length in group_lengths}
    if len(flat_lengths) > 1:
        raise ValueError(f"Attack curves have inconsistent lengths: {lengths}")

    return curves_by_group


def _attack_curves_to_matrix(curves: list[np.ndarray]) -> np.ndarray:
    if not curves:
        return np.empty((0, 0), dtype=float)
    return np.vstack([np.asarray(curve, dtype=float) for curve in curves])


def _run_targetattack_auc_tests(
    output_dir: Path,
    results_dir: Path,
    fc_types: list[str],
    band_mapping: dict[str, str],
    groups: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for fc_folder in fc_types:
        fc_type = fc_folder.upper()
        root = output_dir / "target_attack" / fc_folder
        if not root.exists():
            print(f"警告: 文件夹 {root} 不存在，跳过")
            continue

        for band_folder, band_name in band_mapping.items():
            curves_by_group = _load_attack_curves(root, band_folder, groups=groups)
            if len(curves_by_group) < 3:
                continue

            group_data = {
                group: pd.Series([attack_auc(curve) for curve in curves_by_group.get(group, [])])
                for group in groups
            }
            records.extend(
                _analyze_endpoint(
                    context={
                        "Analysis": "TargetAttack_AUC",
                        "FC_Type": fc_type,
                        "Band": band_name,
                        "Metric": "TargetAttack_AUC",
                    },
                    group_data=group_data,
                    groups=groups,
                )
            )

    output_file = results_dir / "targetattack_显著性分析.csv"
    results_df = _finalize_results(
        records,
        output_file,
        omnibus_scope="all target attack AUC omnibus tests",
        posthoc_scope="all target attack AUC post-hoc tests after significant omnibus",
    )
    _print_result_summary(results_df, output_file)
    return results_df


def _run_targetattack_stepwise_tests(
    output_dir: Path,
    results_dir: Path,
    fc_types: list[str],
    band_mapping: dict[str, str],
    groups: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for fc_folder in fc_types:
        fc_type = fc_folder.upper()
        root = output_dir / "target_attack" / fc_folder
        if not root.exists():
            continue

        for band_folder, band_name in band_mapping.items():
            curves_by_group = _load_attack_curves(root, band_folder, groups=groups)
            if len(curves_by_group) < 3:
                continue

            matrices = {
                group: _attack_curves_to_matrix(curves_by_group.get(group, []))
                for group in groups
            }
            n_steps = next(iter(matrices.values())).shape[1]
            for step in range(n_steps):
                group_data = {
                    group: pd.Series(matrices[group][:, step])
                    for group in groups
                    if matrices[group].size
                }
                records.extend(
                    _analyze_endpoint(
                        context={
                            "Analysis": "TargetAttack_Stepwise_Supplement",
                            "FC_Type": fc_type,
                            "Band": band_name,
                            "Metric": "TargetAttack_NormalizedEglob",
                            "Attack_Step": step,
                        },
                        group_data=group_data,
                        groups=groups,
                    )
                )

    output_file = results_dir / "targetattack_stepwise_补充分析.csv"
    results_df = _finalize_results(
        records,
        output_file,
        omnibus_scope="all target attack stepwise omnibus tests",
        posthoc_scope="all target attack stepwise post-hoc tests after significant omnibus",
    )
    _print_result_summary(results_df, output_file)
    return results_df


def run_targetattack_tests(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
) -> pd.DataFrame:
    """
    Run target attack statistics.

    The main output is subject-level AUC statistics in targetattack_显著性分析.csv.
    Stepwise tests are retained only as FDR-corrected supplementary analysis.
    """
    _ensure_results_dir(results_dir)

    print("=" * 100)
    print("Target Attack 主分析: AUC + omnibus + post-hoc + FDR")
    print("=" * 100)
    auc_df = _run_targetattack_auc_tests(output_dir, results_dir, fc_types, band_mapping, groups)

    print("=" * 100)
    print("Target Attack 补充分析: 分步检验 + omnibus + post-hoc + FDR")
    print("=" * 100)
    _run_targetattack_stepwise_tests(output_dir, results_dir, fc_types, band_mapping, groups)

    return auc_df


def _print_result_summary(results_df: pd.DataFrame, output_file: Path) -> None:
    print(f"\n{'=' * 100}")
    print("分析完成")
    print(f"{'=' * 100}")
    print(f"结果已保存至: {output_file}")
    print(f"总共输出 {len(results_df)} 行统计结果")

    if results_df.empty or "Test_Level" not in results_df.columns:
        return

    for level in ["Omnibus", "Posthoc"]:
        subset = results_df[results_df["Test_Level"] == level]
        if subset.empty:
            continue
        significant = int(subset["Significant_adj"].fillna(False).sum())
        print(f"  {level}: {len(subset)} 行, FDR校正后显著 {significant} 行")


def run_all_statistical_tests() -> None:
    """Run all statistical analysis tasks."""
    run_scalar_metric_tests()
    run_two_way_anova_metrics()
    run_mixed_effects_metrics()
    run_richclub_tests()
    run_targetattack_tests()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统计检验模块化入口")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "scalar", "two-way-anova", "mixed-effects", "richclub", "targetattack"],
        help="执行任务名称，默认 all",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "all":
        run_all_statistical_tests()
    elif args.command == "scalar":
        run_scalar_metric_tests()
    elif args.command == "two-way-anova":
        run_two_way_anova_metrics()
    elif args.command == "mixed-effects":
        run_mixed_effects_metrics()
    elif args.command == "richclub":
        run_richclub_tests()
    elif args.command == "targetattack":
        run_targetattack_tests()


if __name__ == "__main__":
    main()
