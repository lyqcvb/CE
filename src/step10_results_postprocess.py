"""
step10_results_postprocess.py
======================
将结果提取与重组层脚本模块化整合到单一入口。

支持任务：
1. rich-club 汇总表提取
2. target attack 汇总表提取
3. rich-club 个体均值表生成
4. 标量指标重组（均值/SEM）
5. 代表性核心节点提取
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pipeline_config import (
    BAND_LABELS as CONFIG_BAND_MAPPING,
    BANDS as CONFIG_BANDS,
    DEFAULT_CHANNEL_NAMES as CONFIG_DEFAULT_CHANNEL_NAMES,
    FC_TYPES_LOWER as CONFIG_FC_TYPES_LOWER,
    FC_TYPES_UPPER as CONFIG_FC_TYPES_UPPER,
    GROUP_NAMES as CONFIG_GROUPS,
    OUTPUT_DIR as CONFIG_OUTPUT_DIR,
    REPRESENTATIVE_CORE_THRESHOLDS as CONFIG_REPRESENTATIVE_CORE_THRESHOLDS,
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
REPRESENTATIVE_CORE_THRESHOLDS = CONFIG_REPRESENTATIVE_CORE_THRESHOLDS

# 16 导 EEG 常用 10-20 通道顺序（与 NodeIndex 顺序一致时可直接映射）
DEFAULT_CHANNEL_NAMES = CONFIG_DEFAULT_CHANNEL_NAMES


def _ensure_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(exist_ok=True)


def _sorted_csv_files(folder: Path, pattern: str = "*.csv") -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(pattern))


def _read_csv_first_row(csv_path: Path) -> np.ndarray | None:
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception:
        return None

    if df.empty:
        return None

    return pd.to_numeric(df.iloc[0], errors="coerce").to_numpy()


def _read_csv_vector(csv_path: Path) -> np.ndarray | None:
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception:
        return None

    if df.empty:
        return None

    values = pd.to_numeric(pd.Series(df.to_numpy().ravel()), errors="coerce")
    return values.dropna().to_numpy(dtype=np.float64)


def _pad_1d_arrays(values_list: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(values) for values in values_list)
    padded = np.full((len(values_list), max_len), np.nan, dtype=np.float64)
    for idx, values in enumerate(values_list):
        values = values.astype(np.float64)
        padded[idx, : len(values)] = values
    return padded


def _curve_auc(values: np.ndarray) -> float:
    curve = np.asarray(values, dtype=np.float64)
    curve = curve[np.isfinite(curve)]
    if curve.size < 2:
        return np.nan
    x = np.arange(curve.size, dtype=np.float64)
    return float(np.trapezoid(curve, x) / (x[-1] - x[0]))


def _curve_attack_features(values: np.ndarray) -> dict[str, float]:
    curve = np.asarray(values, dtype=np.float64)
    return {
        "AttackAUC": _curve_auc(curve),
        "EarlyDrop_1": 1.0 - curve[1] if curve.size > 1 and np.isfinite(curve[1]) else np.nan,
        "EarlyDrop_2": 1.0 - curve[2] if curve.size > 2 and np.isfinite(curve[2]) else np.nan,
        "EarlyDrop_3": 1.0 - curve[3] if curve.size > 3 and np.isfinite(curve[3]) else np.nan,
        "MaxDrop": float(1.0 - np.nanmin(curve)) if curve.size else np.nan,
    }


def _richclub_curve_features(values: np.ndarray) -> dict[str, float]:
    curve = np.asarray(values, dtype=np.float64)
    valid = curve[np.isfinite(curve)]
    if valid.size == 0:
        return {
            "RichClubMean_all": np.nan,
            "RichClubMean_high_k": np.nan,
            "RichClubMax": np.nan,
            "RichClubPeakK": np.nan,
            "RichClubEnhancedKCount": 0.0,
            "RichClubValidKCount": 0.0,
        }
    valid_indices = np.where(np.isfinite(curve))[0]
    high_start = int(np.floor(valid_indices.size * 0.70))
    high_values = curve[valid_indices[high_start:]]
    peak_k = int(valid_indices[np.nanargmax(curve[valid_indices])] + 1)
    return {
        "RichClubMean_all": float(np.nanmean(valid)),
        "RichClubMean_high_k": float(np.nanmean(high_values)) if high_values.size else np.nan,
        "RichClubMax": float(np.nanmax(valid)),
        "RichClubPeakK": float(peak_k),
        "RichClubEnhancedKCount": float(np.sum(valid > 1.0)),
        "RichClubValidKCount": float(valid.size),
    }


def _resolve_channel_name(row: pd.Series, node_idx: int) -> str:
    """优先使用文件内 ChannelName；缺失或为纯数字时回退为标准通道名。"""
    raw_name = row.get("ChannelName", None)
    name = "" if pd.isna(raw_name) else str(raw_name).strip()

    # 空字符串或纯数字（如 12 / 12.0）都视为无效通道名
    if (not name) or re.fullmatch(r"\d+(?:\.0+)?", name):
        if 0 <= node_idx < len(DEFAULT_CHANNEL_NAMES):
            return DEFAULT_CHANNEL_NAMES[node_idx]
        if 1 <= node_idx <= len(DEFAULT_CHANNEL_NAMES):
            return DEFAULT_CHANNEL_NAMES[node_idx - 1]
        return str(node_idx)

    return name


def extract_richclub_summary(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
    source_folder: str = "rich_club",
    output_prefix: str = "richclub_summary",
) -> dict[str, pd.DataFrame]:
    """提取并保存 rich-club 组均值汇总表。"""
    _ensure_results_dir(results_dir)
    written: dict[str, pd.DataFrame] = {}

    for fc_folder in fc_types:
        print(f"\n{'=' * 80}")
        print(f"处理 {fc_folder}")
        print(f"{'=' * 80}")

        root = output_dir / source_folder / fc_folder
        if not root.exists():
            print(f"警告: 文件夹 {root} 不存在，跳过")
            continue

        all_data: dict[str, np.ndarray] = {}

        for group in groups:
            group_path = root / group
            if not group_path.exists():
                print(f"警告: {group_path} 不存在，跳过")
                continue

            for band_folder, band_name in band_mapping.items():
                band_path = group_path / band_folder
                csv_files = _sorted_csv_files(band_path)
                if not csv_files:
                    print(f"警告: {band_path} 中没有CSV文件")
                    continue

                values_list: list[np.ndarray] = []
                for csv_file in csv_files:
                    values = _read_csv_first_row(csv_file)
                    if values is not None and values.size > 0:
                        values_list.append(values)

                if values_list:
                    mean_richclub = np.nanmean(_pad_1d_arrays(values_list), axis=0)
                    key = f"{band_name}_{group}"
                    all_data[key] = mean_richclub
                    print(f"  {group} - {band_name}: {len(values_list)} 个被试, 节点数={len(mean_richclub)}")

        if not all_data:
            continue

        max_nodes = max(len(v) for v in all_data.values())
        result_dict: dict[str, list[float] | np.ndarray] = {"k (Node Degree)": list(range(1, max_nodes + 1))}

        for band_name in BANDS:
            for group in groups:
                key = f"{band_name}_{group}"
                if key in all_data:
                    data = all_data[key].astype(float)
                    padded_data = np.pad(data, (0, max_nodes - len(data)), constant_values=np.nan)
                    result_dict[key] = padded_data

        result_df = pd.DataFrame(result_dict)
        fc_type = fc_folder.upper()
        output_file = results_dir / f"{output_prefix}_{fc_type}.csv"
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n结果已保存至: {output_file}")
        print(f"数据形状: {result_df.shape[0]} 行 × {result_df.shape[1]} 列")
        written[fc_type] = result_df

    print(f"\n{'=' * 80}")
    print("所有 Rich-club 数据处理完成！")
    print(f"{'=' * 80}")
    return written


def extract_targetattack_summary(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
    source_folder: str = "target_attack",
    output_prefix: str = "targetattack_summary",
) -> dict[str, pd.DataFrame]:
    """提取并保存 target attack 组均值汇总表。"""
    _ensure_results_dir(results_dir)
    written: dict[str, pd.DataFrame] = {}

    for fc_folder in fc_types:
        print(f"\n{'=' * 80}")
        print(f"处理 {fc_folder}")
        print(f"{'=' * 80}")

        root = output_dir / source_folder / fc_folder
        if not root.exists():
            print(f"警告: 文件夹 {root} 不存在，跳过")
            continue

        all_data: dict[str, np.ndarray] = {}

        for group in groups:
            group_path = root / group
            if not group_path.exists():
                print(f"警告: {group_path} 不存在，跳过")
                continue

            for band_folder, band_name in band_mapping.items():
                band_path = group_path / band_folder
                csv_files = _sorted_csv_files(band_path)
                if not csv_files:
                    print(f"警告: {band_path} 中没有CSV文件")
                    continue

                values_list: list[np.ndarray] = []
                for csv_file in csv_files:
                    values = _read_csv_vector(csv_file)
                    if values is not None and values.size > 0:
                        values_list.append(values)

                if values_list:
                    mean_targetattack = np.nanmean(_pad_1d_arrays(values_list), axis=0)
                    key = f"{band_name}_{group}"
                    all_data[key] = mean_targetattack
                    print(f"  {group} - {band_name}: {len(values_list)} 个被试, 数据点数={len(mean_targetattack)}")

        if not all_data:
            continue

        max_points = max(len(v) for v in all_data.values())
        result_dict: dict[str, list[int] | np.ndarray] = {
            "Attack_Step": list(range(max_points))
        }

        for band_name in BANDS:
            for group in groups:
                key = f"{band_name}_{group}"
                if key in all_data:
                    data = all_data[key].astype(float)
                    padded_data = np.pad(data, (0, max_points - len(data)), constant_values=np.nan)
                    result_dict[key] = padded_data

        result_df = pd.DataFrame(result_dict)
        fc_type = fc_folder.upper()
        output_file = results_dir / f"{output_prefix}_{fc_type}.csv"
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n结果已保存至: {output_file}")
        print(f"数据形状: {result_df.shape[0]} 行 × {result_df.shape[1]} 列")
        written[fc_type] = result_df

    print(f"\n{'=' * 80}")
    print("所有 Target Attack 数据处理完成！")
    print(f"{'=' * 80}")
    return written


def extract_targetattack_control_summaries(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
) -> dict[str, dict[str, pd.DataFrame]]:
    """提取 CE/degree/BC/random/adaptive attack 对照曲线均值表。"""
    control_root = output_dir / "target_attack_controls"
    if not control_root.exists():
        print(f"警告: 文件夹 {control_root} 不存在，跳过 attack 对照汇总")
        return {}

    written: dict[str, dict[str, pd.DataFrame]] = {}
    for strategy_dir in sorted(path for path in control_root.iterdir() if path.is_dir()):
        strategy = strategy_dir.name
        written[strategy] = extract_targetattack_summary(
            output_dir=output_dir,
            results_dir=results_dir,
            fc_types=fc_types,
            band_mapping=band_mapping,
            groups=groups,
            source_folder=f"target_attack_controls/{strategy}",
            output_prefix=f"targetattack_{strategy}_summary",
        )
    return written


def aggregate_network_qc(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame | None:
    """合并每个被试/FC/阈值模式的网络质量控制记录。"""
    _ensure_results_dir(results_dir)
    root = output_dir / "network_qc"
    if not root.exists():
        print(f"警告: 文件夹 {root} 不存在，跳过 network QC 汇总")
        return None

    rows: list[pd.DataFrame] = []
    for csv_file in sorted(root.glob("*/*/*/*_qc.csv")):
        try:
            rows.append(pd.read_csv(csv_file))
        except Exception as exc:
            print(f"  [!] 跳过 QC 文件 {csv_file}: {exc}")

    if not rows:
        print("未找到可汇总的 network QC 文件")
        return None

    df = pd.concat(rows, ignore_index=True)
    output_file = results_dir / "network_qc_summary.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    grouped = (
        df.groupby(["FC_Type", "Band", "Group", "ThresholdMode", "ThresholdValue"], dropna=False)
        .agg(
            N=("SubjectID", "nunique"),
            Density_Mean=("Density", "mean"),
            Density_SEM=("Density", "sem"),
            AverageDegree_Mean=("AverageDegree", "mean"),
            ComponentCount_Mean=("ComponentCount", "mean"),
            EdgeCount_Mean=("EdgeCount", "mean"),
            CoreZero_MeanStd_Count=("CoreZero_MeanStd", "sum"),
            AttackInitialEglob_Zero_Count=("AttackInitialEglob", lambda x: int((x <= 0).sum())),
        )
        .reset_index()
    )
    summary_file = results_dir / "network_qc_group_summary.csv"
    grouped.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print(f"Network QC 明细已保存至: {output_file}")
    print(f"Network QC 组汇总已保存至: {summary_file}")
    return df


def aggregate_density_sensitivity(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, pd.DataFrame]:
    """合并 CE/core、rich-club、attack 的多稀疏度敏感性输出。"""
    _ensure_results_dir(results_dir)
    folders = {
        "core_nodes_density": "core_node_density_stability.csv",
        "rich_club_density": "density_richclub_features.csv",
        "target_attack_density": "density_targetattack_features.csv",
    }
    written: dict[str, pd.DataFrame] = {}
    for folder, filename in folders.items():
        root = output_dir / folder
        if not root.exists():
            print(f"警告: 文件夹 {root} 不存在，跳过")
            continue
        frames: list[pd.DataFrame] = []
        for csv_file in sorted(root.glob("*/*/*/*.csv")):
            try:
                frames.append(pd.read_csv(csv_file))
            except Exception as exc:
                print(f"  [!] 跳过 {csv_file}: {exc}")
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        output_file = results_dir / filename
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        written[folder] = df
        print(f"{folder} 汇总已保存至: {output_file}")
    return written


def aggregate_weighted_richclub_stats(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame | None:
    """合并 weighted rich-club 随机归一化和经验 p 值结果。"""
    _ensure_results_dir(results_dir)
    root = output_dir / "rich_club_weighted_stats"
    if not root.exists():
        print(f"警告: 文件夹 {root} 不存在，跳过 weighted rich-club stats 汇总")
        return None
    frames: list[pd.DataFrame] = []
    for csv_file in sorted(root.glob("*/*/*/*.csv")):
        try:
            df = pd.read_csv(csv_file)
            frames.append(
                df.assign(
                    SubjectID=csv_file.stem,
                    Group=csv_file.parents[1].name,
                    Band=BAND_MAPPING.get(csv_file.parent.name, csv_file.parent.name),
                    FC_Type=csv_file.parents[2].name.upper(),
                )
            )
        except Exception as exc:
            print(f"  [!] 跳过 {csv_file}: {exc}")
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    cols = ["SubjectID", "Group", "Band", "FC_Type"] + [
        col for col in out.columns if col not in {"SubjectID", "Group", "Band", "FC_Type"}
    ]
    out = out[cols]
    output_file = results_dir / "weighted_richclub_stats_summary.csv"
    out.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Weighted rich-club stats 汇总已保存至: {output_file}")
    return out


def aggregate_fc_stability(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame | None:
    """合并 epoch-level FC 稳定性结果。"""
    _ensure_results_dir(results_dir)
    root = output_dir / "fc_stability"
    if not root.exists():
        print(f"警告: 文件夹 {root} 不存在，跳过 FC 稳定性汇总")
        return None
    frames = []
    for csv_file in sorted(root.glob("*/*/*.csv")):
        try:
            frames.append(pd.read_csv(csv_file))
        except Exception as exc:
            print(f"  [!] 跳过 {csv_file}: {exc}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    output_file = results_dir / "fc_stability_summary.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"FC 稳定性汇总已保存至: {output_file}")
    return df


def aggregate_layer_similarity(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame | None:
    """合并 LOFC/HOFC/aHOFC 层级相似性结果。"""
    _ensure_results_dir(results_dir)
    root = output_dir / "fc_layer_similarity"
    if not root.exists():
        print(f"警告: 文件夹 {root} 不存在，跳过 FC 层级相似性汇总")
        return None
    frames = []
    for csv_file in sorted(root.glob("*/*/*.csv")):
        try:
            frames.append(pd.read_csv(csv_file))
        except Exception as exc:
            print(f"  [!] 跳过 {csv_file}: {exc}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if "LayerDecoupling" not in df.columns and "Corr_LOFC_HOFC" in df.columns:
        df["LayerDecoupling"] = 1.0 - pd.to_numeric(df["Corr_LOFC_HOFC"], errors="coerce")
    if (
        "HighOrderDecoupling" not in df.columns
        and {"Corr_LOFC_HOFC", "Corr_LOFC_aHOFC"}.issubset(df.columns)
    ):
        df["HighOrderDecoupling"] = 1.0 - df[
            ["Corr_LOFC_HOFC", "Corr_LOFC_aHOFC"]
        ].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    output_file = results_dir / "fc_layer_similarity.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"FC 层级相似性汇总已保存至: {output_file}")
    return df


def compute_richclub_feature_table(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    source_folder: str = "rich_club",
    output_name: str = "richclub_feature_summary.csv",
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
) -> pd.DataFrame | None:
    """计算 RichClubMean_all/high_k/max/peak 等个体特征。"""
    _ensure_results_dir(results_dir)
    rows: list[dict[str, object]] = []
    for fc_folder in fc_types:
        for group in groups:
            for band_folder, band_name in band_mapping.items():
                band_path = output_dir / source_folder / fc_folder / group / band_folder
                for csv_file in _sorted_csv_files(band_path):
                    values = _read_csv_first_row(csv_file)
                    if values is None or values.size == 0:
                        continue
                    rows.append(
                        {
                            "SubjectID": csv_file.stem,
                            "Group": group,
                            "Band": band_name,
                            "FC_Type": fc_folder.upper(),
                            **_richclub_curve_features(values),
                        }
                    )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    output_file = results_dir / output_name
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Rich-club 特征表已保存至: {output_file}")
    return df


def compute_attack_feature_summary(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    source_folder: str = "target_attack_controls",
    output_name: str = "targetattack_feature_summary.csv",
) -> pd.DataFrame | None:
    """量化 attack AUC、EarlyDrop、MaxDrop 和 CE-vs-control AUC 差异。"""
    _ensure_results_dir(results_dir)
    root = output_dir / source_folder
    if not root.exists():
        print(f"警告: 文件夹 {root} 不存在，跳过 attack 特征汇总")
        return None

    rows_by_subject: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for strategy_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        strategy = strategy_dir.name
        for csv_file in sorted(strategy_dir.glob("*/*/*/*.csv")):
            try:
                fc_type = csv_file.parents[2].name.upper()
                group = csv_file.parents[1].name
                band_folder = csv_file.parent.name
                band = BAND_MAPPING.get(band_folder, band_folder)
                values = _read_csv_vector(csv_file)
            except Exception:
                continue
            if values is None or values.size == 0:
                continue
            key = (csv_file.stem, group, band, fc_type)
            row = rows_by_subject.setdefault(
                key,
                {"SubjectID": csv_file.stem, "Group": group, "Band": band, "FC_Type": fc_type},
            )
            features = _curve_attack_features(values)
            for feature_name, feature_value in features.items():
                row[f"{strategy}_{feature_name}"] = feature_value

    rows = list(rows_by_subject.values())
    if not rows:
        return None
    df = pd.DataFrame(rows)
    diff_pairs = {
        "CE_vs_Random_AUC_Diff": ("random_mean_AttackAUC", "ce_AttackAUC"),
        "CE_vs_Degree_AUC_Diff": ("degree_AttackAUC", "ce_AttackAUC"),
        "CE_vs_Betweenness_AUC_Diff": ("betweenness_AttackAUC", "ce_AttackAUC"),
        "CE_vs_AdaptiveCE_AUC_Diff": ("ce_adaptive_AttackAUC", "ce_AttackAUC"),
    }
    for output_col, (left, right) in diff_pairs.items():
        if left in df.columns and right in df.columns:
            df[output_col] = pd.to_numeric(df[left], errors="coerce") - pd.to_numeric(
                df[right],
                errors="coerce",
            )
    friendly_columns = {
        "ce_AttackAUC": "AttackAUC_CE",
        "ce_adaptive_AttackAUC": "AttackAUC_CEAdaptive",
        "degree_AttackAUC": "AttackAUC_Degree",
        "betweenness_AttackAUC": "AttackAUC_Betweenness",
        "random_mean_AttackAUC": "AttackAUC_RandomMean",
        "ce_EarlyDrop_1": "EarlyDrop_1",
        "ce_EarlyDrop_2": "EarlyDrop_2",
        "ce_EarlyDrop_3": "EarlyDrop_3",
        "ce_MaxDrop": "MaxDrop_CE",
    }
    for old_col, new_col in friendly_columns.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    output_file = results_dir / output_name
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Attack 特征表已保存至: {output_file}")
    return df


def compute_core_node_stability(
    threshold: float = 0.30,
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
) -> pd.DataFrame | None:
    """计算被试间、频段间、FC 类型间核心节点稳定性。"""
    _ensure_results_dir(results_dir)
    subject_rows: list[dict[str, object]] = []
    for fc_folder in fc_types:
        for group in groups:
            for band_folder, band_name in band_mapping.items():
                band_path = output_dir / "core_nodes" / fc_folder / group / band_folder
                files = _sorted_csv_files(band_path, "*_core_nodes.csv")
                if not files:
                    continue
                node_values: dict[int, list[int]] = {}
                for csv_file in files:
                    try:
                        df = pd.read_csv(csv_file)
                    except Exception:
                        continue
                    if "NodeIndex" not in df.columns or "IsCoreNode" not in df.columns:
                        continue
                    for _, row in df.iterrows():
                        node_idx = int(row["NodeIndex"])
                        node_values.setdefault(node_idx, []).append(int(row["IsCoreNode"]))
                for node_idx, values in node_values.items():
                    subject_stability = float(np.mean(values)) if values else np.nan
                    subject_rows.append(
                        {
                            "Group": group,
                            "Band": band_name,
                            "FC_Type": fc_folder.upper(),
                            "NodeIndex": node_idx,
                            "SubjectStability": subject_stability,
                            "SubjectStableCoreNode": int(subject_stability >= threshold),
                        }
                    )
    if not subject_rows:
        return None
    subject_df = pd.DataFrame(subject_rows)
    band_df = (
        subject_df.groupby(["Group", "FC_Type", "NodeIndex"], as_index=False)
        .agg(
            BandStability=("SubjectStableCoreNode", "mean"),
            SubjectStability_MeanAcrossBands=("SubjectStability", "mean"),
        )
    )
    fc_df = (
        subject_df.groupby(["Group", "Band", "NodeIndex"], as_index=False)
        .agg(
            FCTypeStability=("SubjectStableCoreNode", "mean"),
            SubjectStability_MeanAcrossFC=("SubjectStability", "mean"),
        )
    )
    overall_df = (
        subject_df.groupby(["Group", "NodeIndex"], as_index=False)
        .agg(
            OverallCoreStability=("SubjectStableCoreNode", "mean"),
            SubjectStability_OverallMean=("SubjectStability", "mean"),
        )
    )
    output_file = results_dir / "core_node_subject_stability.csv"
    subject_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    band_df.to_csv(results_dir / "core_node_band_stability.csv", index=False, encoding="utf-8-sig")
    fc_df.to_csv(results_dir / "core_node_fctype_stability.csv", index=False, encoding="utf-8-sig")
    overall_df.to_csv(
        results_dir / "core_node_overall_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"核心节点稳定性表已保存至: {output_file}")
    return subject_df


def compute_richclub_individual_means(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
    source_folder: str = "rich_club",
    output_prefix: str = "richclub_individual_means",
) -> dict[str, pd.DataFrame]:
    """计算并保存 rich-club 个体均值表（多级表头）。"""
    _ensure_results_dir(results_dir)
    written: dict[str, pd.DataFrame] = {}

    for fc_folder in fc_types:
        print(f"\n{'=' * 80}")
        print(f"处理 {fc_folder}")
        print(f"{'=' * 80}")

        root = output_dir / source_folder / fc_folder
        if not root.exists():
            print(f"警告: 文件夹 {root} 不存在，跳过")
            continue

        all_subject_rows: list[dict[str, object]] = []

        for group in groups:
            group_path = root / group
            if not group_path.exists():
                print(f"警告: {group_path} 不存在，跳过")
                continue

            for band_folder, band_name in band_mapping.items():
                band_path = group_path / band_folder
                csv_files = _sorted_csv_files(band_path)
                if not csv_files:
                    print(f"警告: {band_path} 中没有CSV文件")
                    continue

                for csv_file in csv_files:
                    values = _read_csv_first_row(csv_file)
                    if values is None or values.size == 0:
                        continue

                    all_subject_rows.append(
                        {
                            "SubjectID": csv_file.stem,
                            "Group": group,
                            "Band": band_name,
                            "RichClub_Mean": float(np.nanmean(values)),
                        }
                    )

                print(f"  {group} - {band_name}: 处理了 {len(csv_files)} 个被试")

        if not all_subject_rows:
            continue

        df = pd.DataFrame(all_subject_rows)

        max_rows = 0
        for group in groups:
            for band_name in BANDS:
                subset = df[(df["Band"] == band_name) & (df["Group"] == group)]
                max_rows = max(max_rows, len(subset))

        multi_columns = [("", "Group")]
        for band_name in BANDS:
            for group in groups:
                multi_columns.append((band_name, group))

        output_rows: list[list[object]] = []
        for i in range(max_rows):
            row_data: list[object] = [f"Subject_{i + 1}"]
            for band_name in BANDS:
                for group in groups:
                    subset = df[(df["Band"] == band_name) & (df["Group"] == group)]
                    subset = subset.sort_values("SubjectID").reset_index(drop=True)
                    if i < len(subset):
                        row_data.append(subset.iloc[i]["RichClub_Mean"])
                    else:
                        row_data.append(np.nan)
            output_rows.append(row_data)

        final_df = pd.DataFrame(output_rows, columns=pd.MultiIndex.from_tuples(multi_columns))

        fc_type = fc_folder.upper()
        output_file = results_dir / f"{output_prefix}_{fc_type}.csv"
        final_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n结果已保存至: {output_file}")
        print(f"数据形状: {final_df.shape[0]} 行（被试） × {final_df.shape[1]} 列")
        written[fc_type] = final_df

    print(f"\n{'=' * 80}")
    print("所有 Rich-club 个体均值计算完成！")
    print(f"{'=' * 80}")
    return written


def reorganize_scalar_metrics(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    metric_types: list[str] = [
        "Eglob_bin",
        "Clustering_bin",
        "Eglob_weighted",
        "Clustering_weighted",
        "Eglob_bin_auc",
        "Clustering_bin_auc",
        "MeanConn",
    ],
    fc_types: list[str] = ["LOFC", "HOFC", "aHOFC"],
    bands: list[str] = BANDS,
    groups: list[str] = GROUPS,
) -> pd.DataFrame:
    """重组 all_scalar_metrics_summary 为按指标分组的宽表。"""
    _ensure_results_dir(results_dir)

    input_file = output_dir / "all_scalar_metrics_summary.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_file}")

    df = pd.read_csv(input_file)
    results: dict[str, list[float] | list[str]] = {"Band": list(bands)}
    legacy_metric_types = {
        "Eglob_bin": "Eglob",
        "Clustering_bin": "Cluster",
    }

    for metric_type in metric_types:
        for fc_type in fc_types:
            metric = f"{metric_type}_{fc_type}"
            legacy_metric = f"{legacy_metric_types[metric_type]}_{fc_type}" if metric_type in legacy_metric_types else None
            source_metric = metric if metric in df.columns else legacy_metric

            for group in groups:
                mean_col_name = f"{metric}_{group}_Mean"
                sem_col_name = f"{metric}_{group}_SEM"
                mean_values: list[float] = []
                sem_values: list[float] = []

                for band in bands:
                    if source_metric in df.columns:
                        series = df[(df["Band"] == band) & (df["Group"] == group)][source_metric]
                        mean_values.append(float(series.mean()))
                        sem_values.append(float(series.sem()))
                    else:
                        mean_values.append(np.nan)
                        sem_values.append(np.nan)

                results[mean_col_name] = mean_values
                results[sem_col_name] = sem_values

    results_df = pd.DataFrame(results)
    output_file = results_dir / "重组数据_按指标分组.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("数据重组完成！")
    print(f"结果已保存至: {output_file}")
    print(f"数据形状: {results_df.shape[0]} 行 × {results_df.shape[1]} 列")
    return results_df


def extract_representative_core_nodes(
    threshold: float = 0.30,
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
) -> pd.DataFrame | None:
    """提取并保存代表性核心节点及按FC类型汇总表。"""
    _ensure_results_dir(results_dir)

    print("=" * 100)
    print(f"代表性核心节点提取分析 (阈值: {threshold * 100:.1f}%)")
    print("=" * 100)

    all_results: list[dict[str, object]] = []

    for fc_folder in fc_types:
        fc_type = fc_folder.upper()
        print(f"\n{'=' * 100}")
        print(f"FC类型: {fc_type}")
        print(f"{'=' * 100}")

        root = output_dir / "core_nodes" / fc_folder
        if not root.exists():
            print(f"警告: 文件夹 {root} 不存在，跳过")
            continue

        for band_folder, band_name in band_mapping.items():
            print(f"\n--- 频段: {band_name} ---")

            for group in groups:
                group_path = root / group / band_folder
                csv_files = _sorted_csv_files(group_path, "*_core_nodes.csv")
                if not csv_files:
                    print(f"  警告: {group_path} 中没有核心节点文件")
                    continue

                node_core_counts: Counter[int] = Counter()
                node_names: dict[int, str] = {}
                valid_subjects = 0

                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                    except Exception as exc:
                        print(f"  读取文件 {csv_file} 时出错: {exc}")
                        continue

                    if "NodeIndex" not in df.columns or "IsCoreNode" not in df.columns:
                        print(f"  警告: {csv_file} 缺少 NodeIndex 或 IsCoreNode 列，跳过")
                        continue

                    valid_subjects += 1
                    core_nodes = df[df["IsCoreNode"] == 1]

                    for _, row in core_nodes.iterrows():
                        try:
                            node_idx = int(row["NodeIndex"])
                        except Exception:
                            continue

                        node_name = _resolve_channel_name(row, node_idx)

                        node_core_counts[node_idx] += 1
                        node_names[node_idx] = node_name

                if valid_subjects == 0:
                    print(f"  警告: {group_path} 无有效文件")
                    continue

                node_proportions = {
                    node: count / valid_subjects for node, count in node_core_counts.items()
                }
                representative_nodes = {
                    node: prop for node, prop in node_proportions.items() if prop >= threshold
                }
                sorted_nodes = sorted(
                    representative_nodes.items(), key=lambda x: x[1], reverse=True
                )

                print(f"  {group}: {valid_subjects} 个被试")
                if sorted_nodes:
                    print(f"    代表性核心节点 (比例 >= {threshold * 100:.1f}%): {len(sorted_nodes)} 个")
                    for node_idx, proportion in sorted_nodes:
                        count = node_core_counts[node_idx]
                        node_name = node_names.get(node_idx, str(node_idx))
                        print(
                            f"      节点 {node_idx:2d} ({node_name}): "
                            f"{count:2d}/{valid_subjects} = {proportion * 100:5.1f}%"
                        )

                        all_results.append(
                            {
                                "FC_Type": fc_type,
                                "Band": band_name,
                                "Group": group,
                                "NodeIndex": node_idx,
                                "ChannelName": node_name,
                                "Count": count,
                                "TotalSubjects": valid_subjects,
                                "Proportion": proportion,
                                "Percentage": f"{proportion * 100:.1f}%",
                            }
                        )
                else:
                    print(f"    代表性核心节点 (比例 >= {threshold * 100:.1f}%): 0 个")

    if not all_results:
        print("\n未识别出任何代表性核心节点")
        print(f"\n{'=' * 100}")
        return None

    results_df = pd.DataFrame(all_results)
    detail_file = results_dir / f"representative_core_nodes_threshold_{int(threshold * 100)}.csv"
    results_df.to_csv(detail_file, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 100}")
    print("分析完成！")
    print(f"{'=' * 100}")
    print(f"\n结果已保存至: {detail_file}")
    print(f"共识别出 {len(results_df)} 个代表性核心节点")

    for fc_type in FC_TYPES_UPPER:
        fc_results = results_df[results_df["FC_Type"] == fc_type]
        if fc_results.empty:
            continue

        summary_rows: list[dict[str, object]] = []
        for band in BANDS:
            for group in groups:
                subset = fc_results[(fc_results["Band"] == band) & (fc_results["Group"] == group)]
                if subset.empty:
                    summary_rows.append(
                        {
                            "Band": band,
                            "Group": group,
                            "NumNodes": 0,
                            "RepresentativeNodes": "None",
                        }
                    )
                    continue

                subset_sorted = subset.sort_values("Proportion", ascending=False)
                nodes_list = ", ".join(
                    [
                        f"{row['ChannelName']}({row['Percentage']})"
                        for _, row in subset_sorted.iterrows()
                    ]
                )
                summary_rows.append(
                    {
                        "Band": band,
                        "Group": group,
                        "NumNodes": len(subset),
                        "RepresentativeNodes": nodes_list,
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        summary_file = (
            results_dir
            / f"representative_core_nodes_summary_{fc_type}_threshold_{int(threshold * 100)}.csv"
        )
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
        print(f"  {fc_type} 汇总表已保存至: {summary_file}")

    print(f"\n{'=' * 100}")
    return results_df


def run_all_results_postprocess(threshold: float = 0.30) -> None:
    """顺序执行结果层全部任务。"""
    extract_richclub_summary()
    compute_richclub_individual_means()
    compute_richclub_feature_table()
    extract_richclub_summary(
        source_folder="rich_club_weighted",
        output_prefix="weighted_richclub_summary",
    )
    compute_richclub_individual_means(
        source_folder="rich_club_weighted",
        output_prefix="weighted_richclub_individual_means",
    )
    compute_richclub_feature_table(
        source_folder="rich_club_weighted",
        output_name="weighted_richclub_feature_summary.csv",
    )
    aggregate_weighted_richclub_stats()
    extract_targetattack_summary()
    extract_targetattack_control_summaries()
    compute_attack_feature_summary()
    compute_attack_feature_summary(
        source_folder="ce_ablation_attack",
        output_name="ce_ablation_attack_feature_summary.csv",
    )
    aggregate_density_sensitivity()
    aggregate_fc_stability()
    aggregate_layer_similarity()
    aggregate_network_qc()
    reorganize_scalar_metrics()
    compute_core_node_stability(threshold=threshold)
    thresholds = sorted({threshold, *REPRESENTATIVE_CORE_THRESHOLDS})
    for core_threshold in thresholds:
        extract_representative_core_nodes(threshold=core_threshold)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="结果提取与重组模块化入口")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "all",
            "richclub-summary",
            "richclub-individual-means",
            "targetattack-summary",
            "targetattack-control-summaries",
            "targetattack-features",
            "ce-ablation-attack-features",
            "density-sensitivity",
            "fc-stability",
            "layer-similarity",
            "network-qc",
            "richclub-features",
            "weighted-richclub-summary",
            "weighted-richclub-individual-means",
            "weighted-richclub-features",
            "weighted-richclub-stats",
            "core-node-stability",
            "reorganize-scalar",
            "representative-core-nodes",
        ],
        help="执行任务名称，默认 all",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="代表性核心节点阈值（默认 0.30）",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "all":
        run_all_results_postprocess(threshold=args.threshold)
    elif args.command == "richclub-summary":
        extract_richclub_summary()
    elif args.command == "richclub-individual-means":
        compute_richclub_individual_means()
    elif args.command == "targetattack-summary":
        extract_targetattack_summary()
    elif args.command == "targetattack-control-summaries":
        extract_targetattack_control_summaries()
    elif args.command == "targetattack-features":
        compute_attack_feature_summary()
    elif args.command == "ce-ablation-attack-features":
        compute_attack_feature_summary(
            source_folder="ce_ablation_attack",
            output_name="ce_ablation_attack_feature_summary.csv",
        )
    elif args.command == "density-sensitivity":
        aggregate_density_sensitivity()
    elif args.command == "fc-stability":
        aggregate_fc_stability()
    elif args.command == "layer-similarity":
        aggregate_layer_similarity()
    elif args.command == "network-qc":
        aggregate_network_qc()
    elif args.command == "richclub-features":
        compute_richclub_feature_table()
    elif args.command == "weighted-richclub-summary":
        extract_richclub_summary(
            source_folder="rich_club_weighted",
            output_prefix="weighted_richclub_summary",
        )
    elif args.command == "weighted-richclub-individual-means":
        compute_richclub_individual_means(
            source_folder="rich_club_weighted",
            output_prefix="weighted_richclub_individual_means",
        )
    elif args.command == "weighted-richclub-features":
        compute_richclub_feature_table(
            source_folder="rich_club_weighted",
            output_name="weighted_richclub_feature_summary.csv",
        )
    elif args.command == "weighted-richclub-stats":
        aggregate_weighted_richclub_stats()
    elif args.command == "core-node-stability":
        compute_core_node_stability(threshold=args.threshold)
    elif args.command == "reorganize-scalar":
        reorganize_scalar_metrics()
    elif args.command == "representative-core-nodes":
        extract_representative_core_nodes(threshold=args.threshold)


if __name__ == "__main__":
    main()
