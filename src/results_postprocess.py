"""
results_postprocess.py
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

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = BASE_DIR / "results"

FC_TYPES_LOWER = ["lofc", "hofc", "ahofc"]
FC_TYPES_UPPER = ["LOFC", "HOFC", "AHOFC"]

BAND_MAPPING = {
    "0.5-4(Delta)": "Delta",
    "4-8(Theta)": "Theta",
    "8-13(Alpha)": "Alpha",
    "13-40(Beta)": "Beta",
}
BANDS = ["Delta", "Theta", "Alpha", "Beta"]
GROUPS = ["HC", "MCI", "AD"]

# 16 导 EEG 常用 10-20 通道顺序（与 NodeIndex 顺序一致时可直接映射）
DEFAULT_CHANNEL_NAMES = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
]


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
) -> dict[str, pd.DataFrame]:
    """提取并保存 rich-club 组均值汇总表。"""
    _ensure_results_dir(results_dir)
    written: dict[str, pd.DataFrame] = {}

    for fc_folder in fc_types:
        print(f"\n{'=' * 80}")
        print(f"处理 {fc_folder}")
        print(f"{'=' * 80}")

        root = output_dir / "rich_club" / fc_folder
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
                    mean_richclub = np.nanmean(np.vstack(values_list), axis=0)
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
        output_file = results_dir / f"richclub_summary_{fc_type}.csv"
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
) -> dict[str, pd.DataFrame]:
    """提取并保存 target attack 组均值汇总表。"""
    _ensure_results_dir(results_dir)
    written: dict[str, pd.DataFrame] = {}

    for fc_folder in fc_types:
        print(f"\n{'=' * 80}")
        print(f"处理 {fc_folder}")
        print(f"{'=' * 80}")

        root = output_dir / "target_attack" / fc_folder
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
                    mean_targetattack = np.nanmean(np.vstack(values_list), axis=0)
                    key = f"{band_name}_{group}"
                    all_data[key] = mean_targetattack
                    print(f"  {group} - {band_name}: {len(values_list)} 个被试, 数据点数={len(mean_targetattack)}")

        if not all_data:
            continue

        max_points = max(len(v) for v in all_data.values())
        result_dict: dict[str, list[float] | np.ndarray] = {"Attack_Step": list(range(1, max_points + 1))}

        for band_name in BANDS:
            for group in groups:
                key = f"{band_name}_{group}"
                if key in all_data:
                    data = all_data[key].astype(float)
                    padded_data = np.pad(data, (0, max_points - len(data)), constant_values=np.nan)
                    result_dict[key] = padded_data

        result_df = pd.DataFrame(result_dict)
        fc_type = fc_folder.upper()
        output_file = results_dir / f"targetattack_summary_{fc_type}.csv"
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n结果已保存至: {output_file}")
        print(f"数据形状: {result_df.shape[0]} 行 × {result_df.shape[1]} 列")
        written[fc_type] = result_df

    print(f"\n{'=' * 80}")
    print("所有 Target Attack 数据处理完成！")
    print(f"{'=' * 80}")
    return written


def compute_richclub_individual_means(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
) -> dict[str, pd.DataFrame]:
    """计算并保存 rich-club 个体均值表（多级表头）。"""
    _ensure_results_dir(results_dir)
    written: dict[str, pd.DataFrame] = {}

    for fc_folder in fc_types:
        print(f"\n{'=' * 80}")
        print(f"处理 {fc_folder}")
        print(f"{'=' * 80}")

        root = output_dir / "rich_club" / fc_folder
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
        output_file = results_dir / f"richclub_individual_means_{fc_type}.csv"
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
    metric_types: list[str] = ["MeanConn", "Eglob", "Cluster"],
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

    for metric_type in metric_types:
        for fc_type in fc_types:
            metric = f"{metric_type}_{fc_type}"

            for group in groups:
                mean_col_name = f"{metric}_{group}_Mean"
                sem_col_name = f"{metric}_{group}_SEM"
                mean_values: list[float] = []
                sem_values: list[float] = []

                for band in bands:
                    if metric in df.columns:
                        series = df[(df["Band"] == band) & (df["Group"] == group)][metric]
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
    extract_targetattack_summary()
    reorganize_scalar_metrics()
    extract_representative_core_nodes(threshold=threshold)


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
    elif args.command == "reorganize-scalar":
        reorganize_scalar_metrics()
    elif args.command == "representative-core-nodes":
        extract_representative_core_nodes(threshold=args.threshold)


if __name__ == "__main__":
    main()
