"""
statistical_analysis.py
======================
将统计检验层脚本模块化整合到单一入口。

支持任务：
1. 标量指标组间统计检验
2. rich-club 个体均值统计检验
3. target attack 分步统计检验
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro, ttest_ind

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
COMPARISONS = [("HC", "MCI"), ("HC", "AD"), ("MCI", "AD")]


def _ensure_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(exist_ok=True)


def _sig_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _normality_test(values: pd.Series) -> tuple[float, bool]:
    values = values.dropna()
    if len(values) >= 3:
        _, p_value = shapiro(values)
        return float(p_value), bool(p_value > 0.05)
    return float("nan"), False


def _run_pairwise_test(
    data1: pd.Series, data2: pd.Series, both_normal: bool
) -> tuple[str, float, float]:
    if both_normal:
        stat, p_value = ttest_ind(data1, data2, equal_var=True)
        return "Independent t-test", float(stat), float(p_value)

    stat, p_value = mannwhitneyu(data1, data2, alternative="two-sided")
    return "Mann-Whitney U test", float(stat), float(p_value)


def run_scalar_metric_tests(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """运行标量指标组间统计检验，输出 显著性分析.csv。"""
    _ensure_results_dir(results_dir)

    input_file = output_dir / "all_scalar_metrics_summary.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_file}")

    df = pd.read_csv(input_file)
    metric_columns = [
        col for col in df.columns if col not in ["Group", "Band", "SubjectID"]
    ]
    bands = df["Band"].dropna().unique().tolist()

    results: list[dict[str, object]] = []

    print("=" * 100)
    print("统计检验分析报告")
    print("=" * 100)

    for band in bands:
        print(f"\n{'=' * 100}")
        print(f"频段: {band}")
        print(f"{'=' * 100}\n")

        band_data = df[df["Band"] == band]
        groups = band_data["Group"].dropna().unique().tolist()

        for metric in metric_columns:
            print(f"\n--- 指标: {metric} ---")
            if len(groups) < 2:
                print(f"  警告: 只有一个组，无法进行组间比较")
                continue

            group_data: dict[str, pd.Series] = {}
            normality: dict[str, dict[str, object]] = {}

            for group in groups:
                values = band_data[band_data["Group"] == group][metric].dropna()
                group_data[group] = values
                normal_p, is_normal = _normality_test(values)
                normality[group] = {"p_value": normal_p, "is_normal": is_normal}

                if np.isnan(normal_p):
                    print(f"  Band={band}, Group={group}, Metric={metric}")
                    print(f"    样本量不足 (n={len(values)})，假设不满足正态性")
                else:
                    status = "满足正态性" if is_normal else "不满足正态性"
                    print(f"  Band={band}, Group={group}, Metric={metric}")
                    print(f"    Shapiro-Wilk: p={normal_p:.4f} -> {status}")

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    group1 = groups[i]
                    group2 = groups[j]
                    data1 = group_data[group1]
                    data2 = group_data[group2]

                    if len(data1) < 2 or len(data2) < 2:
                        print(f"  {group1} vs {group2}: 样本量不足，跳过")
                        continue

                    both_normal = bool(
                        normality[group1]["is_normal"] and normality[group2]["is_normal"]
                    )
                    test_type, stat, p_value = _run_pairwise_test(data1, data2, both_normal)

                    mean1, std1 = float(data1.mean()), float(data1.std())
                    mean2, std2 = float(data2.mean()), float(data2.std())
                    sig = _sig_label(p_value)

                    print(f"\n  {group1} vs {group2}:")
                    print(f"    {test_type}: statistic={stat:.4f}, p={p_value:.4f} ({sig})")
                    print(f"    {group1}: Mean={mean1:.4f}, SD={std1:.4f}, n={len(data1)}")
                    print(f"    {group2}: Mean={mean2:.4f}, SD={std2:.4f}, n={len(data2)}")

                    results.append(
                        {
                            "Band": band,
                            "Metric": metric,
                            "Group1": group1,
                            "Group2": group2,
                            "Group1_n": len(data1),
                            "Group2_n": len(data2),
                            "Group1_Mean": mean1,
                            "Group1_SD": std1,
                            "Group2_Mean": mean2,
                            "Group2_SD": std2,
                            "Group1_Normal_p": normality[group1]["p_value"],
                            "Group2_Normal_p": normality[group2]["p_value"],
                            "Test_Type": test_type,
                            "Test_Statistic": stat,
                            "p_value": p_value,
                            "Significant": sig,
                        }
                    )

    results_df = pd.DataFrame(results)
    output_file = results_dir / "显著性分析.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 100}")
    print("分析完成！")
    print(f"{'=' * 100}")
    print(f"结果已保存至: {output_file}")
    print(f"总共进行了 {len(results_df)} 次统计检验")

    if not results_df.empty and "Significant" in results_df.columns:
        print("显著性统计:")
        for key in ["***", "**", "*", "n.s."]:
            print(f"  {key}: {(results_df['Significant'] == key).sum()} 次")

    return results_df


def _load_richclub_individual_means(input_file: Path) -> pd.DataFrame:
    """兼容读取两种格式：多级表头 或 扁平列名。"""
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
    """运行 rich-club 个体均值统计检验，输出 richclub_显著性分析.csv。"""
    _ensure_results_dir(results_dir)

    results: list[dict[str, object]] = []

    print("=" * 100)
    print("Rich-club 系数统计检验分析报告")
    print("=" * 100)

    for fc_type in fc_types:
        print(f"\n{'=' * 100}")
        print(f"FC类型: {fc_type}")
        print(f"{'=' * 100}\n")

        input_file = results_dir / f"richclub_individual_means_{fc_type}.csv"
        if not input_file.exists():
            print(f"警告: 文件 {input_file} 不存在，跳过")
            continue

        df = _load_richclub_individual_means(input_file)
        if df.empty:
            print(f"警告: 文件 {input_file} 未解析出有效列，跳过")
            continue

        for band in bands:
            print(f"\n--- 频段: {band} ---")

            group_data: dict[str, pd.Series] = {}
            normality: dict[str, dict[str, object]] = {}

            for group in groups:
                col_name = f"{band}_{group}"
                if col_name not in df.columns:
                    print(f"  警告: 列 {col_name} 不存在")
                    continue

                values = pd.to_numeric(df[col_name], errors="coerce").dropna()
                group_data[group] = values

                normal_p, is_normal = _normality_test(values)
                normality[group] = {"p_value": normal_p, "is_normal": is_normal}

                if np.isnan(normal_p):
                    print(f"  FC={fc_type}, Band={band}, Group={group}")
                    print(f"    样本量不足 (n={len(values)})，假设不满足正态性")
                else:
                    status = "满足正态性" if is_normal else "不满足正态性"
                    print(f"  FC={fc_type}, Band={band}, Group={group}")
                    print(f"    Shapiro-Wilk: p={normal_p:.4f} -> {status}")

            group_list = [g for g in groups if g in group_data]
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    group1 = group_list[i]
                    group2 = group_list[j]
                    data1 = group_data[group1]
                    data2 = group_data[group2]

                    if len(data1) < 2 or len(data2) < 2:
                        print(f"  {group1} vs {group2}: 样本量不足，跳过")
                        continue

                    both_normal = bool(
                        normality[group1]["is_normal"] and normality[group2]["is_normal"]
                    )
                    test_type, stat, p_value = _run_pairwise_test(data1, data2, both_normal)

                    mean1, std1 = float(data1.mean()), float(data1.std())
                    mean2, std2 = float(data2.mean()), float(data2.std())
                    sig = _sig_label(p_value)

                    print(f"\n  {group1} vs {group2}:")
                    print(f"    {test_type}: statistic={stat:.4f}, p={p_value:.4f} ({sig})")
                    print(f"    {group1}: Mean={mean1:.4f}, SD={std1:.4f}, n={len(data1)}")
                    print(f"    {group2}: Mean={mean2:.4f}, SD={std2:.4f}, n={len(data2)}")

                    results.append(
                        {
                            "FC_Type": fc_type,
                            "Band": band,
                            "Group1": group1,
                            "Group2": group2,
                            "Group1_n": len(data1),
                            "Group2_n": len(data2),
                            "Group1_Mean": mean1,
                            "Group1_SD": std1,
                            "Group2_Mean": mean2,
                            "Group2_SD": std2,
                            "Group1_Normal_p": normality[group1]["p_value"],
                            "Group2_Normal_p": normality[group2]["p_value"],
                            "Test_Type": test_type,
                            "Test_Statistic": stat,
                            "p_value": p_value,
                            "Significant": sig,
                        }
                    )

    results_df = pd.DataFrame(results)
    output_file = results_dir / "richclub_显著性分析.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 100}")
    print("分析完成！")
    print(f"{'=' * 100}")
    print(f"结果已保存至: {output_file}")
    print(f"总共进行了 {len(results_df)} 次统计检验")

    if not results_df.empty and "Significant" in results_df.columns:
        print("显著性统计:")
        for key in ["***", "**", "*", "n.s."]:
            print(f"  {key}: {(results_df['Significant'] == key).sum()} 次")

    return results_df


def run_targetattack_tests(
    output_dir: Path = OUTPUT_DIR,
    results_dir: Path = RESULTS_DIR,
    fc_types: list[str] = FC_TYPES_LOWER,
    band_mapping: dict[str, str] = BAND_MAPPING,
    groups: list[str] = GROUPS,
    comparisons: list[tuple[str, str]] = COMPARISONS,
) -> pd.DataFrame:
    """运行 target attack 分步统计检验，输出 targetattack_显著性分析.csv。"""
    _ensure_results_dir(results_dir)

    all_results: list[dict[str, object]] = []

    print("=" * 100)
    print("Target Attack 显著性差异分析")
    print("=" * 100)

    for fc_folder in fc_types:
        fc_type = fc_folder.upper()

        print(f"\n{'=' * 100}")
        print(f"FC类型: {fc_type}")
        print(f"{'=' * 100}")

        root = output_dir / "target_attack" / fc_folder
        if not root.exists():
            print(f"警告: 文件夹 {root} 不存在，跳过")
            continue

        for band_folder, band_name in band_mapping.items():
            print(f"\n--- 频段: {band_name} ---")
            group_data: dict[str, np.ndarray] = {}

            for group in groups:
                band_path = root / group / band_folder
                if not band_path.exists():
                    print(f"警告: {band_path} 不存在，跳过")
                    continue

                csv_files = sorted(band_path.glob("*.csv"))
                if not csv_files:
                    print(f"警告: {band_path} 中没有CSV文件")
                    continue

                subjects_data: list[np.ndarray] = []
                for csv_file in csv_files:
                    try:
                        values = pd.read_csv(csv_file, header=None).iloc[0].to_numpy(dtype=float)
                        subjects_data.append(values)
                    except Exception as exc:
                        print(f"读取文件 {csv_file} 时出错: {exc}")

                if subjects_data:
                    group_data[group] = np.array(subjects_data)
                    print(
                        f"  {group}: {len(subjects_data)} 个被试, "
                        f"{group_data[group].shape[1]} 个攻击步骤"
                    )

            if len(group_data) != 3:
                continue

            n_steps = group_data["HC"].shape[1]
            for step in range(n_steps):
                step_num = step + 1
                for group1, group2 in comparisons:
                    data1 = group_data[group1][:, step]
                    data2 = group_data[group2][:, step]
                    data1_clean = data1[~np.isnan(data1)]
                    data2_clean = data2[~np.isnan(data2)]

                    if len(data1_clean) < 3 or len(data2_clean) < 3:
                        continue

                    mean1 = float(np.mean(data1_clean))
                    std1 = float(np.std(data1_clean, ddof=1))
                    mean2 = float(np.mean(data2_clean))
                    std2 = float(np.std(data2_clean, ddof=1))

                    _, p_normal1 = stats.shapiro(data1_clean)
                    _, p_normal2 = stats.shapiro(data2_clean)

                    if p_normal1 > 0.05 and p_normal2 > 0.05:
                        stat, p_value = stats.ttest_ind(data1_clean, data2_clean)
                        test_type = "Independent t-test"
                    else:
                        stat, p_value = stats.mannwhitneyu(
                            data1_clean, data2_clean, alternative="two-sided"
                        )
                        test_type = "Mann-Whitney U test"

                    significance = _sig_label(float(p_value))

                    all_results.append(
                        {
                            "FC_Type": fc_type,
                            "Band": band_name,
                            "Attack_Step": step_num,
                            "Group1": group1,
                            "Group2": group2,
                            "Group1_n": len(data1_clean),
                            "Group2_n": len(data2_clean),
                            "Group1_Mean": mean1,
                            "Group1_SD": std1,
                            "Group2_Mean": mean2,
                            "Group2_SD": std2,
                            "Group1_Normal_p": float(p_normal1),
                            "Group2_Normal_p": float(p_normal2),
                            "Test_Type": test_type,
                            "Test_Statistic": float(stat),
                            "p_value": float(p_value),
                            "Significance": significance,
                        }
                    )

                    if significance != "n.s.":
                        print(f"  步骤 {step_num}: {group1} vs {group2}")
                        print(f"    {test_type}: p = {p_value:.4f} {significance}")
                        print(f"    {group1}: {mean1:.4f} ± {std1:.4f}")
                        print(f"    {group2}: {mean2:.4f} ± {std2:.4f}")

    results_df = pd.DataFrame(all_results)
    output_file = results_dir / "targetattack_显著性分析.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 100}")
    print("分析完成！")
    print(f"{'=' * 100}")
    print(f"结果已保存至: {output_file}")
    print(f"总共进行了 {len(results_df)} 次统计检验")

    if not results_df.empty and "Significance" in results_df.columns:
        print("显著性统计:")
        for key in ["***", "**", "*", "n.s."]:
            print(f"  {key}: {(results_df['Significance'] == key).sum()} 次")

    return results_df


def run_all_statistical_tests() -> None:
    """顺序执行统计层全部任务。"""
    run_scalar_metric_tests()
    run_richclub_tests()
    run_targetattack_tests()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统计检验模块化入口")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "scalar", "richclub", "targetattack"],
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
    elif args.command == "richclub":
        run_richclub_tests()
    elif args.command == "targetattack":
        run_targetattack_tests()


if __name__ == "__main__":
    main()
