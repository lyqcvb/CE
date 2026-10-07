"""
Step 21: Figures and auto-generated Markdown tables for the CAUEEG results.

Figures (results/caueeg/figures/*.png, 200 dpi):
fig1_cohort_age, fig2_spectrum, fig3_primary_forest, fig4_measure_comparison,
fig5_attack_curves, fig6_hofc_convergence, fig7_power_adjustment.
Tables: results/caueeg/report_tables.md
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import caueeg_config as cfg  # noqa: E402

# Reference categorical palette, first three slots (validated for all pairs).
COLORS = {"HC": "#2a78d6", "MCI": "#eb6834", "AD": "#1baf7a"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CONTRAST_COLORS = {"MCI_vs_HC": "#eb6834", "AD_vs_HC": "#1baf7a", "AD_vs_MCI": "#2a78d6"}
CONTRAST_LABEL = {"MCI_vs_HC": "aMCI − HC", "AD_vs_HC": "AD − HC", "AD_vs_MCI": "AD − aMCI"}
METRIC_LABEL = {
    "MeanConn": "平均连接", "MeanConn_signed": "平均连接(带符号)", "Eglob_w_norm": "加权效率/零模型",
    "Clust_w_norm": "加权聚类/零模型", "AttackAUCdiff_ce_eff": "CE 攻击 AUC−随机",
}

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9, "legend.frameon": False,
    "lines.linewidth": 2,
})


def _legend(ax, **kw):
    handles = [plt.Line2D([], [], color=COLORS[g], marker="o", lw=2, label=cfg.GROUP_DISPLAY[g]) for g in cfg.GROUPS]
    ax.legend(handles=handles, **kw)


def fig_cohort(paths, matched):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    bins = np.arange(matched["age"].min() - 0.5, matched["age"].max() + 1.5, 2)
    for offset, g in zip((-0.6, 0, 0.6), cfg.GROUPS):
        ages = matched.loc[matched["group"] == g, "age"]
        counts, edges = np.histogram(ages, bins=bins)
        ax.bar(edges[:-1] + 1 + offset, counts, width=0.55, color=COLORS[g],
               label=f"{cfg.GROUP_DISPLAY[g]} (n={len(ages)}, {ages.mean():.1f}±{ages.std():.1f})")
    ax.set_xlabel("年龄（岁）")
    ax.set_ylabel("人数")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("1:1:1 年龄匹配队列", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(paths.figures / "fig1_cohort_age.png", dpi=200)
    plt.close(fig)


def fig_spectrum(paths, matched, spectral):
    psd = pd.read_csv(paths.results / "spectral_psd_global.csv", dtype={"serial": str})
    psd = matched[["serial", "group"]].merge(psd, on="serial")
    freqs = np.array([float(c) for c in psd.columns[2:]])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [1.4, 1]})
    for g in cfg.GROUPS:
        values = np.log10(psd.loc[psd["group"] == g].iloc[:, 2:].to_numpy())
        mean, sem = values.mean(0), values.std(0, ddof=1) / np.sqrt(len(values))
        axes[0].plot(freqs, mean, color=COLORS[g], label=cfg.GROUP_DISPLAY[g])
        axes[0].fill_between(freqs, mean - 1.96 * sem, mean + 1.96 * sem, color=COLORS[g], alpha=0.15, lw=0)
    axes[0].set_xlabel("频率 (Hz)")
    axes[0].set_ylabel("log10 功率 (µV²/Hz)")
    axes[0].set_xlim(1, 45)
    axes[0].set_title("通道平均功率谱（均值±95% CI）", loc="left", color=INK)
    _legend(axes[0], loc="upper right")
    d = matched[["serial", "group"]].merge(spectral, on="serial")
    x = np.arange(len(cfg.BANDS))
    for k, g in enumerate(cfg.GROUPS):
        sub = d[d["group"] == g]
        means = [sub[f"rel_{b}"].mean() for b in cfg.BANDS]
        sems = [sub[f"rel_{b}"].std() / np.sqrt(len(sub)) for b in cfg.BANDS]
        axes[1].bar(x + (k - 1) * 0.27, means, width=0.25, color=COLORS[g], yerr=1.96 * np.array(sems),
                    error_kw={"ecolor": INK_2, "lw": 1})
    axes[1].set_xticks(x, cfg.BAND_NAMES)
    axes[1].set_ylabel("相对功率")
    axes[1].set_title("相对功率", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig2_spectrum.png", dpi=200)
    plt.close(fig)


def _forest(ax, table, labels, title):
    y = np.arange(len(table))[::-1]
    for k, key in enumerate(CONTRAST_LABEL):
        offs = (1 - k) * 0.22
        ax.errorbar(table[f"{key}_est"], y + offs,
                    xerr=[table[f"{key}_est"] - table[f"{key}_ci_lo"], table[f"{key}_ci_hi"] - table[f"{key}_est"]],
                    fmt="o", ms=4, color=CONTRAST_COLORS[key], ecolor=CONTRAST_COLORS[key], elinewidth=1.5,
                    label=CONTRAST_LABEL[key])
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("组间差（标准化，SD 单位，95% CI）")
    ax.set_title(title, loc="left", color=INK, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=3, fontsize=8)


def _endpoint_labels(table):
    labels = []
    for _, row in table.iterrows():
        star = ""
        if "group_p_fdr" in row and pd.notna(row["group_p_fdr"]):
            star = " *" if row["group_p_fdr"] < 0.05 else ""
        labels.append(f"{row['fc']} {METRIC_LABEL.get(row['metric'], row['metric'])}{star}")
    return labels


def fig_primary(paths, primary):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _forest(ax, primary, _endpoint_labels(primary), "AEC-c 主终点（混合模型，校正年龄；* 组效应 FDR<0.05）")
    fig.tight_layout()
    fig.savefig(paths.figures / "fig3_primary_forest.png", dpi=200)
    plt.close(fig)


def fig_measures(paths, desc):
    fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharex=True)
    names = {"aec": "AEC（原 LOFC 定义）", "aecc": "AEC-c（正交化）", "wpli": "wPLI（去偏）"}
    for row, fc in enumerate(("LOFC", "sHOFC")):
        metric = "MeanConn" if fc == "LOFC" else "MeanConn_signed"
        for col, measure in enumerate(cfg.MEASURES):
            ax = axes[row, col]
            sub = desc[(desc["measure"] == measure) & (desc["fc"] == fc) & (desc["metric"] == metric)]
            x = np.arange(len(cfg.BANDS))
            for k, g in enumerate(cfg.GROUPS):
                s = sub[sub["group"] == g].set_index("band").reindex(cfg.BAND_NAMES)
                ax.errorbar(x + (k - 1) * 0.12, s["mean"], yerr=1.96 * s["std"] / np.sqrt(s["count"]),
                            fmt="o-", ms=4, color=COLORS[g], lw=1.5, elinewidth=1)
            ax.set_xticks(x, cfg.BAND_NAMES)
            if row == 0:
                ax.set_title(names[measure], loc="left", color=INK)
            if col == 0:
                ax.set_ylabel(f"{fc} 平均连接" + ("(带符号)" if fc == "sHOFC" else ""))
    _legend(axes[0, 2], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig4_measure_comparison.png", dpi=200)
    plt.close(fig)


def fig_attack(paths, matched):
    curves = pd.read_csv(paths.output / "attack_curves.csv", dtype={"serial": str})
    curves = curves[curves["fc"] == "LOFC"].merge(matched[["serial", "group"]], on="serial")
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    for ax, band in zip(axes, cfg.BAND_NAMES):
        sub = curves[curves["band"] == band]
        for g in cfg.GROUPS:
            for order, style in (("ce", "-"), ("random", ":")):
                m = sub[(sub["group"] == g) & (sub["order"] == order)].groupby("step")["value"].mean()
                ax.plot(m.index, m.values, style, color=COLORS[g], lw=2 if order == "ce" else 1.5)
        ax.set_title(band, loc="left", color=INK)
        ax.set_xlabel("移除节点数")
    axes[0].set_ylabel("相对全局效率")
    handles = [plt.Line2D([], [], color=COLORS[g], lw=2, label=cfg.GROUP_DISPLAY[g]) for g in cfg.GROUPS]
    handles += [plt.Line2D([], [], color=INK_2, lw=2, label="CE 排序"),
                plt.Line2D([], [], color=INK_2, lw=1.5, ls=":", label="随机排序")]
    axes[-1].legend(handles=handles, fontsize=7, loc="upper right")
    fig.suptitle("AEC-c LOFC 目标攻击（固定密度 0.20–0.40 平均）", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig5_attack_curves.png", dpi=200)
    plt.close(fig)


def fig_convergence(paths):
    path = paths.results / "hofc_convergence_summary.csv"
    if not path.exists():
        return
    conv = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, (col, title) in zip(axes, (("r_lofc", "LOFC"), ("r_hofc", "HOFC（绝对值）"), ("r_hofc_signed", "sHOFC"))):
        for g in cfg.GROUPS:
            m = conv[conv["group"] == g].groupby("n_epochs")[f"{col}_mean"].mean()
            ax.plot(m.index * cfg.EPOCH_SEC, m.values, "o-", ms=4, color=COLORS[g])
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("数据长度 (s)")
    axes[0].set_ylabel("与 100 s 矩阵的相关 r")
    _legend(axes[-1], loc="lower right", fontsize=8)
    fig.suptitle("AEC-c 估计收敛性（四频段平均，bootstrap）", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig6_hofc_convergence.png", dpi=200)
    plt.close(fig)


def fig_power(paths, primary, adjusted):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _forest(axes[0], primary, _endpoint_labels(primary), "未校正功率")
    adj = adjusted[adjusted["adjustment"] == "band_logpower"].reset_index(drop=True)
    _forest(axes[1], adj, _endpoint_labels(adj), "校正该频段 log 功率")
    axes[1].get_legend().remove()
    fig.tight_layout()
    fig.savefig(paths.figures / "fig7_power_adjustment.png", dpi=200)
    plt.close(fig)


def _fmt_p(p):
    if pd.isna(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def markdown_tables(paths, t) -> str:
    lines = ["# CAUEEG 结果表（自动生成）", ""]
    lines += ["## 队列", "", t["cohort_summary"].round(2).to_markdown(index=False), ""]
    lines += ["## 预处理去向", "", t["attrition"].to_markdown(index=False), ""]
    prim = t["primary"]
    lines += ["## 主终点（AEC-c，混合模型）", "", "| 网络 | 指标 | n | χ²(2) | p | p_FDR | aMCI−HC | AD−HC | AD−aMCI | 交互 p_FDR |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in prim.iterrows():
        ci = lambda k: f"{r[k + '_est']:.2f} [{r[k + '_ci_lo']:.2f}, {r[k + '_ci_hi']:.2f}]"  # noqa: E731
        lines.append(f"| {r['fc']} | {r['metric']} | {int(r['n_subjects'])} | {r['group_chi2']:.2f} | {_fmt_p(r['group_p'])} | "
                     f"{_fmt_p(r['group_p_fdr'])} | {ci('MCI_vs_HC')} | {ci('AD_vs_HC')} | {ci('AD_vs_MCI')} | {_fmt_p(r['interaction_p_fdr'])} |")
    lines.append("")
    for name, title in (("power", "功率校正"), ("robust", "稳健性"), ("spectral", "频谱"), ("qc", "数据质量"),
                        ("tost", "CE vs 介数（TOST）"), ("order", "组 × 排序方式交互"), ("resampling", "重抽样稳定性"),
                        ("bandwise", "分频段对比（OLS）")):
        if name in t and not t[name].empty:
            lines += [f"## {title}", "", t[name].round(4).to_markdown(index=False), ""]
    return "\n".join(lines)


def main() -> None:
    parser = cfg.add_common_args(argparse.ArgumentParser(description="CAUEEG 图表与结果表"))
    args = parser.parse_args()
    paths = cfg.paths_from_args(args)
    paths.figures.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(paths.results / name, dtype={"serial": str})  # noqa: E731
    matched = read("cohort_matched_111.csv")
    spectral = read("spectral_global.csv")
    primary = read("stats_primary.csv")
    adjusted = read("stats_power_adjusted.csv")
    desc = read("group_descriptives.csv")

    fig_cohort(paths, matched)
    fig_spectrum(paths, matched, spectral)
    fig_primary(paths, primary)
    fig_measures(paths, desc)
    fig_attack(paths, matched)
    fig_convergence(paths)
    fig_power(paths, primary, adjusted)

    tables = {
        "cohort_summary": read("cohort_summary.csv"),
        "attrition": pd.read_csv(paths.results / "cohort_attrition.csv"),
        "primary": primary,
        "power": adjusted[["fc", "metric", "adjustment", "group_p", "group_p_fdr", "AD_vs_HC_est", "AD_vs_HC_attenuation",
                           "MCI_vs_HC_est", "MCI_vs_HC_attenuation"]],
        "robust": read("stats_robustness.csv")[["family", "measure", "fc", "metric", "n_subjects", "group_p", "group_p_fdr",
                                               "MCI_vs_HC_est", "AD_vs_HC_est", "AD_vs_MCI_est"]],
        "spectral": read("stats_spectral.csv")[["variable", "HC_mean", "MCI_mean", "AD_mean", "kruskal_p_fdr", "ols_group_p_fdr",
                                               "ols_AD_vs_HC_est", "ols_MCI_vs_HC_est", "missing_HC_MCI_AD"]],
        "qc": read("stats_qc.csv")[["variable", "HC_mean", "MCI_mean", "AD_mean", "kruskal_p", "kruskal_p_fdr", "ols_group_p_fdr"]],
        "tost": read("stats_ce_vs_betweenness.csv"),
        "order": read("stats_order_interaction.csv"),
        "bandwise": read("stats_bandwise.csv")[["fc", "metric", "band", "group_p", "group_p_fdr", "MCI_vs_HC_est", "AD_vs_HC_est",
                                                "AD_vs_MCI_est"]],
    }
    res_path = paths.results / "stats_resampling_summary.csv"
    if res_path.exists():
        tables["resampling"] = pd.read_csv(res_path)
    (paths.results / "report_tables.md").write_text(markdown_tables(paths, tables), encoding="utf-8")
    print(f"figures in {paths.figures}")


if __name__ == "__main__":
    main()
