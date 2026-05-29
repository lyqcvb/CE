# 脚本输入输出契约

本文档描述当前主流程的输入输出格式。当前范围为四频段、LOFC/HOFC 两类 FC；旧的 aHOFC、Gamma 和 rich-club 输出不属于当前契约。

## 完整复现命令与当前快照

2026-05-29 已用本地 `data/raw/` EEG `.set` 数据完整重跑当前主流程：

```powershell
uv run python -m compileall -q src tests
uv run python src/step00_bandpass_filter.py --skip-existing
uv run python src/step09_batch_pipeline.py
uv run python src/step10_results_postprocess.py
uv run python src/step11_statistical_analysis.py
uv run python src/step12_fc_heatmaps.py
```

当前验证结果：

| 项目 | 当前值 |
| --- | --- |
| `step09` 被试-频段组合 | 352 |
| `step09` 成功 / 失败 | 352 / 0 |
| `output/all_scalar_metrics_summary.csv` | 352 行、101 列 |
| `results/显著性分析.csv` | 779 行、55 列 |
| `results/targetattack_显著性分析.csv` | 20 行、55 列 |
| primary scalar omnibus FDR 显著项 | 13 |
| `TargetAttack_AUC` omnibus FDR 显著项 | 4 |

重跑前快照备份在 `D:\Paper_code\CE_repro_backup_20260529_153050`。49 个关键输出中 45 个哈希一致；`output/all_scalar_metrics_summary.csv` 的表格数据精确一致，但字节哈希不同。`results/fc_heatmaps/*.png` 字节一致，`*.svg` 的哈希差异来自重生成时间和内部元数据。

## 通用路径

| 路径 | 用途 |
| --- | --- |
| `data/raw/{Group} 256 100s/1-40/*.set` | 原始 1-40 Hz EEGLAB 输入 |
| `data/raw/{Group} 256 100s/{band}/*.set` | `step00` 生成的四频段输入 |
| `output/` | 被试级中间结果 |
| `results/` | 组水平汇总、统计和图像 |

`Group` 为 `HC`、`MCI`、`AD`。`band` 为 `1-4(Delta)`、`4-8(Theta)`、`8-12(Alpha)`、`12-30(Beta)`。

## step00_bandpass_filter.py

用途：从 1-40 Hz `.set` 数据生成配置频段 `.set`。

```powershell
uv run python src/step00_bandpass_filter.py --skip-existing
uv run python src/step00_bandpass_filter.py --group HC --skip-existing
```

输入：

```text
data/raw/{Group} 256 100s/1-40/{subject}.set
```

输出：

```text
data/raw/{Group} 256 100s/{band}/{subject}.set
```

## step09_batch_pipeline.py

用途：批量计算 FC、标量指标、核心节点、target attack、稳定性和组均值矩阵。

```powershell
uv run python src/step09_batch_pipeline.py
uv run python src/step09_batch_pipeline.py --group HC --band "1-4(Delta)"
uv run python src/step09_batch_pipeline.py --fc-only
```

主要输出：

```text
output/fc_matrices/{lofc,hofc}/{group}/{band}/{subject}.csv
output/fc_matrices/lofc_signed/{group}/{band}/{subject}.csv
output/core_nodes/{lofc,hofc}/{group}/{band}/{subject}_core_nodes.csv
output/core_nodes_density/{lofc,hofc}/{group}/{band}/{subject}.csv
output/target_attack/{lofc,hofc}/{group}/{band}/{subject}.csv
output/target_attack_controls/{strategy}/{lofc,hofc}/{group}/{band}/{subject}.csv
output/target_attack_density/{lofc,hofc}/{group}/{band}/{subject}.csv
output/ce_ablation_attack/{method}/{lofc,hofc}/{group}/{band}/{subject}.csv
output/network_qc/{lofc,hofc}/{group}/{band}/{subject}_qc.csv
output/fc_stability/{group}/{band}/{subject}.csv
output/fc_layer_similarity/{group}/{band}/{subject}.csv
output/mean/{group}/{band}/{group}_{band}_{lofc|hofc}.csv
output/all_scalar_metrics_summary.csv
```

当前 `output/all_scalar_metrics_summary.csv` 的 live 形态为 352 行、101 列。核心标识列为 `Group`、`Band`、`SubjectID`；主要指标列包含 `Eglob_bin_*`、`Clustering_bin_*`、`MeanConn_*`、weighted metrics、QC metrics、density AUC、CE/Hub overlap、target attack density AUC、FC stability 和 layer similarity。

## step10_results_postprocess.py

用途：把被试级输出汇总成论文可读的结果表。

```powershell
uv run python src/step10_results_postprocess.py
```

主要输出：

```text
results/targetattack_summary_LOFC.csv
results/targetattack_summary_HOFC.csv
results/targetattack_{ce,ce_adaptive,degree,betweenness,random_mean,random_std}_summary_{LOFC,HOFC}.csv
results/targetattack_feature_summary.csv
results/ce_ablation_attack_feature_summary.csv
results/core_node_density_stability.csv
results/density_targetattack_features.csv
results/fc_layer_similarity.csv
results/fc_stability_summary.csv
results/network_qc_summary.csv
results/network_qc_group_summary.csv
results/core_node_subject_stability.csv
results/core_node_band_stability.csv
results/core_node_fctype_stability.csv
results/core_node_overall_stability.csv
results/representative_core_nodes_threshold_{20,30,40}.csv
results/representative_core_nodes_summary_{LOFC,HOFC}_threshold_{20,30,40}.csv
results/重组数据_按指标分组.csv
```

## step11_statistical_analysis.py

用途：对 scalar metrics、target attack 和探索性模型进行统计检验。

```powershell
uv run python src/step11_statistical_analysis.py
```

主要输出：

```text
results/显著性分析.csv
results/two_way_anova_metrics.csv
results/mixed_effects_metrics.csv
results/targetattack_显著性分析.csv
results/targetattack_stepwise_补充分析.csv
```

当前 live 表形态：

| 文件 | 行 | 列 |
| --- | ---: | ---: |
| `results/显著性分析.csv` | 779 | 55 |
| `results/two_way_anova_metrics.csv` | 490 | 15 |
| `results/mixed_effects_metrics.csv` | 294 | 17 |
| `results/targetattack_显著性分析.csv` | 20 | 55 |
| `results/targetattack_stepwise_补充分析.csv` | 286 | 56 |

## step12_fc_heatmaps.py

用途：从 LOFC/HOFC 组均值矩阵生成热图。

```powershell
uv run python src/step12_fc_heatmaps.py
```

输入：

```text
output/mean/{group}/{band}/{group}_{band}_{lofc|hofc}.csv
```

输出：

```text
results/fc_heatmaps/lofc_hofc_heatmap.png
results/fc_heatmaps/lofc_hofc_heatmap.svg
results/fc_heatmaps/lofc_heatmap.png
results/fc_heatmaps/lofc_heatmap.svg
results/fc_heatmaps/hofc_heatmap.png
results/fc_heatmaps/hofc_heatmap.svg
results/fc_heatmaps/colorbar.png
```
