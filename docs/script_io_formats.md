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

## CAUEEG 复现流程（step14–step21）

数据集根目录默认为 `F:\数据\caueeg-dataset`（只读），可用环境变量 `CAUEEG_ROOT` 或参数 `--caueeg-root` 覆盖。输出写到 `{project_root}/{data,output,results}/caueeg/`，`project_root` 默认为 `E:\Paper_code\CE`，可用 `CE_PROJECT_ROOT` 或 `--project-root` 覆盖。这些目录都在 gitignore 中。

```powershell
uv run python src/step14_caueeg_events.py
uv run python src/step15_caueeg_preprocess.py --jobs 10 --skip-existing
uv run python src/step16_caueeg_cohort.py
uv run python src/step17_caueeg_spectral.py
uv run python src/step18_caueeg_connectivity.py --skip-existing
uv run python src/step19_caueeg_network.py
uv run python src/step20_caueeg_statistics.py
uv run python src/step21_caueeg_report.py
```

| 脚本 | 输入 | 输出 |
| --- | --- | --- |
| `step14_caueeg_events.py` | `annotation.json`、`dementia-no-overlap.json`、`event/*.json`、EDF 头 | `data/caueeg/events_corrected/{serial}.json`：`n_samples`、`pause_boundaries`、`ec_spans`、`usable_spans`、`excluded`、标记位（单位均为 200 Hz 采样点，已做暂停校正）；`results/caueeg/candidate_pool.csv`；`preprocess_serials.txt`（只含编号） |
| `step15_caueeg_preprocess.py` | `preprocess_serials.txt`、EDF、校正后的事件 | `data/caueeg/preprocessed/{serial}.npz`：`data`（19×T，float32，µV）、`piece_offsets`、`piece_bounds`、`epoch_starts`、`epoch_piece`、`sfreq`、`channels`；`results/caueeg/preproc_logs/{serial}.json`（含 ICLabel 概率）；`preproc_log.csv`。`status` 取值为 `ok`、`too_many_bad_channels`、`insufficient_epochs` 或 `error: ...` |
| `step16_caueeg_cohort.py` | `candidate_pool.csv`、`preproc_log.csv` | `cohort_matched_111.csv`（triplet、group、age、serial、symptom、split）、`cohort_resamples.csv`（含 draw 1–100）、`cohort_60_85.csv`、`cohort_attrition.csv`、`cohort_summary.csv` |
| `step17_caueeg_spectral.py` | `preprocessed/*.npz` | `spectral_global.csv`：`abs_/logabs_/rel_{band}`、`theta_alpha_ratio`、`iaf_peak`、`iaf_cog`、`ap_offset`、`ap_exponent`、`ap_r2`；`spectral_channels.csv`；`spectral_psd_global.csv`（列为频率，单位 µV²/Hz） |
| `step18_caueeg_connectivity.py` | `preprocessed/*.npz`、`cohort_matched_111.csv`（用于决定收敛分析的对象） | `output/caueeg/fc/{aec,aecc,wpli}/{band}/{serial}.npz`：`full`（19×19，带符号）、`chunks`（5×19×19）、`halves`（2×19×19）；`results/caueeg/hofc_convergence_subject.csv` |
| `step19_caueeg_network.py` | `fc/**.npz` | `output/caueeg/network_metrics.csv`（每行对应 serial×measure×band×fc，指标列见脚本 docstring）；`attack_curves.csv`（仅 AEC-c，包含 ce/betweenness/random 三种排序，20 步） |
| `step20_caueeg_statistics.py` | 上述全部 | `results/caueeg/stats_*.csv`、`hofc_convergence_summary.csv`、`group_descriptives.csv` |
| `step21_caueeg_report.py` | `stats_*.csv` 等 | `results/caueeg/figures/fig1–fig7*.png`、`report_tables.md` |

组标签在代码中写作 `HC`/`MCI`/`AD`，其中 `MCI` 指遗忘型 MCI（aMCI）。
