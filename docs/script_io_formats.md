# 脚本输入输出内容格式说明

本文档按当前 `src/` 代码总结可直接运行脚本的输入、输出路径和内容格式。
算法公式和方法依据见 `docs/algorithm_summary.md`；本文只描述文件契约。

注意：仓库中已有的 `output/`、`results/` 可能由旧版本脚本生成，字段数量可能少于当前代码。
以后重新运行脚本时，以本文档和当前 `src/` 代码为准。
当前配置包含 Delta、Theta、Alpha、Beta、Gamma 五个频段；若只重跑历史四频段数据，
对应被试级行数会少于完整五频段结果。

## 1. 共享命名约定

### 组别目录

| 组别 | 输入目录名 |
| --- | --- |
| `HC` | `data/raw/HC 256 100s/` |
| `MCI` | `data/raw/MCI 256 100s/` |
| `AD` | `data/raw/AD 256 100s/` |

### 频段目录

| 频段标签 | 目录名 | 频率范围 |
| --- | --- | --- |
| `Delta` | `1-4(Delta)` | 1-4 Hz |
| `Theta` | `4-8(Theta)` | 4-8 Hz |
| `Alpha` | `8-12(Alpha)` | 8-12 Hz |
| `Beta` | `12-30(Beta)` | 12-30 Hz |
| `Gamma` | `30-40(Gamma)` | 30-40 Hz |

### FC 类型

| FC 类型 | 文件夹名 | 矩阵方向 |
| --- | --- | --- |
| `LOFC` | `lofc` | 对称矩阵 |
| `HOFC` | `hofc` | 对称矩阵 |
| `aHOFC` | `ahofc` | 有向矩阵 |
| signed LOFC | `lofc_signed` | 对称矩阵，保留正负相关 |

默认通道数为 16，因此大多数 FC 矩阵是 `16 x 16`。底层函数通常按一般
`N x N` 方阵实现，但 `step12_fc_heatmaps.py` 的拼图逻辑固定按 16 节点绘图。

## 2. 输入 EEG 文件格式

脚本读取 EEGLAB `.set` 数据，矩阵方向统一为：

```text
channels x samples
```

当前读取逻辑支持两类 `.set`：

| 格式 | 必需内容 | 说明 |
| --- | --- | --- |
| MAT-style `.set` | `data`，可选 `nbchan`、`pnts`、`srate` | `data` 会被转换为 `float64`，若维度与 `nbchan` 相反会自动转置 |
| HDF5 `.set` 内嵌数据 | `data`，可选 `nbchan`、`pnts`、`srate` | 直接读取 HDF5 中的 `data` |
| HDF5 `.set` + `.fdt` | `nbchan`、`pnts`、`datfile` | `.fdt` 按 `float32` 读取，并 reshape 为 `pnts x nbchan` 后转置 |

`src/step01_eeg_io.py::read_eeg_set()` 默认最多读取前 `25601` 个采样点。

## 3. 脚本级 I/O 总览

### 3.1 `src/step00_bandpass_filter.py`

用途：从 `1-40` 原始数据生成配置中的频段 `.set` 文件，当前包括 Delta、Theta、
Alpha、Beta 和 Gamma。

常用命令：

```powershell
uv run python src/step00_bandpass_filter.py
uv run python src/step00_bandpass_filter.py --group HC
uv run python src/step00_bandpass_filter.py --skip-existing
```

输入：

```text
data/raw/{Group} 256 100s/1-40/*.set
```

输出：

```text
data/raw/{Group} 256 100s/{BandFolder}/{same_subject_file}.set
```

输出 `.set` 是 MAT-style 文件，包含：

| 变量 | 内容 |
| --- | --- |
| `data` | 滤波后的 `channels x samples` 数值矩阵 |
| `nbchan` | 通道数 |
| `pnts` | 采样点数 |
| `srate` | 采样率 |

### 3.2 `src/step09_batch_pipeline.py`

用途：主批处理脚本。读取频段 `.set`，计算 FC 矩阵、核心节点、rich-club、
target attack、稳定性和标量指标。

常用命令：

```powershell
uv run python src/step09_batch_pipeline.py
uv run python src/step09_batch_pipeline.py --group HC --band "1-4(Delta)"
uv run python src/step09_batch_pipeline.py --fc-only
uv run python src/step09_batch_pipeline.py --richclub-only
```

主要参数：

| 参数 | 格式 | 含义 |
| --- | --- | --- |
| `--group` | `HC` / `MCI` / `AD` | 只处理一个组 |
| `--band` | 频段目录名，如 `1-4(Delta)` | 只处理一个频段 |
| `--fc-only` | 开关 | 只写 FC 矩阵和组均值矩阵，不更新标量汇总 |
| `--richclub-only` | 开关 | 从已有 FC 矩阵重算二值归一化 rich-club |
| `--n-rand` | 整数 | rich-club 随机化次数，默认 1000 |
| `--random-attack-repeats` | 整数 | random attack 重复次数，默认 1000 |
| `--fc-stability-epochs` | 整数 | FC 稳定性分段数，默认 5；设为 0 可跳过 |
| `--seed` | 整数 | 随机种子，默认 42 |
| `--skip-existing` | 开关 | 若 LOFC 输出已存在则跳过该被试 |

输入：

```text
data/raw/{GroupDir}/{BandFolder}/*.set
```

主输出：

| 路径 | 内容格式 |
| --- | --- |
| `output/fc_matrices/{fc_type}/{group}/{band}/{subject}.csv` | 无表头 `N x N` FC 矩阵 |
| `output/fc_matrices/lofc_signed/{group}/{band}/{subject}.csv` | 无表头 `N x N` signed LOFC 矩阵，值可为负 |
| `output/mean/{group}/{band}/{group}_{band}_{fc_type}.csv` | 无表头 `16 x 16` 组均值 FC 矩阵 |
| `output/all_scalar_metrics_summary.csv` | 有表头，被试级标量指标宽表 |

核心节点输出：

```text
output/core_nodes/{fc_type}/{group}/{band}/{subject}_core_nodes.csv
```

当前代码输出字段：

```text
NodeIndex
IsCoreNode
IsCoreNode_BinaryCE
IsCoreNode_WeightedCE
IsCoreNodeTopRatio
IsCoreNodeMeanStd
CE
CentralityScore
WeightedCE
WeightedCentralityScore
WeightedCEThreshold
BinaryWeightedCoreOverlap
TopRatioThreshold
MeanStdThreshold
IsDegreeHubTopRatio
IsBetweennessHubTopRatio
IsClosenessHubTopRatio
```

旧版已生成文件可能只有 `NodeIndex,IsCoreNode` 两列；后处理脚本仍兼容这两个基础字段。

曲线类输出：

| 路径 | 内容格式 |
| --- | --- |
| `output/rich_club/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单行曲线；第 `j` 列对应 `k=j+1` |
| `output/rich_club_weighted/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单行 weighted rich-club 曲线 |
| `output/rich_club_weighted_stats/{fc_type}/{group}/{band}/{subject}.csv` | 有表头，列为 `k`、`Phi_w_observed`、`Phi_w_ranknorm`、`Phi_w_random_mean`、`Phi_w_random_std`、`Phi_w_norm`、`Phi_w_empirical_p`、`Phi_w_random_valid_n` |
| `output/target_attack/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单列曲线；第 0 行是初始全局效率归一化值 |
| `output/target_attack_controls/{strategy}/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单列曲线，`strategy` 包括 `ce`、`ce_adaptive`、`degree`、`betweenness`、`random_mean`、`random_std` |
| `output/ce_ablation_attack/{method}/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单列曲线，`method` 包括 `degree`、`betweenness`、`closeness`、`s`、`h`、`ce` |

多密度敏感性输出：

| 路径 | 粒度 | 主要字段 |
| --- | --- | --- |
| `output/core_nodes_density/{fc_type}/{group}/{band}/{subject}.csv` | 每个密度、每个节点一行 | `SubjectID`、`Group`、`Band`、`FC_Type`、`Density`、`NodeIndex`、`CE`、`S_score`、`H_entropy`、`IsCoreNode`、`IsDegreeHub`、`IsBetweennessHub`、`IsClosenessHub`、`TopRatioThreshold`、`CoreFrequency`、`CoreStability`、`StableCoreNode`、`CE_auc`、`S_auc` |
| `output/rich_club_density/{fc_type}/{group}/{band}/{subject}.csv` | 每个密度一行 | `RichClubMean_all`、`RichClubMean_high_k`、`RichClubMax`、`RichClubPeakK`、`RichClubEnhancedKCount`、`RichClubValidKCount` |
| `output/target_attack_density/{fc_type}/{group}/{band}/{subject}.csv` | 每个密度一行 | `AttackAUC`、`EarlyDrop_1`、`EarlyDrop_2`、`EarlyDrop_3`、`MaxDrop` |

质量控制和稳定性输出：

| 路径 | 主要字段 |
| --- | --- |
| `output/network_qc/{fc_type}/{group}/{band}/{subject}_qc.csv` | `SubjectID`、`Group`、`Band`、`FC_Type`、`ThresholdMode`、`ThresholdValue`、`FCMean`、`FCStd`、`FCQ05`、`FCQ25`、`FCQ50`、`FCQ75`、`FCQ95`、`Density`、`AverageDegree`、`EdgeCount`、`ComponentCount`、`LargestComponentSize`、`CoreZero_MeanStd`、`AttackInitialEglob` |
| `output/fc_stability/{group}/{band}/{subject}.csv` | `SubjectID`、`Group`、`Band`、`FC_Type`、`EpochCount`、`MatrixCorr_Mean`、`MatrixCorr_SD`、`EdgeCV_Mean`、`StableEdgeRatio` |
| `output/fc_layer_similarity/{group}/{band}/{subject}.csv` | `SubjectID`、`Group`、`Band`、`Corr_LOFC_HOFC`、`Corr_LOFC_aHOFC`、`Corr_HOFC_aHOFC`、`LayerDecoupling`、`HighOrderDecoupling` |

`output/all_scalar_metrics_summary.csv` 是被试级宽表，固定字段为：

```text
Group,Band,SubjectID
```

其余列按指标前缀展开，主要包括：

```text
Eglob_bin_{FC}
Clustering_bin_{FC}
MeanConn_{FC}
Eglob_weighted_{FC}
Clustering_weighted_{FC}
Density/AverageDegree/EdgeCount/ComponentCount/LargestComponentSize_thr035_{FC}
Eglob_bin_densityXX_{FC}
Clustering_bin_densityXX_{FC}
Density_actual_densityXX_{FC}
Eglob_bin_auc_{FC}
Clustering_bin_auc_{FC}
CoreOverlap_BinaryCE_WeightedCE_{FC}
CE_auc_mean_{FC}
CE_auc_max_{FC}
CoreNode_stability_mean_{FC}
StableCoreNode_count_{FC}
RichClub_auc_{FC}
TargetAttack_auc_across_density_{FC}
FCStability_*_{FC}
Corr_LOFC_HOFC
Corr_LOFC_aHOFC
Corr_HOFC_aHOFC
LayerDecoupling
HighOrderDecoupling
```

其中 `{FC}` 为 `LOFC`、`HOFC` 或 `aHOFC`，`XX` 为密度百分比，如 `10`、`15`、`20`。

### 3.3 `src/step10_results_postprocess.py`

用途：把 `output/` 中的被试级文件汇总为 `results/` 中的组水平和论文用表格。

常用命令：

```powershell
uv run python src/step10_results_postprocess.py
uv run python src/step10_results_postprocess.py richclub-summary
uv run python src/step10_results_postprocess.py representative-core-nodes --threshold 0.30
```

输入：主要来自 `output/`。

命令：

```text
all
richclub-summary
richclub-individual-means
targetattack-summary
targetattack-control-summaries
targetattack-features
ce-ablation-attack-features
density-sensitivity
fc-stability
layer-similarity
network-qc
richclub-features
weighted-richclub-summary
weighted-richclub-individual-means
weighted-richclub-features
weighted-richclub-stats
core-node-stability
reorganize-scalar
representative-core-nodes
```

主要输出：

| 输出文件 | 内容格式 |
| --- | --- |
| `results/richclub_summary_{FC}.csv` | 行为 `k (Node Degree)`，列为 `{Band}_{Group}` 的组均值曲线 |
| `results/richclub_individual_means_{FC}.csv` | 两行表头；一级列为频段，二级列为组别；每列为被试 rich-club 均值 |
| `results/richclub_feature_summary.csv` | 被试级 rich-club 特征宽表 |
| `results/weighted_richclub_summary_{FC}.csv` | weighted rich-club 组均值曲线 |
| `results/weighted_richclub_individual_means_{FC}.csv` | weighted rich-club 被试均值 |
| `results/weighted_richclub_feature_summary.csv` | weighted rich-club 被试级特征 |
| `results/weighted_richclub_stats_summary.csv` | 合并所有 weighted rich-club 随机归一化统计 |
| `results/targetattack_summary_{FC}.csv` | 行为 `Attack_Step`，列为 `{Band}_{Group}` 的组均值攻击曲线 |
| `results/targetattack_{strategy}_summary_{FC}.csv` | 不同攻击策略的组均值曲线 |
| `results/targetattack_feature_summary.csv` | 被试级 attack AUC、EarlyDrop、MaxDrop 和 CE-control 差异 |
| `results/ce_ablation_attack_feature_summary.csv` | CE 消融攻击特征 |
| `results/core_node_density_stability.csv` | 合并 `core_nodes_density` 明细 |
| `results/density_richclub_features.csv` | 合并多密度 rich-club 特征 |
| `results/density_targetattack_features.csv` | 合并多密度 target attack 特征 |
| `results/fc_stability_summary.csv` | 合并 FC epoch 稳定性 |
| `results/fc_layer_similarity.csv` | 合并 FC 层级相似性 |
| `results/network_qc_summary.csv` | 合并被试级网络 QC 明细 |
| `results/network_qc_group_summary.csv` | 按 FC、频段、组别、阈值模式聚合 QC |
| `results/重组数据_按指标分组.csv` | 按指标、FC、组别输出均值和 SEM 的宽表 |
| `results/core_node_subject_stability.csv` | 核心节点被试间稳定性 |
| `results/core_node_band_stability.csv` | 核心节点跨频段稳定性 |
| `results/core_node_fctype_stability.csv` | 核心节点跨 FC 类型稳定性 |
| `results/core_node_overall_stability.csv` | 核心节点总体稳定性 |
| `results/representative_core_nodes_threshold_{threshold_pct}.csv` | 代表性核心节点明细 |
| `results/representative_core_nodes_summary_{FC}_threshold_{threshold_pct}.csv` | 每个 FC 类型的代表性核心节点摘要 |

代表性核心节点明细字段：

```text
FC_Type,Band,Group,NodeIndex,ChannelName,Count,TotalSubjects,Proportion,Percentage
```

### 3.4 `src/step11_statistical_analysis.py`

用途：从标量汇总、rich-club 个体均值和 target attack 曲线生成统计检验结果。

常用命令：

```powershell
uv run python src/step11_statistical_analysis.py
uv run python src/step11_statistical_analysis.py scalar
uv run python src/step11_statistical_analysis.py targetattack
```

命令：

```text
all
scalar
two-way-anova
mixed-effects
richclub
targetattack
```

输入与输出：

| 命令 | 输入 | 输出 |
| --- | --- | --- |
| `scalar` | `output/all_scalar_metrics_summary.csv` | `results/显著性分析.csv` |
| `two-way-anova` | `output/all_scalar_metrics_summary.csv` | `results/two_way_anova_metrics.csv` |
| `mixed-effects` | `output/all_scalar_metrics_summary.csv` | `results/mixed_effects_metrics.csv` |
| `richclub` | `results/richclub_individual_means_{FC}.csv` | `results/richclub_显著性分析.csv` |
| `targetattack` | `output/target_attack/{fc_type}/.../*.csv` | `results/targetattack_显著性分析.csv`、`results/targetattack_stepwise_补充分析.csv` |

`显著性分析.csv`、`richclub_显著性分析.csv`、`targetattack_显著性分析.csv`
使用相近的统计结果格式，主要字段包括：

```text
Analysis
FC_Type
Band
Metric
Attack_Step
HC_n,HC_Mean,HC_SD,HC_Median,HC_IQR
MCI_n,MCI_Mean,MCI_SD,MCI_Median,MCI_IQR
AD_n,AD_Mean,AD_SD,AD_Median,AD_IQR
Test_Level
Test_Type
Test_Statistic
p_raw
Effect_Size
Effect_Size_Type
CI_95_Lower
CI_95_Upper
Levene_p
Equal_Variance
All_Groups_Normal
Group1,Group2
p_value
p_adj
Correction_Method
Correction_Scope
Significance_raw
Significance_adj
Significant_adj
```

`two_way_anova_metrics.csv` 主要字段：

```text
Metric,Effect,N,GroupLevels,BandLevels,F,p_raw,df_num,df_den,EtaSq_partial,p_adj,Significance_raw,Significance_adj,Correction_Method,Correction_Scope
```

`mixed_effects_metrics.csv` 是重复测量 fallback，不是真正的随机截距 MixedLM。
主要字段：

```text
Metric,Model_Type,Model,Effect,N,SubjectN,Correction_Family,F,p_raw,df_num,df_den,EtaSq_partial,p_adj,Significance_raw,Significance_adj,Correction_Method,Correction_Scope
```

其中 `Model_Type` 当前为：

```text
repeated_measures_fallback
```

### 3.5 `src/step12_fc_heatmaps.py`

用途：从组均值 FC 矩阵生成热图图片。

常用命令：

```powershell
uv run python src/step12_fc_heatmaps.py
```

输入：

```text
output/mean/{group}/{band}/{group}_{band}_{network}.csv
```

其中 `network` 为 `lofc`、`hofc`、`ahofc`。

输入矩阵格式：

```text
无表头 16 x 16 数值 CSV
```

输出：

```text
results/fc_heatmaps/lofc_heatmap.png
results/fc_heatmaps/hofc_heatmap.png
results/fc_heatmaps/ahofc_heatmap.png
results/fc_heatmaps/colorbar.png
```

脚本会把每个输入矩阵对角线设为 `NaN`，以便热图中对角线显示为空白。

## 4. 推荐运行顺序

完整生成全部输出时建议：

```powershell
uv run python src/step00_bandpass_filter.py --skip-existing
uv run python src/step09_batch_pipeline.py
uv run python src/step10_results_postprocess.py
uv run python src/step11_statistical_analysis.py
uv run python src/step12_fc_heatmaps.py
```

若只想刷新 FC 矩阵：

```powershell
uv run python src/step09_batch_pipeline.py --fc-only
```

若 FC 已经存在，只想刷新二值归一化 rich-club：

```powershell
uv run python src/step09_batch_pipeline.py --richclub-only
```
