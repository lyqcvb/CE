# 当前论文结果摘要

本结果文档对应当前 live 输出：四频段（Delta、Theta、Alpha、Beta）、两类 FC（LOFC、HOFC），不包含 aHOFC、Gamma 或 rich-club 结果。结果表来自 `output/` 和 `results/`，当前快照已清理旧实验列。

## 0. 2026-05-29 复现验证

本轮已从本地 `data/raw/` EEG `.set` 数据重新生成当前主流程结果：

```powershell
uv run python -m compileall -q src tests
uv run python src/step00_bandpass_filter.py --skip-existing
uv run python src/step09_batch_pipeline.py
uv run python src/step10_results_postprocess.py
uv run python src/step11_statistical_analysis.py
uv run python src/step12_fc_heatmaps.py
```

`step09` 共处理 352 个被试-频段组合，成功 352，失败 0。重跑后
`output/all_scalar_metrics_summary.csv` 为 352 行、101 列；
`results/显著性分析.csv` 为 779 行、55 列；`results/targetattack_显著性分析.csv`
为 20 行、55 列。主 scalar 的 primary omnibus FDR 显著项为 13 个，
TargetAttack_AUC omnibus FDR 显著项为 4 个。

重跑前快照备份在 `D:\Paper_code\CE_repro_backup_20260529_153050`。49 个关键输出中
45 个哈希完全一致；`output/all_scalar_metrics_summary.csv` 虽有字节级哈希差异，
但按表格数据精确相等。热图 PNG 字节一致；SVG 差异来自 Matplotlib 重生成时间和内部元数据。

## 1. 数据规模

`output/all_scalar_metrics_summary.csv` 当前为 352 行、101 列。行数对应 88 名被试 × 4 个频段；标量汇总列中没有 aHOFC 或 rich-club 字段。

主要统计表规模如下：

| 文件 | 行 | 列 | 用途 |
| --- | ---: | ---: | --- |
| `results/显著性分析.csv` | 779 | 55 | 主 scalar 三组检验 |
| `results/targetattack_显著性分析.csv` | 20 | 55 | TargetAttack_AUC 主检验 |
| `results/targetattack_stepwise_补充分析.csv` | 286 | 56 | target attack 分步曲线补充检验 |
| `results/two_way_anova_metrics.csv` | 490 | 15 | group、band、group × band 探索性检验 |
| `results/mixed_effects_metrics.csv` | 294 | 17 | repeated-measures fallback 探索性检验 |
| `results/fc_layer_similarity.csv` | 352 | 5 | LOFC/HOFC 层级相似性 |
| `results/fc_stability_summary.csv` | 704 | 9 | LOFC/HOFC epoch 稳定性 |
| `results/network_qc_summary.csv` | 5632 | 20 | 网络 QC 明细 |
| `results/network_qc_group_summary.csv` | 192 | 13 | 网络 QC 组水平汇总 |
| `results/targetattack_feature_summary.csv` | 704 | 47 | target attack 控制策略特征 |
| `results/ce_ablation_attack_feature_summary.csv` | 704 | 43 | CE 消融攻击特征 |
| `results/core_node_density_stability.csv` | 78848 | 19 | 多密度核心节点明细 |
| `results/density_targetattack_features.csv` | 4928 | 10 | 多密度 target attack 特征 |

代表性核心节点明细为 `representative_core_nodes_threshold_20/30/40.csv`，当前分别为 286、120、25 行，均为 9 列；对应的 `representative_core_nodes_summary_{LOFC,HOFC}_threshold_{20,30,40}.csv` 每个文件为 12 行、4 列。各 target attack 组均值曲线与控制曲线汇总文件均为 17 行、13 列。

## 2. 主结果概括

全部 scalar 统计行中，FDR 校正后显著行为 319/779，其中 omnibus 行 100 条、post-hoc 行 219 条。论文主线应优先报告 primary scalar omnibus 层级：当前该层级 FDR 显著项为 13 个。显著结果集中在 Beta、Alpha、Theta 和 Delta，Beta 最集中。反复出现的指标族包括二值全局效率、二值聚类系数、连接强度、密度敏感性 AUC、weighted topology、网络 QC、FC stability 和 LOFC/HOFC layer similarity。

当前最稳妥的论文主线是：

- HC 相比 AD 通常表现出更高的连接强度、网络整合能力和聚类水平。
- MCI 多数指标位于 HC 与 AD 之间，但不应写成所有指标都严格单调。
- LOFC 和 HOFC 都提供疾病相关网络差异，HOFC 在部分 Delta/Beta 指标上提供额外拓扑轮廓证据。
- 结果解释应以 `MeanConn_*`、`Eglob_bin_*`、`Clustering_bin_*` 和 `TargetAttack_AUC` 为主，density AUC、weighted metrics、FC stability、layer similarity、CE ablation、core-node recurrence 和 network QC 作为敏感性或描述性证据。

## 3. Target Attack

`results/targetattack_显著性分析.csv` 当前 20 行，其中 14 行 FDR 校正后显著，包含 4 条 omnibus 和 10 条 post-hoc。显著结果来自 LOFC 和 HOFC，主要集中在 Beta、Alpha 和 Delta。`TargetAttack_AUC` 仍是主 target-attack 结论指标；`targetattack_stepwise_补充分析.csv` 当前 286 行，其中 130 行 FDR 校正后显著，只用于说明曲线阶段差异，不替代 AUC 主检验。

## 4. FC 稳定性和层级相似性

`results/fc_stability_summary.csv` 显示 LOFC 的 epoch 稳定性整体高于 HOFC。按 FC 类型聚合时，LOFC 的 `MatrixCorr_Mean` 为 0.829，`EdgeCV_Mean` 为 0.383，`StableEdgeRatio` 为 0.337；HOFC 分别为 0.566、0.469 和 0.222。

按组别看，LOFC 的 `MatrixCorr_Mean` 为 HC 0.851、MCI 0.852、AD 0.783；对应 `StableEdgeRatio` 为 0.380、0.353、0.276。HOFC 的 `MatrixCorr_Mean` 为 HC 0.600、MCI 0.603、AD 0.492；对应 `StableEdgeRatio` 为 0.264、0.236、0.165。

`results/fc_layer_similarity.csv` 中，`Corr_LOFC_HOFC` 按组均值为 AD 0.458、HC 0.329、MCI 0.451；`LayerDecoupling` 分别为 AD 0.542、HC 0.671、MCI 0.549。该结果适合用于说明 HOFC 与 LOFC 不是完全重复的信息层，但不应夸大为独立诊断模型。

## 5. 探索性模型

`two_way_anova_metrics.csv` 中 284/490 行达到 FDR 校正显著，主要包括 Group、Band、Model_Additive、Model_Full 和少量 Group:Band 交互。`mixed_effects_metrics.csv` 中 154/294 行达到 FDR 校正显著；该表为 repeated-measures fallback，不应描述为完整随机截距混合效应模型，其用途是检查组间均值、频段内变化和交互趋势是否与主三组检验一致。

## 6. 论文写作边界

当前正文不应写入 aHOFC、rich-club 或 Gamma 结果。若需要说明清理范围，可只在代码/仓库文档中说明这些旧实验已移出主流程；论文正文以实际 Methods 中存在的 LOFC、HOFC、CE/Hub、target attack、FC stability 和 layer similarity 为准。

图件方面，当前 Results 主热图来自 `output/mean/**` 的 LOFC/HOFC 组均值矩阵，由 `src/step12_fc_heatmaps.py` 或 `writing/figures/results/fig3_fc_heatmaps.py` 生成。已删除的 Methods 电极蒙太奇图不再属于 active manuscript。
