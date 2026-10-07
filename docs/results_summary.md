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

## 7. CAUEEG 复现实验（2026-10-07）

流程与参数见 `docs/caueeg_analysis_plan.md`（在组间统计前提交，提交号 bdbfefc），实现见 step14–step21，结果文件在 `results/caueeg/`（gitignored）。下文"显著"均指经过 BH-FDR 校正。

### 7.1 队列与数据质量

- **候选与预处理**：候选记录 724 条，其中 709 条进入预处理，685 条合格。不合格原因：
  - 合格 epoch 不足：18 条，其中 aMCI 15 条；
  - 坏导超过 3 个：6 条。
- **主队列**（1:1:1 匹配）：每组 99 人。
  - 平均年龄 HC 68.98、aMCI 69.21、AD 69.22 岁；三元组内年龄差 ≤2 岁。
  - AD 组中晚发型 64 例、早发型 35 例。
- **敏感性队列**（60–85 岁）：HC 156、aMCI 256、AD 176 人。
- **组间无差异的质控指标**：坏导数、epoch 剔除比例、残余肌电、峰峰值。
- **组间有差异的质控指标**：
  - 被剔除的眼动成分数（FDR=0.04）；
  - 保留的脑成分数（HC 12.6、aMCI 11.9、AD 11.8，FDR=0.03）；
  - 可用 epoch 总数：AD 较多；
  - 选中的 50 个 epoch 在时间上的跨度：aMCI 较长，说明其记录中断更频繁。
- **估计稳定性**：子段稳定性在三组间无差异。AD 并不比 HC 更不稳定，因此"AD 组估计噪声大、相关被衰减"这一解释不成立。

### 7.2 频谱（回应审稿意见 7）

AD 呈现典型的 EEG 慢化，aMCI 位于 HC 与 AD 之间：

| 指标 | HC | aMCI | AD | 组效应 FDR |
| --- | --- | --- | --- | --- |
| 相对 θ 功率 | 0.115 | 0.151 | 0.205 | <0.001 |
| 相对 α 功率 | 0.358 | 0.348 | 0.240 | <0.001 |
| θ/α 比（中位数） | 0.35 | 0.43 | 0.87 | — |
| IAF 重心频率（Hz） | 9.27 | 8.97 | 8.32 | <0.001 |

aMCI 主要表现为 θ 升高、β 降低。非周期参数的组间差异不显著。

### 7.3 主终点（AEC-c，混合模型）

- **组主效应**：8 个主终点均不显著，FDR 均 ≥0.31。
- **group×band 交互**：LOFC MeanConn、LOFC Clust_w_norm、LOFC AttackAUCdiff、sHOFC MeanConn_signed 四项显著，FDR ≤0.001。这说明组间差异随频段改变方向，在频段间平均后相互抵消。
- **按预先规定解释的分频段结果**（单位为 SD）：
  - **AEC-c LOFC 平均连接**：
    - δ 频段：aMCI 比 HC 高 0.52，AD 比 HC 高 0.48；
    - α 频段：aMCI 比 HC 低 0.32，AD 比 HC 低 0.73；
    - β 频段：AD 比 HC 低 0.50。
  - **sHOFC 平均连接**：δ 升高（aMCI 与 AD 均高约 0.52），α 频段 AD 降低 0.36。
  - **零模型归一化拓扑**：效应较小。显著的有：
    - LOFC β 加权效率（AD 低于 HC，−0.42）；
    - LOFC α 加权聚类（AD 低于 HC，−0.55）；
    - sHOFC θ 效率（−0.44）与 θ 聚类（+0.46）。
  - **攻击 AUC 减去随机攻击 AUC**：δ 频段 aMCI 与 AD 更脆弱（约 −0.45 至 −0.48），α 频段 AD 更不脆弱（+0.37）。方向随频段相反，sHOFC 上不显著。
- **功率校正**（回应审稿意见 3、7）：加入该频段 log 功率或相对功率后，主要的连接效应基本保留。分频段功率校正（`stats_power_adjusted_bandwise.csv`，含相对功率版本）是在看到交互显著后补充的分析，不在原计划内，但遵循计划中"交互显著时才解释分频段结果"的规则。
  - AEC-c α 频段 AD−HC：−0.73 降至 −0.65（校正 log 功率）或 −0.55（校正相对功率），FDR 仍 <0.01；
  - δ 频段的升高同样保留。
  - 结论：连接改变不是频谱慢化的简单重复。
- **重抽样**：100 次组内重抽样中，各主终点组主效应 p<0.05 的比例不超过 37%（LOFC MeanConn），与主分析结论一致。

### 7.4 稳健性

- **其他连接指标**：
  - AEC（原 LOFC 定义）重现了 α/β 降低、δ/θ 升高，α 频段 AD−HC 为 −0.76。
  - wPLI 只保留了 δ/θ 升高；α 频段 AD 降低的趋势不显著。
- **HOFC 与 sHOFC**：
  - 原定义 HOFC 的主终点均不显著。
  - sHOFC 有 18%–48% 的边为负，说明原定义取绝对值会改变网络（回应审稿意见 4）。
  - HOFC 的收敛慢于 LOFC：仅用 20 s 数据时，与 100 s 结果的相关为 LOFC 0.53、sHOFC 0.42、HOFC 0.31，三组相同。
- **阈值依赖**（回应审稿意见 5）：
  - AEC-c 在固定密度与 θ=0.35 阈值下均无显著组效应。
  - AEC 的 HOFC 在 θ=0.35 阈值下 AD 的图密度反而更高（+0.31 SD）。原稿中二值拓扑的方向依赖于阈值。
- **CE 的增量价值**（回应审稿意见 6）：
  - 组×排序方式交互 p≥0.99，即无论按 CE、度、介数、接近度还是 TOPSIS 排序，组间差异都相同。
  - CE 与介数排序的 AUC 差异：LOFC 上通过 TOST 等效检验，平均差 −0.0006。
  - 两者前 4 个 hub 的 Jaccard 重叠为 0.79–0.92。
  - 交叉验证攻击的组效应均不显著。
- **60–85 岁队列**（年龄与年龄² 为协变量）：LOFC MeanConn（AD−HC −0.29，FDR=0.003）、LOFC 效率与聚类经零模型归一化后显著。
- **QC 协变量**：加入后，LOFC MeanConn 的组效应 p=0.009，FDR=0.07。

### 7.5 写作边界

- 能稳健支持的结论：
  - 频谱慢化；
  - α/β 频段振幅包络连接降低、δ（和 θ）连接升高，且在去除零延迟成分（AEC-c）和功率校正后仍然成立。
- 不能作为主要结论的内容：
  - HOFC 的组间差异；
  - 二值拓扑；
  - 网络脆弱性；
  - CE 的特异性。
- 只有年龄可作协变量。诊断为临床诊断，没有生物标志物。公开数据可能包含同一受试者的多条记录。
