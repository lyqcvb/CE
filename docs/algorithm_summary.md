# CE 脑网络分析算法说明

本文档记录当前论文口径下的代码实现。主流程只包含 Delta、Theta、Alpha、Beta 四个频段和 LOFC、HOFC 两类功能连接；aHOFC、Gamma 频段和 rich-club 分析已从主流程、兼容导出层、后处理和统计入口中移除。

## 1. 数据和频段

原始输入位于 `data/raw/{Group} 256 100s/1-40/`，`step00_bandpass_filter.py` 从 1-40 Hz 数据生成以下频段：

| 频段 | 目录 | 范围 |
| --- | --- | --- |
| Delta | `1-4(Delta)` | 1-4 Hz |
| Theta | `4-8(Theta)` | 4-8 Hz |
| Alpha | `8-12(Alpha)` | 8-12 Hz |
| Beta | `12-30(Beta)` | 12-30 Hz |

当前配置来自 `src/pipeline_config.py`。后续批处理、后处理、统计和热图均读取同一组配置，避免结果表中混入旧频段或旧 FC 类型。

## 2. 功能连接

### 2.1 LOFC

LOFC 使用 Hilbert 包络相关。对每个通道信号 \(x_i(t)\) 取解析信号并得到振幅包络：

$$
A_i(t)=|\mathcal{H}(x_i(t))|
$$

通道 \(i,j\) 的 LOFC 为包络 Pearson 相关的绝对值：

$$
\operatorname{LOFC}_{ij}=|\operatorname{corr}(A_i,A_j)|
$$

主 LOFC 矩阵保存到 `output/fc_matrices/lofc/`。敏感性检查还保存 signed LOFC 到 `output/fc_matrices/lofc_signed/`，但主图论和统计口径使用绝对值 LOFC。

### 2.2 HOFC

HOFC 使用 LOFC 的拓扑连接轮廓。先对 LOFC 做 Fisher-z 变换，裁剪常数使用代码中的 `np.finfo(np.float64).eps`，即约 \(2.22\times10^{-16}\)：

$$
z_{ij}=\operatorname{arctanh}(\operatorname{clip}(\operatorname{LOFC}_{ij}, -1+\epsilon, 1-\epsilon))
$$

比较节点 \(i\) 和 \(j\) 时，排除自身及彼此对应条目后，计算两个 LOFC 轮廓的绝对 Pearson 相关：

$$
\operatorname{HOFC}_{ij}=|\operatorname{corr}(z_{-i,-j,i}, z_{-i,-j,j})|
$$

HOFC 输出为对称矩阵，保存到 `output/fc_matrices/hofc/`。

## 3. 图论和连接强度指标

固定阈值二值图使用 `THRESHOLD_DEFAULT = 0.35`：

$$
G_{ij}=\mathbf{1}(W_{ij}\geq 0.35), \quad G_{ii}=0
$$

当前 scalar 汇总包括：

- `MeanConn_{FC}`：平均连接强度；LOFC/HOFC 使用上三角非对角权重。
- `Eglob_bin_{FC}`：二值全局效率。
- `Clustering_bin_{FC}`：二值聚类系数均值。
- `Eglob_weighted_{FC}`、`Clustering_weighted_{FC}`：加权网络敏感性指标。
- `Density_thr035_{FC}`、`AverageDegree_thr035_{FC}`、`EdgeCount_thr035_{FC}` 等网络 QC 指标。
- `Eglob_bin_densityXX_{FC}`、`Clustering_bin_densityXX_{FC}` 和对应 AUC：固定密度敏感性结果，密度为 0.10 到 0.40。

`output/all_scalar_metrics_summary.csv` 当前为 352 行、101 列，覆盖 88 名被试、4 个频段和 2 个 FC 类型相关指标；表中没有 aHOFC 或 rich-club 列。

## 4. CE/Hub 节点

`step05_hub_detection.py` 计算基于 centrality entropy 的核心节点。对节点 \(i\)，先结合度、介数和接近中心性得到中心性强度 \(S_i\)，再在邻域内计算归一化 Shannon 熵：

$$
H_i^{norm}=
\frac{-\sum_{k\in \Gamma(i),\ p_{ik}>0} p_{ik}\log_2 p_{ik}}
{\log_2 |\Gamma(i)|}
$$

其中：

$$
p_{ik}=\frac{S_k}{\sum_{m\in\Gamma(i)}S_m}
$$

CE 分数为：

$$
\operatorname{CE}_i=S_iH_i^{norm}
$$

主核心节点使用固定 top ratio，默认 `HUB_TOP_RATIO = 0.20`。输出包括固定阈值核心节点、固定密度核心节点、代表性核心节点和二值 CE 与 weighted CE 的重叠指标。

## 5. Target Attack 和稳定性

Target attack 在二值图上按 CE、degree、betweenness、random mean、adaptive CE 等策略逐步移除节点，并记录归一化全局效率曲线。初始全局效率大于 0 时，主 CE attack 曲线包含 baseline，即 \(\operatorname{TA}(0)=1\)，随后记录逐步移除 \(N\) 个节点后的 \(\operatorname{TA}(s)\)。主要统计指标 `TargetAttack_AUC` 使用归一化梯形积分：

$$
\operatorname{TargetAttack\_AUC}
=
\frac{1}{N}\sum_{s=1}^{N}\frac{\operatorname{TA}(s-1)+\operatorname{TA}(s)}{2}
$$

Random attack 默认重复 `RANDOM_ATTACK_REPEATS = 1000` 次，主流程默认随机种子为 `42`。特征汇总同时保存 early drop 和 max drop。
当初始全局效率为 0 时，代码将整条 attack 曲线和 `TargetAttack_AUC` 置为 0。

FC stability 将已读取 EEG 时间序列分成 `FC_STABILITY_EPOCHS = 5` 个连续片段，分别计算 LOFC/HOFC，再统计 epoch 间矩阵相关、边变异系数和稳定边比例。

LOFC/HOFC layer similarity 使用上三角非对角边向量计算：

$$
\operatorname{Corr}_{LOFC,HOFC}=\operatorname{corr}(\operatorname{vec}(\operatorname{LOFC}), \operatorname{vec}(\operatorname{HOFC}))
$$

`LayerDecoupling = 1 - Corr_LOFC_HOFC`。

## 6. 统计分析

`step11_statistical_analysis.py` 对三组 HC/MCI/AD 进行组间检验。每个指标先做 omnibus 检验：

1. 各组 Shapiro-Wilk 正态性和 Levene 方差齐性通过时使用 one-way ANOVA。
2. 否则使用 Kruskal-Wallis。
3. omnibus 显著后执行 post-hoc，两正态组使用 Welch's t-test，否则使用 Mann-Whitney U。

omnibus 和 post-hoc 分别做 Benjamini-Hochberg FDR 校正。`two_way_anova_metrics.csv` 和 `mixed_effects_metrics.csv` 作为探索性/敏感性统计，不替代主三组检验。

## 7. 热图

`step12_fc_heatmaps.py` 从 `output/mean/{group}/{band}/` 读取 LOFC/HOFC 组均值矩阵，输出组合热图和单网络热图到 `results/fc_heatmaps/`：

```text
lofc_hofc_heatmap.png
lofc_hofc_heatmap.svg
lofc_heatmap.png
lofc_heatmap.svg
hofc_heatmap.png
hofc_heatmap.svg
colorbar.png
```

热图不读取 `writing/` 目录；`src/` 流水线可独立运行。

## 8. 复现验证

2026-05-29 使用当前本地 EEG `.set` 输入从 `step00` 到 `step12` 完整重跑：

```powershell
uv run python -m compileall -q src tests
uv run python src/step00_bandpass_filter.py --skip-existing
uv run python src/step09_batch_pipeline.py
uv run python src/step10_results_postprocess.py
uv run python src/step11_statistical_analysis.py
uv run python src/step12_fc_heatmaps.py
```

重跑验证结果：

- `step09` 总计 352 个被试-频段组合，成功 352，失败 0。
- `output/all_scalar_metrics_summary.csv` 为 352 行、101 列，覆盖 88 名被试和 4 个频段。
- `results/显著性分析.csv` 为 779 行、55 列；其中主 scalar omnibus FDR 显著项为 13 个。
- `results/targetattack_显著性分析.csv` 为 20 行、55 列；其中 TargetAttack_AUC omnibus FDR 显著项为 4 个。
- FC stability 重跑结果与论文正文数值一致：LOFC overall matrix correlation 为 0.829，edge CV 为 0.383，stable edge ratio 为 0.337；HOFC 分别为 0.566、0.469、0.222。

跑前关键结果备份位于 `D:\Paper_code\CE_repro_backup_20260529_153050`。49 个关键文件中 45 个哈希完全一致；`output/all_scalar_metrics_summary.csv` 的表格数据内容完全一致但文件序列化哈希不同，3 个 SVG 热图因重生成时间和内部元数据不同而哈希不同，PNG 热图字节一致。

## 9. CAUEEG 复现流程（step14–step21）

2026-10-07 新增。用于在公开 CAUEEG 数据集上从原始 EDF 重做实验，以回应审稿意见。各项参数在运行统计前已写入 `docs/caueeg_analysis_plan.md`。常量集中在 `src/caueeg_config.py`。旧的 16 导联流程（step00–step12）不改变口径。

### 9.1 事件校正与预处理

- **暂停校正**：EDF 不含暂停期间的数据，但事件采样点仍按包含暂停的时间计数。第 \(k\) 次 `Recording Resumed` 之后的事件 \(t\) 校正为
  $$
  t' = t - \sum_{m\le k} (t^{\mathrm{resume}}_m - t^{\mathrm{pause}}_m)
  $$
  暂停边界 \(t^{\mathrm{pause}}_k - \sum_{m<k}(\cdot)\) 是信号的不连续点，闭眼片段在此处切开。
- **闭眼可用片段**：取闭眼标记到睁眼标记之间的数据。扣除以下时段：
  - 闪光刺激；
  - HV 及其后 30 s；
  - 睡眠或嗜睡标记，直到下一次眼睛状态标记；
  - 伪迹标记前后各 1 s；
  - 电极检查标记前后各 5 s；
  - 记录末尾 5 s。
- **坏导、ICA、epoch 选取**：规则见分析计划第 2 节。要点如下：
  - 平均参考时不纳入坏导；
  - ICA 成分数等于数据秩；
  - ICLabel 概率 ≥0.80 的伪迹成分剔除；
  - 先做 ICA，再插值坏导；
  - 取前 50 个合格的 2 s epoch。

### 9.2 连接指标

对每个连续片段先做带通滤波（4 阶 Butterworth，SOS 形式的 `sosfiltfilt`，零相位），再做 Hilbert 变换，最后按选中的 epoch 拼接样本。记解析信号为 \(X_i(t)\)。

- **AEC**：\(\operatorname{corr}(|X_i|,|X_j|)\)，网络使用其绝对值。
- **AEC-c**（Hipp et al., 2012）：先做正交化
  $$
  Y_{j\perp i}(t)=\operatorname{Im}\!\left(X_j(t)\,\frac{X_i^*(t)}{|X_i(t)|}\right)
  $$
  再计算
  $$
  \operatorname{AECc}_{ij}=\tfrac12\left[\operatorname{corr}(|X_i|,|Y_{j\perp i}|)+\operatorname{corr}(|X_j|,|Y_{i\perp j}|)\right]
  $$
  网络中负值置 0。
- **wPLI**（去偏平方，Vinck et al., 2011）：令 \(I_{ij}(t)=\operatorname{Im}(X_i X_j^*)\)，求和遍历全部拼接样本：
  $$
  \operatorname{wPLI}^2_{\mathrm{deb}}=\frac{(\sum I)^2-\sum I^2}{(\sum|I|)^2-\sum I^2}
  $$
  网络中负值置 0。
- **signed HOFC（sHOFC）**：
  $$
  \operatorname{sHOFC}_{ij}=\operatorname{corr}(z^{s}_{-i,-j,i}, z^{s}_{-i,-j,j})
  $$
  其中 \(z^s=\operatorname{arctanh}(\text{signed LOFC})\)，两层都不取绝对值。拓扑指标用其正部；另外报告带符号均值、负边比例和负边均值。原 HOFC 的定义不变，函数为 `compute_topographical_hofc(lofc, signed=False)`。

### 9.3 网络指标

- **零模型**：
  - 强度保持零模型：Rubinov & Sporns (2011) 的权重重排，权重多重集完全保持，强度序列近似保持，见 `src/network_nulls.py`。
  - 权重置换零模型。
  - 二值图使用 Maslov–Sneppen 换边（每条边换 10 次）。
  - 归一化指标为 \(M/\overline{M}_{\mathrm{null}}\)，每个网络加权零模型 50 个、二值零模型 20 个。
- **固定密度**：\(d\in\{0.20,\dots,0.40\}\)，19 节点连通所需密度约为 \(\ln n/n\approx0.16\)。零权重边不会被选入。density-AUC 为梯形积分除以 0.20。
- **攻击**：
  - 在固定密度图上按 CE、度、介数、接近度、TOPSIS 排序依次移除节点，结局为相对全局效率和相对最大连通分量。
  - \(\mathrm{AUC}=\frac1N\int_0^N c(k)\,dk\)，取各密度的平均。
  - `AttackAUCdiff` 为该 AUC 减去随机顺序 AUC（随机顺序 100 次）。
  - 交叉验证攻击：用一半 epoch 的网络确定节点排序，去攻击另一半 epoch 的网络，再交换两半，取平均。
  - 效率和 LCC 由同一次 BFS 计算（`attack_curves_both`），与 `compute_attack_by_order` 的结果一致。
- **最短路**：`_bfs_distance` 改为布尔矩阵乘法的同步 BFS，`_dijkstra_distance` 改为 `scipy.sparse.csgraph.shortest_path`。数值与原实现一致，有测试覆盖。

### 9.4 统计

- **主模型**：每个终点在频段内 z 标准化，然后拟合
  $$
  y_{sb}=\beta_0+\beta_{g(s)}+\gamma_b+\delta\,\mathrm{age}_s+u_s+\varepsilon_{sb}, \quad u_s\sim N(0,\sigma_u^2)
  $$
  - 实现：statsmodels MixedLM，REML。
  - 组效应：Wald \(\chi^2\)，自由度 2。
  - group×band 交互：在另一个模型中单独做 Wald 检验。
  - 两类检验各自在 8 个主终点内做 BH-FDR。
- **功率校正**：在主模型中加入该频段 log 绝对功率，或加入 specparam 的 offset 与 exponent。衰减比例定义为 \(1-|\beta_{\mathrm{adj}}|/|\beta|\)。
- **CE 与介数比较**：
  - TOST 检验每人 \(\mathrm{AUC}_{CE}-\mathrm{AUC}_{BC}\)，等效界值为 ±0.2×SD(\(\mathrm{AUC}_{BC}\))；
  - 混合模型检验 group×排序方式的交互。
