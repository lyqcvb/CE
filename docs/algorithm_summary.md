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
