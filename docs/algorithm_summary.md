# EEG 脑网络指标计算公式汇总

本文档列出本项目当前代码实际计算的功能连接矩阵、图论指标、Hub/CE 指标、
rich-club 指标、靶向攻击指标以及结果后处理统计量。公式采用 MathType/LaTeX
兼容写法，脑网络指标按流程拆分在 `src/step01_eeg_io.py` 到
`src/step08_summary_metrics.py` 中，`src/brain_metrics.py` 保留为兼容导出层。
其中 `src/step03_connection_strength.py` 位于 FC 矩阵计算之后、图论指标计算之前，
用于直接基于原始 FC 矩阵计算 `MeanConn_*`。
完整处理分析入口按流程命名为 `src/step00_bandpass_filter.py`、
`src/step09_batch_pipeline.py`、`src/step10_results_postprocess.py`、
`src/step11_statistical_analysis.py` 和 `src/step12_fc_heatmaps.py`。

## 近期稳健性增强

当前代码在保留原固定阈值 \(\theta=0.35\) 输出的基础上，新增以下稳健性分析：

1. LOFC 同时保存 signed envelope correlation 到
   `output/fc_matrices/lofc_signed/`；主 LOFC/HOFC/aHOFC 仍沿用绝对相关矩阵。
2. 标量指标新增加权图指标：
   `Eglob_weighted_*`、`Clustering_weighted_*`。
3. 标量指标新增固定稀疏度阈值敏感性分析。默认稀疏度为
   `10%, 15%, 20%, 25%, 30%, 35%, 40%`，并输出
   `Eglob_bin_auc_*`、`Clustering_bin_auc_*`。
4. 每个被试、每个 FC 类型输出网络质量控制记录到 `output/network_qc/`，
   后处理生成 `results/network_qc_summary.csv` 和
   `results/network_qc_group_summary.csv`。QC 字段包括 density、平均度、边数、
   连通分量数、FC 分位数、`mean+std` 规则是否出现 0 核心节点、
   attack 初始全局效率是否为 0。
5. Hub 主判定改为 CE top-ratio，默认 top 20%；`mean+std` 作为严格补充标签保留。
   core-node 文件同时保存 CE、TOPSIS score、degree/BC/closeness 传统 Hub 对照。
6. CE/Hub 新增多稀疏度稳定性，输出到 `output/core_nodes_density/`；
   后处理生成 `results/core_node_density_stability.csv`，包含
   `CoreStability` 和 `StableCoreNode`。
7. rich-club 保留二值归一化结果，同时新增 weighted rich-club 曲线到
   `output/rich_club_weighted/`，并新增 weighted rich-club 随机归一化统计到
   `output/rich_club_weighted_stats/`。
8. 靶向攻击保留原 CE 固定排序曲线到 `output/target_attack/`，并新增
   `ce`、`ce_adaptive`、`degree`、`betweenness`、`random_mean`、`random_std`
   对照曲线到 `output/target_attack_controls/`。
9. CE 消融攻击新增 `degree`、`betweenness`、`closeness`、`s`、`h`、`ce`
   六套攻击曲线到 `output/ce_ablation_attack/`。
10. FC 层级一致性输出到 `output/fc_layer_similarity/`，后处理生成
    `results/fc_layer_similarity.csv`。
11. 统计模块的 FDR 校正新增 `Correction_Family`，标量指标按
   `Scalar primary | FC_Type`、`Scalar weighted topology | FC_Type`、
   `Scalar density AUC | FC_Type`、`Scalar density curve | FC_Type`、
   `Scalar QC | FC_Type` 等比较族分层校正。
12. 统计模块新增 `Metric ~ Group * Band` 的固定效应 two-way ANOVA 输出
    `results/two_way_anova_metrics.csv`，并新增重复测量 fallback 输出
    `results/mixed_effects_metrics.csv`。

## 0. 流程与预处理

### 0.0 结果主次口径

为避免指标过多导致主线分散，当前建议按以下层级解释结果：

```text
主分析:
  MeanConn
  Eglob_bin_auc / Clustering_bin_auc
  CE top 20% 核心节点
  CoreStability 稳定核心节点
  AttackAUC_CE / CE_vs_Random_AUC_Diff / EarlyDrop_1 / EarlyDrop_3
  RichClubMean_high_k / Phi_w_norm
  FC layer similarity / decoupling

稳健性分析:
  fixed threshold theta=0.35 二值指标
  weighted Eglob / weighted Clustering
  Weighted CE 与 BinaryWeightedCoreOverlap
  mean+std 严格核心节点
  density rich-club / density target attack

机制验证和补充分析:
  CE 消融攻击 degree / betweenness / closeness / S-only / H-only / CE
  target attack stepwise 曲线
  FC epoch stability
  core node band / FC type / overall stability
```

### 0.1 目录、组别与频段

共享配置集中在 `src/pipeline_config.py`。当前主流程使用三组被试：

```text
HC   -> data/raw/HC 256 100s/
MCI  -> data/raw/MCI 256 100s/
AD   -> data/raw/AD 256 100s/
```

频段目录和显示标签为：

```text
1-4(Delta)    -> Delta
4-8(Theta)    -> Theta
8-12(Alpha)   -> Alpha
12-30(Beta)   -> Beta
30-40(Gamma)  -> Gamma
```

默认通道顺序为 16 导联：

```text
Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6
```

主流程假定输入矩阵方向为：

$$
X\in\mathbb{R}^{N\times T}
$$

即行为通道/节点，列为采样点。

### 0.2 带通滤波 `step00_bandpass_filter.py`

带通预处理从 `data/raw/{Group} 256 100s/1-40/` 读取 `.set` 文件，输出到对应频段目录。
滤波器为 4 阶 Butterworth 带通滤波器，使用 `scipy.signal.filtfilt` 沿时间轴做零相位前后向滤波。

令采样率为 \(f_s\)，Nyquist 频率为：

$$
f_N=\frac{f_s}{2}
$$

目标频段 \([f_l, f_h]\) 先归一化为：

$$
\omega_l=\frac{f_l}{f_N},\qquad
\omega_h=\frac{f_h}{f_N}
$$

随后构造带通滤波器：

$$
(b,a)=\operatorname{Butterworth}_{order=4}([\omega_l,\omega_h],\operatorname{bandpass})
$$

滤波输出为：

$$
X_{band}=\operatorname{filtfilt}(b,a,X,\operatorname{axis}=time)
$$

代码会检查：

```text
0 < f_l < f_h < f_N
```

并要求时间点数满足 `filtfilt` 的最小长度要求：

$$
T \geq 3\max(|a|,|b|)+1
$$

输出仍保存为 MAT-style `.set`，包含 `data`、`nbchan`、`pnts` 和 `srate`。

### 0.3 EEG 读取 `step01_eeg_io.py`

主流程读取 EEGLAB `.set` 文件时优先识别 HDF5 格式：

1. 若 `.set` 内含 `data`，直接读取。
2. 若 `.set` 内含 `datfile`，读取同目录下外部 `.fdt`，按 `float32` 解析后 reshape 为 `pnts x nbchan`，再转置为 `nbchan x pnts`。
3. 非 HDF5 `.set` 使用 `scipy.io.loadmat` 读取 `data`、`nbchan` 和 `pnts`。

若输入数据方向与 `nbchan` 不一致，且第二维等于 `nbchan`，代码会自动转置为通道 × 时间。
`read_eeg_set` 默认最多截取前 `25601` 个采样点：

$$
T_{use}=\min(25601,\ pnts,\ T)
$$

实际进入 FC 计算的数据为：

$$
X_{use}=X[:,\,1,\ldots,T_{use}]
$$

### 0.4 输入输出维度与文件格式总览

本文档后续公式默认使用以下维度符号：

| 符号 | 含义 | 当前默认值/范围 |
|---|---|---|
| \(N\) | EEG 通道数 / 网络节点数 | 16 |
| \(T\) | 原始或频段数据采样点数 | 由 `.set` 决定，读取时最多使用 25601 |
| \(G\) | 分组数 | 3，HC/MCI/AD |
| \(B\) | 频段数 | 5，Delta/Theta/Alpha/Beta/Gamma |
| \(F\) | FC 类型数 | 3，LOFC/HOFC/aHOFC |
| \(D\) | 固定稀疏度数量 | 7，0.10 到 0.40 |
| \(R\) | 随机攻击 / 随机 rich-club 重复次数 | 默认 1000 |

当前加入 Gamma 后，完整重跑样本数为 HC = 30、MCI = 29、AD = 29。因此被试-频段级标量汇总
`output/all_scalar_metrics_summary.csv` 有 440 行：

$$
440=(30+29+29)\times 5
$$

#### 0.4.1 EEG 输入与带通输出

| 阶段 | 路径 | 文件格式 | 数据维度 |
|---|---|---|---|
| 原始输入 | `data/raw/{Group} 256 100s/1-40/{subject}.set` | EEGLAB `.set`，可为 HDF5 或 MAT-style | 读取后统一为 \(N\times T\) |
| HDF5 外部数据 | 同目录 `.fdt` | `float32` 连续数组 | 先按 `pnts x nbchan` 读入，再转置为 \(N\times T\) |
| 频段输出 | `data/raw/{Group} 256 100s/{band}/{subject}.set` | MAT-style `.set` | `data` 为 \(N\times T\)，并保存 `nbchan`、`pnts`、`srate` |

#### 0.4.2 FC 矩阵输出

| 输出 | 文件格式 | 维度 | 说明 |
|---|---|---|---|
| `output/fc_matrices/lofc/{group}/{band}/{subject}.csv` | 无表头数值 CSV | \(N\times N\)，当前为 `16 x 16` | LOFC 绝对相关矩阵，对称 |
| `output/fc_matrices/hofc/{group}/{band}/{subject}.csv` | 无表头数值 CSV | \(N\times N\)，当前为 `16 x 16` | HOFC 矩阵，对称 |
| `output/fc_matrices/ahofc/{group}/{band}/{subject}.csv` | 无表头数值 CSV | \(N\times N\)，当前为 `16 x 16` | aHOFC 矩阵，允许非对称 |
| `output/fc_matrices/lofc_signed/{group}/{band}/{subject}.csv` | 无表头数值 CSV | \(N\times N\)，当前为 `16 x 16` | signed LOFC，保留正负号 |
| `output/mean/{group}/{band}/{group}_{band}_{fc_type}.csv` | 无表头数值 CSV | \(N\times N\)，当前为 `16 x 16` | 同组、同频段、同 FC 类型的被试均值矩阵 |

矩阵行列顺序均对应默认 16 导联：

```text
Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6
```

对角线在计算图指标前会置 0；热图绘制时会将对角线显示为空白。

#### 0.4.3 被试级标量宽表

| 输出 | 文件格式 | 维度 | 固定标识列 |
|---|---|---|---|
| `output/all_scalar_metrics_summary.csv` | 有表头 CSV 宽表 | 每行 = 1 个 `subject x band`；当前为 `440 x 158` | `Group`、`Band`、`SubjectID` |

其余列为不同 FC 类型、阈值模式和稳健性分析得到的标量指标，例如：

```text
Eglob_bin_LOFC
Clustering_bin_LOFC
MeanConn_LOFC
Eglob_weighted_LOFC
Density_thr035_LOFC
Eglob_bin_density10_LOFC
Clustering_bin_auc_LOFC
RichClub_auc_LOFC
TargetAttack_auc_across_density_LOFC
CEAblationAttackAUC_ce_LOFC
FCStability_MatrixCorrMean_LOFC
Corr_LOFC_HOFC
```

字段数量会随算法输出扩展而变化；统计脚本按列名生成 `Correction_Family`，
不依赖固定总列数。`AttackAUC_CE`、`CE_vs_Random_AUC_Diff` 等固定阈值攻击特征
不在该宽表中，而是在后处理表 `results/targetattack_feature_summary.csv` 中。

#### 0.4.4 核心节点与多密度核心节点输出

| 输出 | 文件格式 | 维度 | 主要字段 |
|---|---|---|---|
| `output/core_nodes/{fc_type}/{group}/{band}/{subject}_core_nodes.csv` | 有表头 CSV | 每个节点一行，当前 `16 x 17` | `NodeIndex`、`IsCoreNode`、`CE`、`CentralityScore`、`WeightedCE`、`TopRatioThreshold`、传统 Hub 对照 |
| `output/core_nodes_density/{fc_type}/{group}/{band}/{subject}.csv` | 有表头 CSV | \(D\times N\) 行，当前 `112 x 19` | `Density`、`NodeIndex`、`CE`、`S_score`、`H_entropy`、`CoreFrequency`、`StableCoreNode`、`CE_auc`、`S_auc` |

`NodeIndex` 使用 0-based 节点编号，对应默认通道顺序。当前固定稀疏度集合为：

```text
0.10 0.15 0.20 0.25 0.30 0.35 0.40
```

旧版已生成的 `core_nodes` 文件可能只包含 `NodeIndex,IsCoreNode` 两列；后处理逻辑仍兼容基础字段。

#### 0.4.5 Rich-club 输出

| 输出 | 文件格式 | 维度 | 说明 |
|---|---|---|---|
| `output/rich_club/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单行 CSV | `1 x K` | 第 \(j\) 列对应 \(k=j+1\) 的二值归一化 rich-club |
| `output/rich_club_weighted/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单行 CSV | `1 x K_w` | weighted rich-club 曲线 |
| `output/rich_club_weighted_stats/{fc_type}/{group}/{band}/{subject}.csv` | 有表头 CSV | 每个有效 k 一行；LOFC/HOFC 当前常见 `14 x 8`，aHOFC 当前常见 `29 x 8` | `k`、`Phi_w_observed`、`Phi_w_ranknorm`、`Phi_w_random_mean`、`Phi_w_random_std`、`Phi_w_norm`、`Phi_w_empirical_p`、`Phi_w_random_valid_n` |
| `output/rich_club_density/{fc_type}/{group}/{band}/{subject}.csv` | 有表头 CSV | \(D\) 行，当前 `7 x 11` | 每个稀疏度一行的 rich-club 曲线特征 |

rich-club 曲线长度 \(K\) 由有效度阈值范围决定，并不固定等于 \(N\)。当某些 k 层级不可计算时，
CSV 中会出现空字段，读入后应视为 NaN。

后处理输出包括：

| 输出 | 文件格式 | 维度含义 |
|---|---|---|
| `results/richclub_summary_{FC}.csv` | 有表头 CSV | 行为 k，列为 `{Band}_{Group}` 的组均值曲线 |
| `results/richclub_individual_means_{FC}.csv` | 两行表头 CSV | 每列为某频段某组的被试 rich-club 均值 |
| `results/richclub_feature_summary.csv` | 有表头 CSV | 每行 = 1 个 `subject x band x fc_type`；当前 `1317 x 10` |
| `results/weighted_richclub_feature_summary.csv` | 有表头 CSV | 每行 = 1 个 `subject x band x fc_type`；当前 `1320 x 10` |
| `results/weighted_richclub_stats_summary.csv` | 有表头 CSV | 合并所有被试的 weighted rich-club k 层级统计；当前 `25080 x 12` |
| `results/density_richclub_features.csv` | 有表头 CSV | 每行 = 1 个 `subject x band x fc_type x density`；当前 `9240 x 11` |

`weighted_richclub_feature_summary.csv` 汇总的是 `output/rich_club_weighted/` 中的
rank-normalized weighted rich-club 曲线；带随机网络归一化、经验 p 值和有效随机次数的
k 层级统计位于 `weighted_richclub_stats_summary.csv`。

#### 0.4.6 Target Attack 输出

| 输出 | 文件格式 | 维度 | 说明 |
|---|---|---|---|
| `output/target_attack/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单列 CSV | \((N+1)\times 1\)，当前 `17 x 1` | CE 固定排序攻击曲线，第 0 行为攻击前归一化效率 1 |
| `output/target_attack_controls/{strategy}/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单列 CSV | \((N+1)\times 1\)，当前 `17 x 1` | `strategy` 包括 `ce`、`ce_adaptive`、`degree`、`betweenness`、`random_mean`、`random_std` |
| `output/ce_ablation_attack/{method}/{fc_type}/{group}/{band}/{subject}.csv` | 无表头单列 CSV | \((N+1)\times 1\)，当前 `17 x 1` | `method` 包括 `degree`、`betweenness`、`closeness`、`s`、`h`、`ce` |
| `output/target_attack_density/{fc_type}/{group}/{band}/{subject}.csv` | 有表头 CSV | \(D\) 行，当前 `7 x 10` | `AttackAUC`、`EarlyDrop_1`、`EarlyDrop_2`、`EarlyDrop_3`、`MaxDrop` |

`Attack_Step=0` 表示尚未移除节点；`Attack_Step=N` 表示所有节点均已按攻击顺序隔离。

后处理输出包括：

| 输出 | 文件格式 | 维度含义 |
|---|---|---|
| `results/targetattack_summary_{FC}.csv` | 有表头 CSV | 行为 `Attack_Step`，列为 `{Band}_{Group}` 的组均值曲线 |
| `results/targetattack_{strategy}_summary_{FC}.csv` | 有表头 CSV | 不同攻击策略的组均值曲线 |
| `results/targetattack_feature_summary.csv` | 有表头 CSV | 每行 = 1 个 `subject x band x fc_type`；当前 `1320 x 47` |
| `results/density_targetattack_features.csv` | 有表头 CSV | 每行 = 1 个 `subject x band x fc_type x density`；当前 `9240 x 10` |

#### 0.4.7 QC、稳定性与层级相似性输出

| 输出 | 文件格式 | 维度 | 主要字段 |
|---|---|---|---|
| `output/network_qc/{fc_type}/{group}/{band}/{subject}_qc.csv` | 有表头 CSV | 每个阈值模式一行；当前样例 `8 x 20` | `ThresholdMode`、`ThresholdValue`、FC 分布分位数、`Density`、`AverageDegree`、`EdgeCount`、连通性字段 |
| `output/fc_stability/{group}/{band}/{subject}.csv` | 有表头 CSV | 每个 FC 类型一行；当前 `3 x 9` | `EpochCount`、`MatrixCorr_Mean`、`MatrixCorr_SD`、`EdgeCV_Mean`、`StableEdgeRatio` |
| `output/fc_layer_similarity/{group}/{band}/{subject}.csv` | 有表头 CSV | 每个被试-频段一行；当前 `1 x 8` | `Corr_LOFC_HOFC`、`Corr_LOFC_aHOFC`、`Corr_HOFC_aHOFC`、`LayerDecoupling`、`HighOrderDecoupling` |

#### 0.4.8 统计检验输出

| 输出 | 文件格式 | 维度 | 说明 |
|---|---|---|---|
| `results/显著性分析.csv` | 有表头 CSV | 当前 `1390 x 55` | 标量指标 omnibus 与 post-hoc 统计 |
| `results/two_way_anova_metrics.csv` | 有表头 CSV | 当前 `760` 行 | `Metric ~ Group * Band` 固定效应分析 |
| `results/mixed_effects_metrics.csv` | 有表头 CSV | 当前 `456` 行 | 重复测量 fallback 探索分析 |
| `results/richclub_显著性分析.csv` | 有表头 CSV | 当前 `18 x 55` | 二值归一化 rich-club 个体均值统计 |
| `results/targetattack_显著性分析.csv` | 有表头 CSV | 当前 `36 x 55` | Target Attack AUC 主统计 |
| `results/targetattack_stepwise_补充分析.csv` | 有表头 CSV | 当前 `444 x 56` | Target Attack 分步曲线补充统计 |

统计表的常见结构包括：

```text
Analysis, Band, Metric,
HC_n, HC_Mean, HC_SD, HC_Median, HC_IQR,
MCI_n, MCI_Mean, MCI_SD, MCI_Median, MCI_IQR,
AD_n, AD_Mean, AD_SD, AD_Median, AD_IQR,
Test_Level, Test_Type, p_raw, p_value, p_adj,
Correction_Method, Correction_Scope, Significant_adj
```

`results/显著性分析.csv` 和 `results/mixed_effects_metrics.csv` 额外保存
`Correction_Family`。`richclub_显著性分析.csv`、`targetattack_显著性分析.csv`
和 `targetattack_stepwise_补充分析.csv` 当前不输出 `Correction_Family`，
而是分别在 rich-club 或 target attack 专表内部执行 FDR。包含 `FC_Type`
或 `Attack_Step` 的统计表会额外保存对应列。写作和结果解释应优先使用
`p_adj` 与 `Significant_adj`，不要仅依据未校正 `p_raw` 下结论。

## 1. 基本符号

EEG 数据矩阵：

$$
X\in\mathbb{R}^{N\times T}
$$

其中 N 为通道/节点数，T 为采样点数。

第 i 个通道在时间 t 的信号：

$$
x_i(t)
$$

加权功能连接矩阵，即 FC 矩阵：

$$
W
$$

二值邻接矩阵：

$$
G
$$

节点数，当前数据通常为 16：

$$
N
$$

二值图中从节点 i 到节点 j 的最短路径长度：

$$
d_{ij}
$$

节点 i 的度：

$$
k_i
$$

Pearson 相关系数：

$$
\operatorname{corr}(a,b)
$$

二值化规则为：

$$
G_{ij}=
\begin{cases}
1, & W_{ij} \geq \theta \\
0, & W_{ij} < \theta
\end{cases},
\qquad G_{ii}=0
$$

当前固定阈值由 `src/pipeline_config.py` 统一配置：

$$
\theta=0.35
$$

其中 `THRESHOLD_RICHCLUB = THRESHOLD_DEFAULT`，即 rich-club 与其他固定阈值二值指标
使用同一个 0.35 阈值。

主稳健性分析同时使用固定稀疏度比例阈值。默认密度集合为：

$$
D=\{0.10,0.15,0.20,0.25,0.30,0.35,0.40\}
$$

对每个 \(d\in D\)，无向图保留上三角中权重最高的 \(\lceil dN(N-1)/2\rceil\)
条边并对称化；有向图保留全部非对角有向边中权重最高的
\(\lceil dN(N-1)\rceil\) 条边。

`LOFC` 和 `HOFC` 按无向图计算，`aHOFC` 按有向图计算。

## 2. 功能连接矩阵

### 2.1 LOFC：Hilbert 包络相关

先对每个通道做 Hilbert 变换，取解析信号振幅作为包络：

$$
z_i(t)=\mathcal{H}\{x_i(t)\}
$$

$$
e_i(t)=|z_i(t)|
$$

`LOFC` 为通道包络之间 Pearson 相关的绝对值：

$$
\operatorname{LOFC}_{ij}=
\left|\operatorname{corr}(e_i,e_j)\right|
$$

$$
\operatorname{LOFC}_{ii}=1
$$

同时保存 signed LOFC 作为符号敏感性检查：

$$
\operatorname{LOFC}^{signed}_{ij}=
\operatorname{corr}(e_i,e_j)
$$

输出路径为 `output/fc_matrices/lofc_signed/{group}/{band}/{subject}.csv`。

### 2.2 Fisher-z 变换

`HOFC` 和 `aHOFC` 计算前会对 FC 矩阵做 Fisher-z 变换：

$$
Z_{ij}=\operatorname{arctanh}
\left(\operatorname{clip}(W_{ij},-1+\varepsilon,1-\varepsilon)\right)
$$

其中 eps 为 `float64` 机器精度，用于避免 arctanh(1) 或 arctanh(-1)。

### 2.3 HOFC：拓扑高阶功能连接

每个节点的 `LOFC` 列向量作为该节点的低阶连接拓扑轮廓。比较节点 i 和 j
时，排除第 i、j 行：

$$
P_i^{\operatorname{LOFC}}(i,j)
=
Z_{\operatorname{LOFC}}[\{1,\ldots,N\}\setminus\{i,j\},\,i]
$$

$$
P_j^{\operatorname{LOFC}}(i,j)
=
Z_{\operatorname{LOFC}}[\{1,\ldots,N\}\setminus\{i,j\},\,j]
$$

`HOFC` 计算为两个拓扑轮廓相关的绝对值：

$$
\operatorname{HOFC}_{ij}
=
\left|
\operatorname{corr}
\left(
P_i^{\operatorname{LOFC}}(i,j),
P_j^{\operatorname{LOFC}}(i,j)
\right)
\right|
$$

$$
\operatorname{HOFC}_{ji}=\operatorname{HOFC}_{ij}
$$

因此 `HOFC` 输出为对称矩阵。

### 2.4 aHOFC：关联高阶功能连接

`aHOFC` 比较 `LOFC` 拓扑轮廓和 `HOFC` 拓扑轮廓：

$$
P_i^{\operatorname{LOFC}}(i,j)
=
Z_{\operatorname{LOFC}}[\{1,\ldots,N\}\setminus\{i,j\},\,i]
$$

$$
P_j^{\operatorname{HOFC}}(i,j)
=
Z_{\operatorname{HOFC}}[\{1,\ldots,N\}\setminus\{i,j\},\,j]
$$

$$
\operatorname{aHOFC}_{ij}
=
\left|
\operatorname{corr}
\left(
P_i^{\operatorname{LOFC}}(i,j),
P_j^{\operatorname{HOFC}}(i,j)
\right)
\right|
$$

由于源轮廓和目标轮廓来自不同层级，aHOFC_ij 不强制等于 aHOFC_ji，
因此 `aHOFC` 输出为非对称矩阵。

### 2.5 FC 层级一致性

层级一致性分析输出到：

```text
output/fc_layer_similarity/{group}/{band}/{subject}.csv
results/fc_layer_similarity.csv
```

每个被试、每个频段计算：

```text
Corr_LOFC_HOFC
Corr_LOFC_aHOFC
Corr_HOFC_aHOFC
LayerDecoupling
HighOrderDecoupling
```

其中 aHOFC 在与无向 LOFC/HOFC 比较时先使用 \((aHOFC+aHOFC^T)/2\) 对称化，
再取上三角非对角元素计算 Pearson 相关。该分析用于评估低阶连接与高阶连接是否解耦。

解耦指数定义为：

$$
\operatorname{LayerDecoupling}
=
1-\operatorname{Corr}_{LOFC,HOFC}
$$

高阶解耦指数定义为：

$$
\operatorname{HighOrderDecoupling}
=
1-\operatorname{mean}
\left(
\operatorname{Corr}_{LOFC,HOFC},
\operatorname{Corr}_{LOFC,aHOFC}
\right)
$$

数值越高表示低阶连接与高阶连接的一致性越弱。由于 Pearson 相关系数理论范围为 [-1, 1]，
该解耦指数理论范围可视为 [0, 2]，负相关对应更强解耦。

## 3. 标量图论指标

`output/all_scalar_metrics_summary.csv` 中当前输出以下标量指标：

```text
Eglob_bin_LOFC        Eglob_bin_HOFC        Eglob_bin_aHOFC
Clustering_bin_LOFC   Clustering_bin_HOFC   Clustering_bin_aHOFC
Eglob_weighted_LOFC   Eglob_weighted_HOFC   Eglob_weighted_aHOFC
Clustering_weighted_LOFC ...
Eglob_bin_auc_LOFC    Clustering_bin_auc_LOFC ...
MeanConn_LOFC         MeanConn_HOFC         MeanConn_aHOFC
Density_thr035_*      AverageDegree_thr035_* EdgeCount_thr035_*
```

### 3.1 二值全局效率 `Eglob_bin_*`

先在二值图 G 上计算最短路径距离 d_ij。不可达节点对贡献 0：

$$
\operatorname{Eglob}_{bin}
=
\frac{1}{N(N-1)}
\sum_{i\neq j}\frac{1}{d_{ij}}
$$

其中：

$$
\frac{1}{d_{ij}}=
\begin{cases}
0, & i \text{ 与 } j \text{ 不可达} \\
\frac{1}{d_{ij}}, & i \text{ 与 } j \text{ 可达}
\end{cases}
$$

`LOFC/HOFC` 在无向二值图上计算，`aHOFC` 保留方向性计算。

固定稀疏度 AUC 对每个 \(d\in D\) 重复计算 `Eglob_bin`，再用标准化梯形积分：

$$
\operatorname{AUC}(m)=
\frac{1}{d_{max}-d_{min}}
\int_{d_{min}}^{d_{max}}m(d)\,dd
$$

离散实现使用相邻稀疏度点的梯形面积。

### 3.2 二值聚类系数 `Clustering_bin_*`

#### 3.2.1 无向二值聚类系数：`LOFC/HOFC`

令 t_i 为节点 i 周围的三角形数量，k_i 为节点度：

$$
C_i=
\begin{cases}
\dfrac{2t_i}{k_i(k_i-1)}, & k_i \geq 2 \\
0, & k_i < 2
\end{cases}
$$

由于 (G^3)_ii = 2t_i，实现等价于：

$$
C_i=
\begin{cases}
\dfrac{(G^3)_{ii}}{k_i(k_i-1)}, & k_i \geq 2 \\
0, & k_i < 2
\end{cases}
$$

网络聚类系数为节点聚类系数的均值：

$$
\operatorname{Clustering}_{bin}
=
\frac{1}{N}\sum_{i=1}^{N}C_i
$$

#### 3.2.2 有向二值聚类系数：`aHOFC`

使用 BCT/Fagiolo 有向二值聚类口径。定义：

$$
S=G+G^{T}
$$

$$
K_i=\sum_{j=1}^{N}S_{ij}
$$

$$
R_i=(G^2)_{ii}
$$

$$
T_i=(S^3)_{ii}
$$

则：

$$
C_i=
\begin{cases}
\dfrac{T_i}{2\left[K_i(K_i-1)-2R_i\right]},
& 2\left[K_i(K_i-1)-2R_i\right] > 0 \\
0, & \text{otherwise}
\end{cases}
$$

$$
\operatorname{Clustering}_{bin}
=
\frac{1}{N}\sum_{i=1}^{N}C_i
$$

`Clustering_bin_auc_*` 同样在固定稀疏度集合上计算二值聚类系数并取标准化 AUC。

### 3.3 加权图指标

加权全局效率 `Eglob_weighted_*` 保留原始连接权重，边长定义为：

$$
\ell_{ij}=\frac{1}{W_{ij}},\qquad W_{ij}>0
$$

加权最短路径仅对 \(W_{ij}>0\) 的边建立；\(W_{ij}=0\) 的连接视为无边，
不参与最短路径计算。

再在加权最短路径距离上计算全局效率。`LOFC/HOFC` 使用无向加权图，
`aHOFC` 使用有向加权图。

加权聚类系数 `Clustering_weighted_*`：无向图使用 Onnela/BCT 口径，
有向图使用 Fagiolo/BCT 风格口径。权重先除以全图最大权重归一化到 [0,1]。

### 3.4 平均连接强度 `MeanConn_*`

该指标不使用二值化图，而直接基于原始 FC 矩阵的连接权重计算。

无向图，即 `LOFC/HOFC`：

$$
\operatorname{MeanConn}
=
\frac{1}{N(N-1)/2}\sum_{i<j}W_{ij}
$$

有向图，即 `aHOFC`，统计全部非对角元素，不纳入自连接：

$$
\operatorname{MeanConn}
=
\frac{1}{N(N-1)}\sum_{i\neq j}W_{ij}
$$

## 4. CE 与 Hub 节点指标

CE/Hub 当前分为三套口径：

```text
fixed threshold CE/Hub    基线分析，基于 theta=0.35 后的二值网络
density CE/Hub            稳健性分析，基于 10%-40% 固定稀疏度二值网络
weighted CE/Hub           稳健性分析，基于原始 FC 权重
```

固定阈值 CE/Hub 作为基线分析，仍基于 \(\theta=0.35\) 后的二值网络 \(G\) 计算。
此外，流程新增多稀疏度 CE/Hub 稳定性分析和加权 CE/Hub 分析，用于检验核心节点识别结果
对阈值选择和二值化处理的稳健性。`LOFC/HOFC` 按无向网络处理；`aHOFC` 按有向网络处理。
二值 CE 入口会再次清理输入：将 NaN/inf 置 0、去除自连接、去除负值，并将所有正值转为 1。

Hub 节点识别输出到：

```text
output/core_nodes/{fc_type}/{group}/{band}/{subject}_core_nodes.csv
output/core_nodes_density/{fc_type}/{group}/{band}/{subject}.csv
```

固定阈值输出保留 `NodeIndex` 和 `IsCoreNode`，并额外保存 binary/weighted CE、
传统中心性 Hub 标签和二者重叠率。

### 4.1 度中心性 `DC`

无向图：

$$
DC_i=\sum_{j=1}^{N}G_{ij}
$$

有向图：

$$
DC_i=\sum_{j=1}^{N}G_{ij}+\sum_{j=1}^{N}G_{ji}
$$

### 4.2 介数中心性 `BC`

当前实现使用二值图非归一化介数中心性。`LOFC/HOFC` 使用无向图，并将
NetworkX 的无向无序对计数乘以 2，以接近 BCT 的有序对计数口径；
`aHOFC` 使用 `DiGraph` 有向图，不再额外乘以 2：

$$
BC_i=\sum_{\substack{s\neq i\neq t}}
\frac{\sigma_{st}(i)}{\sigma_{st}}
$$

其中 sigma_st 为从 s 到 t 的最短路径条数，sigma_st(i) 为经过节点 i 的最短路径条数。
无向 `LOFC/HOFC` 代码实现中：

$$
BC_i=2 \times BC_i^{NetworkX}
$$

有向 `aHOFC` 代码实现中：

$$
BC_i=BC_i^{NetworkX}(\operatorname{DiGraph})
$$

### 4.3 接近中心性 `CC`

当前实现使用 harmonic closeness / nodal efficiency 形式。不可达节点不报错，
也不强制将该节点整体 CC 置为 0，而是在倒数距离求和中自然贡献 0。
`aHOFC` 的最短路径保留有向边方向，按出向可达路径计算：

$$
CC_i=
\frac{1}{N-1}
\sum_{\substack{j\neq i\\d_{ij}<\infty}}
\frac{1}{d_{ij}}
$$

其中不可达节点对不进入求和，等价于对该项贡献 0。若 N <= 1，则 CC_i=0。

### 4.4 中心性指标归一化

将 `DC`、`BC`、`CC` 组成决策矩阵：

$$
X_i=[DC_i,BC_i,CC_i]
$$

对每一列做 min-max 归一化：

$$
X_{ij}^{norm}
=
\frac{X_{ij}-\min_i X_{ij}}
{\max_i X_{ij}-\min_i X_{ij}}
$$

若某列最大值等于最小值，代码将分母置为 1。

### 4.5 CRITIC 客观权重

当前 Hub/CE 流程不使用 AHP 主观权重。`DC`、`BC`、`CC` 的指标权重由
CRITIC 方法从单个网络的归一化节点指标矩阵中客观计算。

对第 j 个指标，先计算其标准差：

$$
\sigma_j=\sqrt{\frac{1}{N}\sum_{i=1}^{N}
\left(X_{ij}^{norm}-\overline{X_j^{norm}}\right)^2}
$$

再计算指标 j 与指标 k 的 Pearson 相关系数 r_jk。CRITIC 信息量为：

$$
C_j=\sigma_j\sum_{k=1}^{3}(1-r_{jk})
$$

客观权重为：

$$
w_j=\frac{C_j}{\sum_{m=1}^{3}C_m}
$$

若某个指标为常数或无有效冲突信息，则其信息量为 0；若所有指标信息量均为 0，
代码退化为三项等权重。

### 4.6 TOPSIS 综合中心性得分 `S`

TOPSIS 排序阶段先对 min-max 后的三指标矩阵继续做按列向量归一化：

$$
Y_{ij}
=
\frac{X_{ij}^{norm}}
{\sqrt{\sum_{i=1}^{N}(X_{ij}^{norm})^2}}
$$

若某列向量范数为 0，代码将该列 `Y` 保持为 0。随后使用 CRITIC 权重加权：

$$
V_{ij}=Y_{ij}w_j
$$

由于 `DC`、`BC`、`CC` 均为效益型指标，TOPSIS 正理想解和负理想解分别为：

$$
V_j^+=\max_i V_{ij}
$$

$$
V_j^-=\min_i V_{ij}
$$

节点 i 到正、负理想解的距离为：

$$
D_i^+=\sqrt{\sum_{j=1}^{3}(V_{ij}-V_j^+)^2}
$$

$$
D_i^-=\sqrt{\sum_{j=1}^{3}(V_{ij}-V_j^-)^2}
$$

节点综合中心性得分使用 TOPSIS 贴近度：

$$
S_i=\frac{D_i^-}{D_i^+ + D_i^-}
$$

若 D_i^+ + D_i^- = 0，则 S_i=0。

### 4.7 局部映射熵 `H`

邻居集合 Gamma(i) 的定义：

$$
\Gamma(i)=
\begin{cases}
\{j\mid G_{ij}>0\}, & \text{LOFC/HOFC} \\
\{j\mid G_{ij}>0 \text{ or } G_{ji}>0\}, & \text{aHOFC，出邻居与入邻居并集}
\end{cases}
$$

邻居中心性映射概率：

$$
p_{ik}=
\frac{S_k}
{\sum_{m\in\Gamma(i)}S_m},
\qquad k\in\Gamma(i)
$$

令 k_i=|\Gamma(i)|。当前实现使用归一化邻域熵，将邻域 Shannon 熵除以
该邻域规模下的最大熵 log_2(k_i)：

$$ 
H_i^{norm}= 
\frac{
-\sum_{\substack{k\in\Gamma(i)\\p_{ik}>0}}
p_{ik}\log_2(p_{ik})
}
{\log_2(k_i)}
$$

因此 H_i^{norm} 的范围为 0 到 1。若 k_i <= 1，或邻居 S 总和为 0，
代码令 H_i^{norm}=0。
`aHOFC` 的局部熵虽然保留有向中心性和有向路径，但邻域集合使用入邻居与出邻居并集，
避免只按单向出边遗漏可影响该节点的入向连接。

### 4.8 节点中心性熵 `CE`

$$
CE_i=S_iH_i^{norm}
$$

若节点邻居数 k_i <= 1、S_i = 0 或邻居 S 总和为 0，则：

$$
CE_i=0
$$

### 4.9 Hub 节点判定 `IsCoreNode`

当前主流程的 `IsCoreNode` 使用 `top_ratio`，默认 top 20%：

$$
K=\max(1,\lceil 0.20N\rceil)
$$

按 \(CE_i\) 从高到低排序，前 \(K\) 个节点记为核心节点。该定义保证每个 16 节点网络
都有可比较数量的核心节点。

严格补充标签 `IsCoreNodeMeanStd` 使用单个被试、单个 FC 图中所有节点 `CE`
的均值加标准差：

$$
\theta_{CE}=\operatorname{mean}(CE)+\operatorname{std}(CE)
$$

$$
\operatorname{IsCoreNodeMeanStd}_i=
\begin{cases}
1, & CE_i>\theta_{CE} \\
0, & CE_i\leq\theta_{CE}
\end{cases}
$$

核心节点文件还输出 `IsDegreeHubTopRatio`、`IsBetweennessHubTopRatio`、
`IsClosenessHubTopRatio`，用于与传统中心性 Hub 进行一致性比较。

### 4.10 多稀疏度核心节点稳定性

核心节点稳定性输出到：

```text
output/core_nodes_density/{fc_type}/{group}/{band}/{subject}.csv
results/core_node_density_stability.csv
```

对默认稀疏度集合 \(D=\{0.10,\ldots,0.40\}\) 中每个密度重新计算：

```text
CE
S_score
H_entropy
IsCoreNode
IsDegreeHub
IsBetweennessHub
IsClosenessHub
```

节点 \(i\) 的核心稳定性为：

$$
\operatorname{CoreStability}_i
=
\frac{1}{|D|}
\sum_{d\in D}
\mathbf{1}\{i\text{ is core at density }d\}
$$

稳定核心节点定义为：

$$
\operatorname{StableCoreNode}_i
=
\mathbf{1}\left\{\operatorname{CoreStability}_i\ge \frac{4}{7}\right\}
$$

这里的 \(4/7\) 来自 `STABLE_CORE_MIN_DENSITIES = 4` 和默认 7 个稀疏度阈值，
即节点至少在 4 个固定稀疏度网络中被选为核心节点才标记为稳定核心。

同时对每个节点输出 `CE_auc` 与 `S_auc`，即 CE 和 TOPSIS score 在稀疏度范围内的标准化 AUC。

### 4.11 Weighted CE / Hub

加权核心节点识别使用原始 FC 权重。`WeightedDegree` 为 strength；
`WeightedBetweenness` 和 `WeightedCloseness` 仅对 \(W_{ij}>0\) 的边建图，
并使用边长 \(\ell_{ij}=1/W_{ij}\)；零权重边视为不存在，不参与最短路径计算。
随后沿用 CRITIC + TOPSIS + 邻域熵流程，得到 `WeightedCE`。
加权 CE 的输入清理会将 NaN/inf 和负权重置 0，去除自连接；`LOFC/HOFC`
会以 \(\max(W,W^T)\) 对称化，`aHOFC` 保留方向性。

固定阈值核心节点文件同时输出：

```text
IsCoreNode_BinaryCE
IsCoreNode_WeightedCE
BinaryWeightedCoreOverlap
```

其中 overlap 为 binary CE 核心节点集合与 weighted CE 核心节点集合的 Jaccard 重叠率。

### 4.12 CE 消融攻击验证

CE 消融攻击输出到：

```text
output/ce_ablation_attack/{method}/{fc_type}/{group}/{band}/{subject}.csv
results/ce_ablation_attack_feature_summary.csv
```

`method` 包括：

```text
degree
betweenness
closeness
s
h
ce
```

每种方法按对应节点分数降序攻击。若 CE 攻击 AUC 低于 degree/BC/CC/S-only/H-only，
说明 CE 识别节点在网络脆弱性上具有额外解释力。

## 5. Rich-club 指标

rich-club 输出到：

```text
output/rich_club/{fc_type}/{group}/{band}/{subject}.csv
output/rich_club_weighted/{fc_type}/{group}/{band}/{subject}.csv
```

二值 rich-club 使用固定阈值 0.35 后的二值图，并通过保持度分布的随机网络进行归一化。
输入矩阵会先清理为二值邻接矩阵：去除自连接、检查 0/1 取值，无向网络会对称化。
度等级 k 根据每个网络的实际最大度动态确定；不可定义的 k 记为 NaN，而不是 0。

如只需从已有 FC 矩阵刷新 rich-club 结果，可运行：

```powershell
uv run python src/step09_batch_pipeline.py --richclub-only
```

刷新后再运行 `src/step10_results_postprocess.py richclub-summary` 和
`src/step10_results_postprocess.py richclub-individual-means` 生成新的汇总表。

### 5.1 无向二值 rich-club：`LOFC/HOFC`

对每个有效度等级 k，取度大于 k 的节点集合：

$$
V_k=\{i\mid degree_i>k\}
$$

$$
N_k=|V_k|
$$

$$
E_k=\text{number of undirected edges inside }V_k
$$

无向 rich-club 系数：

$$
\Phi(k)=
\begin{cases}
\dfrac{E_k}{N_k(N_k-1)/2}, & N_k\geq2 \\
\operatorname{NaN}, & N_k<2
\end{cases}
$$

### 5.2 有向二值 rich-club：`aHOFC`

对每个有效度等级 k，使用总度：

$$
degree_i=out\_degree_i+in\_degree_i
$$

$$
V_k=\{i\mid degree_i>k\}
$$

$$
N_k=|V_k|
$$

$$
E_k=\text{number of directed edges inside }V_k
$$

有向 rich-club 系数：

$$
\Phi(k)=
\begin{cases}
\dfrac{E_k}{N_k(N_k-1)}, & N_k\geq2 \\
\operatorname{NaN}, & N_k<2
\end{cases}
$$

### 5.3 归一化 rich-club

代码默认生成 1000 个保持度分布的二值随机网络，随机重连迭代参数为 10。
每个随机网络计算一条 rich-club 曲线：

$$
\Phi_{random}^{(r)}(k),
\qquad r=1,\ldots,1000
$$

随机网络均值：

$$
\overline{\Phi}_{random}(k)
=
\operatorname{nanmean}_{r}
\Phi_{random}^{(r)}(k)
$$

归一化 rich-club：

$$
\Phi_{norm}(k)=
\begin{cases}
\dfrac{\Phi_{observed}(k)}{\overline{\Phi}_{random}(k)},
& \Phi_{observed}(k)\text{ valid and }\overline{\Phi}_{random}(k)>0 \\
\operatorname{NaN}, & \text{otherwise}
\end{cases}
$$

### 5.4 Weighted rich-club

加权 rich-club 使用原始 FC 权重，但 rich 节点集合仍由二值度 \(degree_i>k\) 定义。
令 rich 节点子图内实际存在 \(E_k\) 条边，其权重和为 \(W_k\)。令全图所有非零边权从大到小排序后，
前 \(E_k\) 条边权重和为 \(W_k^{rank}\)。加权 rich-club 系数为：

$$
\Phi_w^{ranknorm}(k)=\frac{W_k}{W_k^{rank}}
$$

加权 rich-club 还输出随机归一化统计：

```text
output/rich_club_weighted_stats/{fc_type}/{group}/{band}/{subject}.csv
results/weighted_richclub_stats_summary.csv
```

随机基线生成方式为：

1. 取原始权重矩阵的二值支撑图；
2. 保持二值度序列随机重连；
3. 将原图非零权重随机分配到随机网络边上；
4. 重复 `n_rand` 次，默认 1000。

输出字段为：

```text
Phi_w_observed       rich 节点内部原始权重和 W_k
Phi_w_ranknorm       W_k / W_k^{rank}
Phi_w_random_mean    随机网络 Phi_w_ranknorm 均值
Phi_w_random_std     随机网络 Phi_w_ranknorm 标准差
Phi_w_norm           Phi_w_ranknorm / Phi_w_random_mean
Phi_w_empirical_p    P(random >= observed) 的单侧经验 p 值
Phi_w_random_valid_n 某一 k 层级下有效随机曲线数量
```

这里 `observed` 指 `Phi_w_ranknorm`，不是 `Phi_w_observed` 原始权重和。若
`Phi_w_random_mean` 不为有限正数，代码令 `Phi_w_norm` 为 NaN。

后处理命令：

```powershell
uv run python src/step10_results_postprocess.py weighted-richclub-summary
uv run python src/step10_results_postprocess.py weighted-richclub-individual-means
uv run python src/step10_results_postprocess.py weighted-richclub-stats
```

### 5.5 Rich-club 个体均值

`results/richclub_individual_means_{FC}.csv` 中的个体均值为一名被试整条归一化
rich-club 曲线的均值：

$$
\operatorname{RichClubMean}_{subject}
=
\operatorname{nanmean}_{k}
\left(\Phi_{norm,subject}(k)\right)
$$

### 5.6 Rich-club 组均值曲线

`results/richclub_summary_{FC}.csv` 中每个组、每个频段的曲线为被试间均值：

$$
\operatorname{RichClubGroupMean}(k)
=
\operatorname{nanmean}_{subject}
\left(\Phi_{norm,subject}(k)\right)
$$

### 5.7 Rich-club 特征提取

Rich-club 特征表输出到：

```text
results/richclub_feature_summary.csv
results/weighted_richclub_feature_summary.csv
results/density_richclub_features.csv
```

对一条 rich-club 曲线 \(\Phi(k)\)，有效 k 集合记为：

$$
K_{valid}=\{k\mid \Phi(k)\text{ is finite}\}
$$

全曲线均值：

$$
\operatorname{RichClubMean\_all}
=
\operatorname{mean}_{k\in K_{valid}}\Phi(k)
$$

高 k 区间定义为有效 k 中排序最高的 30%：

$$
K_{high}=\text{top 30\% of }K_{valid}
$$

高 k 均值：

$$
\operatorname{RichClubMean\_high\_k}
=
\operatorname{mean}_{k\in K_{high}}\Phi(k)
$$

峰值与峰值位置：

$$
\operatorname{RichClubMax}=\max_{k\in K_{valid}}\Phi(k)
$$

$$
\operatorname{RichClubPeakK}
=
\arg\max_{k\in K_{valid}}\Phi(k)
$$

增强 k 数量当前按 \(\Phi(k)>1\) 计数，表示高于随机网络均值或 rank-normalized
基准，不直接等同于统计显著：

$$
\operatorname{RichClubEnhancedKCount}
=
\sum_{k\in K_{valid}}\mathbf{1}\{\Phi(k)>1\}
$$

weighted rich-club 的统计显著性优先使用 `Phi_w_empirical_p`。若需要统计显著 k 数量，
应基于经验 p 值定义：

$$
\operatorname{RichClubSignificantKCount}
=
\sum_{k\in K_{valid}}\mathbf{1}\{p_{empirical}(k)<0.05\}
$$

多稀疏度 rich-club 特征在每个密度下分别计算上述特征；
跨密度 AUC 指标 `RichClub_auc_*` 在 `output/all_scalar_metrics_summary.csv`
中保存，用于评估 rich-club 结论对阈值选择的稳健性。

## 6. 靶向攻击指标

靶向攻击输出到：

```text
output/target_attack/{fc_type}/{group}/{band}/{subject}.csv
output/target_attack_controls/{strategy}/{fc_type}/{group}/{band}/{subject}.csv
output/target_attack_density/{fc_type}/{group}/{band}/{subject}.csv
results/targetattack_feature_summary.csv
results/density_targetattack_features.csv
```

单被试靶向攻击曲线按纵向一列保存，每一行对应一个 `Attack_Step`。

该指标使用阈值 0.35 后的二值图。代码在攻击入口会清理输入：检查矩阵为方阵，
将 NaN/inf 和负值置 0，去除自连接，并要求输入已经是 0/1 二值邻接矩阵。
`LOFC/HOFC` 按无向二值网络处理，计算前会对称化；`aHOFC` 按有向二值网络处理。

### 6.1 初始全局效率

$$
E_{glob}^{0}=\operatorname{Eglob}_{bin}(G)
$$

`aHOFC` 的全局效率按有向路径计算；`LOFC/HOFC` 的全局效率按对称化后的无向图计算。

### 6.2 攻击顺序

主输出 `output/target_attack/` 使用固定 CE 排序靶向攻击。即在攻击前先计算每个节点的初始 CE_i，再按 CE
降序排序：

$$
order=\operatorname{argsort}(-CE)
$$

实际排序使用 `np.lexsort((index, -degree, -score, -ce))`，因此主键为 CE 降序，
平局时依次按 TOPSIS 综合中心性得分降序、二值度降序、节点索引升序排序。

对照攻击策略同步保存到 `output/target_attack_controls/`：

```text
ce             固定初始 CE 排序
ce_adaptive    每移除一个节点后重新计算 CE
degree         初始二值度排序；aHOFC 使用入度+出度
betweenness    初始 betweenness 排序
random_mean    random attack 多次重复均值
random_std     random attack 多次重复标准差
```

random attack 默认重复 1000 次，可通过 `--random-attack-repeats` 调整。
`random_mean` 和 `random_std` 不是两种节点攻击策略，而是随机节点排列重复后的
逐步均值和样本标准差；`random_std` 的 AUC 仅用于描述随机曲线离散程度，
不应解释为一次独立攻击造成的网络效率。

多稀疏度攻击会在每个固定稀疏度 \(d\in D\) 下重新计算 CE 攻击曲线，并保存：

```text
AttackAUC
EarlyDrop_1
EarlyDrop_2
EarlyDrop_3
MaxDrop
```

### 6.3 节点移除

第 s 步移除排序中的第 s 个节点，即将该节点所有入边和出边清零：

$$
G_{:,node_s}=0
$$

$$
G_{node_s,:}=0
$$
被移除节点保留在矩阵中并变为孤立节点，全局效率仍在原始节点集合上计算。
因此该曲线表示原始网络规模不变时，节点失效导致的全网络通信效率损失。

### 6.4 归一化攻击效率曲线

输出曲线从攻击前开始记录。当 E_{glob}^{0}>0 时，第 0 个点为：

$$
\operatorname{TargetAttack}(0)=1
$$

当已移除 s 个节点且 s \ge 1 时：

$$
\operatorname{TargetAttack}(s)=
\begin{cases}
\dfrac{\operatorname{Eglob}_{bin}(G_{after\ step\ s})}{E_{glob}^{0}},
& E_{glob}^{0}>0 \\
0, & E_{glob}^{0}=0
\end{cases}
$$

若 E_{glob}^{0}=0，则无法归一化，代码直接返回全 0 曲线。

### 6.5 靶向攻击组均值曲线

`results/targetattack_summary_{FC}.csv` 中 `Attack_Step` 表示已移除节点数，从 0
开始；每个组、每个频段、每个攻击步骤的值为：

$$
\operatorname{TargetAttackGroupMean}(s)
=
\operatorname{nanmean}_{subject}
\left(\operatorname{TargetAttack}_{subject}(s)\right)
$$

### 6.6 攻击特征与对照差异

后处理将 attack 曲线压缩为个体级特征：

```text
AttackAUC_CE
AttackAUC_CEAdaptive
AttackAUC_Degree
AttackAUC_Betweenness
AttackAUC_RandomMean
EarlyDrop_1
EarlyDrop_2
EarlyDrop_3
MaxDrop_CE
CE_vs_Random_AUC_Diff
CE_vs_Degree_AUC_Diff
CE_vs_Betweenness_AUC_Diff
CE_vs_AdaptiveCE_AUC_Diff
```

其中：

$$
\operatorname{CE\_vs\_Random\_AUC\_Diff}
=
\operatorname{AUC}_{random}
-
\operatorname{AUC}_{CE}
$$

该值越大，表示 CE 攻击相对随机攻击造成更强的网络效率破坏。
同理，`CE_vs_Degree_AUC_Diff`、`CE_vs_Betweenness_AUC_Diff`、
`CE_vs_AdaptiveCE_AUC_Diff` 均按“对照 AUC - CE AUC”计算。由于曲线是
归一化全局效率，AUC 越低代表网络效率下降越快、攻击破坏性越强。

## 7. 代表性核心节点指标

`results/representative_core_nodes_threshold_30.csv` 和对应汇总表使用核心节点出现比例：

$$
Count_i=
\#\{subject\mid \operatorname{IsCoreNode}_{i,subject}=1\}
$$

$$
TotalSubjects=\text{number of valid subjects}
$$

$$
Proportion_i=\frac{Count_i}{TotalSubjects}
$$

$$
Percentage_i=100\%\times Proportion_i
$$

默认代表性核心节点阈值为 0.30：

$$
\operatorname{RepresentativeCore}_i=
\begin{cases}
1, & Proportion_i\geq0.30 \\
0, & Proportion_i<0.30
\end{cases}
$$

### 7.1 核心节点跨维度稳定性

核心节点稳定性输出到：

```text
results/core_node_subject_stability.csv
results/core_node_band_stability.csv
results/core_node_fctype_stability.csv
results/core_node_overall_stability.csv
```

被试间稳定性：

$$
\operatorname{SubjectStability}_{i,g,b,f}
=
\frac{
\#\{s\in g\mid i\text{ is core in subject }s, band b, FC type f\}
}{
\#\{s\in g\}
}
$$

频段间稳定性：

$$
\operatorname{BandStability}_{i,g,f}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}\{\operatorname{SubjectStability}_{i,g,b,f}\ge\theta_{rep}\}
$$

FC 类型间稳定性：

$$
\operatorname{FCTypeStability}_{i,g,b}
=
\frac{1}{F}
\sum_{f=1}^{F}
\mathbf{1}\{\operatorname{SubjectStability}_{i,g,b,f}\ge\theta_{rep}\}
$$

总体核心稳定性：

$$
\operatorname{OverallCoreStability}_{i,g}
=
\frac{1}{BF}
\sum_{b=1}^{B}\sum_{f=1}^{F}
\mathbf{1}\{\operatorname{SubjectStability}_{i,g,b,f}\ge\theta_{rep}\}
$$

其中 \(\theta_{rep}\) 默认为 0.30，并同时输出 0.20、0.30、0.40 三档代表性节点表。

## 8. FC 稳定性分析

FC 稳定性输出到：

```text
output/fc_stability/{group}/{band}/{subject}.csv
results/fc_stability_summary.csv
```

每个被试的时间序列被切分为若干连续 epoch。每个 epoch 独立计算 LOFC、HOFC、aHOFC。
对同一被试同一 FC 类型的 epoch 矩阵，取非对角边向量，计算两两矩阵相关：

$$
r_{pq}=\operatorname{corr}(\operatorname{vec}(W^{(p)}),\operatorname{vec}(W^{(q)}))
$$

矩阵相关均值和标准差：

$$
\operatorname{MatrixCorr\_Mean}
=
\operatorname{mean}_{p<q}(r_{pq})
$$

$$
\operatorname{MatrixCorr\_SD}
=
\operatorname{std}_{p<q}(r_{pq})
$$

对每条边 \(e\)，跨 epoch 变异系数为：

$$
CV_e=
\begin{cases}
\dfrac{\operatorname{std}_{epoch}(W_e)}{|\operatorname{mean}_{epoch}(W_e)|},
& |\operatorname{mean}_{epoch}(W_e)|\ge\varepsilon \\
\operatorname{NaN}, & |\operatorname{mean}_{epoch}(W_e)|<\varepsilon
\end{cases}
$$

当前 \(\varepsilon=10^{-12}\)。近零均值边的 CV 不参与 `EdgeCV_Mean`。

有效边集合：

$$
E_{valid}=\{e\mid CV_e\text{ is finite}\}
$$

边 CV 均值：

$$
\operatorname{EdgeCV\_Mean}
=
\operatorname{mean}_{e\in E_{valid}}(CV_e)
$$

稳定边比例当前定义为 \(CV_e\le 0.20\) 的有效边比例：

$$
\operatorname{StableEdgeRatio}
=
\frac{\#\{e\in E_{valid}\mid CV_e\le0.20\}}{\#E_{valid}}
$$

## 9. 结果重组统计量

`results/重组数据_按指标分组.csv` 对标量指标按组和频段计算均值与标准误：

$$
\operatorname{Mean}
=
\frac{1}{n}\sum_{i=1}^{n}x_i
$$

$$
\operatorname{SEM}
=
\frac{s}{\sqrt{n}}
$$

其中 s 使用样本标准差：

$$
s=
\sqrt{
\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})^2
}
$$

## 10. 显著性检验

统计检验不改变原始指标，只对指标做组间比较。输出包括
`results/显著性分析.csv`、`results/richclub_显著性分析.csv`、
`results/targetattack_显著性分析.csv` 和
`results/targetattack_stepwise_补充分析.csv`。

### 10.1 正态性检验

每组样本量 n >= 3 时使用 Shapiro-Wilk 检验：

$$
normal=
\begin{cases}
True, & p_{shapiro}>0.05 \\
False, & p_{shapiro}\leq0.05
\end{cases}
$$

若样本量不足 3，或该组数据为常数，视为不满足正态性。

### 10.2 三组总体检验

每个指标先做三组总体检验，再根据总体检验结果决定是否做事后两两比较。
若三组均满足 Shapiro-Wilk 正态性，且 Levene 方差齐性检验满足：

$$
p_{Levene}>0.05
$$

则使用 one-way ANOVA，并输出 eta squared：

$$
\eta^2=\frac{SS_{between}}{SS_{total}}
$$

若正态性或方差齐性任一条件不满足，则使用 Kruskal-Wallis 检验，并输出
epsilon squared：

$$
\epsilon^2=\frac{H-k+1}{n-k}
$$

其中 H 为 Kruskal-Wallis 统计量，k 为组数，n 为总样本量。代码实现会将
该值裁剪到 [0, 1] 区间。

### 10.3 事后两两比较

仅当三组总体检验的原始 p 值 p_raw < 0.05 时，才执行 HC vs MCI、HC vs AD、
MCI vs AD 事后两两比较。

若两组都满足正态性，使用 Welch's t-test：

$$
t=
\frac{\bar{x}_1-\bar{x}_2}
{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}
$$

代码调用 `ttest_ind(data1, data2, equal_var=False)`，并输出 Hedges' g
及其 bootstrap 95% 置信区间。

若任一组不满足正态性，使用双侧 Mann-Whitney U 检验：

$$
U=\operatorname{mannwhitneyu}(x_1,x_2,\operatorname{two\text{-}sided})
$$

代码调用 `mannwhitneyu(data1, data2, alternative="two-sided")`，统计量以
SciPy 返回值为准，并输出
Cliff's delta 及其 bootstrap 95% 置信区间。

### 10.4 多重比较校正

统计表保留原始 p 值 `p_raw`，并使用 Benjamini-Hochberg FDR 生成校正后 p 值
`p_adj`。总体检验和事后检验分别在各自输出表的同一层级内校正。
当前 `Correction_Family` 的主要分类为：

```text
Scalar primary | FC_Type
Scalar weighted topology | FC_Type
Scalar density AUC | FC_Type
Scalar density curve | FC_Type
Scalar QC | FC_Type
Scalar exploratory | FC_Type
layer_similarity
ce_ablation_attack
core_stability | FC_Type
fc_stability | FC_Type
density_sensitivity | FC_Type
richclub_features | FC_Type
weighted_richclub_features | FC_Type
weighted_ce_overlap | FC_Type
Attack features | FC_Type
Exploratory | FC_Type
```

其中 `FC_Type` 为 `LOFC`、`HOFC` 或 `aHOFC/AHOFC`，按源码从指标名末尾解析。
`results/显著性分析.csv` 使用 `Scalar ...` 族名；`mixed_effects_metrics.csv`
使用同一 `_metric_family()` 逻辑中的非 `Scalar` 族名。
显著性标记以校正后 p 值为主：

$$
\operatorname{Sig}(p)=
\begin{cases}
***, & p<0.001 \\
**, & 0.001\leq p<0.01 \\
*, & 0.01\leq p<0.05 \\
n.s., & p\geq0.05
\end{cases}
$$

输出列包括 `p_raw`、`p_adj`、`Correction_Method`、`Significant_adj`、
`Effect_Size`、`Effect_Size_Type`、`CI_95_Lower` 和 `CI_95_Upper`。

### 10.5 Target attack 主分析

Target attack 主统计不再以逐攻击步检验作为核心结果，而是先将每名被试的归一化
效率曲线压缩为曲线下面积 AUC：

$$
AUC=\frac{1}{x_m-x_0}\int_{x_0}^{x_m} TargetAttack(x)\,dx
$$

代码使用离散梯形积分计算 AUC。`results/targetattack_显著性分析.csv` 保存
AUC 的三组总体检验和事后两两比较。逐步检验仅作为补充分析输出到
`results/targetattack_stepwise_补充分析.csv`，并同样执行总体检验和 FDR 校正。

### 10.6 Group x Band 与重复测量模型

标量汇总表额外输出两类模型：

```text
results/two_way_anova_metrics.csv
results/mixed_effects_metrics.csv
```

`two_way_anova_metrics.csv` 使用固定效应探索模型：

$$
Metric \sim Group * Band
$$

该模型用于快速查看组别主效应、频段主效应和交互趋势，但不把 Band 作为被试内重复测量处理，
因此定位为探索性分析。

`mixed_effects_metrics.csv` 在不引入额外依赖的条件下采用重复测量 fallback。
表内 `Model_Type` 固定为 `repeated_measures_fallback`，避免将其误解为完整随机截距
linear mixed-effects model：

```text
Band 和 Group x Band: Subject 固定截距的 nested F test
Group: 每名被试跨频段均值的一元组间 ANOVA
```

对应口径为：

$$
Metric \sim Subject + Band + Group:Band
$$

其中 `Subject` 固定截距用于吸收同一被试跨频段重复测量带来的相关性。
`mixed_effects_metrics.csv` 定位为重复测量探索分析，并输出 FDR 校正结果；
其 `Model_Type` 标识为 `repeated_measures_fallback`，`Correction_Family` 仍沿用
指标所属比较族，并额外按 `Effect` 分层校正。

若后续安装 `statsmodels` 并使用真正随机截距模型，可将其升级为：

$$
Metric \sim Group * Band + (1|Subject)
$$

## 11. 组均值矩阵与 FC 热图

### 11.1 组均值 FC 矩阵

`src/step09_batch_pipeline.py` 在处理完每个组别/频段后，会从
`output/fc_matrices/{fc_type}/{group}/{band}/` 读取该条件下所有被试 FC 矩阵。
当前实现只纳入形状为 `16 x 16` 的矩阵。对每个矩阵元素按被试求算术平均：

$$
\overline{W}_{ij}^{group,band,fc}
=
\frac{1}{M}\sum_{s=1}^{M}W_{ij}^{(s)}
$$

其中 \(M\) 为该组别、该频段、该 FC 类型下有效被试矩阵数。输出到：

```text
output/mean/{group}/{band}/{group}_{band}_{fc_type}.csv
```

### 11.2 FC 热图 `step12_fc_heatmaps.py`

热图脚本读取上述组均值矩阵，并按每一种 FC 类型分别构造一个超级矩阵。行方向为组别：

```text
HC, MCI, AD
```

列方向为频段：

```text
Delta, Theta, Alpha, Beta, Gamma
```

每个子矩阵大小为 \(16\times16\)。设组别数为 \(G=3\)、频段数为 \(B=5\)、节点数为 \(N=16\)，
则每个 FC 类型对应的超级矩阵大小为：

$$
W_{super}\in\mathbb{R}^{(G N)\times(B N)}
=\mathbb{R}^{48\times80}
$$

绘图前将每个子矩阵对角线设置为 NaN，使自连接在热图中显示为空白：

$$
W_{ii}=\operatorname{NaN}
$$

热图使用 `bwr` 色图，固定颜色范围：

$$
v_{min}=0,\qquad v_{max}=1
$$

脚本分别保存三张无文字热图和一张独立颜色条：

```text
results/fc_heatmaps/lofc_heatmap.png
results/fc_heatmaps/hofc_heatmap.png
results/fc_heatmaps/ahofc_heatmap.png
results/fc_heatmaps/colorbar.png
```

若某个组均值矩阵文件缺失，代码会用同尺寸全 NaN 空白矩阵代替该子图。

## 12. 当前不输出的指标

当前主流水线不输出以下指标：

```text
weighted small-worldness sigma
characteristic path length
normalized clustering gamma
normalized path length lambda
```

如果后续需要这些指标，应先在对应的 `src/step*.py` 模块中明确实现公式和随机网络归一化口径，
再加入 `src/step09_batch_pipeline.py` 的输出。

## 13. 主要输出文件

```text
output/fc_matrices/{fc_type}/{group}/{band}/{subject}.csv
output/fc_matrices/lofc_signed/{group}/{band}/{subject}.csv
output/fc_layer_similarity/{group}/{band}/{subject}.csv
output/core_nodes/{fc_type}/{group}/{band}/{subject}_core_nodes.csv
output/core_nodes_density/{fc_type}/{group}/{band}/{subject}.csv
output/rich_club/{fc_type}/{group}/{band}/{subject}.csv
output/rich_club_weighted/{fc_type}/{group}/{band}/{subject}.csv
output/rich_club_weighted_stats/{fc_type}/{group}/{band}/{subject}.csv
output/target_attack/{fc_type}/{group}/{band}/{subject}.csv
output/target_attack_controls/{strategy}/{fc_type}/{group}/{band}/{subject}.csv
output/target_attack_density/{fc_type}/{group}/{band}/{subject}.csv
output/ce_ablation_attack/{method}/{fc_type}/{group}/{band}/{subject}.csv
output/network_qc/{fc_type}/{group}/{band}/{subject}_qc.csv
output/mean/{group}/{band}/
output/all_scalar_metrics_summary.csv
results/fc_heatmaps/{lofc,hofc,ahofc}_heatmap.png
results/fc_heatmaps/colorbar.png
results/重组数据_按指标分组.csv
results/显著性分析.csv
results/richclub_summary_{FC}.csv
results/richclub_individual_means_{FC}.csv
results/weighted_richclub_summary_{FC}.csv
results/weighted_richclub_individual_means_{FC}.csv
results/weighted_richclub_stats_summary.csv
results/richclub_feature_summary.csv
results/weighted_richclub_feature_summary.csv
results/richclub_显著性分析.csv
results/targetattack_summary_{FC}.csv
results/targetattack_{strategy}_summary_{FC}.csv
results/targetattack_feature_summary.csv
results/ce_ablation_attack_feature_summary.csv
results/core_node_density_stability.csv
results/density_richclub_features.csv
results/density_targetattack_features.csv
results/fc_layer_similarity.csv
results/fc_stability_summary.csv
results/two_way_anova_metrics.csv
results/mixed_effects_metrics.csv
results/targetattack_显著性分析.csv
results/targetattack_stepwise_补充分析.csv
results/network_qc_summary.csv
results/network_qc_group_summary.csv
results/core_node_subject_stability.csv
results/core_node_band_stability.csv
results/core_node_fctype_stability.csv
results/core_node_overall_stability.csv
results/representative_core_nodes_threshold_{20,30,40}.csv
results/representative_core_nodes_summary_{FC}_threshold_{20,30,40}.csv
```

## 14. 参考文献

1. Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity:
   Uses and interpretations. *NeuroImage*, 52(3), 1059-1069.
   https://doi.org/10.1016/j.neuroimage.2009.10.003
2. Harvy, J., Sigala, R., Wairagkar, M., Kuppuswamy, A., & Nazarpour, K. (2019).
   Between-frequency topographical and dynamic high-order functional connectivity
   for driving drowsiness assessment. *IEEE Transactions on Neural Systems and
   Rehabilitation Engineering*, 27(3), 358-367.
   https://doi.org/10.1109/TNSRE.2019.2893949
3. Brain Connectivity Toolbox. List of graph measures.
   https://sites.google.com/site/bctnet/list-of-measures
4. Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective
   weights in multiple criteria problems: The CRITIC method. *Computers &
   Operations Research*, 22(7), 763-770.
   https://doi.org/10.1016/0305-0548(94)00059-H
5. Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making:
   Methods and Applications*. Springer.
   https://doi.org/10.1007/978-3-642-48318-9
