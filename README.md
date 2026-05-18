# CE

面向 HC、MCI 和 AD 组的 EEG 脑网络分析流水线。

本项目读取 EEGLAB `.set` 数据，计算功能连接矩阵，提取图论指标，
汇总后处理结果，执行组水平统计检验，并导出 FC 热图。

方法层面的说明、公式和参考文献见
[docs/algorithm_summary.md](docs/algorithm_summary.md)。

## 环境

本仓库使用 Python 3.12 和 `uv`。

```powershell
uv sync
```

主要依赖定义在 `pyproject.toml` 中，包括 `numpy`、`pandas`、`scipy`、
`h5py`、`networkx`、`matplotlib` 和 `tqdm`。

## 频段

项目统一使用以下 EEG 频段目录名称：

| 频段 | 目录 |
| --- | --- |
| Delta | `1-4(Delta)` |
| Theta | `4-8(Theta)` |
| Alpha | `8-12(Alpha)` |
| Beta | `12-30(Beta)` |
| Gamma | `30-40(Gamma)` |

## 功能连接方法

FC 实现遵循 Harvy et al. 2019 描述的 EEG 拓扑功能连接工作流：

- `LOFC`：Hilbert 包络相关。先通过 Hilbert 变换将 EEG 通道信号转换为
  解析振幅，再计算通道包络之间的 Pearson 相关绝对值。
- `HOFC`：拓扑高阶功能连接。对 LOFC 列轮廓进行 Fisher-z 变换，排除自身
  和交叉轮廓条目后计算相关，得到对称的高层拓扑 FC 矩阵。
- `aHOFC`：关联高阶功能连接。对 LOFC 轮廓和 HOFC 轮廓进行 Fisher-z 变换后
  计算相关。由于两个轮廓层级分别承担不同的源/目标角色，结果矩阵是非对称的。

## 图论指标

当前流水线输出的是**二值阈值化**图论指标，统一二值化阈值为 `0.35`。
指标名包含 `_bin`，用于明确这一点。
同时也输出原始 FC 矩阵上的平均连接强度：

- `Eglob_bin_*`：BCT 风格的二值全局效率，基于阈值化图计算。
- `Clustering_bin_*`：BCT 风格的二值聚类系数，基于阈值化图计算；
  LOFC/HOFC 使用无向聚类，aHOFC 使用有向聚类。
- `MeanConn_*`：平均连接强度。LOFC/HOFC 统计上三角连接权重；aHOFC 统计全部非对角有向连接权重。
- Rich-club 输出为归一化的**二值** rich-club 曲线。LOFC/HOFC 使用无向二值
  rich-club，aHOFC 使用有向二值 rich-club。

当前不会输出加权 rich-club 和加权小世界指标。

## 目录结构

```text
src/
  brain_metrics.py          兼容导出层（继续支持 import brain_metrics as bm）
  step00_bandpass_filter.py 从 1-40 Hz 数据生成各频段 .set 文件
  step01_eeg_io.py          EEG 数据读取
  step02_connectivity.py    LOFC / HOFC / aHOFC 连接矩阵
  step03_connection_strength.py 平均连接强度（MeanConn）
  step04_graph_metrics.py   图论指标
  step05_hub_detection.py   CE / Hub 节点识别
  step06_rich_club.py       二值归一化 rich-club
  step07_target_attack.py   靶向攻击模拟
  step08_summary_metrics.py 单被试标量指标汇总
  step09_batch_pipeline.py  批量计算 FC 矩阵和网络指标
  step10_results_postprocess.py 汇总 rich-club、目标攻击和核心节点结果
  step11_statistical_analysis.py 执行组水平统计检验
  step12_fc_heatmaps.py     从组均值矩阵导出 FC 热图

data/
  raw/                      按诊断组和频段组织的输入 EEG 数据

output/                     计算得到的 FC 矩阵、指标和中间 CSV 文件
results/                    最终汇总 CSV 文件和图像
```

## 常用命令

从 `data/raw/{Group} 256 100s/1-40` 生成各频段数据：

```powershell
uv run python src/step00_bandpass_filter.py
uv run python src/step00_bandpass_filter.py --group HC
uv run python src/step00_bandpass_filter.py --skip-existing
```

滤波脚本会从每个 `.set` 文件中读取采样率，并支持 MAT 风格 `.set` 文件以及
引用 `.fdt` 数据的 EEGLAB HDF5 `.set` 文件。输出会保存为本项目流水线使用的
MAT 风格 `.set` 文件。

运行完整批处理流水线：

```powershell
uv run python src/step09_batch_pipeline.py
```

运行单个组和频段：

```powershell
uv run python src/step09_batch_pipeline.py --group HC --band "1-4(Delta)"
```

只生成 FC 矩阵：

```powershell
uv run python src/step09_batch_pipeline.py --fc-only
```

从已有 FC 矩阵刷新 rich-club 输出：

```powershell
uv run python src/step09_batch_pipeline.py --richclub-only
uv run python src/step09_batch_pipeline.py --richclub-only --group AD --band "1-4(Delta)"
```

后处理结果：

```powershell
uv run python src/step10_results_postprocess.py
```

运行统计分析：

```powershell
uv run python src/step11_statistical_analysis.py
```

生成 FC 热图：

```powershell
uv run python src/step12_fc_heatmaps.py
```

## 输出约定

批处理流水线会将结果写入 `output/`：

```text
output/fc_matrices/{fc_type}/{group}/{band}/{subject}.csv
output/core_nodes/{fc_type}/{group}/{band}/{subject}_core_nodes.csv
output/rich_club/{fc_type}/{group}/{band}/{subject}.csv
output/target_attack/{fc_type}/{group}/{band}/{subject}.csv
output/mean/{group}/{band}/
output/all_scalar_metrics_summary.csv
```

后处理和统计分析会将最终表格与图像写入 `results/`。
