# CE

面向 HC、MCI 和 AD 组的 EEG 脑网络分析流水线。当前主实验读取 EEGLAB
`.set` 数据，计算 LOFC/HOFC 功能连接矩阵，提取图论指标、CE/Hub 节点、
靶向攻击和稳定性指标，执行组水平统计检验，并导出 FC 热图。

方法公式见 [docs/algorithm_summary.md](docs/algorithm_summary.md)，脚本输入输出契约见
[docs/script_io_formats.md](docs/script_io_formats.md)，本轮实验结论见
[docs/results_summary.md](docs/results_summary.md)。

## 环境

```powershell
uv sync
```

快速语法和测试检查：

```powershell
uv run python -m compileall -q src tests
uv run pytest -q
```
## 当前实验范围

当前主流程只保留四个频段：

| 频段 | 目录 |
| --- | --- |
| Delta | `1-4(Delta)` |
| Theta | `4-8(Theta)` |
| Alpha | `8-12(Alpha)` |
| Beta | `12-30(Beta)` |

当前主流程只保留两个 FC 类型：

| FC 类型 | 文件夹 | 说明 |
| --- | --- | --- |
| LOFC | `lofc` | Hilbert 包络相关，按无向网络分析 |
| HOFC | `hofc` | LOFC 拓扑轮廓相关，按无向网络分析 |

Gamma 频段、aHOFC 网络和 rich-club 相关实验已从主流程和兼容导出层删除；
`step09`、`step10`、`step11` 不再包含这些旧实验入口。

## 目录结构

```text
src/
  pipeline_config.py        组别、频段、FC 类型和阈值配置
  brain_metrics.py          兼容导出层
  step00_bandpass_filter.py 从 1-40 Hz 数据生成配置频段 .set 文件
  step01_eeg_io.py          EEG 数据读取
  step02_connectivity.py    LOFC / HOFC 连接矩阵
  step03_connection_strength.py 平均连接强度
  step04_graph_metrics.py   图论指标
  step05_hub_detection.py   CE / Hub 节点识别
  step07_target_attack.py   靶向攻击模拟
  step08_summary_metrics.py 被试级标量指标
  step09_batch_pipeline.py  主批处理
  step10_results_postprocess.py 结果汇总
  step11_statistical_analysis.py 统计检验
  step12_fc_heatmaps.py     FC 热图

data/raw/                   输入 EEG 数据
output/                     中间矩阵和被试级输出
results/                    汇总表、统计表和图像
```

## 常用命令

生成四个配置频段数据：

```powershell
uv run python src/step00_bandpass_filter.py --skip-existing
```

运行主批处理：

```powershell
uv run python src/step09_batch_pipeline.py
uv run python src/step09_batch_pipeline.py --group HC --band "1-4(Delta)"
uv run python src/step09_batch_pipeline.py --fc-only
```

后处理、统计和热图：

```powershell
uv run python src/step10_results_postprocess.py
uv run python src/step11_statistical_analysis.py
uv run python src/step12_fc_heatmaps.py
```

CAUEEG 复现流程（数据集只读，输出写到 `{data,output,results}/caueeg/`，详见 `docs/caueeg_analysis_plan.md`、`docs/script_io_formats.md`）：

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

## 主要输出

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

results/显著性分析.csv
results/two_way_anova_metrics.csv
results/mixed_effects_metrics.csv
results/targetattack_显著性分析.csv
results/targetattack_stepwise_补充分析.csv
results/targetattack_feature_summary.csv
results/ce_ablation_attack_feature_summary.csv
results/core_node_density_stability.csv
results/density_targetattack_features.csv
results/fc_layer_similarity.csv
results/fc_stability_summary.csv
results/network_qc_summary.csv
results/network_qc_group_summary.csv
results/representative_core_nodes_threshold_{20,30,40}.csv
results/fc_heatmaps/lofc_hofc_heatmap.{png,svg}
results/fc_heatmaps/{lofc,hofc}_heatmap.{png,svg}
results/fc_heatmaps/colorbar.png
```
