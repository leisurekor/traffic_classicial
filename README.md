# traffic_classicial

面向恶意流量检测研究的仓库，当前以两条稳定主线为核心：

1. `CSV` 主实验线：用于验证模型在标准流特征表格输入上的二分类能力。
2. `CTU13` 图侧主 benchmark：用于验证从原始 `PCAP -> flow -> graph -> feature -> scorer/classifier` 的下游链路可行性。

当前正式报告默认关闭 nuisance-aware：

- `use_nuisance_aware=false`

这份仓库已经包含：

- 代码
- 轻量结果输出
- 关键日志
- 最终实验报告

但不包含超大原始数据和超大中间产物，例如：

- `data/`
- `artifacts/`
- `outputs/runs/`

因此，别的机器克隆仓库后可以直接阅读代码、结果和报告；如果要重新跑实验，需要自行准备数据集。

## 1. 从哪里开始看

如果你是第一次看这个项目，建议按下面顺序阅读：

1. 最终报告：
   - [reports/final_experiment_report.md](reports/final_experiment_report.md)
2. 主结果汇总：
   - [outputs/summary/all_metrics.csv](outputs/summary/all_metrics.csv)
   - [outputs/summary/all_metrics.json](outputs/summary/all_metrics.json)
3. 图侧 benchmark：
   - [results/ctu13_edge_centric_comparison.md](results/ctu13_edge_centric_comparison.md)
   - [results/ctu13_binary_benchmark.md](results/ctu13_binary_benchmark.md)
4. 真实图表：
   - `reports/assets/`
   - `outputs/figures/`
5. 实验日志：
   - `outputs/logs/`

## 2. 当前正式纳入的实验结论

当前正式纳入报告的结果只有两部分：

1. `CICIoT2023` 的 `merge01.csv`
   - 作用：证明模型在标准流特征表格输入上的分类能力。
2. `CTU13` 场景 `48 / 49 / 52`
   - 作用：作为 `PCAP-based graph benchmark`
   - 更准确地说，是“基于原始 PCAP 构建流图后的 benchmark”

注意：

- 这不等于“完整跨数据集 PCAP 主实验矩阵已经完成”
- 当前仓库没有把多数据集完整 PCAP 体系作为已完成结论来声称

## 3. 仓库结构速览

```text
configs/                         统一实验配置样例
docs/                            补充文档
outputs/
  summary/                       轻量汇总指标
  metrics/                       单次实验配置与指标 JSON
  figures/                       PNG 图表
  logs/                          运行日志
  data_inventory/                数据清单与来源索引
reports/
  final_experiment_report.md     最终实验报告
  assets/                        报告内嵌图片副本
results/                         CTU13 等 benchmark/诊断表
scripts/                         入口脚本、汇总脚本、下载脚本
src/traffic_graph/               核心实现
tests/                           回归测试与 smoke tests
```

核心代码大致分布如下：

- `src/traffic_graph/data/`
  - CSV/PCAP 数据准备
- `src/traffic_graph/features/`
  - 特征打包与清洗
- `src/traffic_graph/graph/`
  - 图构建逻辑
- `src/traffic_graph/models/`
  - 模型定义
- `src/traffic_graph/pipeline/`
  - 训练、评估、benchmark、CTU13 图侧实验主逻辑
- `src/traffic_graph/datasets/`
  - 数据集发现、CTU13 manifest 等

## 4. 环境安装

### 4.1 基础依赖

仓库使用 `pyproject.toml` 管理基础依赖。最低要求：

- Python `>=3.10`

安装基础环境：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

如果要跑带图模型或部分 benchmark，建议安装带 `torch` 的可选依赖：

```bash
pip install -e .[gae]
```

### 4.2 快速自检

```bash
python3 -m compileall src tests scripts
python3 -m unittest tests.test_ctu13_pipeline -v
python3 -m unittest tests.test_repro_experiment -v
```

## 5. 数据集准备

### 5.1 CICIoT2023

当前仓库里稳定使用过的 CSV 主结果来自：

- `artifacts/cic_iot2023/Merged01.csv`

如果你要复现实验：

1. 从官方页面下载 `CICIoT2023` 数据集。
2. 准备好包含 `Label` 列的 merge CSV。
3. 推荐放到：
   - `artifacts/cic_iot2023/`

当前示例配置默认使用：

- `artifacts/cic_iot2023/Merged01.csv`

官方来源可参考：

- `https://www.unb.ca/cic/datasets/iotdataset-2023.html`

### 5.2 CICIDS2017

仓库已经有面向 `Wednesday / Thursday / Friday` 的搜索逻辑，但是否能直接跑，取决于你本地是否放置了对应文件。

建议准备：

- CSV 文件
- PCAP 文件

并放在 `data/` 或 `artifacts/` 目录下，脚本会递归搜索。

官方来源可参考：

- `https://www.unb.ca/cic/datasets/ids-2017.html`

### 5.3 CTU13

仓库提供了官方 CTU13 资源下载脚本：

```bash
python3 scripts/download_ctu13.py --scenarios 48 49 52
```

默认会把数据组织到：

- `data/ctu13/raw/`
- `data/ctu13/ctu13_manifest.json`

官方来源可参考：

- `https://www.stratosphereips.org/datasets-ctu13`

注意：

- 当前仓库中最稳定、最正式的 CTU13 结果是图侧 benchmark
- 即 `PCAP-based graph benchmark`
- 并不是“完整公共 mixed-label packet-level 主实验体系”

## 6. 如何运行实验

### 6.1 CSV 可复现实验

最简单的入口：

```bash
python3 scripts/run_csv_experiment.py \
  --config configs/repro_csv.example.yaml
```

如果你想指定自己的 CSV：

```bash
python3 scripts/run_csv_experiment.py \
  --config configs/repro_csv.example.yaml \
  --input /path/to/your.csv \
  --output-dir outputs
```

当前示例配置特点：

- `input_mode: csv`
- `use_nuisance_aware: false`
- `binary_label_mapping` 已预留

### 6.2 PCAP 可复现实验

最简单的入口：

```bash
python3 scripts/run_pcap_experiment.py \
  --config configs/repro_pcap.example.yaml
```

你需要先修改配置中的：

- `benign_inputs`
- `malicious_inputs`

再运行。

当前示例配置默认：

- `input_mode: pcap`
- `use_nuisance_aware: false`
- 以轻量 `packet_limit` 做可运行性验证

### 6.3 CTU13 图侧 benchmark

如果你要复查或重跑当前图侧主 benchmark，重点脚本是：

```bash
python3 scripts/run_ctu13_binary_benchmark.py
python3 scripts/run_ctu13_edge_centric_comparison.py
```

重点输出：

- `results/ctu13_binary_benchmark.csv`
- `results/ctu13_binary_benchmark.md`
- `results/ctu13_edge_centric_comparison.csv`
- `results/ctu13_edge_centric_comparison.md`

### 6.4 一站式数据集实验套件

仓库中也保留了一个更大的实验套件入口：

```bash
python3 scripts/run_repro_dataset_suite.py
```

但这条线会尝试汇总更多数据源与实验组合，开销更大。当前正式报告并不依赖它全部跑完。

## 7. 输出文件怎么看

### 7.1 轻量输出

这些目录已经被保留到 GitHub，适合直接阅读：

- `outputs/summary/`
- `outputs/metrics/`
- `outputs/figures/`
- `outputs/logs/`
- `outputs/data_inventory/`
- `results/`
- `reports/`

### 7.2 大体量中间输出

这些目录默认不进 GitHub：

- `outputs/runs/`
- `outputs/checkpoints/`
- `data/`
- `artifacts/`

原因是：

- 体量大
- 含中间缓存或原始数据
- 不适合直接作为仓库内容长期版本化

## 8. 报告与图片显示

最终报告位置：

- [reports/final_experiment_report.md](reports/final_experiment_report.md)
- [reports/final_experiment_report.html](reports/final_experiment_report.html)

为了保证从别的机器克隆后 markdown 图片能正常显示，报告实际引用的是：

- `reports/assets/`

这里保存的是从原始 `outputs/figures/` 复制过来的只读副本，因此：

- GitHub 网页端能直接显示
- 本地克隆后打开 markdown 也能显示

## 9. 当前最值得看的结果文件

如果你想快速定位最关键结果，优先看这些：

### CSV 主结果

- `outputs/summary/all_metrics.json`
- `outputs/summary/all_metrics.csv`
- `outputs/logs/CICIoT_csv_merge_a_seed*.log`
- `outputs/metrics/CICIoT_csv_merge_a_seed*.json`
- `outputs/figures/CICIoT_csv_merge_a_*`

### CTU13 图侧主 benchmark

- `results/ctu13_edge_centric_comparison.csv`
- `results/ctu13_edge_centric_comparison.md`
- `results/ctu13_binary_benchmark.csv`
- `results/ctu13_binary_benchmark.md`
- `results/ctu13_flow_label_alignment_summary.md`
- `results/ctu13_primary_graph_extraction_summary.md`

## 10. 当前结论边界

请特别注意这几点：

1. 当前正式 CSV 主结果来自 `merge01.csv`
2. 当前正式图侧主 benchmark 来自 `CTU13 48/49/52`
3. CTU13 的结果属于 `PCAP-based graph benchmark`
4. 但这不等于完整跨数据集 PCAP 主实验矩阵已经建立
5. 当前正式报告默认 `use_nuisance_aware=false`

## 11. 常用命令汇总

安装：

```bash
pip install -e .
pip install -e .[gae]
```

CSV：

```bash
python3 scripts/run_csv_experiment.py --config configs/repro_csv.example.yaml
```

PCAP：

```bash
python3 scripts/run_pcap_experiment.py --config configs/repro_pcap.example.yaml
```

CTU13 benchmark：

```bash
python3 scripts/run_ctu13_binary_benchmark.py
python3 scripts/run_ctu13_edge_centric_comparison.py
```

CTU13 下载：

```bash
python3 scripts/download_ctu13.py --scenarios 48 49 52
```

报告生成：

```bash
python3 scripts/generate_final_experiment_report.py
```

测试：

```bash
python3 -m unittest tests.test_ctu13_pipeline -v
python3 -m unittest tests.test_repro_experiment -v
```

## 12. 说明

如果你在另一台机器上克隆后发现某个实验“配置在、代码也在，但数据文件不在”，这通常不是仓库缺文件，而是因为：

- 原始数据和超大中间产物没有被放进 GitHub
- 你需要先按上面的方式准备数据集，再执行对应脚本

当前这份仓库更适合做两件事：

1. 阅读和复核现有正式结果
2. 在自行准备好数据后复现实验流程
