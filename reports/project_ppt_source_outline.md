# 项目总文档大纲版

## 1. 项目概述
- 项目名称：基于流特征与交互图表示的恶意流量检测研究
- 一句话摘要：以 benign-only 异常检测为理论核心，用 CSV 主结果与 CTU13 图主 benchmark 形成双证据链
- 当前正式口径：
  - 只采用 `merge01.csv`
  - 只采用 `CTU13 48 / 49 / 52`
  - `use_nuisance_aware=false`

## 2. 研究背景与问题提出
- 加密流量与复杂背景流量使传统载荷方法受限
- 纯监督分类难以覆盖真实复杂场景
- benign-only 异常检测更符合标签稀缺与背景复杂条件

## 3. 研究目标与核心问题
- CSV line：模型是否具备分类能力？
- Graph line：原始 PCAP 到检测的链路是否可行？图表示是否有结构价值？
- 为什么必须双视角验证

## 4. 整体技术路线与系统构建
- 三层框架：Input / Representation / Detection
- 系统整体流程图
- 从数据到判定的完整链路

## 5. 方法设计
### 5.1 CSV 主线
- StandardScaler + PCA + reconstruction error
- benign-only 训练
- held-out benign percentile threshold

### 5.2 图主线
- Graph AutoEncoder
- encoder / node decoder / edge decoder
- benign 图更可重构，malicious 图残差更大

### 5.3 从 PCAP 到图
- PCAP → flow → endpoint interaction graph
- flow 不只是五元组，而是统计与行为特征对象
- 节点不是纯 IP，而是 endpoint_type + ip + port + protocol
- communication edge 是行为载体
- association edges 表达结构与行为相似性

### 5.4 图级判定思想
- 不采用全图平均异常
- 关注 high-anomaly tail 的局部证据
- 少量关键异常边即可支撑恶意判定

## 6. 研究意义与创新点
- 理论意义：将 benign-only 异常检测推广到图表示
- 方法意义：双视角验证
- 工程意义：真实打通 PCAP 下游图链路
- 应用意义：面向真实复杂网络场景
- 创新点：
  1. benign-only 异常检测扩展到流交互图表示
  2. 边特征是行为编码
  3. 图级检测强调局部异常支持
  4. 双证据链完成验证

## 7. 关键结果与项目结论支撑
### 7.1 merge01.csv
- Accuracy: 0.9756 ± 0.0004
- Precision: 0.9995 ± 0.0000
- Recall: 0.9759 ± 0.0004
- F1: 0.9876 ± 0.0002
- ROC-AUC: 0.9847 ± 0.0006
- PR-AUC: 0.9999 ± 0.0000
- 说明：证明标准流特征输入上的分类能力

### 7.2 CTU13 48 / 49 / 52
- merged F1: 0.9412
- merged Recall: 0.9412
- merged FPR: 0.0294
- merged background hit ratio: 0.2844
- 分场景：
  - 48：F1 0.6667，Recall 0.5000
  - 49：F1 0.8571，Recall 1.0000
  - 52：F1 1.0000，Recall 1.0000
- 说明：证明原始 PCAP 下游图链路可行

### 7.3 结论边界
- 能证明什么
- 不能证明什么

## 8. 研究探索、失败分析与方法边界
- nuisance-aware：动机合理，但未形成正式主线
- two-stage micrograph：proposal / extraction 是主要瓶颈
- episode-graph：episode stitching / sessionization 未稳定成功
- 当前正式主线仍是 `single-stage edge_temporal_binary_v2`

## 9. 局限性与未来工作
- 未形成完整跨数据集 PCAP 主实验矩阵
- background / unknown 仍是主要残余问题
- 未来可继续推进 nuisance-aware、图级证据建模与更系统 PCAP 验证

## 10. 结尾页可直接使用的总结
- 项目已形成稳定 CSV 主结果
- 项目已形成 CTU13 图侧主 benchmark
- nuisance-aware 当前不是正式结果的必要项
- CTU13 是 PCAP-based graph benchmark，不是完整 PCAP 主实验体系

