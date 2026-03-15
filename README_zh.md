# STELLAR：面向恶意流量识别的相似度感知卫星联邦学习框架

> **发表于：** IEEE Transactions on Information Forensics and Security，第 21 卷，第 1766–1780 页，2026 年。
> DOI: [10.1109/TIFS.2026.3659044](https://doi.org/10.1109/TIFS.2026.3659044)

STELLAR 是一个面向低轨（LEO）卫星星座网络的联邦学习仿真框架，任务目标为**网络入侵检测（恶意流量分类）**。其核心贡献是**多维相似度感知分组策略**：通过综合模型参数相似度、损失曲线相似度和预测分布相似度，将卫星动态聚类为协作训练小组，同时兼顾星间链路的间歇性、卫星能量约束和通信拓扑的高动态性。

[[English README]](README.md)

---

## 系统架构

```mermaid
flowchart TD
    subgraph CFG["配置层 (YAML)"]
        C1["星座选择\nIridium-like: 6 轨道 × 11 颗 = 66 颗\nOneWeb: 18 轨道 × ~36 颗 = 651 颗"]
        C2["数据集\nCICIDS-2017 / 自定义 CSV"]
        C3["FL 超参数\n轮次 · 学习率 · batch_size · α · μ"]
        C1 --- C2 --- C3
    end

    subgraph DATA["数据层"]
        DG["数据生成器\nRealTrafficGenerator / CICIDS2017Generator"]
        NI["Dirichlet Non-IID 分区\nalpha · 区域相似性"]
        SD["每颗卫星本地数据分片\n（轨道相关分布）"]
        DG --> NI --> SD
    end

    subgraph NET["卫星网络仿真"]
        TLE["TLE 轨道力学\nSkyfield · 实时位置计算"]
        ISL["动态星间链路拓扑\n可配置最大通信距离"]
        EM["太阳能能量模型\nbattery_level(t)"]
        TLE --> ISL --> EM
    end

    subgraph FL["联邦学习核心 — 单轮迭代"]
        subgraph CLI["卫星客户端 ×N"]
            LT["本地训练\nMLP · SGD · local_epochs 轮"]
            EC["能量感知调度\n每 batch 检查电量"]
            LT --- EC
        end

        subgraph SG["STELLAR 相似度分组 (每 K 轮刷新)"]
            SP["参数相似度\n余弦距离 · 权重 α=0.4"]
            SL["损失相似度\n损失差倒数 · 权重 β=0.3"]
            SH["预测相似度\nHellinger 距离 · 权重 γ=0.3"]
            CS["综合相似度\nS = α·S_param + β·S_loss + γ·S_pred"]
            GF["贪婪分组\n跳数半径 · max_group_size · threshold"]
            SP & SL & SH --> CS --> GF
        end

        subgraph AGG["传播感知聚合"]
            IA["轨道内 FedAvg\n仅聚合组代表节点"]
            GS["地面站聚合\nISL 中继 · 可见窗口调度"]
            GA["全局 FedAvg\n跨地面站加权平均"]
            IA --> GS --> GA
        end

        CLI --> SG
        SG --> AGG
        GA -->|"广播全局模型"| CLI
    end

    subgraph OUT["评估与输出"]
        MT["准确率 · F1 (macro/weighted)\n精确率 · 召回率（逐轮）"]
        EN["训练能耗 · 通信能耗\n按轨道细分"]
        MT --- EN
    end

    CFG -->|"num_satellites · tle_file"| NET
    CFG -->|"csv_path · alpha"| DATA
    CFG -->|"rounds · lr · mu"| FL
    SD --> CLI
    NET --> FL
    AGG --> OUT
```

---

## 背景与动机

传统联邦学习算法（FedAvg、FedProx）面向地面网络设计，无法直接应用于卫星网络，原因包括：

- **星间链路间歇性**：卫星相对运动导致链路随时建立/断开
- **非均匀 Non-IID 数据分布**：不同轨道面覆盖区域差异显著
- **严格能量约束**：卫星依赖太阳能，电量受光照周期影响剧烈
- **高传播时延**：卫星与地面站之间通信窗口短暂且不可预测

STELLAR 的三项核心设计：

1. **相似度感知分组** — 三维综合相似度（参数 + 损失 + 预测分布）驱动卫星动态聚类
2. **传播受限聚合** — 模型更新仅在可达跳数范围内传播，遵守链路可用性约束
3. **能量感知调度** — 基于实时电量估计门控训练与传输

---

## 项目结构

```
stellar/
├── configs/                         # 实验配置文件（YAML）
│   ├── similarity_grouping_config.yaml  # STELLAR 算法配置
│   ├── fedavg_config.yaml               # FedAvg 基线配置
│   ├── fedprox_config.yaml              # FedProx 基线配置
│   ├── propagation_fedavg_config.yaml   # 传播受限 FedAvg 配置
│   ├── propagation_fedprox_config.yaml  # 传播受限 FedProx 配置
│   ├── sda_fl_config.yaml               # SDA-FL 基线配置
│   ├── oneweb_config.yaml               # OneWeb 大规模星座配置（651 颗）
│   ├── Iridium_TLEs.txt                 # Iridium NEXT TLE 轨道根数（66 颗）
│   ├── oneweb_TLEs.txt                  # OneWeb TLE 轨道根数（651 颗）
│   └── energy_config.yaml               # 卫星能量模型参数
│
├── data_simulator/                  # 数据集加载与 Non-IID 分区
│   ├── real_traffic_generator.py        # 自定义 CSV 流量数据加载器
│   ├── cicids2017_generator.py          # CICIDS-2017 数据集加载器
│   ├── non_iid_generator.py             # Dirichlet Non-IID 数据分区
│   └── network_traffic_generator.py     # 合成流量数据生成器
│
├── fl_core/                         # 联邦学习核心组件
│   ├── client/
│   │   ├── satellite_client.py          # 卫星节点本地训练客户端
│   │   └── fedprox_client.py            # FedProx 近端项客户端
│   ├── aggregation/
│   │   ├── intra_orbit.py               # 轨道内聚合逻辑
│   │   ├── ground_station.py            # 地面站聚合
│   │   └── global_aggregator.py         # 全局模型聚合
│   └── models/
│       ├── real_traffic_model.py        # MLP 分类器（流量分类）
│       └── hybrid_traffic_model.py      # 混合自编码器 + 分类器模型
│
├── simulation/                      # 卫星网络仿真
│   ├── network_model.py                 # TLE 轨道力学与星间链路模型
│   ├── topology_manager.py              # 动态拓扑管理与谱聚类分组
│   ├── energy_model.py                  # 太阳能与电池仿真
│   ├── comm_scheduler.py                # 通信调度
│   └── network_manager.py              # 网络状态管理
│
├── experiments/                     # 实验运行脚本
│   ├── baseline_experiment.py           # 基础实验类（所有算法共用）
│   ├── grouping_experiment.py           # STELLAR（相似度分组）
│   ├── fedavg_experiment.py             # 标准 FedAvg
│   ├── fedprox_experiment.py            # 标准 FedProx
│   ├── propagation_fedavg_experiment.py # 传播受限 FedAvg
│   ├── propagation_fedprox_experiment.py# 传播受限 FedProx
│   ├── sda_fl_experiment.py             # SDA-FL（GAN 数据增强联邦学习）
│   ├── async_experiment.py              # 异步联邦学习变体
│   ├── run_fair_comparison_satfl.py     # 主对比实验脚本
│   ├── run_oneweb_comparison.py         # OneWeb 大规模星座实验
│   ├── run_ablation_study.py            # 消融实验
│   ├── run_robustness_study_v2.py       # 鲁棒性实验
│   ├── run_async_study.py               # 异步实验
│   └── run_theta_sweep.py               # 超参数扫描实验
│
├── visualization/                   # 可视化工具
│   ├── visualization.py                 # 训练曲线绘制
│   └── comparison_visualization.py      # 多算法对比绘图
│
├── tests/                           # 单元测试
├── data/                            # 数据集目录（见 data/README.md）
├── requirements.txt
└── setup.py
```

---

## 环境配置

**依赖环境：** Python 3.9+，PyTorch 1.13+

```bash
git clone https://github.com/lyb88999/STELLAR.git
cd STELLAR
pip install -r requirements.txt
```

或以开发模式安装为 Python 包：

```bash
pip install -e .
```

> **注意：** 首次运行时，Skyfield 会自动下载 `de421.bsp` 星历文件（约 16 MB）并缓存到本地，无需手动下载。

---

## 数据集准备

STELLAR 支持两种数据集格式：

### 方式一：自定义流量 CSV（默认）

准备一个包含数值特征列和 `Label` 标签列的 CSV 文件，放置到 `data/` 目录：

```
data/traffic_data.csv
```

`Label` 列支持：
- 二分类标签（`0` = 正常，`1` = 恶意）
- 多类字符串标签（自动 one-hot 编码后二值化）

### 方式二：CICIDS-2017 数据集

从加拿大新不伦瑞克大学下载 CICIDS-2017 数据集：

```
https://www.unb.ca/cic/datasets/ids-2017.html
```

下载后将 `MachineLearningCVE` 文件夹解压到 `data/` 目录：

```
data/MachineLearningCVE/
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

然后使用 `configs/cicids2017_config.yaml` 作为配置文件。

---

## 快速开始

### 第一步：配置数据路径

编辑对应配置文件，将 `csv_path` 改为你的数据集路径：

```yaml
# configs/similarity_grouping_config.yaml
data:
  dataset: "real_traffic"
  csv_path: "data/traffic_data.csv"   # <-- 修改为实际路径
```

### 第二步：运行全量对比实验（STELLAR vs. 基线）

```bash
python -m experiments.run_fair_comparison_satfl \
    --rounds 20 \
    --output_dir comparison_results/
```

### 第三步：单独运行某个算法

```bash
# STELLAR（本文提出的方法）
python -c "
from experiments.grouping_experiment import SimilarityGroupingExperiment
exp = SimilarityGroupingExperiment('configs/similarity_grouping_config.yaml')
exp.prepare_data()
exp.setup_clients()
stats = exp.train()
"

# FedAvg 基线
python -c "
from experiments.fedavg_experiment import FedAvgExperiment
exp = FedAvgExperiment('configs/fedavg_config.yaml')
exp.prepare_data()
exp.setup_clients()
stats = exp.train()
"

# FedProx 基线
python -c "
from experiments.fedprox_experiment import FedProxExperiment
exp = FedProxExperiment('configs/fedprox_config.yaml')
exp.prepare_data()
exp.setup_clients()
stats = exp.train()
"

# SDA-FL 基线
python -c "
from experiments.sda_fl_experiment import SDAFLExperiment
exp = SDAFLExperiment('configs/sda_fl_config.yaml')
exp.prepare_data()
exp.setup_clients()
stats = exp.train()
"
```

### 切换到 OneWeb 大规模星座（651 颗卫星）

```bash
python -c "
from experiments.grouping_experiment import SimilarityGroupingExperiment
exp = SimilarityGroupingExperiment('configs/oneweb_config.yaml')
exp.prepare_data()
exp.setup_clients()
stats = exp.train()
"
```

---

## 配置参数说明

所有实验共用的核心参数：

```yaml
fl:
  num_satellites: 66          # 卫星总数
  num_orbits: 6               # 轨道面数量
  satellites_per_orbit: 11    # 每轨道卫星数
  num_rounds: 20              # 联邦学习通信轮次
  round_interval: 600         # 每轮仿真时间间隔（秒）

data:
  dataset: "real_traffic"     # 数据集类型：real_traffic | cicids2017
  csv_path: "data/traffic_data.csv"
  iid: false                  # true = IID 分布；false = Non-IID（Dirichlet）
  alpha: 0.5                  # Dirichlet 参数（仅 non-IID 时生效）
  test_size: 0.2              # 测试集比例

client:
  batch_size: 32
  local_epochs: 5
  learning_rate: 0.01
  momentum: 0.9

network:
  tle_file: "configs/Iridium_TLEs.txt"   # TLE 文件路径（决定星座构型）
  max_distance: 4000.0                    # 最大星间通信距离（km）

aggregation:
  min_updates: 2              # 触发轨道内聚合所需的最少更新数
  max_staleness: 300.0        # 最大允许延迟（秒）
  timeout: 600.0              # 聚合超时时间（秒）
  weighted_average: true      # 是否按样本数加权平均
```

### STELLAR 专属参数

```yaml
group:
  max_distance: 2              # 相似度搜索的跳数半径
  max_group_size: 5            # 每个分组最大卫星数
  similarity_threshold: 0.5   # 加入分组所需的最低综合相似度
  similarity_refresh_rounds: 5 # 每隔 N 轮重新计算分组
  weights:
    alpha: 0.4                 # 参数相似度权重
    beta: 0.3                  # 损失相似度权重
    gamma: 0.3                 # 预测分布相似度权重
```

### FedProx 专属参数

```yaml
fedprox:
  mu: 0.01    # 近端项系数（控制本地模型偏离全局模型的程度）
```

### 鲁棒性测试参数

```yaml
robustness:
  parameter_noise_level: 0.1  # 参数噪声强度（0.0 表示禁用）
  noise_start_round: 5        # 从第几轮开始注入噪声
```

---

## 支持的算法

| 算法 | 实验类 | 配置文件 | 说明 |
|---|---|---|---|
| **STELLAR** | `SimilarityGroupingExperiment` | `similarity_grouping_config.yaml` | 本文提出：相似度感知分组联邦学习 |
| FedAvg | `FedAvgExperiment` | `fedavg_config.yaml` | McMahan et al. (2017) |
| FedProx | `FedProxExperiment` | `fedprox_config.yaml` | Li et al. (2020)，加近端项约束 |
| Prop-FedAvg | `LimitedPropagationFedAvg` | `propagation_fedavg_config.yaml` | FedAvg + 跳数受限传播 |
| Prop-FedProx | `LimitedPropagationFedProx` | `propagation_fedprox_config.yaml` | FedProx + 跳数受限传播 |
| SDA-FL | `SDAFLExperiment` | `sda_fl_config.yaml` | 基于 GAN 的合成数据增强联邦学习 |

---

## 进阶实验

### 消融实验

评估各相似度维度对 STELLAR 性能的贡献：

```bash
python -m experiments.run_ablation_study
```

### 鲁棒性实验

测试在参数传输噪声下的性能退化：

```bash
python -m experiments.run_robustness_study_v2
```

### 超参数灵敏度分析

扫描 `alpha`、`beta`、`gamma` 权重组合对性能的影响：

```bash
python -m experiments.run_theta_sweep
```

### 异步联邦学习实验

```bash
python -m experiments.run_async_study
```

### OneWeb 大规模对比实验

在 651 颗卫星的 OneWeb 星座上运行对比：

```bash
python -m experiments.run_oneweb_comparison
```

---

## 运行测试

```bash
pytest tests/ -v
```

---

## 引用

如果本代码对你的研究有帮助，请引用我们的论文：

```bibtex
@article{li2026stellar,
  author  = {Li, Yubo and Zhang, Li and Li, Kai and Su, Haoru},
  journal = {IEEE Transactions on Information Forensics and Security},
  title   = {STELLAR: Similarity-Based Satellite Federated Learning for Malicious Traffic Recognition},
  year    = {2026},
  volume  = {21},
  pages   = {1766--1780},
  doi     = {10.1109/TIFS.2026.3659044}
}
```

---

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。
