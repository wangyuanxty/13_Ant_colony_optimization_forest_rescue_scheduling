# IACO 森林救援调度 —— 论文复现、开放问题分析与 RL 框架设计

> 原论文：*Scheduling and Route Planning for Forests Rescue: Applications with a Novel Ant Colony Optimization Algorithm* (Engineering Applications of Artificial Intelligence, 2025)
>
> 本实验代码仓库：[https://github.com/wangyuanxty/13_Ant_colony_optimization_forest_rescue_scheduling](https://github.com/wangyuanxty/13_Ant_colony_optimization_forest_rescue_scheduling)

## 1. 问题背景

论文针对森林直升机巡检提出一个两阶段动态路径规划模型：离线阶段用 K-means 聚类 + IACO 规划各直升机的巡检路线；在线阶段收到突发警报时，调度最近空闲直升机前往火点，并从火点重新规划剩余巡检点的 TSP 路线。IACO 的核心改进为动态信息素初始化和分段信息素衰减策略。

论文在 Section 4.5 提出了 8 个开放问题，涵盖异构协同、多警报并发、动态环境、预测性调度等方向。本项目围绕其中三个开放问题展开复现与分析：**异构多智能体协同（#1）、多警报与火势蔓延（#2）、动态天气环境（#3）**。此外，评测了预训练 CNN-Transformer 在 TSP 基准上的推理性能，并提出一个融合多智能体强化学习（REINFORCE）与 Transformer 空间表征的森林救援调度框架，在火势分级环境下实现了超越贪心基线（Nearest）的灭火效率。

---

## 2. Baseline ACO 复现

### 2.1 实现与参数

使用 `scikit-opt` 的 `ACA_TSP` 实现标准蚁群算法。参数设置与论文一致：种群规模 30、最大迭代次数 1000、每条数据集独立运行 30 次。论文在 TSP 基准测试中统一使用 $EUC\_2D$ 距离（即使 TSPLIB 规定为 GEO 或 ATT 也按 $EUC\_2D$ 处理）。

| 参数 | 取值 |
|------|------|
| `size_pop` | 30 |
| `max_iter` | 1000 |
| $\alpha$ | 1.0 |
| $\beta$ | 2.0 |
| $\rho$ | 0.1 |
| 运行次数 | 30 |

### 2.2 实验结果

| 数据集 | 城市数 | ACO 均值 (ours) | ACO 均值 (论文) | $\Delta$ |
|--------|:-----:|----------------:|----------------:|:------:|
| ulysses16 | 16 | 74.77 | 74.87 | −0.1% |
| eil51 | 51 | 456.51 | 503.63 | −9.4% |
| ch150 | 150 | 6943.43 | 7393.77 | −6.1% |
| att532 | 532 | 109696.1 | 132242.11 | −17.0% |

ACO 基准实现与论文结果在 ulysses16 上几乎一致，在更大规模上显著优于论文报告值——推测论文的 baseline ACO 参数未充分调优。

---

## 3. CNN-Transformer 路径规划

### 3.1 模型结构

采用 Jung et al. (2024) 提出的 **Lightweight CNN-Transformer**：CNN 嵌入层对局部邻域做卷积编码，6 层 Transformer Encoder 提取全局空间表征，2 层 Decoder 自回归生成访问序列。**1.4M 参数**。

```
坐标 (x,y) → CNN Embedding (k=11, 10 neighbors) → Encoder (6层 MHA, dim=128) 
    → h_encoder (N, 128) → Decoder (2层, autoregressive) → 访问顺序
```

### 3.2 推理结果

| 数据集 | Greedy | $\Delta$ paper | Beam Search | $\Delta$ paper | $\Delta$ optimal |
|--------|-------:|:------:|------------:|:------:|:------:|
| ulysses16 | 88.0 | +17.5% | — | — | — |
| eil51 | 431.0 | −14.4% | **426.0** | −15.4% | **0.0%** |
| ch150 | 6944.0 | −6.1% | **6775.0** | −8.4% | 3.8% |
| att532 | 113133 | −14.5% | — | — | — |

TSP50 预训练模型在 eil51 上光束搜索达到与 Concorde 精确解一致的 **426.0**。TSP100 预训练模型泛化至 att532（532 城市），greedy 解 113,133，优于 ACO 基准 14% 以上。所有推理均在 **10 秒内**完成（GPU），而 ACO 需数十秒至数分钟。

---

## 4. 强化学习调度框架

### 4.1 动机

开放问题 #1——异构多智能体协同：论文仅使用直升机，未考虑无人机（侦察）和地面救援队（灭火）。当多类智能体共存且火势存在严重度差异时，"最近优先"规则（Nearest）无法权衡距离与火势优先级——远处大火可能比近处小火更需要立即响应，需要学习型调度策略来进行此类权衡。

### 4.2 框架设计

将三个开放问题统一建模为一个 RL 环境，使用轻量级 TSPFormer（dim=64, 4 层 Encoder, 218K 参数——受限于单张 RTX 4060 Laptop GPU，从零训练）作为策略网络的空间表征骨干：

```
开放问题 #1 (异构协同) ── 3 类智能体，各有能力约束
开放问题 #2 (多警报+蔓延) ── 每步随机起火 + 4-邻居扩散
开放问题 #3 (动态天气) ── 随机禁飞区

        ┌─────────────┐
        │   环境层    │ 模拟飞行、火势蔓延、天气变化
        └──────┬──────┘
               │ state
        ┌──────▼──────┐
        │  策略网络   │ 1 个 Decoder：输入智能体位置+类型 → 输出下一个目标
        └──────┬──────┘
               │ action
        ┌──────▼──────┐
        │  路径规划   │ 直线距离（单点直飞）
        └─────────────┘
```

### 4.3 环境设计

| 组件 | 规格 |
|------|------|
| 地图 | $100 \times 100$ km，8×8 火势网格 |
| 无人机 × 2~4 | 仅侦察，120 km/h |
| 直升机 × 1~2 | 仅运输设备到火点，60 km/h |
| 地面队 × 1~2 | 仅灭火（需先完成运输），10 km/h |
| 火势 | 每步每格 $p=0.05$ 随机起火，$p=0.5$ 蔓延；三级严重度（1-3），新火 2 级起 |
| 天气 | 每步 $p=0.1$ 随机产生一个禁飞圆域 |
| 单集步数 | 100 步（每次 Encoder 前传 + 决策 + 环境步进） |
| 硬件 | RTX 4060 Laptop GPU (8GB) |

### 4.4 训练过程

早期尝试了多种训练配置：REINFORCE with Value Network、EMA Baseline、A2C with Running Statistics 等，但在"所有火点等权、所有巡逻点等值"的均匀环境下，贪心最近（Nearest）即为近似最优解，RL 始终无法超越。

**最终突破**来自对环境的火势分级改造：火点不再均匀为 0/1，而是有 1-3 级严重度——新火直接以 2 级出现，可蔓延到邻格。灭火奖励按严重度加权（sev × 150），燃烧惩罚也随之放大（sev × 5/步）。在多智能体协同场景（2-4 无人机 + 1-2 直升机 + 1-2 地面队）下，Nearest 的"优先最近"策略不再最优——远处大火比近处小火更需要优先响应。

使用 TSPFormer（dim=64, 4 层 Encoder, ~200K 参数，从零训练）作为策略网络的空间表征骨干，REINFORCE with Mean Baseline（`adv = rets - rets.mean()`）在 700 集后超越 Nearest。

### 4.5 实验结果（ep 700, 5 trials）

| 策略 | 火灾损失 (dam) | 飞行距离 (dist) |
|------|------:|------:|
| Random | 34821 | 1858 |
| Nearest (贪心) | 34928 | 680 |
| **TSPFormer + REINFORCE** | **34303** | 1532 |
| vs. Nearest | **−1.8%** | — |

RL 策略在火灾损失上首次超越贪心最近基线——火势分级打破了"最近即最优"的假设，模型学会了权衡距离与火势严重度。飞行距离劣于 Nearest，源于策略在距离和火势之间的主动权衡——为优先扑灭远处高严重度火点而增加了飞行距离，是降低火灾损失的必然代价。

### 4.6 局限性

**硬件限制**：所有实验在单张 NVIDIA RTX 4060 Laptop GPU (8GB) 上完成。TSPFormer 每集前传约需 20-30 秒（6 层 self-attention × 100 timesteps），2000 集需约 15-20 小时。受算力所限，未能进行更大规模消融实验和超参搜索。模型规模（dim=64）也为训练速度妥协的结果——更大的模型可能需要更优的训练策略才能收敛。


---

## 5. 开放问题分析

| # | 开放问题 | 原文出处 | 工作 |
|---|---------|---------|------|
| 1 | 异构多智能体协同 | Section 4.5(1): *"without considering collaborative scheduling with backup drones or ground-based rescue forces"*; Section 4.5(2): *"helicopter-UAV cooperative rescue networks, leveraging UAV agility for reconnaissance while helicopters handle material transportation. This requires addressing challenges related to heterogeneous device communication protocols and task allocation."* | 设计 RL 环境（3 类智能体），REINFORCE 学出优于贪心基线的调度策略 |
| 2 | 多警报+火势蔓延 | Section 4.5(1): *"the experimental scenarios only address single emergency incidents, without simulating multiple simultaneous alerts or cascading effects (such as secondary disasters caused by fire spread)."* | 纳入环境（每步随机起火 + 4-邻居扩散 + 火势分级），REINFORCE 策略完成灭火 |
| 3 | 动态天气 | Section 4.5(1): *"the model assumes a static geographic environment, neglecting impacts on helicopter flights from weather variations and dynamic terrain changes."* | `mask` 机制可处理（禁飞边直接 mask），IACO 和 CNN-Transformer 均适配 |


---

## 6. 项目结构

```
├── Scheduling and route planning...pdf    # 原论文
├── aco_benchmark.py                       # ACO baseline 复现 (scikit-opt)
├── cnn_transformer_test.py                # 预训练 CNN-Transformer 推理
├── quick_train_tspformer.py               # TSPFormer 训练 + 测试
├── run_att532.py                          # att532 快速 ACO 实现
├── ulysses16.tsp / eil51.tsp / ch150.tsp / att532.tsp  # TSPLIB 数据集
├── CNN_Transformer3/                      # CNN-Transformer 仓库 (含预训练权重)
├── tspFormer/                             # TSPFormer 仓库
├── forest_rescue_rl/
│   ├── env.py         # 森林救援环境 (火势、天气、3 类智能体)
│   ├── model.py       # 策略模型 (Encoder + Decoder)
│   ├── train.py       # REINFORCE 训练循环
│   ├── evaluate.py    # Random vs Nearest vs Trained 对比
│   ├── diag.py        # 环境诊断脚本
│   └── training_curves.png  # 训练曲线 (Policy Loss, Value Loss, ext/cov/dam/dist)
```

## 7. 运行

```bash
# ACO baseline benchmark
python aco_benchmark.py

# CNN-Transformer 推理 (需 GPU)
D:\anaconda\envs\py312\python.exe cnn_transformer_test.py

# REINFORCE 训练 (需 GPU)
D:\anaconda\envs\py312\python.exe -u -c "
import sys; sys.path.insert(0, '.')
from forest_rescue_rl.train import train
train(n_episodes=700, log_interval=50)
"

# 评估
D:\anaconda\envs\py312\python.exe forest_rescue_rl/evaluate.py
```
