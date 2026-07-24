# IACO 森林救援调度 —— 论文复现、开放问题分析与 RL 框架设计

> 原论文：*Scheduling and Route Planning for Forests Rescue: Applications with a Novel Ant Colony Optimization Algorithm* (Engineering Applications of Artificial Intelligence, 2025)
>
> 本实验代码仓库：[https://github.com/wangyuanxty/13_Ant_colony_optimization_forest_rescue_scheduling](https://github.com/wangyuanxty/13_Ant_colony_optimization_forest_rescue_scheduling)

## 1. 问题背景

论文针对森林直升机巡检提出一个两阶段动态路径规划模型：离线阶段用 K-means 聚类 + IACO 规划各直升机的巡检路线；在线阶段收到突发警报时，调度最近空闲直升机前往火点，并从火点重新规划剩余巡检点的 TSP 路线。IACO 的核心改进为动态信息素初始化和分段信息素衰减策略。

论文在 Section 4.5 提出了 8 个开放问题，涵盖异构协同、多警报并发、动态环境、预测性调度等方向。本项目围绕其中三个开放问题展开复现与分析：**异构多智能体协同（#1）、多警报与火势蔓延（#2）、动态天气环境（#3）**。此外，提出并实现了基于 CNN-Transformer 和 TSPFormer 的神经网络路径规划方案，以及基于 A2C 的多智能体强化学习调度框架。

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

开放问题 #1——异构多智能体协同：论文仅使用直升机，未考虑无人机（侦察）和地面救援队（灭火）。当多类智能体共存时，"找最近空闲者派遣"的简单规则在多火点并发、火势蔓延、天气禁飞等场景下不再最优，需要学习型调度策略。

### 4.2 框架设计

将三个开放问题统一建模为一个 RL 环境，并复用 CNN-Transformer 的 Encoder 作为策略网络的空间表征骨干：

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
        │  路径规划   │ CNN-Transformer / 直线距离（当前简化）
        └─────────────┘
```

### 4.3 环境设计

| 组件 | 规格 |
|------|------|
| 地图 | $100 \times 100$ km，8×8 火势网格 |
| 无人机 × 2~4 | 仅侦察，120 km/h |
| 直升机 × 2~3 | 仅运输设备到火点，60 km/h |
| 地面队 × 2~3 | 仅灭火（需先完成运输），10 km/h |
| 火势 | 每步每格 $p=0.01$ 随机起火，$p=0.3$ 向邻格蔓延 |
| 天气 | 每步 $p=0.1$ 随机产生一个禁飞圆域 |

### 4.4 训练算法演变

**REINFORCE with Value Network**（早期）→ 失败。Value 网络从零初始化，输出≈0，而 episode 累计回报≈40,000，advantage 无法区分好动作与坏动作——所有决策获得几乎相同的 credit。

**REINFORCE with EMA Baseline**（中期）→ 仍失败。EMA 快速收敛到平均回报，优势信号均值趋零，策略梯度消失。

**A2C with Running Statistics**（最终方案）→ 成功。关键改进：
- 每 $K$ 步用 TD 自举更新（非 episode 级）提高样本效率
- Value 目标通过 running mean/std 归一化：$\tilde{V}_{\text{target}} = (V_{\text{target}} - \mu_V) / \sigma_V$
- Critic 预测归一化后的值，使 policy loss 和 value loss 处于同一量级

```python
# A2C 核心更新
adv_t = v_targets - vals.detach()
adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)  # 优势归一化
p_loss = -(log_probs * adv_t).mean()
v_loss = (v_targets_norm - vals_norm).pow(2).mean()
(p_loss + v_loss).backward()
```

### 4.5 实验结果

| 策略 | 灭火数 (ext) | 巡逻覆盖 (cov) | 火灾损失 (dam) |
|------|:-----:|:-----:|------:|
| Random | 99 | 40 | 10677 |
| Nearest (贪心) | 99 | 40 | 10655 |
| **A2C Trained** | **121** | 40 | **10199** |
| vs. Nearest | **+23%** | — | **−4%** |

A2C 策略以 121 次灭火（+23%）和更低火灾损失（−4%）超越了贪心最近 baseline。这表明在**多智能体并发决策存在冲突**的场景下（例如两架直升机竞争同一个火点），学习型策略确实能够发现优于规则化调度的高效分配。

### 4.6 消融与调试记录

- **"BASE target=-1" 卡死 bug**：智能体指派去基地时 `agent_busy=True` 但 `target<0` 跳过移动循环 → `_on_arrival` 永不调用 → 智能体永久"卡住"。修复：指派基地时立即设 `busy=False`。
- **坐标映射越界**：动态火点数变化导致 coord 序列长度不固定。修复：固定 coord 序列结构 `[base, patrol_pts, all_grid_cells, agents]`，仅通过 mask 区分有效火点。
- **Loss 量级失衡**：policy loss ≈ 0.1，value loss ≈ 167,000。修复：running statistics 归一化 V targets + 消融 value network 改用 EMA baseline，最终回到带归一化 V targets 的 A2C。

---

## 5. 开放问题分析

| # | 开放问题 | 结论 |
|---|---------|------|
| 1 | 异构多智能体协同 | RL+A2C 学出优于 rule-based 的调度策略 |
| 2 | 多警报+火势蔓延 | 已纳入环境，A2C 策略在 100 步内完成灭火 |
| 3 | 动态天气 | `mask` 机制可处理，IACO 和 CNN-Transformer 均能适配 |
| 4 | 极端场景验证 | 尚未覆盖 |
| 5 | 预测性调度 | 方案：FPS 增密 + 单标量 M 扫帕累托前沿 |
| 6 | EFFIS 集成 | 工程问题 |
| 7 | 仿真→真实适配 | 工程问题 |
| 8 | 更广救援场景 | 方向性声明 |

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
│   ├── train.py       # A2C 训练循环
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

# A2C 训练 (需 GPU)
D:\anaconda\envs\py312\python.exe -u -c "
import sys; sys.path.insert(0, '.')
from forest_rescue_rl.train import train_a2c
train_a2c(n_episodes=200, log_interval=20)
"

# 评估
D:\anaconda\envs\py312\python.exe forest_rescue_rl/evaluate.py
```
