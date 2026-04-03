# DRPM-NSCV / Baseline: HERLN（Du Y, Liu X, Liang W, et al. Hawkes based Representation Learning for Reasoning over Scale-free Community-structured Temporal Knowledge Graphs[C]. Rambow O, Wanner L, Apidianaki M, et al. Proceedings of the 31st International Conference on Computational Linguistics. Abu Dhabi, UAE: Association for Computational Linguistics, 2025: 2935-2946.）

本项目为硕士学位论文《基于时态知识图谱推理的事件预测技术研究》的实验代码仓库，覆盖模型训练、推理评估、消融实验、参数敏感性实验，以及论文表 5.12~5.17 的复现实验与结果整理。

核心模型为 DRPM-NSCV (Dynamic Relation Prediction Model Guided by Neighborhood Sampling and Clustering Validation)，实现了时态知识图谱的实体预测与关系预测，并包含全局关系动态、复制-生成机制、关系上下文先验、频域分解与正交正则化等模块。

## 2.1 项目简介

本研究面向社区结构明显的时态知识图谱，解决实体/关系表示随时间演化与关系预测不稳定的问题。模型由表示学习与预测解码两阶段构成，重点包括：历史子图编码、社区结构融合、关系频域分解与正则约束、关系上下文先验、全局关系动态与复制-生成机制。

## 2.2 系统环境

表 5.1 实验用服务器具体硬件参数：

| 硬件配置 | 实验室工作站 | 云服务器实例 |
| --- | --- | --- |
| CPU 型号 | Intel Core i9-13900K @ 3.00GHz | Intel Xeon Silver 4214R @ 2.40GHz |
| CPU 核心数 | 24 核心 (8P + 16E) | 12 vCPU |
| GPU 型号 | NVIDIA RTX 4090 | NVIDIA RTX 3090 |
| 显存 | 24 GB | 24 GB |
| 内存 | 64 GB | 90 GB |
| 磁盘 | 1 TB | 80 GB |

表 5.2 实验用服务器相关依赖库说明：

| 名称 | 版本 | 说明 |
| --- | --- | --- |
| PyTorch | 2.1.2 | 深度学习框架，支持 GPU 加速计算 |
| CUDA | 11.8 | 并行计算平台 |
| NumPy | 1.26.4 | 多维数组与数值计算支持 |
| scikit-learn | 1.5.2 | 数据预处理与模型评估 |
| DGL | 1.1.2 (cu118) | 图神经网络库 |

推荐运行环境：

- OS: Ubuntu 22.04 / Windows 10 或 11
- Python: 3.10
- CUDA: 11.8
- PyTorch: 2.1.2
- GPU: NVIDIA RTX 3090 24GB 及以上

## 2.3 依赖安装说明

**AUTODL服务器环境配置**：
服务器默认环境已预装 CUDA 11.8 + PyTorch 2.1.2，直接安装 DGL：

```bash
export DGLBACKEND=pytorch
pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.1/cu118/repo.html
pip install -r requirement.txt
```

**本地环境配置**：

使用 conda 创建环境：

```bash
conda create -n logcl python=3.10
conda activate logcl
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

然后安装 DGL 和其他依赖：

```bash
export DGLBACKEND=pytorch
pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.1/cu118/repo.html
pip install -r requirement.txt
```

> **注**：[requirement.txt](requirement.txt) 包含当前版本的完整依赖清单，确保 CUDA、PyTorch 和 DGL 版本与环境一致。

## 2.4 数据说明

数据位于 [data](data) 目录，包含 ICEWS14s、ICEWS18、WIKI、YAGO 等数据集，每个数据集目录包含：

- entity2id.txt
- relation2id.txt
- train.txt / valid.txt / test.txt
- train.csv (用于社区类图与统计信息)

数据格式按四元组 (s, r, o, t) 存储，时间字段用于快照划分与历史窗口构建。

数据来源为公开时态知识图谱数据集的官方发布版本。若无法分发原始数据，请保留样例文件并说明限制。

## 2.5 如何运行项目

从零开始执行的推荐流程如下：

1) 安装依赖
2) 准备数据并确认目录结构 [data](data)
3) 配置参数 (见 [src/config.py](src/config.py))
4) 运行训练
5) 运行测试/评估
6) 生成表格与图像

最小训练示例：

```bash
cd src
python main.py -d WIKI --self-loop --layer-norm --weight 0.5 --theta 1 --relation-prediction --relation-evaluation --task-weight 0.0 --gpu 0 --freq-reg 5e-4 --alpha 10 --n-epochs 50 --temporal-gating --time-embedding-alpha 0.4
```

训练结果保存在 [checkpoints](checkpoints) 与 [checkpoints_服务器](checkpoints_服务器) 下按日期分层的目录。

## 2.6 各实验复现方法

本仓库已将论文表 5.12~5.17 的复现实验整理到 [experiments](experiments)：

- 表 5.12 (ICEWS14 / ICEWS18): [experiments/table5_12对比实验（ICEWS14](experiments/table5_12对比实验（ICEWS14)
- 表 5.13 (WIKI / YAGO): [experiments/table5_13对比实验（WIKI](experiments/table5_13对比实验（WIKI)
- 表 5.14 (time-embedding-alpha): [experiments/table5_14时间嵌入系数](experiments/table5_14时间嵌入系数)
- 表 5.15~5.17 (消融): [experiments/table5_15_16_17消融实验](experiments/table5_15_16_17消融实验)

每个实验目录包含 `run.sh`、`args.log`、`experiment.log`、`result.csv`，可直接复现对应实验：

```bash
bash experiments/table5_12对比实验（ICEWS14/icews14/run.sh
bash experiments/table5_13对比实验（WIKI/wiki/run.sh
bash experiments/table5_14时间嵌入系数/alpha_0.4/run.sh
bash experiments/table5_15_16_17消融实验/ftt_freq_reg/run.sh
```

## 2.7 图表生成说明

论文表格与图像的源数据与绘图脚本集中在 [results](results) 与 [results/相关表格.md](results/相关表格.md)。

如果需要更新图表，请先运行对应 `experiments/*/run.sh`，再根据 [results](results) 下的脚本或已有表格文件整理输出。

## 2.8 常见问题说明
- torch 2.4.1 启动失败，考虑是dgl包太新了
- DGL 无法导入: 确认 `DGLBACKEND=pytorch` 且 DGL 与 PyTorch 版本匹配。
- 显存不足: 降低 `--n-hidden`、`--n-layers` 或缩短历史长度 `--train-history-len`。