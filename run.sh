#!/bin/bash

# ==============================================================================
# run.sh
# 用于统一运行主要实验或全流程复现（这里以 WIKI 为例配置，可按需修改 Dataset）。
# 本文件列出了基线模型（Baseline）和各类消融实验的具体命令。
# 所有的模型状态与日志结果会自动保存在 checkpoints/ 目录下。
# ==============================================================================

# 进入源码目录
cd src || exit 1

echo "========================================================="
echo "               全面实验与消融测试脚本复现                "
echo "========================================================="

# ==============================
# 【Baseline】
# 包含所有完整模块的基础版本：开启 FFT（freq-reg=5e-4），正交正则（weight=0.5）等
# ==============================
echo -e "\n>>> 运行 Baseline (完整模型，无剥离)..."
# 注意：原全量测试可能为 n-epochs 50，您可以根据机器配置调整
python main.py -d WIKI \
    --self-loop --layer-norm \
    --weight 0.5 --theta 1 \
    --relation-prediction \
    --relation-evaluation \
    --task-weight 0.0 \
    --gpu 0 \
    --freq-reg 5e-4 \
    --alpha 10 \
    --n-epochs 50 \
    --temporal-gating \
    --time-embedding-alpha 0.9

# ==============================
# 【消融 A: w/o FFT关系嵌入频域分解】
# 移除频域正则化模块：将 --freq-reg 设为 0.0，禁用关系嵌入的频域约束。
# ==============================
echo -e "\n>>> 运行 消融 A (w/o FFT频域分解)..."
python main.py -d WIKI \
    --self-loop --layer-norm \
    --weight 0.5 --theta 1 \
    --relation-prediction \
    --relation-evaluation \
    --task-weight 0.0 \
    --gpu 0 \
    --freq-reg 0.0 \
    --alpha 10 \
    --n-epochs 50 \
    --temporal-gating \
    --time-embedding-alpha 0.9

# ==============================
# 【消融 B: w/o 关系上下文先验】
# 禁用关系上下文先验的线性层模块（--no-rel-context-prior），移除关系语义增强机制。
# 备注：为了避免由于 task-weight 和 relation-evaluation 引发的致命冲突，需确保实体预测任务正常。
# 此处使用文档特别指定的命令，并在 20 epochs 时验证稳定性。
# ==============================
echo -e "\n>>> 运行 消融 B (w/o 关系上下文先验)..."
python main.py -d WIKI \
    --self-loop --layer-norm \
    --weight 0.5 --theta 1 \
    --gpu 0 \
    --freq-reg 5e-4 \
    --alpha 10 \
    --n-epochs 20 \
    --temporal-gating \
    --time-embedding-alpha 0.9 \
    --no-rel-context-prior

# ==============================
# 【消融 C: w/o 关系正交正则化】
# 移除关系正交约束：将 --weight 变为 0.0，取消对关系向量几何独立性的强制约束。
# ==============================
echo -e "\n>>> 运行 消融 C (w/o 关系正交正则化)..."
python main.py -d WIKI \
    --self-loop --layer-norm \
    --weight 0.0 --theta 1 \
    --relation-prediction \
    --relation-evaluation \
    --task-weight 0.0 \
    --gpu 0 \
    --freq-reg 5e-4 \
    --alpha 10 \
    --n-epochs 50 \
    --temporal-gating \
    --time-embedding-alpha 0.05

# ==============================
# 【终极消融：测试最纯净的骨干网络】
# 同时关闭 A、B、C 等新引入模块，验证基干表现。
# 移除 FFT 频域分解 (--freq-reg 0.0) 
# 移除 关系上下文先验 (--no-rel-context-prior 并去冲突) 
# 移除 正交正则 (--weight 0.0)
# ==============================
echo -e "\n>>> 运行 终极消融 (骨干网络测试)..."
python main.py -d WIKI \
    --self-loop --layer-norm \
    --weight 0.0 --theta 1 \
    --gpu 0 \
    --freq-reg 0.0 \
    --alpha 10 \
    --n-epochs 20 \
    --temporal-gating \
    --time-embedding-alpha 0.05 \
    --no-rel-context-prior

echo "========================================================="
echo "所有实验已按顺序执行完毕。请前往 checkpoints 目录进行分析比对。"
echo "========================================================="
