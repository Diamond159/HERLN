# 论文-代码-脚本-结果映射说明（paper_mapping）

---

## 1. 总体流程映射

论文方法总体链路：
1. 时态四元组按时间切分快照 + 历史窗口构建
2. 历史子图构建（含反向边、时间信息）并可选构建线图/概率矩阵
3. 社区图构建并对实体嵌入做 TAGConv 增强
4. HawkesRGCN 编码历史图，得到时序实体表示
5. 关系动态更新 + FFT 频域正则
6. ConvTransR 关系解码 + 关系上下文先验 + 复制-生成增强（可选）
7. 训练/验证/测试，并将结果写入 checkpoints 目录

代码主链路定位：
- 训练总入口：src/main.py:152
- 数据按时间切分：src/main.py:166, src/main.py:167, src/main.py:168, src/utils.py:639
- 历史子图构建调用：src/main.py:315, src/main.py:401
- 模型主类：model/rrgcn.py:144
- Hawkes 层：model/hrgcn.py:7
- 结果写出：src/main.py:147
- checkpoint 目录生成：src/main.py:182

---

## 2. 分节精确映射（对应论文 4.3.3.1 ~ 4.3.3.5）

## 2.1 4.3.3.1 历史结构构建与边级增强层

论文要点：
- 时间切分快照 Gi
- 反向关系构造（r + |R|）
- 历史窗口 history(tk)
- 多快照合并 GH + 时间衰减
- 边级增强：线图与概率矩阵

代码对应：
- 快照切分：src/utils.py:639（split_by_time）
- 历史窗口在训练/测试中切片：src/main.py:84, src/main.py:305-313（训练同逻辑）
- 子图构建（含反向边、边类型、时间）：src/utils.py:319
- 反向关系构造：src/utils.py:302-315（prepare_triples_with_inverse）
- 历史图合并与时间权重 max_time - time + 1：src/utils.py:374
- 概率矩阵 PM_PD：src/utils.py:216
- 线图构建：src/utils.py:266, src/utils.py:285
- 在主流程开启边级增强（可选开关）：src/main.py:87, src/main.py:315, src/main.py:401 + src/config.py:123

实现说明：
- 代码中的 merge_graphs 使用边时间重映射 new_graph.edata['time'] = max_time - time + 1，与论文中的时间衰减思想一致。
- 线图和概率矩阵通过 --enable-line-graph 进行门控，满足“可增强可关闭”的实验对照需求。

## 2.2 4.3.3.2 社区/类别结构融合层

论文要点：
- 基于社区标签构建 Gcls
- 仅连接同社区且训练图中有连接的实体
- TAGConv 多跳聚合
- L2 归一化稳定训练

代码对应：
- 社区标签读取（train.csv 中 modularity_class）：src/utils.py:825
- 社区图构建（同社区且存在训练连接）：src/utils.py:852
- 社区图加入自环：src/utils.py:870
- 在训练入口构建 class_g：src/main.py:255
- TAGConv 应用于实体嵌入：model/rrgcn.py:239, model/rrgcn.py:298
- 融合后归一化：model/rrgcn.py:299

实现说明：
- new_class_graph 的边构造规则与论文式 (4.6) 对齐：仅保留同社区并在训练事实中出现的实体对。
- TAGConv 在老版本 DGL 场景下含兼容兜底（TAGConv 不可用时回退 GraphConv）：model/rrgcn.py:6-9。

## 2.3 4.3.3.3 基于 Hawkes 过程的时序图卷积编码层

论文要点：
- 边注意力 αij
- Hawkes 指数衰减核 κ(Δt)
- 消息传递 + 时间-注意力联合加权
- 跨层门控/重置门融合

代码对应：
- Hawkes 层定义：model/hrgcn.py:7
- 边注意力：model/hrgcn.py:86
- 时间衰减核（delta 与 edge_time）：model/hrgcn.py:28, model/hrgcn.py:103-104
- 消息构造 [entity || relation]：model/hrgcn.py:111-116
- 联合加权聚合（softmax(k*e)）：model/hrgcn.py:120
- skip connect 跨层融合：model/hrgcn.py:35-39, model/hrgcn.py:63, model/hrgcn.py:73
- 全局重置门融合（最终输出与初始表示融合）：model/rrgcn.py:338-342

实现说明：
- HawkesRGCNLayer 在 reduce_func 中直接使用 softmax(k * e) 作为邻居权重，与论文式 (4.12) 的联合权重思想一致。
- 代码中重置门在 RecurrentRGCN.forward 末端执行，作用与论文式 (4.14) 一致。

## 2.4 4.3.3.4 关系演化增强与正则约束层

论文要点：
- 关系-实体交互
- GRU 更新关系表示
- FFT 频域分解 + 低/高频分离
- 频率正则损失 Lfreq

代码对应：
- 全局关系动态模块实现：model/relation_dynamics.py:10
- 关系动态模块在主模型中初始化与调用：model/rrgcn.py:269, model/rrgcn.py:347
- 频域分解（FFT）与低/高频掩码：model/rrgcn.py:400-409
- 频率正则损失 relation_freq_reg：model/rrgcn.py:411
- 训练时将频率正则加入总损失：src/main.py:406, src/main.py:414
- 控制参数：src/config.py:29（freq-reg）, src/config.py:31（alpha）

实现说明：
- relation_freq_reg 与论文“低高频分离 + 高频抑制”目标一致，损失项按 args.freq_reg 加权进入总损失。
- GlobalRelationDynamics 模块以 GRUCell 为核心，执行关系时序更新并归一化。

## 2.5 4.3.3.5 关系解码与预测层

论文要点：
- ConvTransR 关系解码
- 实体对上下文先验（加性偏置）
- 复制-生成混合解码

代码对应：
- 关系解码器 ConvTransR：model/rrgcn.py:246
- 关系上下文先验 MLP（rel_prior）：model/rrgcn.py:252
- 先验加性偏置到关系 logits：model/rrgcn.py:456-457（预测）, model/rrgcn.py:492-499（训练）
- 复制-生成模块实现：model/copy_generation_decoder.py:12
- 复制-生成在预测中融合：model/rrgcn.py:434-444
- 复制-生成在训练中作为辅助损失：model/rrgcn.py:470-478
- 开关参数：src/config.py:94-97（上下文先验）, src/config.py:138-141（copy-generation）, src/config.py:136（relation-dynamics）

实现说明：
- 关系预测路径支持三种形态：
  1) 原始 ConvTransR
  2) ConvTransR + 上下文先验
  3) ConvTransR + 复制生成（并可与先验叠加）
- 与论文“加性偏置 + 混合解码”的结构一致。

---

## 3. 论文消融与脚本映射

## 3.1 快速演示脚本（现场审查）
- 脚本：demo.sh
- 目标：快速跑通主流程
- 命令入口：demo.sh:18
- 输出：checkpoints 年/月/日/小时/序号 目录结构（由 src/main.py:182 创建）

## 3.2 全流程复现实验脚本
- 脚本：run.sh
- 包含：Baseline、A/B/C 消融、终极消融
- Baseline：run.sh:22
- 消融 A（w/o FFT，freq-reg=0.0）：run.sh:40
- 消融 B（w/o 关系上下文先验，去除 relation task 强依赖开关）：run.sh:61
- 消融 C（w/o 正交约束，weight=0.0）：run.sh:76
- 终极消融（同时关闭关键增强）：run.sh:98

## 3.3 消融项到代码开关的精确映射
- w/o FFT 关系嵌入频域分解：--freq-reg 0.0
  - 影响位置：src/main.py:414（总损失中频率项权重为 0）
- w/o 关系上下文先验：--no-rel-context-prior
  - 影响位置：model/rrgcn.py:252（不构建 rel_prior）与 model/rrgcn.py:456 / 492（不加先验偏置）
- w/o 关系正交正则化：--weight 0.0
  - 影响位置：该参数在当前实现中被写入模型配置（model/rrgcn.py:169），但活跃前向路径未直接使用（相关融合代码位于注释区 model/rrgcn.py:391-392）。
  - 审查解释：命令层面保留该消融配置，与论文设置保持一致；若需严格代码等价消融，建议后续将该参数接回有效计算图。

---

## 4. 输出结果文件映射

训练/评估结果主要产物：
- 参数记录：checkpoints/.../args.log（src/main.py:192）
- 训练日志：checkpoints/.../train.log（src/main.py:200）
- 测试日志：checkpoints/.../test.log（src/main.py:198）
- 最优模型：checkpoints/.../best.pt（src/main.py:194）
- 末轮模型：checkpoints/.../last.pt（src/main.py:501）
- 排名明细：checkpoints/.../ranks_raw.txt, ranks_filter.txt（src/main.py:132, src/main.py:137）
- 关系评估结果：checkpoints/.../results.csv（src/main.py:147）

补充可视化与论文图表脚本：
- 时间嵌入参数图：results/第五章图/plot_time_embedding_alpha_5.3.py
- 消融图与其他图表脚本：results/第五章图 下相关文件

---

## 5. 损失函数与训练算法 4.4 的数据信息补充

本节用于补充论文式 (4.31)~(4.33) 和算法 4.4 在代码中的可核验数据来源与参数映射。

### 5.1 数据集规模（来自各数据集 stat.txt）

- WIKI：12554 24 232（data/WIKI/stat.txt）
- ICEWS14s：7128 230 0（data/ICEWS14s/stat.txt）
- ICEWS18：23033 256 0（data/ICEWS18/stat.txt）
- YAGO：10623 10 189（data/YAGO/stat.txt）

说明：
- 第一列对应实体数量 |V|，与 data.num_nodes 一致（由 src/knowledge_graph.py 读取 entity2id.txt 得到）。
- 第二列对应关系数量 |R|，训练时关系预测维度使用 2|R|（包含反向关系）。
- 第三列通常用于时间相关统计；部分数据为 0。实际训练快照由 src/utils.py:639 的 split_by_time 基于四元组时间戳动态切分，审查时以该流程为准。

### 5.2 论文符号与代码变量映射

- D_train, D_valid：src/main.py:166-168 中 train_list, valid_list
- L（历史窗口）：args.train_history_len（src/config.py:157）
- N_epochs：args.n_epochs（src/config.py:11）
- eta（学习率）：args.lr（src/config.py:78）
- lambda_e, lambda_r：args.task_weight 与 (1-args.task_weight)（src/main.py:413）
- lambda_freq：args.freq_reg（src/config.py:29, src/main.py:413）
- lambda_Lie：通过 model.get_loss 内部的 Lie 正则项并入 loss_rel（model/rrgcn.py:522-525）

### 5.3 关系预测损失与实现对照

论文描述：式 (4.31) 给出二元交叉熵形式的关系预测损失。

当前实现：
- 代码实际使用 FocalLoss(gamma=2) 作为实体与关系损失（model/rrgcn.py:211-212）。
- 关系损失计算在 model/rrgcn.py:513（主路径）与 model/rrgcn.py:518（异常回退路径）。

审查结论：
- 论文公式可视为目标函数表达；工程实现采用 FocalLoss 以提升长尾类别稳定性。
- 若需与论文“严格 BCE 版本”完全一致，应在实验说明中单独标注该实现差异或提供 BCE 对照实验。

### 5.4 频率正则与总损失映射

- 频率正则 relation_freq_reg 对应论文式 (4.32)：model/rrgcn.py:411
- 训练中计算频率正则：src/main.py:406
- 总损失：src/main.py:413

对应关系：
- 代码总损失为：L_total = w_alt * (lambda_e * L_ent + lambda_r * L_rel) + lambda_freq * L_freq
- 其中 w_alt 为交替训练时的损失权重（非交替训练时为 1.0）。
- Lie 正则在 get_loss 内部并入 L_rel，因此主训练循环外层未额外显式相加。

### 5.5 算法 4.4 训练步骤与代码对位

1. 初始化模型参数：src/main.py:209-247
2. 构建类别图 G_cls：src/main.py:253-255
3. epoch 循环：src/main.py:365
4. 滑动窗口采样与历史图构建：src/main.py:394-401
5. 前向与损失：model/rrgcn.py:292, model/rrgcn.py:461
6. 频率正则与总损失：src/main.py:406, src/main.py:413
7. 反向传播、梯度裁剪、优化器更新：src/main.py:418-421
8. 优化器类型 AdamW：src/main.py:251（默认）与 src/main.py:345（两阶段训练分组学习率）

---


---

## 6. 完整审查清单

- 论文 4.3.3.1：看 src/utils.py:639, src/utils.py:319, src/utils.py:374, src/utils.py:266
- 论文 4.3.3.2：看 src/utils.py:825, src/utils.py:852, model/rrgcn.py:239, model/rrgcn.py:298
- 论文 4.3.3.3：看 model/hrgcn.py:7, model/hrgcn.py:86, model/hrgcn.py:92, model/rrgcn.py:338
- 论文 4.3.3.4：看 model/relation_dynamics.py:10, model/rrgcn.py:400, model/rrgcn.py:411, src/main.py:414
- 论文 4.3.3.5：看 model/rrgcn.py:246, model/rrgcn.py:252, model/copy_generation_decoder.py:12, model/rrgcn.py:434
- 现场演示：看 demo.sh
- 全流程复现：看 run.sh
- 输出核验：看 checkpoints/*/results.csv 与 args.log

---

