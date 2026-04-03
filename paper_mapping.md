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

创新点补充（面向 DRPM-NSCV 描述）：
- 邻域采样与聚类验证：通过“历史窗口 + 快照子图”的邻域构建与 train.csv 的社区标签构图，形成可核验的结构先验；既保留局部邻域动态，又用同社区且存在训练事实的实体对进行结构验证与约束。
- FFT 频域解耦：关系嵌入执行频域分解，将长期稳定语义（低频）与短期波动（高频）显式拆分；alpha 控制低通强度以稳定长期趋势。
- 频域正则抑噪：在训练损失中加入高频抑制与低高频分离正则，缓解频谱纠缠与噪声扩散。
- 关系上下文先验：基于主体-客体实体对的先验分布偏好，对关系 logits 施加加性偏置，为解码器提供结构归纳偏置，降低对纯解码器统计关联的依赖。

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
 - 社区标签来源于 train.csv 的 modularity_class，实现“聚类验证”的结构先验来源。

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
 - FFT 分解与 alpha 低通掩码共同实现“长期趋势/短期波动”的显式解耦。

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
 - 上下文先验在预测与训练阶段均以加性偏置融入关系 logits，用于引入结构归纳偏置。

## 2.6 变量与公式逐项映射（4.3.3.1 ~ 4.3.4）

本节补充“论文符号 -> 代码变量/函数 -> 调用位置”的逐项映射，便于从公式直接追到实现。

### 2.6.1 4.3.3.1 历史结构构建与边级增强层

| 论文符号/概念 | 代码变量/函数 | 说明 | 位置 |
|---|---|---|---|
| $G_i=\{(s,r,o,\tau)\mid \tau=t_i\}$ | `train_list` / `valid_list` / `test_list` | 按时间切分后的快照序列，每个元素是某个时间步的三元组集合 | `src/main.py`（`run_experiment` 中 `utils.split_by_time(...)`） |
| 快照切分中的 $t_i$ | `t`, `latest_t` | `split_by_time` 按第 4 列时间戳分桶，遇到时间变化就开启新快照 | `src/utils.py`（`split_by_time`） |
| $(s,r,o)$ | `snapshot` 中的 `train[:3]` | 每条四元组仅取前三列进入快照（三元组） | `src/utils.py`（`split_by_time`） |
| 反向关系 $(o, r+|R|, s)$ | `prepare_triples_with_inverse`、`inverse_triples[:,1] += num_rels` | 将关系 ID 偏移 `+num_rels` 区分正/反向关系 | `src/utils.py`（`prepare_triples_with_inverse`） |
| 历史窗口 $history(t_k)=[G_{k-h+1},...,G_k]$ | `input_list` | 训练时按 `train_sample_num` 截取历史窗口；测试时按 `args.test_history_len` 截取 | `src/main.py`（训练循环与 `test`） |
| 历史窗口长度 $h$ | `args.train_history_len` / `args.test_history_len` | 训练/测试窗口长度；`-1` 表示使用全部历史 | `src/config.py` 与 `src/main.py` |
| 历史子图列表 | `history_glist` | 将窗口内每个快照经 `build_sub_graph` 转为 DGL 子图列表 | `src/main.py` |
| 边类型 $r$ | `g.edata['type']` | 子图中每条边保存关系类型（含反向关系） | `src/utils.py`（`build_sub_graph`） |
| 边时间标签 | `g.edata['time']` | 子图阶段先写入快照索引 `idx`；合并后重映射为相对时间 | `src/utils.py`（`build_sub_graph`、`merge_graphs`） |
| 合并历史图 $G_H=\bigcup_i G_i$ | `merge_graphs(...)` / `history_graph` | 将历史窗口多图拼接为统一历史图输入编码器 | `src/utils.py` 与 `model/rrgcn.py`（`forward`） |
| 时间权重/衰减索引 $w_i=max(t)-t_i+1$ | `new_graph.edata['time'] = max_time - time + 1` | 统一历史图上使用“越旧值越大”的时间距离，供 Hawkes 衰减核使用 | `src/utils.py`（`merge_graphs`） |
| 概率矩阵 PM_PD | `cal_pmpd` / `pm_pd` / `g.pm_pd` | 节点-边关联稀疏矩阵，头节点 `+1`，尾节点 `-1` | `src/utils.py`（`cal_pmpd`、`build_line_graph_and_pm`、`build_sub_graph`） |
| 线图 | `graph.line_graph(backtracking=False)` / `g.line_graph_obj` | 边级结构增强，线图节点对应原图边 | `src/utils.py`（`build_line_graph_and_pm`） |
| 边级增强开关 | `args.enable_line_graph` | 控制是否在子图构建时附加线图与概率矩阵 | `src/config.py`、`src/main.py` |

### 2.6.2 4.3.3.2 社区/类别结构融合层

| 论文符号/概念 | 代码变量/函数 | 说明 | 位置 |
|---|---|---|---|
| 社区划分 $C=\{c_1,...,c_K\}$ | `node2class`, `class2node` | 从 `train.csv` 的 `modularity_class` 读取社区标签 | `src/utils.py`（`analyse_class`） |
| 社区图 $G_{cls}$ | `class_g = new_class_graph(node2class, train_list)` | 训练阶段构建的类别图/社区图 | `src/main.py` |
| 社区边集 $E_{cls}$ | `edges.add((s,o))`（条件：`node2class[s] == node2class[o]`） | 只保留“同社区且在训练事实出现”的实体对 | `src/utils.py`（`new_class_graph`） |
| 自环 | `dgl.add_self_loop(g)` | 与算法步骤一致，增强节点自身保真 | `src/utils.py`（`new_class_graph`） |
| 初始实体嵌入 $E^{(0)}$ | `self.emb_ent` | 模型可学习实体参数矩阵 | `model/rrgcn.py`（`RecurrentRGCN.__init__`） |
| TAGConv 增强 $E_{aug}^{(0)}$ | `self.classgcn(class_g, current_ent_emb)` | 在社区图上对实体嵌入做传播增强 | `model/rrgcn.py`（`forward`） |
| 归一化 $F_{normalize}(\cdot)$ | `F.normalize(current_ent_emb)` | 当 `layer_norm=True` 时启用归一化 | `model/rrgcn.py`（`forward`） |

### 2.6.3 4.3.3.3 Hawkes 时序图卷积编码层

| 论文符号/概念 | 代码变量/函数 | 说明 | 位置 |
|---|---|---|---|
| 历史图输入 $H_t$ | `g_list` / `history_graph` | `g_list` 先按窗口构建，再合并为 `history_graph` 送入编码器 | `src/main.py`、`model/rrgcn.py` |
| 实体表示 $e_i^{(l)}$ | `g.ndata['h']` / `node_repr` | Hawkes 层内节点隐藏状态 | `model/hrgcn.py` |
| 关系表示 $r_{ij}$ | `self.rel_emb.index_select(0, edges.data['type'])` | 通过边类型索引关系向量 | `model/hrgcn.py`（`edge_attention`、`msg_func`） |
| 边注意力 $\alpha_{ij}$ | `edges.data['e']`（来自 `attn_fc` + LeakyReLU） | 注意力输入为 `[src_h, dst_h, rel_emb]` 拼接 | `model/hrgcn.py`（`edge_attention`） |
| 衰减参数 $\delta$ | `self.delta` | 可学习的 Hawkes 衰减率参数 | `model/hrgcn.py` |
| 相对时间差 $\Delta t_{ij}$ | `edge_time = edges.data['time']` | 使用历史图重映射时间作为衰减输入 | `model/hrgcn.py`（`msg_func`） |
| 衰减核 $\kappa_{ij}=\exp(-\delta\Delta t)$ | `k = (-1 * edge_time * self.delta)` 后进入 `softmax(k*e)` | 实现中用 logit 形式与注意力相乘后 softmax 归一化 | `model/hrgcn.py`（`msg_func`、`reduce_func`） |
| 消息 $m_{i\to j}$ | `msg = torch.cat((node, relation), dim=1); msg = torch.mm(msg, self.weight_neighbor)` | 对 `[e_i || r_{ij}]` 做线性变换 | `model/hrgcn.py`（`msg_func`） |
| 联合加权聚合 | `h = torch.sum(k * nodes.mailbox['msg'], dim=1)` | 邻居消息按 `softmax(k*e)` 权重聚合 | `model/hrgcn.py`（`reduce_func`） |
| 跨层门控（skip） | `skip_weight = sigmoid(prev_h W + b)`；`node_repr = skip_weight*node_repr + (1-skip_weight)*prev_h` | 防止深层过平滑 | `model/hrgcn.py`（`forward`） |
| 全局重置门 | `reset_gate` + `reset_gate1` + `new_ent_emb = new*weight + current*(1-weight)` | 最终层输出与初始实体表示融合 | `model/rrgcn.py`（`forward`） |

### 2.6.4 4.3.3.4 关系演化增强与频率正则

| 论文符号/概念 | 代码变量/函数 | 说明 | 位置 |
|---|---|---|---|
| 当前关系嵌入 $R_t^{(0)}$ | `current_rel_emb` | 由 `self.emb_rel` 初始化，可归一化 | `model/rrgcn.py`（`forward`） |
| 关系动态模块 | `self.global_rel_dynamics` | 可选 ERD-Net 全局关系动态建模模块 | `model/rrgcn.py`（初始化与 `forward`） |
| 关系-实体交互 $I_r$ | `rel_ent_input`（基于 `avg_ent_feature` + `rel_ent_interaction`） | 当前实现为简化交互（平均实体特征），再经 MLP 变换 | `model/relation_dynamics.py`（`forward`） |
| GRU 更新 $h_{r,t}$ | `self.rel_gru(rel_input, ...)` / `self.rel_h0` | 关系嵌入通过 GRUCell 做时间更新并归一化 | `model/relation_dynamics.py`（`forward`） |
| 频域变换 FFT | `freq_domain = fft.fft(rel, dim=1)` | 对关系嵌入按隐层维度做频域分解 | `model/rrgcn.py`（`_relation_fft_components`） |
| 低/高频掩码 $M_{low},M_{high}$ | `low_mask = exp(-abs_freqs * self.alpha)`，`high_mask = 1-low_mask` | `alpha` 控制低通强度 | `model/rrgcn.py`（`_relation_fft_components`） |
| 低/高频分量 $\hat R_{low},\hat R_{high}$ | `low_freq`, `high_freq` | 分别用于分离约束与高频抑制 | `model/rrgcn.py`（`_relation_fft_components`） |
| 频率正则 $L_{freq}$ | `relation_freq_reg()` 返回 `(separation + high_intensity) * norm_factor` | `separation=-||low-high||_2`，`high_intensity=||high||_2` | `model/rrgcn.py` |
| 频率正则系数 $\lambda_{freq}$ | `args.freq_reg` | 训练总损失中乘到 `loss_freq` | `src/config.py`、`src/main.py` |
| 低通强度系数 $\alpha_f$ | `args.alpha` -> `self.alpha` | 由 CLI 传入模型，作用于 FFT 掩码 | `src/config.py`、`src/main.py`、`model/rrgcn.py` |

### 2.6.5 4.3.3.5 关系解码与预测层

| 论文符号/概念 | 代码变量/函数 | 说明 | 位置 |
|---|---|---|---|
| 时间步实体表示 $E_t$ | `embedding` / `pre_emb` | `forward` 输出的最终实体表示 | `model/rrgcn.py`（`predict`、`get_loss`） |
| 演化关系表示 $R_t^{dyn}$ | `r_emb` | `forward` 返回的关系表示（可含动态更新） | `model/rrgcn.py` |
| ConvTransR 解码 | `self.rdecoder.forward(...)` | 关系预测基础分支 | `model/rrgcn.py` |
| 基础关系得分 $S_{base}$ | `score_rel`（先验注入前） | 来自 `rdecoder` 的关系 logits | `model/rrgcn.py` |
| 实体对特征 $[e_s||e_o]$ | `pair_feat = torch.cat([x_s, x_o], dim=-1)` | 上下文先验输入 | `model/rrgcn.py` |
| 先验 logits $l_{prior}$ | `prior_logits = self.rel_prior(pair_feat)` | MLP 生成关系偏置 | `model/rrgcn.py` |
| 最终关系得分 $S_{rel}=S_{base}+\lambda_{prior}l_{prior}$ | `score_rel = score_rel + self.rel_prior_weight * prior_logits` | 加性偏置注入 | `model/rrgcn.py` |
| 先验权重 $\lambda_{prior}$ | `self.rel_prior_weight`（来自 `args.rel_prior_weight`） | 控制先验强度 | `src/config.py`、`model/rrgcn.py` |
| 复制-生成分支 | `self.copy_gen_decoder` / `copy_scores` | 与基础分支线性融合增强关系预测 | `model/rrgcn.py`、`model/copy_generation_decoder.py` |
| 融合系数（工程实现） | 常数 `0.3`（预测融合）与 `0.1`（训练 copy loss 权重） | 当前代码中使用固定权重；`copy_gen_alpha` 主要传入 copy 模块内部 | `model/rrgcn.py` |

### 2.6.6 4.3.4 训练目标与算法 4.4 映射

| 论文符号/概念 | 代码变量/函数 | 说明 | 位置 |
|---|---|---|---|
| 训练快照序列 $D_{train}=\{G_1,...,G_T\}$ | `train_list` | 由训练集按时间切分快照得到 | `src/main.py` |
| 验证集 $D_{valid}$ | `valid_list` | 验证快照序列 | `src/main.py` |
| 轮数 $N_{epochs}$ | `args.n_epochs` | 训练 epoch 数 | `src/config.py`、`src/main.py` |
| 学习率 $\eta$ | `args.lr` | AdamW 主学习率 | `src/config.py`、`src/main.py` |
| 历史长度 $L$ | `args.train_history_len` | 滑动窗口历史长度 | `src/config.py`、`src/main.py` |
| 实体损失 $L_{ent}$ | `loss_e` | `model.get_loss(...)` 返回 | `model/rrgcn.py`、`src/main.py` |
| 关系损失 $L_{rel}$ | `loss_r` | `model.get_loss(...)` 返回；内部可叠加 Lie 正则 | `model/rrgcn.py`、`src/main.py` |
| 频率正则 $L_{freq}$ | `loss_freq = model.relation_freq_reg()` | 外层训练循环每步计算 | `src/main.py` |
| 任务权重 $\lambda_e,\lambda_r$ | `args.task_weight` 与 `1-args.task_weight` | 总损失中实体/关系分支加权 | `src/config.py`、`src/main.py` |
| 频率权重 $\lambda_{freq}$ | `args.freq_reg` | 对 `loss_freq` 进行加权 | `src/config.py`、`src/main.py` |
| Lie 权重 $\lambda_{Lie}$ | `args.lie_ent_weight` / `args.lie_rel_weight` / `args.lie_pair_weight` | 在 `lie_regularizer(...)` 内生效并并入 `loss_rel` | `src/config.py`、`model/rrgcn.py` |
| 总损失 $L_{total}$ | `loss = loss_weight*(task_weight*loss_e + (1-task_weight)*loss_r) + freq_reg*loss_freq` | 与交替训练权重 `loss_weight` 共同作用 | `src/main.py` |
| 梯度裁剪 | `clip_grad_norm_(model.parameters(), args.grad_norm)` | 防止梯度爆炸 | `src/main.py` |
| 优化器 | `torch.optim.AdamW(...)` | 默认优化器；两阶段训练可分组学习率 | `src/main.py` |

补充说明（实现差异，建议在论文实验说明中注明）：
- 论文式（4.31）写作 BCE 形式，当前代码的实体/关系主损失使用 `FocalLoss(gamma=2)`。
- 论文算法中的关系-实体交互（按关系聚合邻接实体）在当前 `GlobalRelationDynamics` 中采用了“平均实体特征 + MLP”的简化实现。
- 论文式（4.29）中的动态复制权重在当前主路径中未显式按样本学习，预测融合采用固定系数（`0.3`）。

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

