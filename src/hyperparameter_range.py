# src/hyperparameter_range.py

hp_range = {
    "n_hidden": [200],  # 实体和关系的嵌入维度
    "n_layers": [2],    # HRGCN 层数
    "dropout": [0.2],   # 每层的 Dropout 概率
    "n_bases": [None],  # 基数数量（如果适用）
    "lr": [0.001],      # Adam 优化器的学习率
    "conv_kernels": [50],  # ConvTransE 的卷积核数量
    "kernel_size": [(2, 3)],  # ConvTransE 的卷积核大小
    "random_seeds": [123],  # 随机种子
    "edge_weight": [1],  # 边的默认权重
}