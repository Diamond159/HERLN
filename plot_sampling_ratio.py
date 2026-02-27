import matplotlib.pyplot as plt
import matplotlib

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据
x = [0.0, 0.2, 0.35, 0.4, 0.6, 0.8, 1.0]
x_labels = ['0', '0.2', '0.35', '0.4', '0.6', '0.8', '1']
icews14_mrr = [50.28, 50.55, 51.05, 51.17, 50.71, 50.75, 51.31]
icews18_mrr = [37.55, 37.92, 37.94, 37.77, 37.68, 37.56, 37.99]

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制 ICEWS14 MRR
ax1.plot(x, icews14_mrr, linestyle='--', marker='o', label='ICEWS14 MRR', color='#1f77b4')
ax1.set_xlabel('采样比率', fontsize=12)
ax1.set_ylabel('MRR (%)', fontsize=12)
ax1.set_title('(a) ICEWS14', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.grid(True, linestyle='--', alpha=0.5)

# 绘制 ICEWS18 MRR
ax2.plot(x, icews18_mrr, linestyle='--', marker='s', label='ICEWS18 MRR', color='#ff7f0e')
ax2.set_xlabel('采样比率', fontsize=12)
ax2.set_ylabel('MRR (%)', fontsize=12)
ax2.set_title('(b) ICEWS18', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels)
ax2.grid(True, linestyle='--', alpha=0.5)

# 调整布局
plt.tight_layout()

# 保存
plt.savefig('sampling_ratio_mrr.png', dpi=300)
plt.show()
