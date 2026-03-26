import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 优先使用黑体，保证中文正常显示
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# Data Preparation
# X-axis: time-embedding-alpha values
alpha_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Y-axis: Metrics
raw_mrr = [0.518073, 0.519796, 0.523831, 0.520741, 0.518126, 0.519806]
raw_hits1 = [0.367250, 0.371727, 0.376611, 0.372677, 0.368335, 0.369692]
raw_hits3 = [0.602225, 0.598291, 0.604396, 0.601954, 0.601140, 0.603175]
raw_hits10 = [0.825940, 0.827432, 0.826753, 0.827974, 0.822684, 0.822412]

# Convert to percentage scale (multiplying by 100) to match common paper styles
mrr_data = [x * 100 for x in raw_mrr]
hits1_data = [x * 100 for x in raw_hits1]
hits3_data = [x * 100 for x in raw_hits3]
hits10_data = [x * 100 for x in raw_hits10]

# Set global font size suitable for papers
plt.rcParams.update({'font.size': 14})

# Create 2x2 subplot layout
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Define a generic plotting function to ensure consistent style
def plot_metric(ax, x_data, y_data, title, color, marker, y_label):
    ax.plot(x_data, y_data, marker=marker, color=color, linewidth=1.5, markersize=8)
    ax.set_xlabel(r'混合系数$\alpha$', fontsize=16) # Using LaTeX for alpha symbol
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=16)
    
    # Grid settings matching the reference style
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    
    # Tick params
    ax.set_xticks(x_data)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Dynamic Y-axis limits with padding
    y_min, y_max = min(y_data), max(y_data)
    padding = (y_max - y_min) * 0.2
    if padding == 0: padding = 0.5
    ax.set_ylim(y_min - padding, y_max + padding)

# 1. MRR (Top-Left, Blue theme)
plot_metric(axs[0, 0], alpha_values, mrr_data, '(a) MRR', '#4c72b0', 'o', 'MRR (%)')

# 2. Hits@1 (Top-Right, Orange theme)
# Note: Hits@1 typically follows MRR closely but is strictly harder
plot_metric(axs[0, 1], alpha_values, hits1_data, '(b) Hits@1', '#dd8452', 's', 'Hits@1 (%)')

# 3. Hits@3 (Bottom-Left, Green theme)
plot_metric(axs[1, 0], alpha_values, hits3_data, '(c) Hits@3', '#55a868', '^', 'Hits@3 (%)')

# 4. Hits@10 (Bottom-Right, Red theme)
plot_metric(axs[1, 1], alpha_values, hits10_data, '(d) Hits@10', '#c44e52', 'd', 'Hits@10 (%)')

# Adjust layout to prevent overlapping labels
fig.tight_layout(pad=1.5)

# Save the figure
output_filename = './results/plot_time_embedding_alpha_5.3.svg'
plt.savefig(output_filename, format='svg')
print(f"Plot saved to {output_filename}")