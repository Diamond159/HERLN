import matplotlib.pyplot as plt
import numpy as np

# Set the style to match the reference image generally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# Data Preparation
datasets = ['ICEWS14', 'ICEWS18', 'WIKI', 'YAGO']
metrics = ['MRR', 'H@1', 'H@3', 'H@10']

# Model configurations (Legend labels)
# Translating the Chinese labels to English or keeping as is depending on preference.
# Based on the user prompt's "Model Configuration":
# DRM-MR (完整) -> DRM-MR (Full)
# w/o FFT频域分解 -> w/o FFT Decomp
# w/o 关系上下文先验 -> w/o Context Prior
# w/o 关系正交正则化 -> w/o Ortho Reg
# 全部消融 -> All Ablated

configs = [
    'DRM-MR (Full)',
    'w/o FFT', 
    'w/o Context', # Shortened for better fit
    'w/o Orthogonal', # Shortened for better fit
    'Ablation All'
]

# Patterns for bars to match reference style (hatched patterns)
# The reference image has: blue (lines), orange (vertical lines), green (horizontal lines), red (diagonal lines)
# We will try to mimic these.
patterns = ['//', '||', '--', '\\\\', 'xx']
colors = ['#5b9bd5', '#ed7d31', '#70ad47', '#eb5e5e', '#a5a5a5']

# Data Structure: data[dataset][metric] = [val_config1, val_config2, ...]
# Order of configs: DRM-MR (Full), w/o FFT, w/o Context, w/o Orthogonal, All Ablated

data = {
    'ICEWS14': {
        'MRR':  [52.10, 51.92, 52.09, 51.71, 51.49],
        'H@1':  [37.32, 37.20, 37.42, 37.01, 36.56],
        'H@3':  [59.84, 59.67, 59.84, 59.53, 59.84],
        'H@10': [82.43, 82.38, 82.43, 82.55, 82.39]
    },
    'ICEWS18': {
        'MRR':  [51.91, 51.73, 51.89, 51.68, 51.45],
        'H@1':  [36.84, 36.65, 36.82, 36.59, 36.32],
        'H@3':  [59.90, 59.72, 59.88, 59.65, 59.51],
        'H@10': [82.85, 82.67, 82.83, 82.71, 82.51]
    },
    'WIKI': {
        'MRR':  [99.45, 99.41, 99.43, 99.38, 99.32],
        'H@1':  [99.10, 99.06, 99.08, 99.03, 98.96],
        'H@3':  [99.77, 99.73, 99.75, 99.70, 99.64],
        'H@10': [99.96, 99.95, 99.95, 99.94, 99.92]
    },
    'YAGO': {
        'MRR':  [99.08, 99.04, 99.06, 99.01, 98.94],
        'H@1':  [98.67, 98.63, 98.65, 98.60, 98.53],
        'H@3':  [99.35, 99.31, 99.33, 99.28, 99.20],
        'H@10': [100.00, 99.99, 99.99, 99.98, 99.97]
    }
}

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

bar_width = 0.15
x = np.arange(len(metrics))

for i, dataset in enumerate(datasets):
    ax = axes[i]
    dataset_data = data[dataset]
    
    # Extract data for this dataset into a matrix (config x metric)
    values = [[dataset_data[m][c] for m in metrics] for c in range(len(configs))]
    
    # Plot bars for each configuration
    for j, config_name in enumerate(configs):
        offset = (j - len(configs)/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values[j], 
                      width=bar_width, 
                      label=config_name, 
                      color=colors[j], 
                      edgecolor='black', 
                      linewidth=0.5,
                      hatch=patterns[j] * 2, # Density of hatch
                      alpha=0.9)
    
    ax.set_title(dataset, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Metric', fontsize=14)
    
    # Adjust Y-axis limits to make differences visible
    # Find min and max for this dataset
    all_vals = [val for sublist in values for val in sublist]
    min_val = min(all_vals)
    max_val = max(all_vals)
    padding = (max_val - min_val) * 0.2
    
    # For H@10 in WIKI/YAGO which is near 100, we might need specific scaling
    # But automatic scaling with some lower bound usually works best to show "zoom"
    if min_val > 90:
        ax.set_ylim(min_val - 0.5, 100.2)
    elif min_val > 50: # ICEWS
         ax.set_ylim(min_val - 2, max_val + 2)
    else:
        ax.set_ylim(min_val - 5, max_val + 5)
        
    ax.tick_params(axis='y', labelsize=12)

    # Add Legend to the first plot (or all, but commonly just one or shared)
    # The reference image has legends in each plot.
    ax.legend(loc='best', fontsize=10, framealpha=0.5)

plt.tight_layout()
plt.savefig('./results/ablation_study_plot_5.4.svg', format='svg')
print("Plot saved as /results/ablation_study_plot_5.4.svg")