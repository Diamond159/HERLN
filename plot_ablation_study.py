import matplotlib.pyplot as plt
import numpy as np

# Set font to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei for Chinese
plt.rcParams['axes.unicode_minus'] = False     # Ensure minus signs are displayed correctly

def plot_ablation_study():
    # Data from Table 1
    # Models/Configurations
    models = [
        "完整TDSRN-AGEP模型",
        "移除静态图组件",
        "移除对比损失",
        "移除双流架构",
        "移除对比损失+双流"
    ]
    
    # Dataset 1: ICEWS14
    icews14_mrr = [51.17, 49.45, 50.44, 50.28, 50.08]
    # Dataset 2: ICEWS18
    icews18_mrr = [37.94, 35.51, 37.54, 37.55, 37.47]
    
    # Data from Table 2
    # Dataset 3: ICEWS05-15
    icews0515_mrr = [58.90, 56.92, 58.06, 57.88, 57.65]
    # Dataset 4: GDELT
    gdelt_mrr = [25.30, 23.68, 25.04, 25.03, 24.99]

    # Combine datasets for plotting structure
    # We want X-axis to be Datasets, and bars to be Models
    datasets = ['ICEWS14', 'ICEWS18', 'ICEWS05-15', 'GDELT']
    
    # Re-organize data: list of lists where each sublist is a model's scores across datasets
    # model_data[i] corresponds to models[i]
    model_data = []
    for i in range(len(models)):
        row = [icews14_mrr[i], icews18_mrr[i], icews0515_mrr[i], gdelt_mrr[i]]
        model_data.append(row)

    # Plotting configuration
    x = np.arange(len(datasets))  # the label locations
    width = 0.15  # the width of the bars
    
    # Patterns for bars to match the style (hatching)
    patterns = ['/', 'x', '.', 'o', '*'] # Different patterns for different models
    colors = ['#4e79a7', '#59a14f', '#edc948', '#f28e2b', '#e15759'] # Tableau-like colors
    
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create bars for each model
    rects = []
    # Calculate offset for each bar to center the group
    total_width = width * len(models)
    start_offset = -total_width / 2 + width / 2

    for i, model_name in enumerate(models):
        offset = start_offset + i * width
        rect = ax.bar(x + offset, model_data[i], width, label=model_name, 
                      color=colors[i], edgecolor='black', hatch=patterns[i], alpha=0.9)
        rects.append(rect)
        
        # Add value labels on top of bars
        ax.bar_label(rect, padding=3, fmt='%.2f', fontsize=8, rotation=90)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MRR (%)', fontsize=12)
    ax.set_xlabel('数据集', fontsize=12)
    # ax.set_title('消融实验性能对比 (MRR)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('ablation_study_mrr.png', dpi=300)
    print("Plot saved as ablation_study_mrr.png")
    # plt.show()

if __name__ == "__main__":
    plot_ablation_study()
