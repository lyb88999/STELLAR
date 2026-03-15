import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def setup_fonts(language='en'):
    """Setup fonts based on language"""
    if language == 'zh':
        # Chinese font settings
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'Heiti TC']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        # English font settings
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = True

def get_labels(language='en'):
    """Return dictionary of labels in specified language"""
    if language == 'zh':
        return {
            'title_single': '单组件效应分析 (Single Component Analysis)',
            'title_necessity': '组件必要性分析 (Component Necessity Analysis)',
            'xlabel': '通信轮次',
            'ylabel': '准确率 (%)',
            'suptitle': '消融实验分析 (Ablation Study)',
            'experiments': {
                'Baseline (All)': 'Baseline (完整版)',
                'Only Parameter': '仅参数相似度',
                'Only Loss': '仅损失值',
                'Only Prediction': '仅预测结果',
                'No Parameter': '去参数相似度',
                'No Loss': '去损失值',
                'No Prediction': '去预测结果'
            }
        }
    else:
        return {
            'title_single': 'Single Component Analysis',
            'title_necessity': 'Component Necessity Analysis',
            'xlabel': 'Communication Round',
            'ylabel': 'Accuracy (%)',
            'suptitle': 'Ablation Study Analysis',
            'experiments': {
                'Baseline (All)': 'Baseline (All)',
                'Only Parameter': 'Only Parameter',
                'Only Loss': 'Only Loss',
                'Only Prediction': 'Only Prediction',
                'No Parameter': 'No Parameter',
                'No Loss': 'No Loss',
                'No Prediction': 'No Prediction'
            }
        }

def smooth_curve(points, factor=0.8):
    """
    使用指数加权移动平均进行平滑
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_ablation_results(csv_path, output_path="ablation_analysis.png", smooth_factor=0.6, language='en'):
    """
    绘制消融实验结果，包含平滑处理和分组展示
    """
    # Setup style first (this resets rcParams)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Then setup fonts to override style defaults
    setup_fonts(language)
    labels = get_labels(language)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 定义颜色映射
    colors = {
        'Baseline (All)': '#000000',  # 黑色，加粗
        'Only Parameter': '#e74c3c',  # 红
        'Only Loss': '#2ecc71',       # 绿
        'Only Prediction': '#f39c12', # 橙
        'No Parameter': '#9b59b6',    # 紫
        'No Loss': '#3498db',         # 蓝
        'No Prediction': '#1abc9c'    # 青
    }
    
    # 定义线型和宽度 (增强黑白打印辨识度)
    exp_styles = {
        'Baseline (All)':  {'linestyle': '-',  'marker': '*', 'linewidth': 3.0, 'markevery': 2},
        'Only Parameter':  {'linestyle': '--', 'marker': '^', 'linewidth': 2.0, 'markevery': 2},
        'Only Loss':       {'linestyle': ':',  'marker': 's', 'linewidth': 2.0, 'markevery': 2},
        'Only Prediction': {'linestyle': '-.', 'marker': 'D', 'linewidth': 2.0, 'markevery': 2},
        'No Parameter':    {'linestyle': '--', 'marker': 'v', 'linewidth': 2.0, 'markevery': 2},
        'No Loss':         {'linestyle': ':',  'marker': 'x', 'linewidth': 2.0, 'markevery': 2},
        'No Prediction':   {'linestyle': '-.', 'marker': 'o', 'linewidth': 2.0, 'markevery': 2}
    }
    
    # 分组定义 (使用原始英文Key来筛选数据)
    groups = {
        'single': {
            'title_key': 'title_single',
            'experiments': ['Baseline (All)', 'Only Parameter', 'Only Loss', 'Only Prediction']
        },
        'necessity': {
            'title_key': 'title_necessity',
            'experiments': ['Baseline (All)', 'No Parameter', 'No Loss', 'No Prediction']
        }
    }
    
    axes = [ax1, ax2]
    
    for ax, group_info in zip(axes, groups.values()):
        title = labels[group_info['title_key']]
        experiments = group_info['experiments']
        
        for exp_name in experiments:
            # 筛选数据
            exp_data = df[df['Experiment'] == exp_name].sort_values('Round')
            if exp_data.empty:
                continue
                
            rounds = exp_data['Round'].values
            accuracy = exp_data['Accuracy'].values
            
            # 平滑处理
            if smooth_factor > 0:
                accuracy_smooth = smooth_curve(accuracy, smooth_factor)
            else:
                accuracy_smooth = accuracy
                
            # 获取样式
            color = colors.get(exp_name, '#7f8c8d')
            # 使用特定样式，如果没有定义则使用默认
            default_style = {'linestyle': '-', 'marker': None, 'linewidth': 1.5, 'markevery': 1}
            style = exp_styles.get(exp_name, default_style)
            
            # 获取显示标签
            display_label = labels['experiments'].get(exp_name, exp_name)
            
            # 绘制平滑曲线 (带Marker和不同线型)
            ax.plot(rounds, accuracy_smooth, 
                   label=display_label,
                   color=color,
                   linestyle=style['linestyle'],
                   marker=style['marker'],
                   linewidth=style['linewidth'],
                   markevery=style['markevery'], # 避免Marker太密集
                   markersize=6,
                   alpha=0.9)
            
            # 绘制原始数据的散点（淡淡的），可选，为了清晰可以注释掉或保留
            # ax.scatter(rounds, accuracy, color=color, alpha=0.1, s=10)
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(labels['xlabel'], fontsize=12)
        ax.set_ylabel(labels['ylabel'], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, loc='best')
        
        # 设置y轴范围，自动调整但留有余地
        try:
            y_min = df[df['Experiment'].isin(experiments)]['Accuracy'].min()
            y_max = df[df['Experiment'].isin(experiments)]['Accuracy'].max()
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
        except:
            pass # Handle empty data gracefully

    plt.suptitle(labels['suptitle'], fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存图片
    # Save in multiple formats for high quality
    # 1. High-Res PNG (600 DPI)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    
    # 2. Vector Formats (PDF/SVG)
    base_name = os.path.splitext(output_path)[0]
    plt.savefig(f"{base_name}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{base_name}.svg", format='svg', bbox_inches='tight')
    
    print(f"[{language}] Analysis plots saved to: {output_path} (+ .pdf, .svg)")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制消融实验分析图 (中英双语)")
    parser.add_argument("--csv", type=str, required=True, help="CSV文件路径")
    parser.add_argument("--output_dir", type=str, default="experiments/results/plots_dual_lang", help="输出目录")
    parser.add_argument("--smooth", type=float, default=0.6, help="平滑因子 (0-1)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"错误: 文件 {args.csv} 不存在")
    else:
        # 英文版
        en_dir = os.path.join(args.output_dir, "english_paper")
        os.makedirs(en_dir, exist_ok=True)
        plot_ablation_results(args.csv, os.path.join(en_dir, "ablation_analysis.png"), args.smooth, 'en')
        
        # 中文版
        zh_dir = os.path.join(args.output_dir, "chinese_thesis")
        os.makedirs(zh_dir, exist_ok=True)
        plot_ablation_results(args.csv, os.path.join(zh_dir, "ablation_analysis.png"), args.smooth, 'zh')
