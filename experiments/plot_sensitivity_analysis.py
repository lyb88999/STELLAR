
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
import glob
import re

def setup_fonts(language='en'):
    """Setup fonts based on language"""
    # Setup style first (this resets rcParams)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if language == 'zh':
        # Chinese font settings (Mac friendly)
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
            'title': r'权重 $\alpha$ (参数相似度) 敏感性分析',
            'xlabel': r'权重 $\alpha$ (结构)',
            'ylabel': '最终收敛准确率 (%)',
            'legend_label': '平均准确率 (最后5轮)',
            'best_label': '最佳 $\\alpha={}$\n({:.2f}%)',
            'error_read': '  - 读取失败: {}, 错误: {}',
            'error_format': "错误: 'Experiment' 列格式不正确，应包含 'Alpha=X.X'",
            'processing': '正在处理 {} 个输入文件...',
            'read_ok': '  - 已读取: {} ({} 行)'
        }
    else:
        return {
            'title': r'Sensitivity Analysis of Weight $\alpha$ (Parameter Similarity)',
            'xlabel': r'Weight $\alpha$ (Structure)',
            'ylabel': 'Final Convergence Accuracy (%)',
            'legend_label': 'Avg. Accuracy (Last 5 Rounds)',
            'best_label': 'Optimal $\\alpha={}$\n({:.2f}%)',
            'error_read': '  - Read failed: {}, Error: {}',
            'error_format': "Error: 'Experiment' column format incorrect, should contain 'Alpha=X.X'",
            'processing': 'Processing {} input files...',
            'read_ok': '  - Read: {} ({} rows)'
        }

def plot_sensitivity_analysis(input_files, output_path, language='en'):
    """
    读取一个或多个CSV文件，合并数据，计算最后几轮的平均精度，绘制参数敏感性分析图。
    """
    # 设置字体和标签
    setup_fonts(language)
    labels = get_labels(language)
    
    all_data = []

    # 1. 循环读取所有输入文件
    print(labels['processing'].format(len(input_files)))
    for file_path in input_files:
        try:
            df = pd.read_csv(file_path)
            
            # 清理列名（去除空格）
            df.columns = df.columns.str.strip()
            # 标准化列名
            column_mapping = {
                'round': 'Round',
                'accuracy': 'Accuracy',
                'experiment': 'Experiment'
            }
            df.rename(columns=lambda x: column_mapping.get(x.lower(), x), inplace=True)
            
            all_data.append(df)
            print(labels['read_ok'].format(file_path, len(df)))
        except Exception as e:
            print(labels['error_read'].format(file_path, e))

    if not all_data:
        print("Error: No valid data read.")
        return

    # 合并所有数据
    full_df = pd.concat(all_data, ignore_index=True)

    # 2. 数据预处理
    # 从 "Alpha=0.2" 提取数值 0.2
    try:
        # 使用正则表达式增加鲁棒性
        def extract_alpha(s):
             match = re.search(r'Alpha=([0-9.]+)', str(s))
             return float(match.group(1)) if match else None

        full_df['Alpha_Value'] = full_df['Experiment'].apply(extract_alpha)
        # 移除解析失败的行
        full_df = full_df.dropna(subset=['Alpha_Value'])
        
    except Exception as e:
        print(labels['error_format'])
        print(f"Details: {e}")
        return

    # 3. 提取收敛阶段数据
    max_round = full_df['Round'].max()
    converged_rounds = 5
    start_round = max_round - converged_rounds + 1
    
    subset = full_df[full_df['Round'] >= start_round]

    # 4. 聚合计算 (Mean & Std)
    sensitivity_df = subset.groupby('Alpha_Value')['Accuracy'].agg(['mean', 'std']).reset_index()

    print("\n--- Final Statistics ---")
    print(sensitivity_df)

    # 5. 绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制带误差棒的折线图
    plt.errorbar(
        sensitivity_df['Alpha_Value'], 
        sensitivity_df['mean'], 
        yerr=sensitivity_df['std'], 
        fmt='-o',               # 线型
        color='black',          # 主线条黑色
        ecolor='gray',          # 误差棒灰色
        elinewidth=2,           # 误差棒粗细
        capsize=5,              # 误差棒帽子大小
        linewidth=2.5,          # 主线条粗细
        markersize=8,           # 数据点大小
        markerfacecolor='#D62728', # 红色填充，突出显示
        markeredgecolor='black',
        label=labels['legend_label']
    )

    # 6. 自动寻找并标记最高点
    best_idx = sensitivity_df['mean'].idxmax()
    best_alpha = sensitivity_df.loc[best_idx, 'Alpha_Value']
    best_acc = sensitivity_df.loc[best_idx, 'mean']

    # 格式化最佳标签
    best_text = labels['best_label'].format(best_alpha, best_acc)

    plt.annotate(
        best_text, 
        xy=(best_alpha, best_acc), 
        xytext=(best_alpha, best_acc + 0.5), # Increased offset from 0.15 to 0.5
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        ha='center', fontsize=12, fontweight='bold', color='#D62728'
    )

    # 7. 图表美化
    plt.title(labels['title'], fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(labels['xlabel'], fontsize=13)
    plt.ylabel(labels['ylabel'], fontsize=13)
    
    plt.xticks(sensitivity_df['Alpha_Value'].unique())
    
    # 自动调整Y轴范围
    y_min = sensitivity_df['mean'].min()
    y_max = sensitivity_df['mean'].max()
    margin = (y_max - y_min) * 0.6 # 留出足够空间给文字标签
    plt.ylim(y_min - margin, y_max + margin)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower center', fontsize=11, frameon=True)
    plt.tight_layout()

    # 8. 保存
    # Save in multiple formats for high quality
    # 1. High-Res PNG (600 DPI)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    
    # 2. Vector Formats (PDF/SVG)
    base_name = os.path.splitext(output_path)[0]
    plt.savefig(f"{base_name}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{base_name}.svg", format='svg', bbox_inches='tight')
    
    print(f"\n[{language}] Plot saved to: {output_path} (+ .pdf, .svg)")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制多实验参数敏感性分析图 (双语版)")
    
    # 支持多个输入文件 (nargs='+') or single csv flag for backward compatibility
    parser.add_argument("-i", "--inputs", nargs='+', 
                        help="输入的一个或多个CSV文件路径 (例如: data1.csv data2.csv)")
    parser.add_argument("--csv", type=str, help="兼容旧参数 (单个CSV)")
    
    parser.add_argument("-o", "--output_dir", type=str, default="experiments/results/plots_dual_lang", 
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 收集所有文件
    all_files = []
    if args.inputs:
        for f in args.inputs:
            files = glob.glob(f)
            if files:
                all_files.extend(files)
            else:
                print(f"Warning: File not found {f}")
    if args.csv:
         if os.path.exists(args.csv):
             all_files.append(args.csv)
            
    if not all_files:
        print("Error: No valid input files.")
    else:
        # Generate English version
        en_dir = os.path.join(args.output_dir, "english_paper")
        os.makedirs(en_dir, exist_ok=True)
        plot_sensitivity_analysis(all_files, os.path.join(en_dir, "sensitivity_analysis_summary.png"), 'en')
        
        # Generate Chinese version
        zh_dir = os.path.join(args.output_dir, "chinese_thesis")
        os.makedirs(zh_dir, exist_ok=True)
        plot_sensitivity_analysis(all_files, os.path.join(zh_dir, "sensitivity_analysis_summary.png"), 'zh')
