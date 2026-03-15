#!/usr/bin/env python3
"""
Dual-Language Plotting Tool for Async Experiments
Generates plots for both Chinese Thesis and English Paper.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
import sys

# Configure fonts for Chinese support
def setup_fonts(language='en'):
    plt.style.use('seaborn-v0_8-whitegrid')
    if language == 'zh':
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = True

def get_labels(language='en'):
    if language == 'zh':
        return {
            'acc_title': '不同异步场景下的模型准确率对比', # More specific than "分类准确率对比"
            'acc_ylabel': '准确率 (%)',
            'f1_title': 'F1 分数 (加权)',
            'f1_ylabel': 'F1 分数',
            'loss_title': '测试损失收敛',
            'loss_ylabel': '测试损失',
            'util_title': '重度异步场景下的信息利用率分析', # More specific
            'util_ylabel': '参与聚合的地面站数量',
            'fresh_label': '及时更新 (Fresh)',
            'stale_label': '滞后更新 (Stale/Recovered)',
            'xlabel': '通信轮次',
            'total_recovered': '累计回收滞后更新: {}',
            'scenarios': {
                'Synchronous (Baseline)': '同步基准 (Baseline)',
                'Mild Async (30% Delay)': '轻度异步 (30% 延迟)',
                'Severe Async (60% Delay)': '重度异步 (60% 延迟)'
            }
        }
    else:
        return {
            'acc_title': 'Accuracy Comparison under Different Async Scenarios',
            'acc_ylabel': 'Accuracy (%)',
            'f1_title': 'F1 Score (Weighted)',
            'f1_ylabel': 'F1 Score',
            'loss_title': 'Test Loss Convergence',
            'loss_ylabel': 'Test Loss',
            'util_title': 'Information Utilization Analysis in Severe Async',
            'util_ylabel': 'Number of Participating Stations',
            'fresh_label': 'Fresh Updates',
            'stale_label': 'Stale Updates (Recovered)',
            'xlabel': 'Communication Round', # Consistent with other plots
            'total_recovered': 'Total Stale Recovered: {}',
            'scenarios': {
                'Synchronous (Baseline)': 'Synchronous (Baseline)',
                'Mild Async (30% Delay)': 'Mild Async (30% Delay)',
                'Severe Async (60% Delay)': 'Severe Async (60% Delay)'
            }
        }

def save_plot(path, language):
    """Save plot in triple format: PNG(600dpi), PDF, SVG"""
    plt.tight_layout()
    # 1. High-Res PNG
    plt.savefig(path, dpi=600, bbox_inches='tight')
    # 2. PDF
    base = os.path.splitext(path)[0]
    plt.savefig(f"{base}.pdf", format='pdf', bbox_inches='tight')
    # 3. SVG
    plt.savefig(f"{base}.svg", format='svg', bbox_inches='tight')
    print(f"[{language}] Saved: {path} (+ .pdf, .svg)")
    plt.close()

def plot_async_results(csv_path, output_dir, language='en'):
    setup_fonts(language)
    labels = get_labels(language)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    experiments = df['Experiment'].unique()
    
    # --- 1. Plot Accuracy Comparison (Single Plot) ---
    plt.figure(figsize=(10, 6))
    col = 'Accuracy'
    title = labels['acc_title']
    ylabel = labels['acc_ylabel']
    
    # Colors/Styles for consistent plotting
    colors = ['#34495e', '#3498db', '#e74c3c', '#f1c40f']
    styles = ['-', '--', '-.', ':']
    
    for i, exp_name in enumerate(experiments):
        exp_data = df[df['Experiment'] == exp_name].sort_values('Round')
        if exp_data.empty: continue
        
        display_name = labels['scenarios'].get(exp_name, exp_name)
        
        plt.plot(exp_data['Round'], exp_data[col], 
               label=display_name,
               color=colors[i % len(colors)],
               linestyle=styles[i % len(styles)],
               linewidth=2.5 if 'Severe' in exp_name else 2.0,
               marker='o', markevery=2) # Added markers for single plot
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(labels['xlabel'], fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)

    save_plot(os.path.join(output_dir, "async_accuracy_comparison.png"), language)

    # --- 2. Plot Utilization (Stacked Bar) ---
    # Draw strictly for Severe Async scenario if available (most interesting)
    target_exp = next((e for e in experiments if 'Severe' in e), None)
    if not target_exp and len(experiments) > 0:
        target_exp = experiments[-1] # Fallback
        
    if target_exp and 'Fresh' in df.columns and 'Stale' in df.columns:
        exp_data = df[df['Experiment'] == target_exp].sort_values('Round')
        rounds = exp_data['Round']
        fresh = exp_data['Fresh']
        stale = exp_data['Stale']
        
        # Only plot if we have actual data (sum > 0)
        if fresh.sum() > 0 or stale.sum() > 0:
            plt.figure(figsize=(10, 6))
            
            plt.bar(rounds, fresh, label=labels['fresh_label'], color='#2ecc71', alpha=0.8, edgecolor='white', width=0.8)
            plt.bar(rounds, stale, bottom=fresh, label=labels['stale_label'], color='#e74c3c', alpha=0.8, hatch='//', edgecolor='white', width=0.8)
            
            display_name_util = labels['scenarios'].get(target_exp, target_exp)
            plt.title(labels['util_title'].format(display_name_util), fontsize=14, fontweight='bold')
            plt.xlabel(labels['xlabel'], fontsize=12)
            plt.ylabel(labels['util_ylabel'], fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Annotation
            total_recovered = int(stale.sum())
            plt.text(0.02, 0.95, labels['total_recovered'].format(total_recovered), 
                     transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'))
            
            save_plot(os.path.join(output_dir, "async_utilization_analysis.png"), language)
    else:
        print(f"[{language}] Skip Utilization Plot: Missing Fresh/Stale columns or no data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Async Experiment Results")
    parser.add_argument("--csv", type=str, required=True, help="Path to async results CSV")
    parser.add_argument("--output_dir", type=str, default="experiments/results/plots_dual_lang", help="Output root directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"File not found: {args.csv}")
        sys.exit(1)
        
    # English
    en_dir = os.path.join(args.output_dir, "english_paper")
    os.makedirs(en_dir, exist_ok=True)
    plot_async_results(args.csv, en_dir, 'en')
    
    # Chinese
    zh_dir = os.path.join(args.output_dir, "chinese_thesis")
    os.makedirs(zh_dir, exist_ok=True)
    plot_async_results(args.csv, zh_dir, 'zh')
