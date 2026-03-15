#!/usr/bin/env python3
"""
Dual-Language Plotting Tool for Comparisons
Generates plots for both Chinese Thesis (Big Paper) and English Paper (Small Paper).
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import logging
from pathlib import Path
import matplotlib as mpl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('plot_paper')

def setup_fonts(language='en'):
    """Setup fonts based on language"""
    if language == 'zh':
        # Chinese font settings
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Heiti TC']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        # English font settings
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = True

def get_labels(language='en'):
    """Return dictionary of labels in specified language"""
    if language == 'zh':
        return {
            'accuracy_title': '不同算法分类准确率对比',
            'accuracy_ylabel': '准确率 (%)',
            'f1_title': '宏平均 F1 分数对比',
            'f1_ylabel': 'F1 分数 (%)',
            'loss_title': '训练损失收敛对比',
            'loss_ylabel': '训练损失',
            'train_energy_title': '训练能耗对比',
            'train_energy_ylabel': '训练能耗 (Wh)',
            'comm_energy_title': '通信能耗对比',
            'comm_energy_ylabel': '通信能耗 (Wh)',
            'total_energy_title': '总能耗对比',
            'total_energy_ylabel': '总能耗 (Wh)',
            'active_sats_title': '活跃卫星数量对比',
            'active_sats_ylabel': '活跃卫星数量',
            'efficiency_title': '瞬时能效比 (准确率/本轮能耗)',
            'efficiency_ylabel': '能效比 (%/Wh)',
            'cumulative_efficiency_title': '累积能效比 (准确率/累积能耗)',
            'cumulative_efficiency_ylabel': '能效比 (%/Wh)',
            'round_xlabel': '通信轮次',
            'algorithms': {
                'SDA-FL': 'SDA-FL', # Keep English names usually
                'FedProx': 'FedProx',
                'FedAvg': 'FedAvg',
                'Similarity': 'STELLAR (本文方法)'
            }
        }
    else:
        return {
            'accuracy_title': 'Classification Accuracy Performance Analysis',
            'accuracy_ylabel': 'Accuracy (%)',
            'f1_title': 'Macro-averaged F1-Score Performance Evaluation',
            'f1_ylabel': 'F1-Score (%)',
            'loss_title': 'Convergence Analysis: Training Loss Comparison',
            'loss_ylabel': 'Training Loss',
            'train_energy_title': 'Computational Energy Consumption Analysis',
            'train_energy_ylabel': 'Training Energy (Wh)',
            'comm_energy_title': 'Communication Energy Cost Analysis',
            'comm_energy_ylabel': 'Communication Energy (Wh)',
            'total_energy_title': 'Total Energy Efficiency Assessment',
            'total_energy_ylabel': 'Total Energy (Wh)',
            'active_sats_title': 'Resource Utilization: Active Satellite Count',
            'active_sats_ylabel': 'Number of Active Satellites',
            'efficiency_title': 'Energy-Performance Efficiency Analysis',
            'efficiency_ylabel': 'Efficiency (%/Wh)',
            'cumulative_efficiency_title': 'Cumulative Energy-Performance Efficiency',
            'cumulative_efficiency_ylabel': 'Efficiency (%/Wh)',
            'round_xlabel': 'Communication Round',
            'algorithms': {
                'SDA-FL': 'SDA-FL',
                'FedProx': 'FedProx',
                'FedAvg': 'FedAvg',
                'Similarity': 'STELLAR (Ours)'
            }
        }

def calculate_communication_overhead(stats):
    """Auxiliary calculation for communication overhead"""
    if not stats: return np.array([])
    try:
        return np.cumsum(stats['energy_stats']['communication_energy'])
    except:
        return np.array([])

def plot_single_metric(stats_dict, metric_key, ax, label_map, color_map, marker_map, style):
    """Helper to plot a single line"""
    for algo_name, stats in stats_dict.items():
        if not stats: continue
        
        # Determine data extraction logic
        if metric_key == 'accuracy':
            data = stats.get('accuracies', [])
        elif metric_key == 'f1':
            data = stats.get('f1_macros', [])
        elif metric_key == 'loss':
            data = stats.get('losses', [])
        elif metric_key == 'train_energy':
            data = stats['energy_stats'].get('training_energy', [])
        elif metric_key == 'comm_energy':
            # Special case: cumulative communication energy
            data = calculate_communication_overhead(stats)
        elif metric_key == 'total_energy':
            data = stats['energy_stats'].get('total_energy', [])
        elif metric_key == 'active_sats':
            data = stats['satellite_stats'].get('training_satellites', [])
        elif metric_key == 'efficiency':
            # Instantaneous Efficiency: Accuracy / Round Energy
            accs = stats.get('accuracies', [])
            energies = stats['energy_stats'].get('total_energy', [])
            data = [a / (e + 1e-10) for a, e in zip(accs, energies)]
        elif metric_key == 'cumulative_efficiency':
            # Cumulative Efficiency: Accuracy / Cumulative Energy
            accs = stats.get('accuracies', [])
            energies = np.cumsum(stats['energy_stats'].get('total_energy', []))
            data = [a / (e + 1e-10) for a, e in zip(accs, energies)]
        else:
            data = []

        if len(data) == 0: continue

        ax.plot(data, 
                color=color_map[algo_name],
                marker=marker_map[algo_name],
                label=label_map[algo_name],
                linewidth=style.get('linewidth', 2),
                markersize=style.get('marker_size', 6))

def plot_figures(stats_dict, output_dir, language='en'):
    """Generates all standard comparison plots"""
    
    # Setup style
    setup_fonts(language)
    labels = get_labels(language)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Styles
    colors = {'SDA-FL': 'purple', 'FedProx': 'g', 'FedAvg': 'b', 'Similarity': 'r'}
    markers = {'SDA-FL': '*', 'FedProx': 'o', 'FedAvg': 's', 'Similarity': '^'}
    style_cfg = {'linewidth': 2.5, 'marker_size': 7}
    
    # List of plots to generate
    # (metric_key, title_key, ylabel_key, filename)
    plots_config = [
        ('accuracy', 'accuracy_title', 'accuracy_ylabel', 'accuracy_comparison.png'),
        ('f1', 'f1_title', 'f1_ylabel', 'f1_comparison.png'),
        ('loss', 'loss_title', 'loss_ylabel', 'loss_comparison.png'),
        ('train_energy', 'train_energy_title', 'train_energy_ylabel', 'training_energy_comparison.png'),
        ('comm_energy', 'comm_energy_title', 'comm_energy_ylabel', 'communication_energy_comparison.png'),
        ('total_energy', 'total_energy_title', 'total_energy_ylabel', 'total_energy_comparison.png'),
        ('active_sats', 'active_sats_title', 'active_sats_ylabel', 'active_satellites_comparison.png'),
        ('efficiency', 'efficiency_title', 'efficiency_ylabel', 'efficiency_comparison.png'),
        ('cumulative_efficiency', 'cumulative_efficiency_title', 'cumulative_efficiency_ylabel', 'energy_efficiency_cumulative.png'),
    ]

    for metric, t_key, y_key, fname in plots_config:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Map algorithm names to descriptive labels
        algo_label_map = labels['algorithms']
        
        plot_single_metric(stats_dict, metric, ax, algo_label_map, colors, markers, style_cfg)
        
        plt.title(labels[t_key], fontsize=14, fontweight='bold')
        plt.xlabel(labels['round_xlabel'], fontsize=12)
        plt.ylabel(labels[y_key], fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=10)
        plt.tight_layout()
        # Save in multiple formats for high quality
        # 1. High-Res PNG (600 DPI for line art)
        plt.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches='tight')
        
        # 2. Vector Formats (PDF/SVG) - Resolution independent
        base_name = os.path.splitext(fname)[0]
        plt.savefig(os.path.join(output_dir, f"{base_name}.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{base_name}.svg"), format='svg', bbox_inches='tight')
        
        plt.close()
        logger.info(f"Saved {language} plot: {fname}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Dual-Language Comparison Plots")
    parser.add_argument('--data', type=str, required=True, help="Path to experiment_data.pkl")
    parser.add_argument('--output', type=str, default='experiments/results/plots', help="Base output directory")
    args = parser.parse_args()

    # Load Data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Organize stats into a clean dictionary
    # The structure of experiment_data.pkl is usually {'satfl': stats, 'fedprox': stats, ...}
    stats_dict = {
        'SDA-FL': data.get('satfl'),
        'FedProx': data.get('fedprox'),
        'FedAvg': data.get('fedavg'),
        'Similarity': data.get('similarity')
    }

    # Generate English Plots (Small Paper)
    en_dir = os.path.join(args.output, 'english_paper')
    logger.info("Generating English plots...")
    plot_figures(stats_dict, en_dir, language='en')

    # Generate Chinese Plots (Big Thesis)
    zh_dir = os.path.join(args.output, 'chinese_thesis')
    logger.info("Generating Chinese plots...")
    plot_figures(stats_dict, zh_dir, language='zh')
    
    logger.info("All plots generated successfully!")

if __name__ == "__main__":
    main()
