#!/usr/bin/env python3
"""
公平对比实验 - 比较SDA-FL、FedAvg、FedProx与STELLAR算法
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
import argparse
import seaborn as sns
from datetime import datetime
from pathlib import Path
from experiments.sda_fl_experiment import SDAFLExperiment
from experiments.propagation_fedavg_experiment import LimitedPropagationFedAvg
from experiments.propagation_fedprox_experiment import LimitedPropagationFedProx
from experiments.grouping_experiment import SimilarityGroupingExperiment
from visualization.visualization import Visualization

import pickle
import json
import copy
import argparse
from pathlib import Path

# 设置matplotlib不使用中文
plt.rcParams['font.sans-serif'] = ['Arial']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comparison_with_sda_fl.log")
    ]
)
logger = logging.getLogger('comparison_with_sda_fl')

def run_experiment(config_path, experiment_class):
    """运行单个实验"""
    logger.info(f"使用配置 {config_path} 运行 {experiment_class.__name__} 实验")
    
    try:
        experiment = experiment_class(config_path)
        
        # 准备数据
        experiment.prepare_data()
        
        # 设置客户端
        experiment.setup_clients()
        
        # 执行训练并获取统计信息
        stats = experiment.train()
        
        # 记录一些关键指标
        if 'accuracies' in stats and stats['accuracies']:
            max_acc = max(stats['accuracies'])
            logger.info(f"实验最高准确率: {max_acc:.2f}%")
        
        if 'satellite_stats' in stats and 'training_satellites' in stats['satellite_stats']:
            avg_sats = np.mean(stats['satellite_stats']['training_satellites'])
            logger.info(f"平均参与卫星数: {avg_sats:.2f}")
        
        return stats, experiment
    
    except Exception as e:
        logger.error(f"运行实验 {experiment_class.__name__} 出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def calculate_communication_overhead(stats):
    """计算通信开销"""
    try:
        return np.cumsum(stats['energy_stats']['communication_energy'])
    except Exception as e:
        logger.error(f"计算通信开销时出错: {str(e)}")
        return np.array([0])

def calculate_convergence_speed(accuracies, target_accuracy=None):
    """计算收敛速度 - 达到目标准确率所需轮次"""
    if not accuracies:
        return float('inf')
        
    if target_accuracy is None:
        # 如果未指定目标准确率，使用最终准确率的90%
        target_accuracy = 0.9 * max(accuracies)
    
    # 找到第一个达到或超过目标准确率的轮次
    for round_num, acc in enumerate(accuracies):
        if acc >= target_accuracy:
            return round_num + 1
    
    # 如果没有达到目标，返回总轮次
    return len(accuracies)

def create_comparison_plots(satfl_stats, fedprox_stats, fedavg_stats, similarity_stats, output_dir, 
                           satfl_exp=None, fedprox_exp=None, fedavg_exp=None, similarity_exp=None,
                           custom_style=None, show_grid=True, figure_format='png', dpi=150):
    """
    创建对比图表，支持自定义图表样式
    """
    # 获取实际参与的卫星数量
    satfl_sats = np.mean(satfl_stats['satellite_stats']['training_satellites'])
    fedprox_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 准备图表标题 - 简洁版本
    title_suffix = ""
    
    # 应用自定义样式
    style = {
        'figsize': (10, 6),
        'title_fontsize': 14,
        'label_fontsize': 12,
        'tick_fontsize': 10,
        'legend_fontsize': 10,
        'linewidth': 2,
        'marker_size': 6,
        'grid_alpha': 0.3,
        'grid_linestyle': '--',
        'save_format': figure_format,
        'dpi': dpi
    }
    
    if custom_style:
        style.update(custom_style)
    
    # 设置默认样式
    plt.rcParams.update({
        'font.size': style['label_fontsize'],
        'axes.titlesize': style['title_fontsize'],
        'axes.labelsize': style['label_fontsize'],
        'xtick.labelsize': style['tick_fontsize'],
        'ytick.labelsize': style['tick_fontsize'],
        'legend.fontsize': style['legend_fontsize']
    })
    
    # 算法样式定义
    algo_styles = {
        'SDA-FL': {'color': 'purple', 'marker': '*', 'label': 'SDA-FL'},
        'FedProx': {'color': 'g', 'marker': 'o', 'label': 'FedProx'},
        'FedAvg': {'color': 'b', 'marker': 's', 'label': 'FedAvg'},
        'Similarity': {'color': 'r', 'marker': '^', 'label': 'STELLAR'}
    }
    
    # 1. 准确率对比
    plt.figure(figsize=style['figsize'])
    plt.plot(satfl_stats['accuracies'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_stats['accuracies'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['accuracies'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['accuracies'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Classification Accuracy Performance Analysis {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 1.1 F1 Score 对比 (Macro)
    plt.figure(figsize=style['figsize'])
    if 'f1_macros' in satfl_stats and satfl_stats['f1_macros']:
        plt.plot(satfl_stats['f1_macros'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_macros' in fedprox_stats and fedprox_stats['f1_macros']:
        plt.plot(fedprox_stats['f1_macros'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_macros' in fedavg_stats and fedavg_stats['f1_macros']:
        plt.plot(fedavg_stats['f1_macros'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_macros' in similarity_stats and similarity_stats['f1_macros']:
        plt.plot(similarity_stats['f1_macros'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Macro-averaged F1-Score Performance Evaluation {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('F1-Score (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_macro_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 1.2 精确率对比 (Macro)
    plt.figure(figsize=style['figsize'])
    if 'precision_macros' in satfl_stats and satfl_stats['precision_macros']:
        plt.plot(satfl_stats['precision_macros'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_macros' in fedprox_stats and fedprox_stats['precision_macros']:
        plt.plot(fedprox_stats['precision_macros'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_macros' in fedavg_stats and fedavg_stats['precision_macros']:
        plt.plot(fedavg_stats['precision_macros'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_macros' in similarity_stats and similarity_stats['precision_macros']:
        plt.plot(similarity_stats['precision_macros'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Macro-averaged Precision Performance Analysis {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Precision (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_macro_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 1.3 召回率对比 (Macro)
    plt.figure(figsize=style['figsize'])
    if 'recall_macros' in satfl_stats and satfl_stats['recall_macros']:
        plt.plot(satfl_stats['recall_macros'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_macros' in fedprox_stats and fedprox_stats['recall_macros']:
        plt.plot(fedprox_stats['recall_macros'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_macros' in fedavg_stats and fedavg_stats['recall_macros']:
        plt.plot(fedavg_stats['recall_macros'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_macros' in similarity_stats and similarity_stats['recall_macros']:
        plt.plot(similarity_stats['recall_macros'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Macro-averaged Recall Performance Evaluation {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Recall (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_macro_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 1.4 综合性能指标对比（在一张图中显示多个指标）
    plt.figure(figsize=(14, 8))
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: 准确率
    axes[0, 0].plot(satfl_stats['accuracies'], 
                    color=algo_styles['SDA-FL']['color'], 
                    marker=algo_styles['SDA-FL']['marker'], 
                    label=algo_styles['SDA-FL']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[0, 0].plot(fedprox_stats['accuracies'], 
                    color=algo_styles['FedProx']['color'], 
                    marker=algo_styles['FedProx']['marker'], 
                    label=algo_styles['FedProx']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[0, 0].plot(fedavg_stats['accuracies'], 
                    color=algo_styles['FedAvg']['color'], 
                    marker=algo_styles['FedAvg']['marker'], 
                    label=algo_styles['FedAvg']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[0, 0].plot(similarity_stats['accuracies'], 
                    color=algo_styles['Similarity']['color'], 
                    marker=algo_styles['Similarity']['marker'], 
                    label=algo_styles['Similarity']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[0, 0].set_title('Classification Accuracy')
    axes[0, 0].set_xlabel('Communication Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    if show_grid:
        axes[0, 0].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图2: F1 Score (Macro)
    if all('f1_macros' in stats and stats['f1_macros'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[0, 1].plot(satfl_stats['f1_macros'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(fedprox_stats['f1_macros'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(fedavg_stats['f1_macros'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(similarity_stats['f1_macros'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[0, 1].set_title('Macro F1-Score')
    axes[0, 1].set_xlabel('Communication Round')
    axes[0, 1].set_ylabel('F1-Score (%)')
    if show_grid:
        axes[0, 1].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图3: 精确率 (Macro)
    if all('precision_macros' in stats and stats['precision_macros'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[1, 0].plot(satfl_stats['precision_macros'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(fedprox_stats['precision_macros'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(fedavg_stats['precision_macros'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(similarity_stats['precision_macros'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[1, 0].set_title('Macro Precision')
    axes[1, 0].set_xlabel('Communication Round')
    axes[1, 0].set_ylabel('Precision (%)')
    if show_grid:
        axes[1, 0].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图4: 召回率 (Macro)
    if all('recall_macros' in stats and stats['recall_macros'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[1, 1].plot(satfl_stats['recall_macros'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 1].plot(fedprox_stats['recall_macros'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 1].plot(fedavg_stats['recall_macros'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 1].plot(similarity_stats['recall_macros'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[1, 1].set_title('Macro Recall')
    axes[1, 1].set_xlabel('Communication Round')
    axes[1, 1].set_ylabel('Recall (%)')
    if show_grid:
        axes[1, 1].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 添加总标题和图例
    fig.suptitle(f'Comprehensive Classification Performance Assessment {title_suffix}', fontsize=16, y=0.98)
    
    # 在图的底部添加通用图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)
    plt.savefig(f"{output_dir}/comprehensive_metrics_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 2. 损失函数对比
    plt.figure(figsize=style['figsize'])
    plt.plot(satfl_stats['losses'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_stats['losses'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['losses'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['losses'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Convergence Analysis: Training Loss Comparison {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Training Loss')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 3. 能耗对比 - 训练能耗
    plt.figure(figsize=style['figsize'])
    plt.plot(satfl_stats['energy_stats']['training_energy'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_stats['energy_stats']['training_energy'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['energy_stats']['training_energy'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['energy_stats']['training_energy'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Computational Energy Consumption Analysis {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Training Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_energy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 4. 能耗对比 - 通信能耗
    plt.figure(figsize=style['figsize'])
    plt.plot(satfl_stats['energy_stats']['communication_energy'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_stats['energy_stats']['communication_energy'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['energy_stats']['communication_energy'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['energy_stats']['communication_energy'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Communication Energy Cost Analysis {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Communication Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/communication_energy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 5. 能耗对比 - 总能耗
    plt.figure(figsize=style['figsize'])
    plt.plot(satfl_stats['energy_stats']['total_energy'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_stats['energy_stats']['total_energy'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['energy_stats']['total_energy'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['energy_stats']['total_energy'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Total Energy Efficiency Assessment {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Total Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_energy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 6. 能效比对比(准确率/能耗)
    satfl_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                      zip(satfl_stats['accuracies'], satfl_stats['energy_stats']['total_energy'])]
    fedprox_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                        zip(fedprox_stats['accuracies'], fedprox_stats['energy_stats']['total_energy'])]
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                       zip(fedavg_stats['accuracies'], fedavg_stats['energy_stats']['total_energy'])]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_stats['energy_stats']['total_energy'])]
    
    plt.figure(figsize=style['figsize'])
    plt.plot(satfl_efficiency, 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_efficiency, 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_efficiency, 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_efficiency, 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Energy-Performance Efficiency Analysis {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Efficiency (%/Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/efficiency_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 7. 活跃卫星数量对比
    import matplotlib as mpl
    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=style['figsize'])
    
    # 直接绘制各算法的训练卫星数据，不使用偏移
    plt.plot(satfl_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    
    plt.title(f'Resource Utilization: Active Satellite Count {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Number of Active Satellites')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_satellites_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 8. 通信开销对比
    plt.figure(figsize=style['figsize'])
    satfl_comm = calculate_communication_overhead(satfl_stats)
    fedprox_comm = calculate_communication_overhead(fedprox_stats)
    fedavg_comm = calculate_communication_overhead(fedavg_stats)
    similarity_comm = calculate_communication_overhead(similarity_stats)
    
    plt.plot(satfl_comm, 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_comm, 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_comm, 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_comm, 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Cumulative Communication Cost Assessment {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Cumulative Communication Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/communication_overhead.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 9. 能效比对比 (准确率/累积能耗)
    plt.figure(figsize=style['figsize'])
    
    # 计算能效比 - 每单位能量获得的准确率
    satfl_cumulative_energy = np.cumsum(satfl_stats['energy_stats']['total_energy'])
    fedprox_cumulative_energy = np.cumsum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_cumulative_energy = np.cumsum(fedavg_stats['energy_stats']['total_energy'])
    similarity_cumulative_energy = np.cumsum(similarity_stats['energy_stats']['total_energy'])
    
    satfl_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                      zip(satfl_stats['accuracies'], satfl_cumulative_energy)]
    fedprox_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                       zip(fedprox_stats['accuracies'], fedprox_cumulative_energy)]
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                       zip(fedavg_stats['accuracies'], fedavg_cumulative_energy)]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_cumulative_energy)]
    
    plt.plot(satfl_efficiency, 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_efficiency, 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_efficiency, 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_efficiency, 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Cumulative Energy-Performance Efficiency {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Efficiency (%/Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_efficiency_cumulative.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 10. 加权指标对比
    # 10.1 F1 Score (Weighted) 对比
    plt.figure(figsize=style['figsize'])
    if 'f1_weighteds' in satfl_stats and satfl_stats['f1_weighteds']:
        plt.plot(satfl_stats['f1_weighteds'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_weighteds' in fedprox_stats and fedprox_stats['f1_weighteds']:
        plt.plot(fedprox_stats['f1_weighteds'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_weighteds' in fedavg_stats and fedavg_stats['f1_weighteds']:
        plt.plot(fedavg_stats['f1_weighteds'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_weighteds' in similarity_stats and similarity_stats['f1_weighteds']:
        plt.plot(similarity_stats['f1_weighteds'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Weighted F1-Score Performance Evaluation {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('F1-Score (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_weighted_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 10.2 精确率 (Weighted) 对比
    plt.figure(figsize=style['figsize'])
    if 'precision_weighteds' in satfl_stats and satfl_stats['precision_weighteds']:
        plt.plot(satfl_stats['precision_weighteds'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_weighteds' in fedprox_stats and fedprox_stats['precision_weighteds']:
        plt.plot(fedprox_stats['precision_weighteds'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_weighteds' in fedavg_stats and fedavg_stats['precision_weighteds']:
        plt.plot(fedavg_stats['precision_weighteds'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_weighteds' in similarity_stats and similarity_stats['precision_weighteds']:
        plt.plot(similarity_stats['precision_weighteds'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Weighted Precision Performance Analysis {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Precision (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_weighted_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 10.3 召回率 (Weighted) 对比
    plt.figure(figsize=style['figsize'])
    if 'recall_weighteds' in satfl_stats and satfl_stats['recall_weighteds']:
        plt.plot(satfl_stats['recall_weighteds'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_weighteds' in fedprox_stats and fedprox_stats['recall_weighteds']:
        plt.plot(fedprox_stats['recall_weighteds'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_weighteds' in fedavg_stats and fedavg_stats['recall_weighteds']:
        plt.plot(fedavg_stats['recall_weighteds'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_weighteds' in similarity_stats and similarity_stats['recall_weighteds']:
        plt.plot(similarity_stats['recall_weighteds'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Weighted Recall Performance Evaluation {title_suffix}')
    plt.xlabel('Communication Round')
    plt.ylabel('Recall (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_weighted_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 10.4 加权指标综合对比图
    # 创建2x2的子图显示加权指标
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: F1 Score (Weighted)
    if all('f1_weighteds' in stats and stats['f1_weighteds'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[0, 0].plot(satfl_stats['f1_weighteds'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 0].plot(fedprox_stats['f1_weighteds'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 0].plot(fedavg_stats['f1_weighteds'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 0].plot(similarity_stats['f1_weighteds'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[0, 0].set_title('Weighted F1-Score')
    axes[0, 0].set_xlabel('Communication Round')
    axes[0, 0].set_ylabel('F1-Score (%)')
    if show_grid:
        axes[0, 0].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图2: 精确率 (Weighted)
    if all('precision_weighteds' in stats and stats['precision_weighteds'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[0, 1].plot(satfl_stats['precision_weighteds'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(fedprox_stats['precision_weighteds'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(fedavg_stats['precision_weighteds'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(similarity_stats['precision_weighteds'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[0, 1].set_title('Weighted Precision')
    axes[0, 1].set_xlabel('Communication Round')
    axes[0, 1].set_ylabel('Precision (%)')
    if show_grid:
        axes[0, 1].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图3: 召回率 (Weighted)
    if all('recall_weighteds' in stats and stats['recall_weighteds'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[1, 0].plot(satfl_stats['recall_weighteds'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(fedprox_stats['recall_weighteds'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(fedavg_stats['recall_weighteds'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(similarity_stats['recall_weighteds'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[1, 0].set_title('Weighted Recall')
    axes[1, 0].set_xlabel('Communication Round')
    axes[1, 0].set_ylabel('Recall (%)')
    if show_grid:
        axes[1, 0].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图4: 准确率（作为参考）
    axes[1, 1].plot(satfl_stats['accuracies'], 
                    color=algo_styles['SDA-FL']['color'], 
                    marker=algo_styles['SDA-FL']['marker'], 
                    label=algo_styles['SDA-FL']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[1, 1].plot(fedprox_stats['accuracies'], 
                    color=algo_styles['FedProx']['color'], 
                    marker=algo_styles['FedProx']['marker'], 
                    label=algo_styles['FedProx']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[1, 1].plot(fedavg_stats['accuracies'], 
                    color=algo_styles['FedAvg']['color'], 
                    marker=algo_styles['FedAvg']['marker'], 
                    label=algo_styles['FedAvg']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[1, 1].plot(similarity_stats['accuracies'], 
                    color=algo_styles['Similarity']['color'], 
                    marker=algo_styles['Similarity']['marker'], 
                    label=algo_styles['Similarity']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[1, 1].set_title('Classification Accuracy')
    axes[1, 1].set_xlabel('Communication Round')
    axes[1, 1].set_ylabel('Accuracy (%)')
    if show_grid:
        axes[1, 1].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 添加总标题和图例
    fig.suptitle(f'Weighted Classification Performance Assessment {title_suffix}', fontsize=16, y=0.98)
    
    # 在图的底部添加通用图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)
    plt.savefig(f"{output_dir}/weighted_metrics_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    logger.info(f"图表生成完成，保存在 {output_dir}/ 目录")

def save_experiment_data(output_dir, satfl_stats, fedprox_stats, fedavg_stats, similarity_stats, timestamp):
    """保存实验数据到pickle文件和生成简要报告"""
    
    def prepare_for_serialization(stats_dict):
        # 深复制以避免修改原始数据
        import copy
        serializable_dict = copy.deepcopy(stats_dict)
        
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        return convert_tensors(serializable_dict)
    
    # 安全获取最大值的函数
    def safe_max(lst, default=0.0):
        return max(lst) if lst and len(lst) > 0 else default
    
    def safe_min(lst, default=0.0):
        return min(lst) if lst and len(lst) > 0 else default
    
    # 确保output_dir是Path对象
    from pathlib import Path
    output_dir = Path(output_dir)
    
    # 创建raw_data目录
    raw_data_dir = output_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据进行序列化
    experiment_data = {
        'satfl': prepare_for_serialization(satfl_stats) if satfl_stats else {},
        'fedprox': prepare_for_serialization(fedprox_stats) if fedprox_stats else {},
        'fedavg': prepare_for_serialization(fedavg_stats) if fedavg_stats else {},
        'similarity': prepare_for_serialization(similarity_stats) if similarity_stats else {},
        'timestamp': timestamp
    }
    
    # 保存到pickle文件
    with open(raw_data_dir / "experiment_data.pkl", 'wb') as f:
        pickle.dump(experiment_data, f)
    
    # 生成简要报告
    with open(output_dir / "summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"实验比较总结 - {timestamp}\n")
        f.write("="*50 + "\n\n")
        
        f.write("最高准确率:\n")
        f.write(f"  SDA-FL: {safe_max(satfl_stats['accuracies'] if satfl_stats else []):.2f}%\n")
        f.write(f"  FedProx: {safe_max(fedprox_stats['accuracies'] if fedprox_stats else []):.2f}%\n")
        f.write(f"  FedAvg: {safe_max(fedavg_stats['accuracies'] if fedavg_stats else []):.2f}%\n")
        f.write(f"  STELLAR: {safe_max(similarity_stats['accuracies'] if similarity_stats else []):.2f}%\n")
        f.write("\n")
        
        # 检查是否有新指标数据
        has_new_metrics = any([
            stats and 'f1_macros' in stats for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]
            if stats is not None
        ])
        
        if has_new_metrics:
            f.write("最高F1分数 (Macro):\n")
            f.write(f"  SDA-FL: {safe_max(satfl_stats.get('f1_macros', []) if satfl_stats else []):.2f}%\n")
            f.write(f"  FedProx: {safe_max(fedprox_stats.get('f1_macros', []) if fedprox_stats else []):.2f}%\n")
            f.write(f"  FedAvg: {safe_max(fedavg_stats.get('f1_macros', []) if fedavg_stats else []):.2f}%\n")
            f.write(f"  STELLAR: {safe_max(similarity_stats.get('f1_macros', []) if similarity_stats else []):.2f}%\n")
        else:
            f.write("注意: 本次实验数据不包含详细的分类指标（F1、精确率、召回率）\n")
            f.write("如需这些指标，请重新运行实验。\n")
        
        # 实验状态
        f.write("\n实验状态:\n")
        f.write(f"  SDA-FL: {'成功' if satfl_stats else '失败'}\n")
        f.write(f"  FedProx: {'成功' if fedprox_stats else '失败'}\n")
        f.write(f"  FedAvg: {'成功' if fedavg_stats else '失败'}\n")
        f.write(f"  STELLAR: {'成功' if similarity_stats else '失败'}\n")
    
    print(f"实验数据已保存到: {output_dir}")

def load_experiment_data(data_dir):
    """加载保存的实验数据"""
    # 优先尝试加载pickle格式
    # pickle_path = os.path.join(data_dir, 'raw_data', 'experiment_data.pkl')
    # if os.path.exists(pickle_path):
    #     try:
    #         with open(pickle_path, 'rb') as f:
    #             return pickle.load(f)
    #     except Exception as e:
    #         logger.error(f"无法加载pickle数据: {str(e)}")
    
    # 尝试加载JSON格式
    logger.info(f"尝试加载JSON格式数据: {data_dir}")
    json_path = os.path.join(data_dir, 'raw_data', 'experiment_data.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"无法加载JSON数据: {str(e)}")
    
    raise FileNotFoundError(f"在 {data_dir}/raw_data/ 中找不到实验数据文件")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行公平对比实验或重新绘制已有实验的图表')
    parser.add_argument('--target-sats', type=int, default=0,
                      help='目标卫星数量 (0表示使用STELLAR的平均卫星数)')
    parser.add_argument('--fedprox-mu', type=float, default=0.01,
                      help='FedProx的接近性参数μ')
    parser.add_argument('--config-dir', type=str, default='configs',
                      help='配置文件目录')
    parser.add_argument('--satfl-noise-dim', type=int, default=100,
                      help='SDA-FL的噪声维度')
    parser.add_argument('--satfl-samples', type=int, default=1000,
                      help='SDA-FL生成的合成样本数量')
    
    # 添加重新绘图相关参数
    parser.add_argument('--replot', action='store_true',
                      help='重新绘图模式，不运行实验，只加载已有数据并重新绘制图表')
    parser.add_argument('--data-dir', type=str, default=None,
                      help='数据目录(用于重新绘图模式)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='图表输出目录(默认为data-dir，仅用于重新绘图模式)')
    parser.add_argument('--format', type=str, default='png',
                      choices=['png', 'pdf', 'svg', 'jpg'],
                      help='图表保存格式(仅用于重新绘图模式)')
    parser.add_argument('--dpi', type=int, default=150,
                      help='图表DPI(仅用于重新绘图模式)')
    parser.add_argument('--no-grid', action='store_true',
                      help='不显示网格(仅用于重新绘图模式)')
    
    return parser.parse_args()

def generate_comparison_report(satfl_stats, fedprox_stats, fedavg_stats, similarity_stats, 
                              satfl_config_path, fedprox_config_path, similarity_config_path, output_dir):
    """生成详细的比较报告"""
    
    # 加载配置文件
    def load_config(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"无法加载配置文件 {config_path}: {str(e)}")
            return {}
    
    satfl_config = load_config(satfl_config_path) if isinstance(satfl_config_path, str) else satfl_config_path
    fedprox_config = load_config(fedprox_config_path) if isinstance(fedprox_config_path, str) else fedprox_config_path
    similarity_config = load_config(similarity_config_path) if isinstance(similarity_config_path, str) else similarity_config_path
    
    # 检查是否有新指标数据
    has_new_metrics = any([
        'f1_macros' in stats for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]
        if stats is not None
    ])
    
    report_lines = [
        "# 公平对比报告: SDA-FL vs FedProx vs FedAvg vs STELLAR",
        "",
    ]
    
    if not has_new_metrics:
        report_lines.extend([
            "## ⚠️ 重要说明",
            "",
            "当前实验数据不包含详细的分类指标（F1分数、精确率、召回率）。",
            "这些指标在之前的实验运行中没有被计算和保存。",
            "如需获取完整的分类指标分析，请重新运行实验：",
            "",
            "```bash",
            "python experiments/run_fair_comparison_satfl.py --num_rounds 20",
            "```",
            "",
        ])
    
    # 实验设置部分
    report_lines.extend([
        "## 实验设置",
        f"- SDA-FL 参数 - 合成样本数: {satfl_config.get('sda_fl', {}).get('num_synthetic_samples', 'N/A')}",
        f"- FedProx 参数 μ: {fedprox_config.get('fedprox', {}).get('mu', 'N/A')}",
        f"- 总轮次: {satfl_config.get('fl', {}).get('num_rounds', 'N/A')}",
        "",
    ])
    
    # 参与卫星数量
    report_lines.extend([
        "## 参与卫星数量",
    ])
    
    algorithms = {
        "SDA-FL": satfl_stats,
        "FedProx": fedprox_stats,
        "FedAvg": fedavg_stats,
        "STELLAR": similarity_stats
    }
    
    for name, stats in algorithms.items():
        if stats and 'satellite_stats' in stats and stats['satellite_stats']['training_satellites']:
            avg_satellites = sum(stats['satellite_stats']['training_satellites']) / len(stats['satellite_stats']['training_satellites'])
            report_lines.append(f"- {name} 平均训练卫星数: {avg_satellites:.2f}")
        else:
            report_lines.append(f"- {name} 平均训练卫星数: 无数据")
    
    report_lines.append("")
    
    # 分类性能指标
    report_lines.extend([
        "## 分类性能指标",
        "### 准确率性能",
    ])
    
    for name, stats in algorithms.items():
        if stats and 'accuracies' in stats and stats['accuracies']:
            max_acc = max(stats['accuracies'])
            report_lines.append(f"- {name} 最高准确率: {max_acc:.2f}%")
        else:
            report_lines.append(f"- {name} 最高准确率: 无数据")
    
    if has_new_metrics:
        # 只在有新指标时添加这些部分
        report_lines.extend([
            "",
            "### F1分数性能 (Macro)",
        ])
        
        for name, stats in algorithms.items():
            if stats and 'f1_macros' in stats and stats['f1_macros']:
                max_f1 = max(stats['f1_macros'])
                report_lines.append(f"- {name} 最高F1分数: {max_f1:.2f}%")
            else:
                report_lines.append(f"- {name} 最高F1分数: 无数据")
        
        report_lines.extend([
            "",
            "### 精确率性能 (Macro)",
        ])
        
        for name, stats in algorithms.items():
            if stats and 'precision_macros' in stats and stats['precision_macros']:
                max_precision = max(stats['precision_macros'])
                report_lines.append(f"- {name} 最高精确率: {max_precision:.2f}%")
            else:
                report_lines.append(f"- {name} 最高精确率: 无数据")
        
        report_lines.extend([
            "",
            "### 召回率性能 (Macro)",
        ])
        
        for name, stats in algorithms.items():
            if stats and 'recall_macros' in stats and stats['recall_macros']:
                max_recall = max(stats['recall_macros'])
                report_lines.append(f"- {name} 最高召回率: {max_recall:.2f}%")
            else:
                report_lines.append(f"- {name} 最高召回率: 无数据")
    else:
        # 没有新指标时的说明
        report_lines.extend([
            "",
            "### F1分数、精确率、召回率",
            "由于当前实验数据不包含详细分类指标，这些指标无法显示。",
            "请重新运行实验以获取完整的性能分析。",
        ])
    
    # 性能对比分析
    report_lines.extend([
        "",
        "## 性能对比分析",
    ])
    
    # 准确率对比
    satfl_acc = max(satfl_stats['accuracies']) if satfl_stats['accuracies'] else 0
    fedprox_acc = max(fedprox_stats['accuracies']) if fedprox_stats['accuracies'] else 0
    fedavg_acc = max(fedavg_stats['accuracies']) if fedavg_stats['accuracies'] else 0
    similarity_acc = max(similarity_stats['accuracies']) if similarity_stats['accuracies'] else 0
    
    report_lines.extend([
        f"- SDA-FL vs FedProx (准确率): {satfl_acc - fedprox_acc:+.2f}%",
        f"- SDA-FL vs FedAvg (准确率): {satfl_acc - fedavg_acc:+.2f}%",
        f"- SDA-FL vs STELLAR (准确率): {satfl_acc - similarity_acc:+.2f}%",
        f"- FedProx vs FedAvg (准确率): {fedprox_acc - fedavg_acc:+.2f}%",
        f"- FedProx vs STELLAR (准确率): {fedprox_acc - similarity_acc:+.2f}%",
        f"- STELLAR vs FedAvg (准确率): {similarity_acc - fedavg_acc:+.2f}%",
    ])
    
    if has_new_metrics:
        # F1分数对比
        satfl_f1 = max(satfl_stats.get('f1_macros', [0])) if satfl_stats.get('f1_macros') else 0
        fedprox_f1 = max(fedprox_stats.get('f1_macros', [0])) if fedprox_stats.get('f1_macros') else 0
        fedavg_f1 = max(fedavg_stats.get('f1_macros', [0])) if fedavg_stats.get('f1_macros') else 0
        similarity_f1 = max(similarity_stats.get('f1_macros', [0])) if similarity_stats.get('f1_macros') else 0
        
        report_lines.extend([
            "",
            f"- SDA-FL vs FedProx (F1分数): {satfl_f1 - fedprox_f1:+.2f}%",
            f"- SDA-FL vs FedAvg (F1分数): {satfl_f1 - fedavg_f1:+.2f}%",
            f"- SDA-FL vs STELLAR (F1分数): {satfl_f1 - similarity_f1:+.2f}%",
            f"- FedProx vs FedAvg (F1分数): {fedprox_f1 - fedavg_f1:+.2f}%",
            f"- FedProx vs STELLAR (F1分数): {fedprox_f1 - similarity_f1:+.2f}%",
            f"- STELLAR vs FedAvg (F1分数): {similarity_f1 - fedavg_f1:+.2f}%",
        ])
    
    # 继续添加其他部分（能耗、效率等）
    report_lines.extend([
        "",
        "## 能耗",
    ])
    
    for name, stats in algorithms.items():
        if stats and 'energy_stats' in stats and stats['energy_stats']['total_energy']:
            total_energy = sum(stats['energy_stats']['total_energy'])
            report_lines.append(f"- {name} 总能耗: {total_energy:.2f} Wh")
        else:
            report_lines.append(f"- {name} 总能耗: 无数据")
    
    # 效率指标
    report_lines.extend([
        "",
        "## 效率指标",
        "### 卫星效率（每卫星准确率）",
    ])
    
    for name, stats in algorithms.items():
        if stats and 'satellite_stats' in stats and stats['satellite_stats']['training_satellites']:
            avg_satellites = sum(stats['satellite_stats']['training_satellites']) / len(stats['satellite_stats']['training_satellites'])
            if stats['accuracies']:
                max_acc = max(stats['accuracies'])
                efficiency = max_acc / avg_satellites
                report_lines.append(f"- {name} 每卫星准确率: {efficiency:.2f}%")
        else:
            report_lines.append(f"- {name} 每卫星准确率: 无数据")
    
    if has_new_metrics:
        report_lines.extend([
            "",
            "### 卫星F1效率（每卫星F1分数）",
        ])
        
        for name, stats in algorithms.items():
            if stats and 'satellite_stats' in stats and stats['satellite_stats']['training_satellites']:
                avg_satellites = sum(stats['satellite_stats']['training_satellites']) / len(stats['satellite_stats']['training_satellites'])
                if stats.get('f1_macros'):
                    max_f1 = max(stats['f1_macros'])
                    f1_efficiency = max_f1 / avg_satellites
                    report_lines.append(f"- {name} 每卫星F1分数: {f1_efficiency:.2f}%")
                else:
                    report_lines.append(f"- {name} 每卫星F1分数: 无数据")
            else:
                report_lines.append(f"- {name} 每卫星F1分数: 无数据")
    
    # 能源效率
    report_lines.extend([
        "",
        "### 能源效率",
    ])
    
    for name, stats in algorithms.items():
        if stats and 'energy_stats' in stats and stats['energy_stats']['total_energy'] and stats['accuracies']:
            total_energy = sum(stats['energy_stats']['total_energy'])
            max_acc = max(stats['accuracies'])
            energy_efficiency = max_acc / total_energy
            report_lines.append(f"- {name} 能源效率: {energy_efficiency:.4f}%/Wh")
        else:
            report_lines.append(f"- {name} 能源效率: 无数据")
    
    # 收敛速度
    report_lines.extend([
        "",
        "## 收敛速度",
        "### 准确率收敛速度",
    ])
    
    for name, stats in algorithms.items():
        if stats and 'accuracies' in stats and stats['accuracies']:
            target_acc = max(stats['accuracies']) * 0.9  # 90%的最高准确率
            convergence_round = calculate_convergence_speed(stats['accuracies'], target_acc)
            report_lines.append(f"- {name} 达到90%最高准确率轮次: {convergence_round}")
        else:
            report_lines.append(f"- {name} 达到90%最高准确率轮次: 无数据")
    
    if has_new_metrics:
        report_lines.extend([
            "",
            "### F1分数收敛速度",
        ])
        
        for name, stats in algorithms.items():
            if stats and 'f1_macros' in stats and stats['f1_macros']:
                target_f1 = max(stats['f1_macros']) * 0.9  # 90%的最高F1分数
                convergence_round = calculate_convergence_speed(stats['f1_macros'], target_f1)
                report_lines.append(f"- {name} 达到90%最高F1分数轮次: {convergence_round}")
            else:
                report_lines.append(f"- {name} 达到90%最高F1分数轮次: 无数据")
    
    # 总结
    report_lines.extend([
        "",
        "## 总结",
        "### SDA-FL",
        "- **优势**: 使用合成数据增强训练，在数据稀缺情况下能提高准确率。",
        "- **劣势**: 需要训练GAN模型，增加了计算复杂度和能耗。",
        "",
        "### FedProx",
        "- **优势**: 在非IID数据分布下更稳定的收敛性，对数据异质性更鲁棒。",
        "- **劣势**: 额外的计算开销，需要调整接近性参数μ。",
        "",
        "### FedAvg",
        "- **优势**: 简单，计算开销低。",
        "- **劣势**: 在非IID数据上可能发散，对异质性数据敏感。",
        "",
        "### STELLAR",
        "- **优势**: 高效的资源利用，适应数据分布特点，每卫星性能更好。",
        "- **劣势**: 需要计算数据相似度的开销，实现更复杂。",
        "",
        "### 结论",
    ])
    
    # 找出最佳算法
    best_acc_algo = max(algorithms.keys(), key=lambda name: max(algorithms[name]['accuracies']) if algorithms[name] and algorithms[name].get('accuracies') else 0)
    best_acc = max(algorithms[best_acc_algo]['accuracies']) if algorithms[best_acc_algo] and algorithms[best_acc_algo].get('accuracies') else 0
    
    report_lines.append(f"**准确率**: {best_acc_algo}在准确率上表现最好 ({best_acc:.2f}%)。")
    
    if has_new_metrics:
        best_f1_algo = max(algorithms.keys(), key=lambda name: max(algorithms[name].get('f1_macros', [0])) if algorithms[name] and algorithms[name].get('f1_macros') else 0)
        best_f1 = max(algorithms[best_f1_algo].get('f1_macros', [0])) if algorithms[best_f1_algo] and algorithms[best_f1_algo].get('f1_macros') else 0
        report_lines.append(f"**F1分数**: {best_f1_algo}在F1分数上表现最好 ({best_f1:.2f}%)。")
    
    # 找出最节能的算法
    best_efficiency_algo = None
    best_efficiency = 0
    for name, stats in algorithms.items():
        if stats and 'satellite_stats' in stats and stats['satellite_stats']['training_satellites']:
            avg_satellites = sum(stats['satellite_stats']['training_satellites']) / len(stats['satellite_stats']['training_satellites'])
            if stats['accuracies']:
                max_acc = max(stats['accuracies'])
                efficiency = max_acc / avg_satellites
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_efficiency_algo = name
    
    if best_efficiency_algo:
        report_lines.append(f"**资源效率**: {best_efficiency_algo}在资源效率上表现最好 ({best_efficiency:.2f}%/satellite)。")
    
    if has_new_metrics:
        report_lines.append(f"\n{best_acc_algo}在分类性能（准确率和F1分数）方面表现最佳，而{best_efficiency_algo}在资源效率方面更优。")
    else:
        report_lines.append(f"\n{best_acc_algo}在准确率方面表现最佳，而{best_efficiency_algo}在资源效率方面更优。")
        report_lines.extend([
            "",
            "**注意**: 要获得F1分数、精确率、召回率等详细分类指标的完整分析，",
            "请使用最新版本的代码重新运行实验。",
        ])
    
    # 保存报告
    from pathlib import Path
    output_dir = Path(output_dir)
    report_path = output_dir / "comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"详细比较报告已保存到: {report_path}")

def create_modified_config(base_config_path, target_satellite_count, output_path):
    """
    创建修改后的配置文件，设置目标卫星数量
    """
    try:
        # 读取基础配置
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 修改传播配置
        if 'propagation' not in config:
            config['propagation'] = {}
            
        # 移除强制约束，让各算法能由 hops 决定实际触达的卫星数，不再受限于极小值（如 24）
        config['propagation']['max_satellites'] = 651
        
        # 保留原有的跳数设置，如果未设置则给定一个合理默认值（按需）
        if 'hops' not in config['propagation']:
            # 根据卫星数量调整跳数（仅作为后备方案）
            if target_satellite_count <= 10:
                config['propagation']['hops'] = 1
            elif target_satellite_count <= 20:
                config['propagation']['hops'] = 2
            else:
                config['propagation']['hops'] = 3
        
        # 处理client配置，移除不支持的参数
        if 'client' in config and 'optimizer' in config['client']:
            # FedProx的ClientConfig不支持optimizer参数，所以移除它
            config['client'].pop('optimizer', None)
            logger.info("从client配置中移除了'optimizer'参数")
        
        # 保存修改后的配置
        with open(output_path, 'w') as f:
            yaml.dump(config, f)
            
        logger.info(f"已创建修改后的配置文件：{output_path}")
        logger.info(f"- 目标卫星数：{target_satellite_count}")
        logger.info(f"- 传播跳数：{config['propagation']['hops']}")
        
        return output_path
    except Exception as e:
        logger.error(f"创建修改后的配置文件时出错：{str(e)}")
        return base_config_path

def create_satfl_config(args, target_satellite_count, output_path="configs/sda_fl_config.yaml"):
    """创建SDA-FL配置文件"""
    try:
        # 基于baseline_config.yaml创建SDA-FL配置
        with open("configs/baseline_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # 添加SDA-FL特有配置
        config['sda_fl'] = {
            'noise_dim': args.satfl_noise_dim,
            'num_synthetic_samples': args.satfl_samples,
            'dp_epsilon': 1.0,
            'dp_delta': 1e-5,
            'pseudo_threshold': 0.8,
            'initial_rounds': 3,
            'gan_epochs': 50,
            'gan_samples_per_client': 100,
            'regenerate_interval': 5
        }
        
        # 保存配置
        with open(output_path, 'w') as f:
            yaml.dump(config, f)
            
        logger.info(f"已创建SDA-FL配置文件：{output_path}")
        logger.info(f"- 噪声维度：{args.satfl_noise_dim}")
        logger.info(f"- 合成样本数：{args.satfl_samples}")
        
        return output_path
    except Exception as e:
        logger.error(f"创建SDA-FL配置文件时出错：{str(e)}")
        return None

def run_fair_comparison():
    """运行公平比较实验"""
    args = parse_args()
    
    if args.replot:
        # 重新绘图模式
        if not args.data_dir:
            logger.error("重新绘图模式需要指定 --data-dir 参数")
            exit(1)
            
        try:
            # 设置输出目录
            output_dir = args.output_dir if args.output_dir else args.data_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # 加载实验数据
            logger.info(f"从 {args.data_dir} 加载实验数据")
            data = load_experiment_data(args.data_dir)
            
            # 提取各个算法的统计数据
            satfl_stats = data['satfl']
            fedprox_stats = data['fedprox']
            fedavg_stats = data['fedavg']
            similarity_stats = data['similarity']
            
            # 重新绘制图表
            logger.info(f"开始重新生成图表")
            create_comparison_plots(
                satfl_stats,
                fedprox_stats, 
                fedavg_stats, 
                similarity_stats, 
                output_dir,
                show_grid=not args.no_grid,
                figure_format=args.format,
                dpi=args.dpi
            )
            
            logger.info(f"图表重绘完成，保存在 {output_dir}/")
            
        except Exception as e:
            logger.error(f"重新绘制图表时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            exit(1)
        return
    
    logger.info("=== 开始公平比较实验 ===")
    
    # 1. 运行STELLAR实验
    logger.info("\n=== 运行STELLAR实验 ===")
    similarity_stats, similarity_exp = run_experiment(
        "configs/similarity_grouping_config.yaml", 
        SimilarityGroupingExperiment
    )
    
    if not similarity_stats:
        logger.error("STELLAR实验失败")
        return
    
    # 获取STELLAR使用的卫星数量
    similarity_sats = similarity_stats['satellite_stats']['training_satellites']
    avg_similarity_sats = np.mean(similarity_sats)
    logger.info(f"STELLAR平均使用卫星数: {avg_similarity_sats:.2f}")
    
    # 2. 为FedProx、FedAvg和SDA-FL创建配置文件
    target_sats = int(avg_similarity_sats) if args.target_sats == 0 else args.target_sats
    logger.info(f"为SDA-FL、FedProx和FedAvg设置目标卫星数: {target_sats}")
    
    # 创建配置目录
    os.makedirs("configs/temp", exist_ok=True)
    
    # 为SDA-FL创建配置
    # 直接使用准备好的 sda_fl_config.yaml，它已经是为最新实验（例如 OneWeb）配置好的
    satfl_config = create_modified_config(
        "configs/sda_fl_config.yaml",
        target_sats,
        f"configs/temp/sdafl_{target_sats}sats.yaml"
    )
    
    # 为FedProx和FedAvg创建配置
    fedprox_config = create_modified_config(
        "configs/propagation_fedprox_config.yaml",
        target_sats,
        f"configs/temp/fedprox_{target_sats}sats.yaml"
    )
    
    fedavg_config = create_modified_config(
        "configs/propagation_fedavg_config.yaml",
        target_sats,
        f"configs/temp/fedavg_{target_sats}sats.yaml"
    )
    
    # 3. 运行SDA-FL实验
    logger.info(f"\n=== 运行SDA-FL实验 (目标卫星数: {target_sats}) ===")
    satfl_stats, satfl_exp = run_experiment(
        satfl_config, 
        SDAFLExperiment
    )
    
    if not satfl_stats:
        logger.error("SDA-FL实验失败")
        return
    
    # 4. 运行有限传播FedProx实验
    logger.info(f"\n=== 运行有限传播FedProx实验 (目标卫星数: {target_sats}) ===")
    fedprox_stats, fedprox_exp = run_experiment(
        fedprox_config, 
        LimitedPropagationFedProx
    )
    
    if not fedprox_stats:
        logger.error("有限传播FedProx实验失败")
        return
    
    # 5. 运行有限传播FedAvg实验
    logger.info(f"\n=== 运行有限传播FedAvg实验 (目标卫星数: {target_sats}) ===")
    fedavg_stats, fedavg_exp = run_experiment(
        fedavg_config, 
        LimitedPropagationFedAvg
    )
    
    if not fedavg_stats:
        logger.error("有限传播FedAvg实验失败")
        return
    
    # 6. 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"comparison_results/with_satfl_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存实验数据到文件，以便后续可以重新绘制图表
    save_experiment_data(
        output_dir,
        satfl_stats=satfl_stats,
        fedprox_stats=fedprox_stats,
        fedavg_stats=fedavg_stats,
        similarity_stats=similarity_stats,
        timestamp=timestamp
    )
    
    # 7. 生成对比报告和图表
    create_comparison_plots(
        satfl_stats,
        fedprox_stats, 
        fedavg_stats, 
        similarity_stats, 
        output_dir, 
        satfl_exp=satfl_exp,
        fedprox_exp=fedprox_exp, 
        fedavg_exp=fedavg_exp, 
        similarity_exp=similarity_exp
    )
    
    generate_comparison_report(
        satfl_stats, 
        fedprox_stats, 
        fedavg_stats, 
        similarity_stats, 
        satfl_config, 
        fedprox_config, 
        "configs/similarity_grouping_config.yaml",  # 使用实际的配置文件路径
        output_dir
    )
    
    # 8. 打印关键指标
    print_key_metrics(satfl_stats, fedprox_stats, fedavg_stats, similarity_stats)
    
    logger.info(f"公平比较实验完成，结果保存在 {output_dir}/")
    logger.info(f"实验原始数据已保存，可使用 'python run_fair_comparison_satfl.py --replot --data-dir {output_dir}' 重新绘制图表")
    
    return output_dir

def print_key_metrics(satfl_stats, fedprox_stats, fedavg_stats, similarity_stats):
    """打印关键指标总结"""
    print("\n" + "="*80)
    print("关键指标总结")
    print("="*80)
    
    # 检查是否有新指标数据
    has_new_metrics = any([
        'f1_macros' in stats for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]
    ])
    
    if not has_new_metrics:
        print("\n⚠️  注意：当前实验数据不包含F1分数、精确率、召回率等详细分类指标")
        print("   这些指标在之前的实验运行中没有被计算和保存")
        print("   如需获取完整的分类指标，请重新运行实验")
        print("   运行命令：python experiments/run_fair_comparison_satfl.py --num_rounds 20")
        print()
    
    algorithms = {
        "SDA-FL": satfl_stats,
        "FedProx": fedprox_stats, 
        "FedAvg": fedavg_stats,
        "STELLAR": similarity_stats
    }
    
    print("### 准确率性能")
    best_accuracy = 0
    best_acc_algo = ""
    for name, stats in algorithms.items():
        if stats and 'accuracies' in stats and stats['accuracies']:
            max_acc = max(stats['accuracies'])
            print(f"- {name} 最高准确率: {max_acc:.2f}%")
            if max_acc > best_accuracy:
                best_accuracy = max_acc
                best_acc_algo = name
        else:
            print(f"- {name} 最高准确率: 无数据")
    
    if has_new_metrics:
        # 只有在有新指标数据时才显示这些部分
        print("\n### F1分数性能 (Macro)")
        best_f1 = 0
        best_f1_algo = ""
        for name, stats in algorithms.items():
            if stats and 'f1_macros' in stats and stats['f1_macros']:
                max_f1 = max(stats['f1_macros'])
                print(f"- {name} 最高F1分数: {max_f1:.2f}%")
                if max_f1 > best_f1:
                    best_f1 = max_f1
                    best_f1_algo = name
            else:
                print(f"- {name} 最高F1分数: 无数据")

        print("\n### 精确率性能 (Macro)")
        for name, stats in algorithms.items():
            if stats and 'precision_macros' in stats and stats['precision_macros']:
                max_precision = max(stats['precision_macros'])
                print(f"- {name} 最高精确率: {max_precision:.2f}%")
            else:
                print(f"- {name} 最高精确率: 无数据")

        print("\n### 召回率性能 (Macro)")
        for name, stats in algorithms.items():
            if stats and 'recall_macros' in stats and stats['recall_macros']:
                max_recall = max(stats['recall_macros'])
                print(f"- {name} 最高召回率: {max_recall:.2f}%")
            else:
                print(f"- {name} 最高召回率: 无数据")
    
    # 能耗对比
    print("\n### 能耗对比")
    for name, stats in algorithms.items():
        if stats and 'energy_stats' in stats and stats['energy_stats']['total_energy']:
            total_energy = sum(stats['energy_stats']['total_energy'])
            print(f"- {name} 总能耗: {total_energy:.2f} Wh")
        else:
            print(f"- {name} 总能耗: 无数据")
    
    # 效率分析
    print("\n### 效率分析")
    for name, stats in algorithms.items():
        if stats and 'satellite_stats' in stats and stats['satellite_stats']['training_satellites']:
            avg_satellites = sum(stats['satellite_stats']['training_satellites']) / len(stats['satellite_stats']['training_satellites'])
            if stats['accuracies']:
                max_acc = max(stats['accuracies'])
                efficiency = max_acc / avg_satellites
                print(f"- {name} 每卫星准确率: {efficiency:.2f}%/satellite")
            else:
                print(f"- {name} 每卫星准确率: 无法计算")
        else:
            print(f"- {name} 每卫星准确率: 无数据")
    
    print("\n### 最佳算法")
    print(f"- 准确率最佳: {best_acc_algo} ({best_accuracy:.2f}%)")
    
    if has_new_metrics and best_f1_algo:
        print(f"- F1分数最佳: {best_f1_algo} ({best_f1:.2f}%)")
    
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    
    if args.replot:
        # 重新绘图模式 - 已在函数中实现
        run_fair_comparison()
    else:
        # 正常实验模式
        run_fair_comparison()
