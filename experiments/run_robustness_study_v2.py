import argparse
import yaml
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from experiments.grouping_experiment import SimilarityGroupingExperiment
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_experiment(base_config_path, weights, rounds, noise_level, name):
    """运行单个实验"""
    logger = logging.getLogger(__name__)
    
    # 创建临时配置文件
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改权重
    if 'group' not in config:
        config['group'] = {}
    config['group']['weights'] = weights
    
    # 添加鲁棒性配置
    config['robustness'] = {
        'parameter_noise_level': noise_level,
        'noise_start_round': 5
    }
    
    # 禁用早停
    if 'early_stopping' in config:
        config['early_stopping']['enabled'] = False
        
    # 设置轮数
    config['fl']['num_rounds'] = rounds
    
    temp_config_path = f"configs/temp_robustness_{name}_{int(time.time())}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
        
    try:
        # 初始化并运行实验
        exp = SimilarityGroupingExperiment(config_path=temp_config_path)
        stats = exp.run()
        return stats
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def plot_robustness_comparison(results, save_path):
    """绘制鲁棒性对比图"""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        colors = {'Only Parameter': '#e74c3c', 'All Metrics': '#2ecc71'}
        styles = {'Only Parameter': '--', 'All Metrics': '-'}
        
        for res in results:
            name = res['Name']
            stats = res['Stats']
            if not stats.get('accuracies'):
                continue
                
            rounds = range(1, len(stats['accuracies']) + 1)
            
            # Accuracy Plot
            ax1.plot(rounds, stats['accuracies'], 
                    label=name, color=colors.get(name, 'blue'), 
                    linestyle=styles.get(name, '-'), linewidth=2.5)
            
            # Stability Plot (Rolling Std Dev of Accuracy)
            # 计算准确率的滑动标准差作为稳定性的度量
            acc_series = pd.Series(stats['accuracies'])
            stability = acc_series.rolling(window=3).std().fillna(0)
            
            ax2.plot(rounds, stability, 
                    label=name, color=colors.get(name, 'blue'),
                    linestyle=styles.get(name, '-'), linewidth=2.5)

        ax1.set_title('Accuracy under Parameter Noise', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Stability (Rolling Std Dev)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy Fluctuation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 标注噪声开始区域
        for ax in [ax1, ax2]:
            ax.axvline(x=5, color='gray', linestyle=':', alpha=0.5)
            # transform=ax.get_xaxis_transform() makes x data coords, but y 0-1 relative to axis
            ax.text(5.2, 0.95, 'Noise Start', color='gray', transform=ax.get_xaxis_transform())

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"鲁棒性对比图已保存至 {save_path}")
    except Exception as e:
        print(f"绘图出错: {str(e)}")

def run_robustness_study_v2(base_config_path, rounds=20, noise_level=0.5):
    logger = setup_logging()
    logger.info(f"开始鲁棒性实验 V2 (Noise Level={noise_level})")
    
    # Load default weights from config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    default_weights = base_config.get('group', {}).get('weights', {"alpha": 0.4, "beta": 0.3, "gamma": 0.3})
    logger.info(f"使用配置中的默认权重作为 All Metrics: {default_weights}")

    experiments = [
        {
            "name": "Only Parameter",
            "weights": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}
        },
        {
            "name": "All Metrics",
            "weights": default_weights
        }
    ]
    
    results = []
    
    for exp_cfg in experiments:
        logger.info(f"\n=== 运行实验: {exp_cfg['name']} ===")
        stats = run_experiment(base_config_path, exp_cfg['weights'], rounds, noise_level, exp_cfg['name'])
        
        results.append({
            "Name": exp_cfg['name'],
            "Stats": stats
        })
        
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_robustness_comparison(results, f"logs/robustness_v2_comparison_{timestamp}.png")
    
    # 保存CSV
    data = []
    for res in results:
        stats = res['Stats']
        if stats.get('accuracies'):
            for r, acc in enumerate(stats['accuracies']):
                data.append({
                    'Experiment': res['Name'],
                    'Round': r + 1,
                    'Accuracy': acc
                })
    pd.DataFrame(data).to_csv(f"logs/robustness_v2_results_{timestamp}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/similarity_grouping_config.yaml")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.5)
    args = parser.parse_args()
    
    run_robustness_study_v2(args.config, args.rounds, args.noise)
