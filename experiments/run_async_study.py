
import argparse
import yaml
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from experiments.async_experiment import AsyncGroupingExperiment
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("AsyncStudy")

def run_single_experiment(base_config_path, name, delay_prob, max_delay, staleness_alpha, rounds):
    """运行单个异步实验场景"""
    logger = logging.getLogger("AsyncStudy")
    logger.info(f"\n=== Running Scenario: {name} ===")
    logger.info(f"Params: DelayProb={delay_prob}, MaxDelay={max_delay}, Alpha={staleness_alpha}")
    
    # 创建临时配置 (主要是为了传递给父类, 虽然AsyncExperiment额外参数是通过init传的)
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['fl']['num_rounds'] = rounds
    # 禁用早停
    if 'early_stopping' in config:
        config['early_stopping']['enabled'] = False
        
    temp_config_path = f"configs/temp_async_{int(time.time())}_{name.replace(' ', '_')}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
        
    try:
        exp = AsyncGroupingExperiment(
            temp_config_path, 
            delay_prob=delay_prob, 
            max_delay=max_delay, 
            staleness_alpha=staleness_alpha
        )
        stats = exp.run() # 修正: 调用run()以确保先初始化(setup_clients)
        return stats
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def plot_metrics_comparison(results, save_path):
    """绘制多指标对比图 (Accuracy, F1, Loss)"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['black', 'blue', 'red', 'orange']
    styles = ['-', '--', '-.', ':']
    
    metrics = [
        ('accuracies', 'Accuracy (%)', 'Accuracy'),
        ('f1_weighteds', 'F1 Score (Weighted)', 'F1 Score'),
        ('losses', 'Test Loss', 'Convergence (Loss)')
    ]
    
    for idx, (key, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        for i, res in enumerate(results):
            name = res['Name']
            stats = res['Stats']
            data = stats.get(key, [])
            if not data: continue
            
            rounds = range(1, len(data) + 1)
            ax.plot(rounds, data, 
                     label=name, 
                     color=colors[i % len(colors)],
                     linestyle=styles[i % len(styles)],
                     linewidth=2.5 if 'Baseline' in name else 2.0)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend() # Only legend in first plot to save space

    plt.tight_layout()
    # Save in multiple formats for high quality
    # 1. High-Res PNG (600 DPI)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    # 2. Vector Formats (PDF/SVG)
    base_name = os.path.splitext(save_path)[0]
    plt.savefig(f"{base_name}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{base_name}.svg", format='svg', bbox_inches='tight')
    
    print(f"Metrics plot saved to {save_path} (+ .pdf, .svg)")

def plot_utilization(results, save_path):
    """绘制陈旧信息利用率图 (Stacked Bar/Area) - 仅针对Severe Async场景"""
    # 找到 Severe Async 场景 (通常是最后一个或含有 'Severe' 的)
    target_res = next((r for r in results if 'Severe' in r['Name']), None)
    if not target_res:
        target_res = results[-1] # Fallback to last one
    
    name = target_res['Name']
    stats = target_res['Stats']
    fresh = stats.get('fresh_counts', [])
    stale = stats.get('stale_counts', [])
    
    if not fresh or not stale:
        print("No utilization data found.")
        return

    rounds = range(1, len(fresh) + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # Stacked Bar
    plt.bar(rounds, fresh, label='Fresh Updates', color='#2ecc71', alpha=0.8)
    plt.bar(rounds, stale, bottom=fresh, label='Stale Updates (Recovered)', color='#e74c3c', alpha=0.8, hatch='//')
    
    plt.title(f'Information Utilization in {name}', fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Number of Participating Stations', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add text annotation
    total_recovered = sum(stale)
    plt.text(0.02, 0.95, f'Total Stale Updates Recovered: {total_recovered}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    # Save in multiple formats for high quality
    # 1. High-Res PNG (600 DPI)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    # 2. Vector Formats (PDF/SVG)
    base_name = os.path.splitext(save_path)[0]
    plt.savefig(f"{base_name}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{base_name}.svg", format='svg', bbox_inches='tight')
    
    print(f"Utilization plot saved to {save_path} (+ .pdf, .svg)")

def run_async_study(config_path, rounds=20):
    logger = setup_logging()
    
    scenarios = [
        {
            "name": "Synchronous (Baseline)",
            "delay_prob": 0.0,
            "max_delay": 0,
            "staleness_alpha": 0.0
        },
        {
            "name": "Mild Async (30% Delay)",
            "delay_prob": 0.3,
            "max_delay": 2, # 延迟1-2轮
            "staleness_alpha": 0.5
        },
        {
            "name": "Severe Async (60% Delay)",
            "delay_prob": 0.6,
            "max_delay": 4, # 延迟1-4轮
            "staleness_alpha": 0.5
        }
    ]
    
    results = []
    
    for sc in scenarios:
        stats = run_single_experiment(
            config_path, 
            sc['name'], 
            sc['delay_prob'], 
            sc['max_delay'], 
            sc['staleness_alpha'],
            rounds
        )
        results.append({
            "Name": sc['name'],
            "Stats": stats
        })
        
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV
    csv_data = []
    for res in results:
        name = res['Name']
        accs = res['Stats']['accuracies']
        f1s = res['Stats'].get('f1_weighteds', [0]*len(accs))
        losses = res['Stats'].get('losses', [0]*len(accs))
        fresh = res['Stats'].get('fresh_counts', [0]*len(accs))
        stale = res['Stats'].get('stale_counts', [0]*len(accs))
        
        for r, acc in enumerate(accs):
            csv_data.append({
                "Experiment": name,
                "Round": r + 1,
                "Accuracy": acc,
                "F1": f1s[r] if r < len(f1s) else 0,
                "Loss": losses[r] if r < len(losses) else 0,
                "Fresh": fresh[r] if r < len(fresh) else 0,
                "Stale": stale[r] if r < len(stale) else 0
            })
    
    csv_path = f"logs/async_results_{timestamp}.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    # Plot Metrics
    plot_metrics_path = f"logs/async_metrics_{timestamp}.png"
    plot_metrics_comparison(results, plot_metrics_path)
    
    # Plot Utilization
    plot_util_path = f"logs/async_utilization_{timestamp}.png"
    plot_utilization(results, plot_util_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oneweb_config.yaml")
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    
    run_async_study(args.config, args.rounds)
