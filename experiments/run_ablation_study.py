import argparse
import logging
import os
import yaml
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from grouping_experiment import SimilarityGroupingExperiment

def setup_logging(log_dir="logs/ablation"):
    """设置日志"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ablation_study_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger("AblationStudy")

def run_experiment(config_path, weights, rounds=None):
    """运行单个实验"""
    # 读取配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新权重
    if 'group' not in config:
        config['group'] = {}
    
    config['group']['weights'] = weights
    
    # 如果指定了轮数，覆盖配置
    if rounds:
        config['fl']['num_rounds'] = rounds
        # 禁用早停以确保运行指定轮数
        if 'early_stopping' in config:
            config['early_stopping']['enabled'] = False
            
    # 保存临时配置文件
    temp_config_path = f"configs/temp_ablation_{int(time.time())}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
        
    try:
        # 运行实验
        experiment = SimilarityGroupingExperiment(temp_config_path)
        stats = experiment.run()
        return stats
    finally:
        # 清理临时文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def run_ablation_study(base_config_path, rounds=20):
    """运行消融实验"""
    logger = setup_logging()
    logger.info("开始消融实验 (Ablation Study)")
    
    # 读取基础配置以获取默认权重
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    default_weights = base_config.get('group', {}).get('weights', {"alpha": 0.4, "beta": 0.3, "gamma": 0.3})
    logger.info(f"使用配置中的默认权重作为Baseline: {default_weights}")

    # 定义实验配置
    experiments = [
        # 基准 (Baseline) - 使用配置文件中的权重
        {"name": "Baseline (All)", "weights": default_weights},
        
        # 单一组件 (Single Component)
        {"name": "Only Parameter", "weights": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}},
        {"name": "Only Loss", "weights": {"alpha": 0.0, "beta": 1.0, "gamma": 0.0}},
        {"name": "Only Prediction", "weights": {"alpha": 0.0, "beta": 0.0, "gamma": 1.0}},
        
        # 移除单一组件 (Leave One Out)
        {"name": "No Parameter", "weights": {"alpha": 0.0, "beta": 0.5, "gamma": 0.5}},
        {"name": "No Loss", "weights": {"alpha": 0.5, "beta": 0.0, "gamma": 0.5}},
        {"name": "No Prediction", "weights": {"alpha": 0.5, "beta": 0.5, "gamma": 0.0}},
    ]
    
    results = []
    
    for exp in experiments:
        logger.info(f"\n=== 运行实验: {exp['name']} ===")
        logger.info(f"权重: {exp['weights']}")
        
        start_time = time.time()
        stats = run_experiment(base_config_path, exp['weights'], rounds)
        duration = time.time() - start_time
        
        # 提取关键指标（最后一轮）
        final_acc = stats['accuracies'][-1] if stats.get('accuracies') else 0
        final_f1 = stats['f1_weighteds'][-1] if stats.get('f1_weighteds') else 0
        
        # 记录结果
        result = {
            "Experiment": exp['name'],
            "Alpha": exp['weights']['alpha'],
            "Beta": exp['weights']['beta'],
            "Gamma": exp['weights']['gamma'],
            "Accuracy": final_acc,
            "F1 Score": final_f1,
            "Duration": duration,
            "Stats": stats  # 保存完整统计信息用于绘图
        }
        results.append(result)
        logger.info(f"实验完成 - 准确率: {final_acc:.2f}%, F1: {final_f1:.4f}")
        
    # 保存结果
    # 1. 保存摘要结果
    df_summary = pd.DataFrame([{k: v for k, v in r.items() if k != 'Stats'} for r in results])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"logs/ablation/ablation_results_{timestamp}.csv"
    df_summary.to_csv(csv_path, index=False)
    logger.info(f"消融实验摘要结果已保存至 {csv_path}")
    
    # 2. 保存详细的每轮结果
    detailed_results = []
    for res in results:
        exp_name = res['Experiment']
        stats = res['Stats']
        if stats.get('accuracies'):
            for round_idx, acc in enumerate(stats['accuracies']):
                detailed_results.append({
                    'Experiment': exp_name,
                    'Round': round_idx + 1,
                    'Accuracy': acc,
                    'F1_Weighted': stats['f1_weighteds'][round_idx] if stats.get('f1_weighteds') and len(stats['f1_weighteds']) > round_idx else 0
                })
    
    if detailed_results:
        df_detailed = pd.DataFrame(detailed_results)
        detailed_csv_path = f"logs/ablation/ablation_detailed_results_{timestamp}.csv"
        df_detailed.to_csv(detailed_csv_path, index=False)
        logger.info(f"消融实验详细每轮结果已保存至 {detailed_csv_path}")
    
    # 生成对比图
    plot_comparison(results, f"logs/ablation/ablation_comparison_{timestamp}.png", "Ablation Study")
    
    return df_summary

def run_sensitivity_study(base_config_path, rounds=20):
    """运行敏感性分析"""
    logger = setup_logging()
    logger.info("开始敏感性分析 (Sensitivity Study)")
    
    # 定义实验配置 - 改变Alpha (Parameter Similarity) 的权重，其他两个均分剩余权重
    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    results = []
    
    for alpha in alpha_values:
        remaining = 1.0 - alpha
        beta = remaining / 2
        gamma = remaining / 2
        
        weights = {"alpha": alpha, "beta": beta, "gamma": gamma}
        name = f"Alpha={alpha:.1f}"
        
        logger.info(f"\n=== 运行实验: {name} ===")
        logger.info(f"权重: {weights}")
        
        start_time = time.time()
        stats = run_experiment(base_config_path, weights, rounds)
        duration = time.time() - start_time
        
        final_acc = stats['accuracies'][-1] if stats.get('accuracies') else 0
        final_f1 = stats['f1_weighteds'][-1] if stats.get('f1_weighteds') else 0
        
        result = {
            "Experiment": name,
            "Alpha": alpha,
            "Beta": beta,
            "Gamma": gamma,
            "Accuracy": final_acc,
            "F1 Score": final_f1,
            "Duration": duration,
            "Stats": stats
        }
        results.append(result)
        logger.info(f"实验完成 - 准确率: {final_acc:.2f}%, F1: {final_f1:.4f}")
        
    # 保存结果
    # 1. 保存摘要结果
    df_summary = pd.DataFrame([{k: v for k, v in r.items() if k != 'Stats'} for r in results])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"logs/ablation/sensitivity_results_{timestamp}.csv"
    df_summary.to_csv(csv_path, index=False)
    logger.info(f"敏感性分析摘要结果已保存至 {csv_path}")
    
    # 2. 保存详细的每轮结果
    detailed_results = []
    for res in results:
        exp_name = res['Experiment']
        stats = res['Stats']
        if stats.get('accuracies'):
            for round_idx, acc in enumerate(stats['accuracies']):
                detailed_results.append({
                    'Experiment': exp_name,
                    'Round': round_idx + 1,
                    'Accuracy': acc,
                    'F1_Weighted': stats['f1_weighteds'][round_idx] if stats.get('f1_weighteds') and len(stats['f1_weighteds']) > round_idx else 0
                })
    
    if detailed_results:
        df_detailed = pd.DataFrame(detailed_results)
        detailed_csv_path = f"logs/ablation/sensitivity_detailed_results_{timestamp}.csv"
        df_detailed.to_csv(detailed_csv_path, index=False)
        logger.info(f"敏感性分析详细每轮结果已保存至 {detailed_csv_path}")
    
    # 生成对比图
    plot_comparison(results, f"logs/ablation/sensitivity_comparison_{timestamp}.png", "Sensitivity Analysis (Varying Alpha)")
    
    return df_summary

def plot_comparison(results, save_path, title):
    """绘制对比图"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # 颜色和标记
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
        markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'h']
        
        for i, res in enumerate(results):
            stats = res['Stats']
            if not stats.get('accuracies'):
                continue
                
            rounds = range(1, len(stats['accuracies']) + 1)
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # 准确率
            plt.plot(rounds, stats['accuracies'], 
                    color=color, marker=marker, markersize=4,
                    label=res['Experiment'])
        
        plt.title(f'{title} - Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"对比图已保存至 {save_path}")
        
    except Exception as e:
        print(f"绘图出错: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行相似度权重消融实验")
    parser.add_argument("--mode", type=str, default="ablation", choices=["ablation", "sensitivity"],
                      help="实验模式: ablation (消融实验) 或 sensitivity (敏感性分析)")
    parser.add_argument("--config", type=str, default="configs/similarity_grouping_config.yaml",
                      help="基础配置文件路径")
    parser.add_argument("--rounds", type=int, default=20,
                      help="每个实验运行的轮数")
    
    args = parser.parse_args()
    
    if args.mode == "ablation":
        run_ablation_study(args.config, args.rounds)
    else:
        run_sensitivity_study(args.config, args.rounds)
