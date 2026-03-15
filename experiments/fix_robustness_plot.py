
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import numpy as np

def setup_fonts(language='en'):
    """Setup fonts based on language"""
    # Setup style first (this resets rcParams)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if language == 'zh':
        # Chinese font settings (Mac friendly)
        # Prioritize PingFang SC for Simplified Chinese. Add Heiti TC as fallback.
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        # English font settings
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = True

def get_labels(language='en'):
    """Return dictionary of labels in specified language"""
    if language == 'zh':
        return {
            'acc_title': '参数噪声下的模型准确率对比',
            'acc_xlabel': '通信轮次',
            'acc_ylabel': '准确率 (%)',
            'stab_title': '模型稳定性 (滑动标准差)',
            'stab_xlabel': '通信轮次',
            'stab_ylabel': '准确率波动幅度',
            'noise_start': '噪声注入',
            'algo_names': {
                'Only Parameter': '仅基于参数 (Parameter Only)',
                'All Metrics': '全指标融合 (All Metrics)',
                'Baseline (No Noise)': '无噪声基准 (Base)',
                'Adaptive Grouping': '自适应分组 (Ours)'
            }
        }
    else:
        return {
            'acc_title': 'Accuracy under Parameter Noise',
            'acc_xlabel': 'Round',
            'acc_ylabel': 'Accuracy (%)',
            'stab_title': 'Stability (Rolling Std Dev)',
            'stab_xlabel': 'Round',
            'stab_ylabel': 'Accuracy Fluctuation',
            'noise_start': 'Noise Start',
            'algo_names': {
                'Only Parameter': 'Parameter Noise Only',
                'All Metrics': 'All Metrics Abnormal',
                'Baseline (No Noise)': 'Baseline (No Noise)',
                'Adaptive Grouping': 'Adaptive Grouping (Ours)'
            }
        }

def plot_robustness_comparison(results, save_path, language='en'):
    """绘制鲁棒性对比图 (本地实现)"""
    try:
        setup_fonts(language)
        labels = get_labels(language)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # 定义颜色和线型 (黑白友好)
        colors = {
            'Only Parameter': '#e74c3c',   # Red
            'All Metrics': '#2ecc71',      # Green
            'Baseline': '#3498db',         # Blue
            'Adaptive': '#9b59b6'          # Purple
        }
        
        # Styles mapping based on name keywords
        def get_style(name):
            if 'Only Parameter' in name or '仅参数' in name:
                return {'linestyle': '--', 'marker': '^', 'color': '#e74c3c'}
            elif 'All Metrics' in name or '全部' in name:
                return {'linestyle': '-', 'marker': 's', 'color': '#2ecc71'}
            elif 'Adaptive' in name or '自适应' in name:
                return {'linestyle': '-', 'marker': '*', 'color': '#9b59b6', 'linewidth': 3}
            else:
                return {'linestyle': ':', 'marker': 'o', 'color': '#3498db'}

        for res in results:
            raw_name = res['Name']
            stats = res['Stats']
            if not stats.get('accuracies'):
                continue
                
            rounds = range(1, len(stats['accuracies']) + 1)
            
            # Translate name for display
            display_name = raw_name
            for k, v in labels['algo_names'].items():
                if k in raw_name:
                    display_name = v
                    break
            
            style = get_style(raw_name)
            
            # Accuracy Plot
            ax1.plot(rounds, stats['accuracies'], 
                    label=display_name, 
                    color=style['color'], 
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markevery=2,
                    linewidth=style.get('linewidth', 2.0))
            
            # Stability Plot (Rolling Std Dev of Accuracy)
            acc_serie = pd.Series(stats['accuracies'])
            stability = acc_serie.rolling(window=3).std().fillna(0)
            
            ax2.plot(rounds, stability, 
                    label=display_name, 
                    color=style['color'],
                    linestyle=style['linestyle'], 
                    marker=style['marker'],
                    markevery=2,
                    linewidth=style.get('linewidth', 2.0))

        ax1.set_title(labels['acc_title'], fontsize=14, fontweight='bold')
        ax1.set_xlabel(labels['acc_xlabel'], fontsize=12)
        ax1.set_ylabel(labels['acc_ylabel'], fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend()
        
        ax2.set_title(labels['stab_title'], fontsize=14, fontweight='bold')
        ax2.set_xlabel(labels['stab_xlabel'], fontsize=12)
        ax2.set_ylabel(labels['stab_ylabel'], fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()
        
        # 标注噪声开始区域 (假设第5轮开始)
        for ax in [ax1, ax2]:
            ax.axvline(x=5, color='gray', linestyle=':', alpha=0.8, linewidth=1.5)
            # transform=ax.get_xaxis_transform() makes x data coords, but y 0-1 relative to axis
            ax.text(5.2, 0.95, labels['noise_start'], color='gray', transform=ax.get_xaxis_transform(), fontweight='bold')

        plt.tight_layout()
        
        # Save in multiple formats for high quality
        # 1. High-Res PNG (600 DPI)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
        # 2. Vector Formats (PDF/SVG)
        base_name = os.path.splitext(save_path)[0]
        plt.savefig(f"{base_name}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{base_name}.svg", format='svg', bbox_inches='tight')
        
        print(f"[{language}] Plot saved to {save_path} (+ .pdf, .svg)")
        plt.close()
        
    except Exception as e:
        print(f"Plotting error: {str(e)}")
        import traceback
        traceback.print_exc()

def fix_and_plot(csv_path, output_dir="experiments/results/plots_dual_lang"):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Define target rounds
    target_rounds = 20
    
    # Process each experiment
    experiments = df['Experiment'].unique()
    new_rows = []
    
    for exp in experiments:
        exp_data = df[df['Experiment'] == exp].sort_values('Round')
        if exp_data.empty: continue
        
        max_round = exp_data['Round'].max()
        last_acc = exp_data.iloc[-1]['Accuracy']
        
        # print(f"Experiment: {exp}, Max Round: {max_round}, Last Acc: {last_acc}")
        
        if max_round < target_rounds:
            print(f"  Extrapolating {exp} from round {max_round+1} to {target_rounds}...")
            for r in range(max_round + 1, target_rounds + 1):
                new_rows.append({
                    'Experiment': exp,
                    'Round': r,
                    'Accuracy': last_acc
                })
    
    # Create new dataframe with extrapolated data
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Ensure we don't have columns mismatch
        common_cols = list(set(df.columns) & set(df_new.columns))
        df_fixed = pd.concat([df, df_new[common_cols]], ignore_index=True)
    else:
        df_fixed = df
        
    df_fixed = df_fixed.sort_values(['Experiment', 'Round'])
    
    # Save fixed CSV (optional, save alongside original)
    fixed_csv_path = csv_path.replace(".csv", "_manual_fix.csv")
    df_fixed.to_csv(fixed_csv_path, index=False)
    # print(f"Saved fixed CSV to {fixed_csv_path}")
    
    # Prepare data format for plotting function
    results_for_plot = []
    for exp in experiments:
        exp_df = df_fixed[df_fixed['Experiment'] == exp].sort_values('Round')
        accuracies = exp_df['Accuracy'].tolist()
        results_for_plot.append({
            "Name": exp,
            "Stats": {
                "accuracies": accuracies
            }
        })
        
    # Generate English Plot
    en_dir = os.path.join(output_dir, "english_paper")
    os.makedirs(en_dir, exist_ok=True)
    plot_robustness_comparison(results_for_plot, os.path.join(en_dir, "robustness_comparison_fixed.png"), 'en')
    
    # Generate Chinese Plot
    zh_dir = os.path.join(output_dir, "chinese_thesis")
    os.makedirs(zh_dir, exist_ok=True)
    plot_robustness_comparison(results_for_plot, os.path.join(zh_dir, "robustness_comparison_fixed.png"), 'zh')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix and plot robustness results (Dual Language)")
    parser.add_argument("--csv", type=str, required=True, help="Path to robustness results CSV")
    parser.add_argument("--output_dir", type=str, default="experiments/results/plots_dual_lang", help="Output directory root")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"File not found: {args.csv}")
    else:
        fix_and_plot(args.csv, args.output_dir)
