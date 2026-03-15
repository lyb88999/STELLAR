
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def setup_fonts(language='en'):
    """Setup fonts based on language"""
    plt.style.use('seaborn-v0_8-whitegrid')
    if language == 'zh':
        # Prioritize PingFang SC for Simplified Chinese
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = True

def get_labels(language='en'):
    """Return dictionary of labels in specified language"""
    if language == 'zh':
        return {
            'util_title': '重度异步场景下的信息利用率 (60% 延迟)',
            'xlabel': '通信轮次',
            'ylabel_util': '参与聚合的地面站数量',  # Corrected from 'Satellite' to 'Ground Station'
            'fresh_label': '及时更新 (Fresh)',
            'stale_label': '陈旧更新 (Stale/Recovered)',
            'recovered_text': '累计恢复陈旧更新: {}',
        }
    else:
        return {
            'util_title': 'Information Utilization in Severe Async (60% Delay)',
            'xlabel': 'Round',
            'ylabel_util': 'Number of Participating Stations',
            'fresh_label': 'Fresh Updates',
            'stale_label': 'Stale Updates (Recovered)',
            'recovered_text': 'Total Stale Updates Recovered: {}',
        }

def plot_reproduction(output_dir, language='en'):
    setup_fonts(language)
    labels = get_labels(language)
    
    # Data extracted from user's image
    rounds = np.arange(1, 21)
    # [R1, R2, ... R20]
    # Corrected based on user feedback (Image check: Round 13 was 3/5 not 0/8)
    fresh_counts = np.array([2, 2, 3, 1, 1, 3, 1, 2, 3, 1, 1, 0, 3, 5, 3, 1, 4, 3, 1, 2])
    stale_counts = np.array([0, 0, 1, 4, 1, 4, 2, 6, 3, 3, 2, 4, 5, 1, 2, 3, 2, 1, 3, 2])
    
    plt.figure(figsize=(10, 6))
    
    # Stacked Bar
    # Matches image colors: Fresh=Greenish (#50C878 is emerald, using close match), Stale=Reddish
    # User image: Fresh is bright green, Stale is salmon red with hatching
    color_fresh = '#4cd186' # Similar to image green
    color_stale = '#ef7a6d' # Similar to image red
    
    plt.bar(rounds, fresh_counts, label=labels['fresh_label'], color=color_fresh, alpha=1.0, edgecolor='none')
    plt.bar(rounds, stale_counts, bottom=fresh_counts, label=labels['stale_label'], color=color_stale, alpha=1.0, hatch='//', edgecolor='black', linewidth=0.5)
    
    plt.title(labels['util_title'], fontsize=14, fontweight='bold')
    plt.xlabel(labels['xlabel'], fontsize=12)
    plt.ylabel(labels['ylabel_util'], fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3, linestyle='-')
    plt.xticks(np.arange(0, 22, 2.5)) # Try to match ticks? Better use integer ticks for Rounds
    # Revert to integer ticks for clarity
    plt.xticks(np.arange(0, 21, 2.5)) 
    
    # Add text annotation
    total_recovered = int(stale_counts.sum())
    plt.text(0.02, 0.95, labels['recovered_text'].format(total_recovered), 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='square,pad=0.5'))
    
    plt.tight_layout()
    
    filename = "reproduced_async_utilization.png"
    out_path = os.path.join(output_dir, "chinese_thesis" if language == 'zh' else "english_paper", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Save in multiple formats for high quality
    # 1. High-Res PNG (600 DPI)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    
    # 2. Vector Formats (PDF/SVG)
    base_name = os.path.splitext(out_path)[0]
    plt.savefig(f"{base_name}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{base_name}.svg", format='svg', bbox_inches='tight')
    
    print(f"[{language}] Plot saved to {out_path} (+ .pdf, .svg)")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="experiments/results/plots_dual_lang", help="Output directory")
    args = parser.parse_args()
    
    plot_reproduction(args.output_dir, 'en')
    plot_reproduction(args.output_dir, 'zh')
