#!/usr/bin/env python3
"""
Theta (Similarity Threshold) Sweep Experiment
Analysis of trade-off between similarity threshold, model performance, and resource utilization.
"""

import os
import sys
import yaml
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from copy import deepcopy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.grouping_experiment import SimilarityGroupingExperiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("theta_sweep.log")
    ]
)
logger = logging.getLogger('theta_sweep')

def run_sweep():
    # Base config
    base_config_path = "configs/cicids2017_config.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Sweep parameters
    thetas = [0.6, 0.7, 0.8, 0.9, 0.95]
    results = {
        'theta': thetas,
        'accuracy': [],
        'energy': [],
        'representatives': []
    }
    
    output_dir = Path("experiments/results/theta_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for theta in thetas:
        logger.info(f"\n=== Running for Theta = {theta} ===")
        
        # Create temp config
        current_config = deepcopy(base_config)
        current_config['group']['similarity_threshold'] = theta
        
        # Ensure correct output path to avoid overwriting
        temp_config_path = output_dir / f"config_theta_{theta}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(current_config, f)
            
        try:
            # Initialize experiment
            exp = SimilarityGroupingExperiment(str(temp_config_path))
            
            # Run training
            exp.prepare_data()
            exp.setup_clients()
            stats = exp.train()
            
            # Extract metrics
            # 1. Max Accuracy
            max_acc = max(stats['accuracies']) if stats['accuracies'] else 0.0
            
            # 2. Total Energy (Training + Communication)
            total_energy = stats['energy_stats']['total_energy'][-1] if stats['energy_stats']['total_energy'] else 0.0
            
            # 3. Average Representatives (Active Satellites)
            # STELLAR's "training_satellites" tracks active reps per round
            avg_reps = np.mean(stats['satellite_stats']['training_satellites']) if stats['satellite_stats']['training_satellites'] else 0.0
            
            results['accuracy'].append(max_acc)
            results['energy'].append(total_energy)
            results['representatives'].append(avg_reps)
            
            logger.info(f"Theta {theta} Results: Acc={max_acc:.2f}%, Energy={total_energy:.2f}Wh, Avg Reps={avg_reps:.2f}")
            
        except Exception as e:
            logger.error(f"Failed run for theta {theta}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Append 0 for failed runs to keep list length consistent
            results['accuracy'].append(0)
            results['energy'].append(0)
            results['representatives'].append(0)
            
    # Save raw results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "sweep_results.csv", index=False)
    logger.info(f"Saved results to {output_dir}/sweep_results.csv")
    
    # Generate Plots
    generate_plots(results, output_dir)

def generate_plots(results, output_dir):
    plt.rcParams.update({'font.size': 12})
    
    thetas = results['theta']
    accs = results['accuracy']
    energies = results['energy']
    reps = results['representatives']
    
    # Figure 1: Accuracy and Energy vs Theta (Dual Y-axis)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Similarity Threshold (θ)')
    ax1.set_ylabel('Accuracy (%)', color=color)
    line1 = ax1.plot(thetas, accs, color=color, marker='o', linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Total Energy Consumption (Wh)', color=color)
    line2 = ax2.plot(thetas, energies, color=color, marker='s', linewidth=2, linestyle='--', label='Energy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('Trade-off: Model Performance vs Energy Consumption')
    plt.tight_layout()
    plt.savefig(output_dir / "tradeoff_accuracy_energy.png", dpi=300)
    plt.close()
    
    # Figure 2: Representatives vs Theta
    plt.figure(figsize=(8, 6))
    plt.plot(thetas, reps, color='tab:green', marker='^', linewidth=2)
    plt.xlabel('Similarity Threshold (θ)')
    plt.ylabel('Avg. Number of Representatives (M)')
    plt.title('Resource Utilization Analysis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "theta_vs_representatives.png", dpi=300)
    plt.close()
    
    logger.info("Plots generated successfully.")

if __name__ == "__main__":
    run_sweep()
