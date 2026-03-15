#!/usr/bin/env python3
"""
OneWeb Scale Experiment - Fair Comparison of SDA-FL, FedAvg, FedProx, and STELLAR
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.run_fair_comparison_satfl import (
    run_experiment,
    create_comparison_plots,
    save_experiment_data,
    generate_comparison_report,
    SDAFLExperiment,
    LimitedPropagationFedAvg,
    LimitedPropagationFedProx,
    SimilarityGroupingExperiment
)
from experiments.baseline_experiment import BaselineExperiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("oneweb_comparison.log")
    ]
)
logger = logging.getLogger('oneweb_comparison')

def main():
    parser = argparse.ArgumentParser(description='Run OneWeb Scale Fair Comparison Experiment')
    parser.add_argument('--rounds', type=int, default=20, help='Number of communication rounds')
    parser.add_argument('--config', type=str, default='configs/oneweb_config.yaml', help='Path to OneWeb config file')
    parser.add_argument('--output-dir', type=str, default='experiments/results/oneweb_comparison', help='Output directory')
    parser.add_argument('--experiment', type=str, default='all', 
                      choices=['all', 'satfl', 'fedprox', 'fedavg', 'similarity'],
                      help='Specific experiment to run (default: all)')
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing data to resume
    raw_data_path = output_dir / "raw_data" / "experiment_data.pkl"
    existing_data = {}
    if raw_data_path.exists():
        logger.info(f"Found existing data at {raw_data_path}, attempting to resume...")
        try:
            import pickle
            with open(raw_data_path, 'rb') as f:
                existing_data = pickle.load(f)
            logger.info("Successfully loaded existing data.")
        except Exception as e:
            logger.warning(f"Failed to load existing data: {e}")

    logger.info(f"Starting OneWeb Scale Experiment with {args.rounds} rounds")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Selected experiment(s): {args.experiment}")

    # Initialize stats with existing data or None
    satfl_stats = existing_data.get('satfl')
    fedprox_stats = existing_data.get('fedprox')
    fedavg_stats = existing_data.get('fedavg')
    similarity_stats = existing_data.get('similarity')
    
    # Keep track of experiment objects (cannot be loaded from pickle easily, so might be None if skipped)
    satfl_exp = None
    fedprox_exp = None
    fedavg_exp = None
    similarity_exp = None

    try:
        # 1. Run SDA-FL
        if args.experiment in ['all', 'satfl']:
            if not satfl_stats:
                logger.info("=== Running SDA-FL ===")
                satfl_stats, satfl_exp = run_experiment(args.config, SDAFLExperiment)
            else:
                logger.info("=== Skipping SDA-FL (Already completed) ===")
        
        # 2. Run FedProx
        if args.experiment in ['all', 'fedprox']:
            if not fedprox_stats:
                logger.info("=== Running FedProx ===")
                fedprox_stats, fedprox_exp = run_experiment(args.config, LimitedPropagationFedProx)
            else:
                logger.info("=== Skipping FedProx (Already completed) ===")

        # 3. Run FedAvg
        if args.experiment in ['all', 'fedavg']:
            if not fedavg_stats:
                logger.info("=== Running FedAvg ===")
                fedavg_stats, fedavg_exp = run_experiment(args.config, LimitedPropagationFedAvg)
            else:
                logger.info("=== Skipping FedAvg (Already completed) ===")

        # 4. Run STELLAR (Similarity Grouping)
        if args.experiment in ['all', 'similarity']:
            if not similarity_stats:
                logger.info("=== Running STELLAR ===")
                similarity_stats, similarity_exp = run_experiment(args.config, SimilarityGroupingExperiment)
            else:
                logger.info("=== Skipping STELLAR (Already completed) ===")

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Attempting to save partial results...")
    
    finally:
        # Generate timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save data (even if partial)
        logger.info("Saving experiment data...")
        save_experiment_data(output_dir, satfl_stats, fedprox_stats, fedavg_stats, similarity_stats, timestamp)

        # Create plots (only if we have some data)
        if any([satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
            logger.info("Generating plots...")
            try:
                create_comparison_plots(
                    satfl_stats, fedprox_stats, fedavg_stats, similarity_stats,
                    output_dir,
                    satfl_exp, fedprox_exp, fedavg_exp, similarity_exp,
                    custom_style={'title_suffix': '(OneWeb Scale)'}
                )
            except Exception as e:
                logger.error(f"Failed to create plots: {e}")

        # Generate report
        try:
            generate_comparison_report(
                satfl_stats, fedprox_stats, fedavg_stats, similarity_stats,
                args.config, args.config, args.config, # Use same config for all as base
                output_dir
            )
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

    logger.info("OneWeb Experiment Process Completed!")

if __name__ == "__main__":
    main()
