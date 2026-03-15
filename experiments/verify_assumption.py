#!/usr/bin/env python3
"""
STELLAR Assumption Verification Script
Empirically validates Assumption 3: Gradient Similarity ∝ Metric Similarity
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import logging
from tqdm import tqdm
from itertools import combinations
import threading
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.grouping_experiment import SimilarityGroupingExperiment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VerifyAssumption')

class AssumptionVerifier(SimilarityGroupingExperiment):
    def __init__(self, config_path, target_round=10):
        super().__init__(config_path)
        self.target_round = target_round
        
    def flatten_model(self, model_state):
        """Flatten model parameters into a single vector"""
        vec = []
        for key in model_state:
            if 'weight' in key or 'bias' in key:
                vec.append(model_state[key].view(-1).cpu().numpy())
        return np.concatenate(vec)

    def run_verification(self):
        """Run standard training until target round, then verify assumption"""
        logger.info(f"Running FL training until Round {self.target_round}...")
        
        # 1. Train until target round
        # We manually loop to control the stop point
        for self.current_round in range(self.target_round):
            logger.info(f"--- Round {self.current_round + 1}/{self.target_round} ---")
            # Standard training step (simplified)
            self._init_orbit_structures(0) # Init logic
            # Just do one round of normal training to advance model state
            # For verification, we actually just need the model to be somewhat converged/trained
            # so random initialization isn't the only thing we measure.
            # Here we skip actual full training loop for speed if verifying on pre-trained,
            # but usually we want "realistic" middle-stage model.
            # To save time, let's assume valid initialization is enough or run simplified:
            super().run() # This runs FULL experiment. 
            # Wait, we can't easily break super().run().
            # Strategy: Overwrite 'train' or just run partial logic.
            # Better Strategy: Just use the initial model if round=0, or implement a partial stepper.
            
            # Since inheriting/modifying run() is complex, let's just use the current model state
            # assuming we loaded a checkpoint OR just run from scratch for 'target_round' iterations.
            break # For now, let's verify on Round 1 (Initial + 1) or just Initial. 
            # User asked for "Round 10". Doing true Round 10 requires running 10 rounds.
            # Let's Implement a "run_n_rounds" helper if needed, but for now let's modify logic:
    
    def prepare_data_at_round(self, rounds_to_run=5):
        """Run simulation for N rounds (or load checkpoint) to get a realistic mid-training model."""
        # Absolute path to ensure consistency
        ckpt_dir = os.path.abspath("checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"verification_model_r{rounds_to_run}.pth")
        
        logger.info(f"Checking for checkpoint at: {ckpt_path}")
        
        if os.path.exists(ckpt_path):
            logger.info(f"Subsequent Run: Found checkpoint. Loading...")
            try:
                # IMPORTANT: Must initialize environment even if loading model
                self.prepare_data()
                self.setup_clients()
                
                self.model.load_state_dict(torch.load(ckpt_path))
                logger.info("Checkpoint loaded successfully. Skipping pre-training.")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Re-running training...")
                os.remove(ckpt_path)
                self._run_training_and_save(rounds_to_run, ckpt_path)
        else:
            logger.info(f"First Run: Checkpoint not found. Training for {rounds_to_run} rounds...")
            self._run_training_and_save(rounds_to_run, ckpt_path)

    def _run_training_and_save(self, rounds, path):
        self.config['fl']['num_rounds'] = rounds
        super().run() 
        torch.save(self.model.state_dict(), path)
        logger.info(f"Pre-training complete. Model saved to {path}.")

    def collect_gradients_and_stats(self):
        """For ALL satellites, compute Gradients and Similarity Metrics"""
        logger.info("Collecting gradients from ALL satellites (Simulating Full Participation)...")
        
        satellite_data = {}
        
        # 1. Get list of all satellites
        all_sats = list(self.clients.keys())
        # subset for speed if needed, but user said "all 646" is feasible
        # Let's limit to 100-200 if 646 is too slow, but user asked for "all".
        # 646 * Forward/Backward might take a few minutes.
        
        global_model_vec = self.flatten_model(self.model.state_dict())
        
        pbar = tqdm(all_sats, desc="Computing Gradients")
        
        for sat_id in pbar:
            client = self.clients[sat_id]
            
            # A. Compute Gradient (Mock update step)
            # We want \nabla F_i(w). 
            # In FL, w_new = w - lr * grad => grad = (w - w_new) / lr
            # So we perform one local epoch and diff the weights.
            w_old = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            # Force client to use global model
            client.model.load_state_dict(w_old)
            
            # Train 20 epochs to get enough loss history for Trend Correlation
            # Short epochs (5) produce noisy loss curves dominated by SGD variance.
            # Using 20 epochs allows the loss to stabilize and reveal real convergence patterns.
            original_epochs = client.config.local_epochs
            client.config.local_epochs = 20
            
            # Run training
            train_stats = client.train(round_number=0)
            
            # Restore config
            client.config.local_epochs = original_epochs
            
            # Extract loss history
            loss_hist = train_stats['summary'].get('train_loss', [])
            # Fallback if empty
            if not loss_hist:
                loss_hist = [0.0, 0.0] 
            
            w_new = client.model.state_dict()
            
            # Calculate Pseudo-Gradient Vector
            grads = []
            for k in w_new:
                if 'weight' in k or 'bias' in k:
                    # g \approx w_old - w_new (ignoring LR scale as it cancels out in correlation)
                    g = w_old[k] - w_new[k]
                    grads.append(g.view(-1).cpu().numpy())
            
            grad_vec = np.concatenate(grads)
            
            # B. Compute Similarity Features (using existing Experiment logic helpers)
            # Parameters: w_new (already got)
            # Loss: need Reference Loss
            # Prediction: need Reference Output
            
            # We reuse the logic from `grouping_experiment.py`
            # But we need "Reference" to compare AGAINST.
            # Actually, we need pairwise. So we just store the features.
            
            # 1. Parameter Feature: w_new vector
            param_vec = self.flatten_model(w_new)
            
            # 2. Loss Feature: Use History for trend correlation
            # We already captured loss_hist above
            
            # 3. Prediction Feature: Output on reference batch
            # We need a shared reference batch for all clients to compare logits
            if not hasattr(self, 'ref_input'):
                # Better strategy: Infer likely input shape from the client
                try:
                    # Strategy 1: Grab from dataset
                    if client.dataset and len(client.dataset) > 0:
                        sample, _ = client.dataset[0]
                        # Create a small batch (size 2) to avoid BatchNorm issues
                        self.ref_input = sample.unsqueeze(0).repeat(2, 1).to(client.device)
                        logger.info(f"Initialized ref_input from dataset: {self.ref_input.shape}")
                    # Strategy 2: Infer from Model Layer
                    elif hasattr(client.model, 'fc1') and isinstance(client.model.fc1, torch.nn.Linear):
                        in_dim = client.model.fc1.in_features
                        self.ref_input = torch.randn(2, in_dim).to(client.device)
                        logger.info(f"Initialized ref_input from Linear layer: {self.ref_input.shape}")
                    elif hasattr(client.model, 'conv1'):
                        self.ref_input = torch.randn(2, 1, 28, 28).to(client.device)
                        logger.info(f"Initialized ref_input for CNN: {self.ref_input.shape}")
                    else:
                        # Fallback
                        self.ref_input = torch.randn(2, 3, 32, 32).to(client.device)
                        logger.warning(f"Fallback ref_input: {self.ref_input.shape}")
                except Exception as e:
                    logger.error(f"Failed to init ref_input: {e}")
                    # Last resort
                    self.ref_input = torch.randn(2, 10).to(client.device)

            # Switch to eval mode to avoid BatchNorm error with small batch
            client.model.eval()
            with torch.no_grad():
                logits = client.model(self.ref_input)
                # Take the first sample from the batch
                pred_vec = torch.softmax(logits, dim=1)[0].cpu().numpy().flatten()
            
            # Revert to training mode if needed (though we reload weights anyway)
            # client.model.train()

            satellite_data[sat_id] = {
                'grad': grad_vec,
                'param': param_vec,
                'loss_hist': loss_hist,
                'pred': pred_vec
            }
            
            satellite_data[sat_id] = {
                'grad': grad_vec,
                'param': param_vec,
                'loss_hist': loss_hist,
                'pred': pred_vec
            }
        
        # DEBUG: Print first 5 loss histories to verify data quality
        debug_ids = list(satellite_data.keys())[:5]
        for did in debug_ids:
            logger.info(f"Sample Loss Hist ({did}): {satellite_data[did]['loss_hist']}")
            
        return satellite_data

    def compute_pairwise_correlations(self, data, alpha=0.2, beta=0.4, gamma=0.4):
        """Compute pairwise Sim(i,j) vs Dist(Grad_i, Grad_j)"""
        ids = list(data.keys())
        n = len(ids)
        pairs = list(combinations(ids, 2))
        
        # Subsample pairs if too many (200k is fine, but lets be safe)
        if len(pairs) > 50000:
            logger.info(f"Subsampling pairs from {len(pairs)} to 50000 for plotting clarity...")
            import random
            random.shuffle(pairs)
            pairs = pairs[:50000]
            
        logger.info(f"Computing metrics for {len(pairs)} pairs with weights: α={alpha}, β={beta}, γ={gamma}...")
        
        results = {
            'sim_total': [],
            'dist_grad': [],
            'sim_param': [],
            'sim_loss': [],
            'sim_pred': []
        }
        
        for id1, id2 in tqdm(pairs, desc="Pairwise Calc"):
            d1, d2 = data[id1], data[id2]
            
            # 1. Gradient Distance (Ground Truth)
            # Euclidean distance between gradient vectors
            g_dist = np.linalg.norm(d1['grad'] - d2['grad'])
            
            # 2. Similarity Metrics
            # A. Parameter Cosine Similarity
            p1, p2 = d1['param'], d2['param']
            cos_sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-10)
            sim_param = (cos_sim + 1) / 2
            
            # B. Loss Similarity (Simple Inverse Distance - Robust)
            # Reverted from vector-based Cosine to scalar comparison.
            # Rationale: SGD noise makes epoch-to-epoch trends unreliable.
            # Comparing mean loss levels is stable and effective.
            l1, l2 = d1['loss_hist'], d2['loss_hist']
            mean_loss1 = np.mean(l1) if len(l1) > 0 else 0.0
            mean_loss2 = np.mean(l2) if len(l2) > 0 else 0.0
            
            # Scale factor 50 to make differences meaningful in [0,1] range
            diff = abs(mean_loss1 - mean_loss2)
            sim_loss = 1.0 / (1.0 + diff * 50.0)
            
            # C. Prediction Similarity (Hellinger)
            # Match Paper: 1 - Mean Hellinger per sample (Here defined on one REF sample for speed in verification loop)
            # But the logic is consistent: 1 - Hellinger
            # Hellinger(p, q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2
            pr1, pr2 = d1['pred'], d2['pred']
            sqrt_p = np.sqrt(pr1)
            sqrt_q = np.sqrt(pr2)
            # This is vector-wise for the flattened batch output
            # Actually, per sample is better, but here we treat the whole ref batch output as one distribution-ish vector
            # STRICTLY SPEAKING, we should average Hellinger per sample.
            # However, since pr1 is FLATTENED logits/probs of Reference Batch, we must be careful.
            # In collect step: 'pred_vec = torch.softmax(logits, dim=1)[0]' -> ONE SAMPLE only.
            # So pr1 is a single probability vector for one sample.
            # Then Hellinger is simply:
            h_dist = (1.0 / np.sqrt(2.0)) * np.linalg.norm(sqrt_p - sqrt_q)
            sim_pred = max(0.0, min(1.0, 1.0 - h_dist))
            
            # Total Similarity
            sim_total = alpha * sim_param + beta * sim_loss + gamma * sim_pred
            
            results['sim_total'].append(sim_total)
            results['dist_grad'].append(g_dist)
            results['sim_param'].append(sim_param)
            results['sim_loss'].append(sim_loss)
            results['sim_pred'].append(sim_pred)
            
        # Diagnostic Logging
        for key in ['sim_total', 'sim_param', 'sim_loss', 'sim_pred']:
            vals = np.array(results[key])
            logger.info(f"Metric {key}: Mean={vals.mean():.4f}, Std={vals.std():.4f}, Min={vals.min():.4f}, Max={vals.max():.4f}")
            
        return results

    def plot_verification(self, results, output_dir):
        """Generate the publication-ready scatter plot"""
        # Fonts
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial', 'DejaVu Sans']
        
        x = np.array(results['sim_total'])
        y = np.array(results['dist_grad'])
        
        # Transform x to Dissimilarity for the theoretical claim:
        # ||g_i - g_j|| <= phi(1 - Sim)
        x_dissim = 1 - x
        
        plt.figure(figsize=(8, 6))
        
        # Scatter Plot
        # Use density-based coloring if points are dense
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([x_dissim, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x_dissim, y, z = x_dissim[idx], y[idx], z[idx]
            plt.scatter(x_dissim, y, c=z, s=10, cmap='viridis', alpha=0.6)
        except:
            plt.scatter(x_dissim, y, color='#2980b9', alpha=0.3, s=5)
            
        # Linear Fit: y = kx + b
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_dissim, y)
        line_x = np.array([min(x_dissim), max(x_dissim)])
        line_y = slope * line_x + intercept
        
        plt.plot(line_x, line_y, color='#c0392b', linewidth=3, linestyle='--')
        
        # Annotate Statistics
        # Use raw string for latex to avoid escape issues
        stats_text = (
            f"Pearson r: {r_value:.4f}\n"
            f"Slope $\kappa$: {slope:.2f}\n"
            f"Checking $\phi(\delta) \\approx {slope:.1f}\delta$"
        )
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 fontsize=12, fontweight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.title('Empirical Verification of Assumption 3\n(Gradient Diff vs. Metric Dissimilarity)', fontsize=14, fontweight='bold')
        plt.xlabel(r'Metric Dissimilarity $\delta = 1 - \text{Sim}(i,j)$', fontsize=12)
        plt.ylabel(r'Gradient Difference $\|\nabla F_i - \nabla F_j\|_2$', fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save high quality
        fname = os.path.join(output_dir, "assumption_verification.png")
        plt.savefig(fname, dpi=600, bbox_inches='tight')
        
        base = os.path.splitext(fname)[0]
        plt.savefig(f"{base}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{base}.svg", format='svg', bbox_inches='tight')
        
        logger.info(f"Verification Plot Saved: {fname}")
        logger.info(f"Correlation: {r_value}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/oneweb_config.yaml')
    parser.add_argument('--rounds', type=int, default=3, help="Pre-training rounds before verification (Use early round like 3 for non-converged model)")
    parser.add_argument('--alpha', type=float, default=0.2, help="Weight for Parameter Similarity")
    parser.add_argument('--beta', type=float, default=0.4, help="Weight for Loss Similarity")
    parser.add_argument('--gamma', type=float, default=0.4, help="Weight for Prediction Similarity")
    parser.add_argument('--output', default='experiments/results/plots_dual_lang/english_paper')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    verifier = AssumptionVerifier(args.config)
    
    # 1. Prepare Model (Train for a bit)
    verifier.prepare_data_at_round(args.rounds)
    
    # 2. Collect Data
    data = verifier.collect_gradients_and_stats()
    
    # 3. Compute Metrics
    res = verifier.compute_pairwise_correlations(data, args.alpha, args.beta, args.gamma)
    
    # 4. Plot
    verifier.plot_verification(res, args.output)

if __name__ == "__main__":
    main()
