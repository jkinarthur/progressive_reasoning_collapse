"""
Experiment 2: Validating Critical Compression Depth (Theorem 3)

This experiment validates the existence of a critical compression depth k* that
separates the safe regime from the collapse regime.

Key Outputs:
- Phase transition plot showing k* boundary
- Safe vs. collapse region visualization
- Monotone ordering validation
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).parent.parent))

from config import get_experiment_config, RESULTS_DIR, FIGURES_DIR, ModelConfig, COMPRESSION_DEPTHS
from data_loader import RecommendationDataModule
from models.carr_model import CARRModel, create_model
from metrics.collapse_metrics import CollapseMetricsComputer, CriticalDepthFinder
from training.trainer import RecommendationMetrics, save_results


class Experiment2:
    """Experiment 2: Validating Critical Compression Depth"""
    
    def __init__(
        self,
        dataset_name: str = "ml-1m",
        device: str = "cuda",
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.config = get_experiment_config("exp2_critical_depth", dataset_name)
        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="recency"
        )
        
        self.collapse_metrics = CollapseMetricsComputer()
        self.rec_metrics = RecommendationMetrics()
        
        # Thresholds for defining collapse
        self.R_threshold = 0.1  # Minimum acceptable R score
        self.S_threshold = 0.05  # Minimum acceptable evidence survival
        
        self.results = {
            'R_by_depth': {},
            'S_by_depth': {},
            'metrics_by_depth': {},
            'critical_depth': None,
            'safe_depths': [],
            'collapse_depths': []
        }
    
    @torch.no_grad()
    def evaluate_compression_depth(
        self,
        model: torch.nn.Module,
        compression_depth: int,
        num_batches: int = 50
    ) -> Dict[str, float]:
        """Evaluate model at a specific compression depth"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_R_scores = []
        all_S_scores = []
        
        test_loader = self.data_module.test_dataloader()
        
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward with compression
            outputs = model(
                input_ids,
                attention_mask,
                compression_depth=compression_depth,
                return_hidden_states=True
            )
            
            # Recommendation metrics
            logits = outputs['logits'][:, -1, :]
            predictions = F.softmax(logits, dim=-1)
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            # Collapse score (R)
            hidden_states = outputs.get('last_hidden_state')
            if hidden_states is not None:
                R_result = self.collapse_metrics.compute_reasoning_collapse_score(hidden_states)
                all_R_scores.append(R_result['reasoning_collapse_score'])
            
            # Evidence survival (S)
            if 'ablated_input_ids' in batch:
                ablated_ids = batch['ablated_input_ids'].to(self.device)
                ablated_mask = batch['ablated_attention_mask'].to(self.device)
                
                outputs_ablated = model(
                    ablated_ids,
                    ablated_mask,
                    compression_depth=compression_depth
                )
                prob_ablated = F.softmax(outputs_ablated['logits'][:, -1, :], dim=-1)
                
                survival = self.collapse_metrics.compute_evidence_survival(
                    predictions.to(self.device), prob_ablated
                )
                all_S_scores.append(survival['evidence_survival'].mean().item())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.rec_metrics.compute_metrics(all_predictions, all_targets)
        metrics['avg_R'] = np.nanmean(all_R_scores) if all_R_scores else 0.0
        metrics['avg_S'] = np.nanmean(all_S_scores) if all_S_scores else 0.0
        
        return metrics
    
    def find_critical_depth(self) -> Dict:
        """Find the critical compression depth k*"""
        
        R_by_depth = list(self.results['R_by_depth'].values())
        S_by_depth = list(self.results['S_by_depth'].values())
        depths = list(self.results['R_by_depth'].keys())
        
        finder = CriticalDepthFinder(
            threshold_R=self.R_threshold,
            threshold_S=self.S_threshold
        )
        
        result = finder.find_critical_depth(R_by_depth, S_by_depth)
        
        # Map back to actual depth values
        if result['critical_depth'] < len(depths):
            k_star = depths[result['critical_depth']]
        else:
            k_star = depths[-1]
        
        safe_depths = [depths[i] for i in result['safe_depths'] if i < len(depths)]
        collapse_depths = [depths[i] for i in result['collapse_depths'] if i < len(depths)]
        
        return {
            'critical_depth': k_star,
            'safe_depths': safe_depths,
            'collapse_depths': collapse_depths
        }
    
    def validate_monotonicity(self) -> Dict[str, bool]:
        """
        Validate that earlier compression cannot outperform later compression
        (Theorem 3, part iii)
        """
        depths = sorted(self.results['R_by_depth'].keys())
        
        R_monotone = True
        S_monotone = True
        
        for i in range(len(depths) - 1):
            k1, k2 = depths[i], depths[i+1]
            
            # k1 < k2 (earlier compression)
            # Should have R(k1) <= R(k2) and S(k1) <= S(k2)
            if self.results['R_by_depth'][k1] > self.results['R_by_depth'][k2]:
                R_monotone = False
            if self.results['S_by_depth'][k1] > self.results['S_by_depth'][k2]:
                S_monotone = False
        
        return {
            'R_monotone': R_monotone,
            'S_monotone': S_monotone,
            'both_monotone': R_monotone and S_monotone
        }
    
    def run(
        self,
        compression_depths: List[int] = None,
        num_batches: int = 50
    ) -> Dict:
        """Run Experiment 2"""
        
        print("=" * 60)
        print("Experiment 2: Validating Critical Compression Depth")
        print("=" * 60)
        
        if compression_depths is None:
            compression_depths = COMPRESSION_DEPTHS
        
        # Create model
        model = CARRModel(
            num_items=self.data_module.num_items,
            config=self.config.model
        ).to(self.device)
        
        # Evaluate at each compression depth
        for depth in tqdm(compression_depths, desc="Sweeping compression depths"):
            metrics = self.evaluate_compression_depth(model, depth, num_batches)
            
            self.results['R_by_depth'][depth] = metrics['avg_R']
            self.results['S_by_depth'][depth] = metrics['avg_S']
            self.results['metrics_by_depth'][depth] = metrics
        
        # Find critical depth
        critical_result = self.find_critical_depth()
        self.results['critical_depth'] = critical_result['critical_depth']
        self.results['safe_depths'] = critical_result['safe_depths']
        self.results['collapse_depths'] = critical_result['collapse_depths']
        
        # Validate monotonicity
        monotonicity = self.validate_monotonicity()
        self.results['monotonicity_validation'] = monotonicity
        
        print(f"\nCritical compression depth k* = {self.results['critical_depth']}")
        print(f"Safe depths: {self.results['safe_depths']}")
        print(f"Collapse depths: {self.results['collapse_depths']}")
        print(f"Monotonicity validation: {monotonicity}")
        
        # Save results
        save_results(self.results, f"exp2_{self.dataset_name}")
        
        return self.results
    
    def plot_results(self, save_dir: Path = FIGURES_DIR):
        """Generate plots for Experiment 2"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        depths = sorted(self.results['R_by_depth'].keys())
        R_values = [self.results['R_by_depth'][d] for d in depths]
        S_values = [self.results['S_by_depth'][d] for d in depths]
        
        # Plot 1: Phase Transition Diagram
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # R(k) plot
        ax1 = axes[0]
        colors = ['green' if d in self.results['safe_depths'] else 'red' for d in depths]
        ax1.bar(depths, R_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=self.R_threshold, color='orange', linestyle='--', label=f'Threshold R̲={self.R_threshold}')
        
        if self.results['critical_depth']:
            ax1.axvline(x=self.results['critical_depth'], color='purple', linestyle='-', 
                       linewidth=2, label=f'k*={self.results["critical_depth"]}')
        
        ax1.set_xlabel('Compression Depth k')
        ax1.set_ylabel('Final R Score $\\mathcal{R}_L^{(k)}$')
        ax1.set_title('Reasoning Collapse by Compression Depth')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # S(k) plot
        ax2 = axes[1]
        ax2.bar(depths, S_values, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=self.S_threshold, color='orange', linestyle='--', label=f'Threshold S̲={self.S_threshold}')
        
        if self.results['critical_depth']:
            ax2.axvline(x=self.results['critical_depth'], color='purple', linestyle='-', 
                       linewidth=2, label=f'k*={self.results["critical_depth"]}')
        
        ax2.set_xlabel('Compression Depth k')
        ax2.set_ylabel('Final Evidence Survival $S_L^{(k)}(\\tau)$')
        ax2.set_title('Evidence Survival by Compression Depth')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add legend for safe/collapse
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Safe regime'),
            Patch(facecolor='red', alpha=0.7, label='Collapse regime')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp2_phase_transition_{self.dataset_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'exp2_phase_transition_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Quality vs. R/S Trade-off
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ndcg_values = [self.results['metrics_by_depth'][d].get('NDCG@10', 0) for d in depths]
        
        ax1 = axes[0]
        ax1.scatter(ndcg_values, R_values, c=depths, cmap='viridis', s=100)
        for i, d in enumerate(depths):
            ax1.annotate(f'k={d}', (ndcg_values[i], R_values[i]), fontsize=8)
        ax1.set_xlabel('NDCG@10')
        ax1.set_ylabel('R Score')
        ax1.set_title('Recommendation Quality vs. Intent Preservation')
        ax1.axhline(y=self.R_threshold, color='red', linestyle='--', alpha=0.5)
        
        ax2 = axes[1]
        ax2.scatter(ndcg_values, S_values, c=depths, cmap='viridis', s=100)
        for i, d in enumerate(depths):
            ax2.annotate(f'k={d}', (ndcg_values[i], S_values[i]), fontsize=8)
        ax2.set_xlabel('NDCG@10')
        ax2.set_ylabel('Evidence Survival')
        ax2.set_title('Recommendation Quality vs. Evidence Sensitivity')
        ax2.axhline(y=self.S_threshold, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp2_quality_tradeoff_{self.dataset_name}.pdf', dpi=300)
        plt.savefig(save_dir / f'exp2_quality_tradeoff_{self.dataset_name}.png', dpi=300)
        plt.close()
        
        # Plot 3: Monotonicity Validation
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(depths, R_values, 'o-', label='R Score', color='blue', markersize=8)
        ax.plot(depths, S_values, 's-', label='Evidence Survival', color='green', markersize=8)
        
        if self.results['critical_depth']:
            ax.axvline(x=self.results['critical_depth'], color='red', linestyle='--', 
                      linewidth=2, label=f'Critical Depth k*={self.results["critical_depth"]}')
        
        ax.set_xlabel('Compression Depth k')
        ax.set_ylabel('Score')
        ax.set_title('Monotone Phase Transition (Theorem 3)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotation about monotonicity
        mono_text = "✓ Monotone" if self.results['monotonicity_validation']['both_monotone'] else "✗ Not strictly monotone"
        ax.text(0.02, 0.98, mono_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp2_monotonicity_{self.dataset_name}.pdf', dpi=300)
        plt.savefig(save_dir / f'exp2_monotonicity_{self.dataset_name}.png', dpi=300)
        plt.close()
        
        print(f"Plots saved to {save_dir}")


def run_experiment_2(
    datasets: List[str] = ['ml-1m'],
    device: str = "cuda",
    num_batches: int = 50
):
    """Run Experiment 2 across datasets"""
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)
        
        exp = Experiment2(dataset_name=dataset, device=device)
        results = exp.run(num_batches=num_batches)
        exp.plot_results()
        
        all_results[dataset] = results
    
    return all_results


if __name__ == "__main__":
    results = run_experiment_2(datasets=['ml-1m'], num_batches=20)
