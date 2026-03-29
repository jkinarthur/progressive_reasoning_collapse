"""
Experiment 1: Observing Progressive Reasoning Collapse

This experiment validates the core theoretical claims:
- R(l) decreases monotonically after compression (Theorem 1)
- S_l(τ) decreases monotonically after compression (Theorem 2)
- Standard metrics may not reveal structural reasoning collapse

Key Outputs:
- Layerwise R(l) plots for different compression depths
- Layerwise S_l(τ) plots for different τ strategies
- Comparison of collapse severity vs. recommendation quality
"""

import os
import sys
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from config import get_experiment_config, DATASET_CONFIGS, RESULTS_DIR, FIGURES_DIR, ModelConfig
from data_loader import RecommendationDataModule
from models.carr_model import CARRModel, FixedCompressionModel, FullLLMModel, create_model
from metrics.collapse_metrics import (
    CollapseMetricsComputer,
    LayerwiseCollapseAnalyzer,
    compute_monotonicity_violation,
    compute_exponential_decay_fit
)
from training.trainer import Trainer, RecommendationMetrics, save_results


class Experiment1:
    """Experiment 1: Observing Progressive Reasoning Collapse"""
    
    def __init__(
        self,
        dataset_name: str = "ml-1m",
        device: str = "cuda",
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load config
        self.config = get_experiment_config("exp1_progressive_collapse", dataset_name)
        
        # Initialize data
        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="recency"  # Default τ strategy
        )
        
        # Initialize metrics
        self.collapse_metrics = CollapseMetricsComputer()
        self.rec_metrics = RecommendationMetrics()
        
        # Results storage
        self.results = {
            'layerwise_R': {},
            'layerwise_S': {},
            'recommendation_metrics': {},
            'monotonicity': {},
            'exponential_decay_fit': {}
        }
    
    def create_models(self) -> Dict[str, torch.nn.Module]:
        """Create models with different compression configurations"""
        
        num_items = self.data_module.num_items
        model_config = self.config.model
        
        models = {
            'full_llm': FullLLMModel(num_items, model_config),
            'fixed_early': FixedCompressionModel(num_items, ModelConfig(compression_depth=3)),
            'fixed_mid': FixedCompressionModel(num_items, ModelConfig(compression_depth=6)),
            'fixed_late': FixedCompressionModel(num_items, ModelConfig(compression_depth=9)),
            'carr': CARRModel(num_items, model_config)
        }
        
        for model in models.values():
            model.to(self.device)
        
        return models
    
    @torch.no_grad()
    def analyze_layerwise_collapse(
        self,
        model: torch.nn.Module,
        model_name: str,
        compression_depth: Optional[int] = None,
        num_batches: int = 20
    ) -> Dict[str, List[float]]:
        """
        Analyze reasoning collapse score R(l) across layers
        
        Returns R(l) for each layer to verify Theorem 1 (monotonic decrease after compression)
        """
        model.eval()
        
        layerwise_R = defaultdict(list)
        layerwise_trace_w = defaultdict(list)
        layerwise_trace_b = defaultdict(list)
        
        test_loader = self.data_module.test_dataloader()
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Analyzing {model_name}")):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Get all hidden states
            outputs = model(
                input_ids,
                attention_mask,
                compression_depth=compression_depth,
                return_hidden_states=True
            )
            
            hidden_states_list = outputs.get('all_hidden_states', [])
            
            for layer_idx, hidden_states in enumerate(hidden_states_list):
                R_result = self.collapse_metrics.compute_reasoning_collapse_score(hidden_states)
                
                layerwise_R[layer_idx].append(R_result['reasoning_collapse_score'])
                layerwise_trace_w[layer_idx].append(R_result['trace_sigma_w'])
                layerwise_trace_b[layer_idx].append(R_result['trace_sigma_b'])
        
        # Average across batches
        avg_R = {l: np.nanmean(vals) for l, vals in layerwise_R.items()}
        avg_trace_w = {l: np.nanmean(vals) for l, vals in layerwise_trace_w.items()}
        avg_trace_b = {l: np.nanmean(vals) for l, vals in layerwise_trace_b.items()}
        
        return {
            'layerwise_R': avg_R,
            'layerwise_trace_w': avg_trace_w,
            'layerwise_trace_b': avg_trace_b,
            'compression_depth': compression_depth
        }
    
    @torch.no_grad()
    def analyze_evidence_survival(
        self,
        model: torch.nn.Module,
        model_name: str,
        tau_strategies: List[str] = ['recency', 'minority', 'random'],
        compression_depths: List[Optional[int]] = [None, 3, 6, 9],
        num_batches: int = 20
    ) -> Dict[str, Dict]:
        """
        Analyze evidence survival S_l(τ) for different τ strategies and compression depths
        
        Validates Theorem 2: S_l(τ) should decrease after compression
        """
        model.eval()
        
        survival_results = {}
        
        for tau_strategy in tau_strategies:
            # Create data loader with this τ strategy
            data_module = RecommendationDataModule(
                self.config.dataset,
                batch_size=self.config.training.batch_size,
                tau_strategy=tau_strategy
            )
            test_loader = data_module.test_dataloader()
            
            survival_by_depth = {depth: [] for depth in compression_depths}
            
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"{model_name} - {tau_strategy}")):
                if batch_idx >= num_batches:
                    break
                
                if 'ablated_input_ids' not in batch:
                    continue
                
                input_ids_full = batch['input_ids'].to(self.device)
                input_ids_ablated = batch['ablated_input_ids'].to(self.device)
                mask_full = batch['attention_mask'].to(self.device)
                mask_ablated = batch['ablated_attention_mask'].to(self.device)
                
                for depth in compression_depths:
                    # Get predictions for full history
                    outputs_full = model(
                        input_ids_full,
                        mask_full,
                        compression_depth=depth
                    )
                    prob_full = F.softmax(outputs_full['logits'][:, -1, :], dim=-1)
                    
                    # Get predictions for ablated history
                    outputs_ablated = model(
                        input_ids_ablated,
                        mask_ablated,
                        compression_depth=depth
                    )
                    prob_ablated = F.softmax(outputs_ablated['logits'][:, -1, :], dim=-1)
                    
                    # Compute evidence survival
                    survival = self.collapse_metrics.compute_evidence_survival(
                        prob_full, prob_ablated
                    )
                    
                    survival_by_depth[depth].append(
                        survival['evidence_survival'].mean().item()
                    )
            
            # Average
            survival_results[tau_strategy] = {
                depth: np.mean(vals) if vals else 0.0
                for depth, vals in survival_by_depth.items()
            }
        
        return survival_results
    
    @torch.no_grad()
    def evaluate_recommendation_quality(
        self,
        model: torch.nn.Module,
        compression_depth: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate standard recommendation metrics"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(self.data_module.test_dataloader(), desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            outputs = model(input_ids, attention_mask, compression_depth=compression_depth)
            logits = outputs['logits'][:, -1, :]
            predictions = F.softmax(logits, dim=-1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return self.rec_metrics.compute_metrics(all_predictions, all_targets)
    
    def run(self, num_batches: int = 50) -> Dict:
        """Run full Experiment 1"""
        
        print("=" * 60)
        print("Experiment 1: Observing Progressive Reasoning Collapse")
        print("=" * 60)
        
        # Create models
        models = self.create_models()
        
        # Analyze each model
        for model_name, model in models.items():
            print(f"\nAnalyzing {model_name}...")
            
            # Get compression depth for this model
            if hasattr(model, 'fixed_compression_depth'):
                compression_depth = model.fixed_compression_depth
            else:
                compression_depth = None
            
            # Layerwise R(l) analysis
            R_analysis = self.analyze_layerwise_collapse(
                model, model_name, compression_depth, num_batches
            )
            self.results['layerwise_R'][model_name] = R_analysis
            
            # Evidence survival analysis
            S_analysis = self.analyze_evidence_survival(
                model, model_name, num_batches=min(num_batches, 20)
            )
            self.results['layerwise_S'][model_name] = S_analysis
            
            # Recommendation quality
            rec_metrics = self.evaluate_recommendation_quality(model, compression_depth)
            self.results['recommendation_metrics'][model_name] = rec_metrics
            
            # Monotonicity analysis
            R_values = list(R_analysis['layerwise_R'].values())
            if compression_depth and compression_depth < len(R_values):
                post_compression_R = R_values[compression_depth:]
                violation_rate = compute_monotonicity_violation(post_compression_R)
                decay_fit = compute_exponential_decay_fit(post_compression_R)
            else:
                violation_rate = 0.0
                decay_fit = {'rho': float('nan'), 'r_squared': 0.0}
            
            self.results['monotonicity'][model_name] = {
                'violation_rate': violation_rate,
                'is_monotonic': violation_rate == 0.0
            }
            self.results['exponential_decay_fit'][model_name] = decay_fit
        
        # Save results
        save_results(self.results, f"exp1_{self.dataset_name}")
        
        return self.results
    
    def plot_results(self, save_dir: Path = FIGURES_DIR):
        """Generate plots for Experiment 1"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Layerwise R(l) for different models
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for model_name, R_analysis in self.results['layerwise_R'].items():
            R_values = list(R_analysis['layerwise_R'].values())
            layers = list(R_analysis['layerwise_R'].keys())
            
            ax.plot(layers, R_values, marker='o', label=model_name)
            
            # Mark compression depth
            compression_depth = R_analysis.get('compression_depth')
            if compression_depth:
                ax.axvline(x=compression_depth, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Reasoning Collapse Score R(l)')
        ax.set_title('Progressive Reasoning Collapse Across Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp1_layerwise_R_{self.dataset_name}.pdf', dpi=300)
        plt.savefig(save_dir / f'exp1_layerwise_R_{self.dataset_name}.png', dpi=300)
        plt.close()
        
        # Plot 2: Evidence Survival by Compression Depth
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        tau_strategies = ['recency', 'minority', 'random']
        
        for idx, tau in enumerate(tau_strategies):
            ax = axes[idx]
            
            for model_name, S_analysis in self.results['layerwise_S'].items():
                if tau in S_analysis:
                    depths = list(S_analysis[tau].keys())
                    survival = [S_analysis[tau][d] for d in depths]
                    
                    depth_labels = ['None' if d is None else str(d) for d in depths]
                    ax.plot(range(len(depths)), survival, marker='o', label=model_name)
                    ax.set_xticks(range(len(depths)))
                    ax.set_xticklabels(depth_labels)
            
            ax.set_xlabel('Compression Depth')
            ax.set_ylabel('Evidence Survival S(τ)')
            ax.set_title(f'τ Strategy: {tau}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp1_evidence_survival_{self.dataset_name}.pdf', dpi=300)
        plt.savefig(save_dir / f'exp1_evidence_survival_{self.dataset_name}.png', dpi=300)
        plt.close()
        
        # Plot 3: Collapse vs. Recommendation Quality Trade-off
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        for model_name in self.results['recommendation_metrics'].keys():
            ndcg = self.results['recommendation_metrics'][model_name].get('NDCG@10', 0)
            
            # Use average R score as collapse measure
            R_values = list(self.results['layerwise_R'][model_name]['layerwise_R'].values())
            avg_R = np.mean(R_values) if R_values else 0
            
            ax.scatter(ndcg, avg_R, s=100, label=model_name)
        
        ax.set_xlabel('NDCG@10 (Recommendation Quality)')
        ax.set_ylabel('Average R Score (Intent Preservation)')
        ax.set_title('Recommendation Quality vs. Reasoning Preservation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp1_quality_vs_collapse_{self.dataset_name}.pdf', dpi=300)
        plt.savefig(save_dir / f'exp1_quality_vs_collapse_{self.dataset_name}.png', dpi=300)
        plt.close()
        
        print(f"Plots saved to {save_dir}")


def run_experiment_1(
    datasets: List[str] = ['ml-1m'],
    device: str = "cuda",
    num_batches: int = 50
):
    """Run Experiment 1 across multiple datasets"""
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)
        
        exp = Experiment1(dataset_name=dataset, device=device)
        results = exp.run(num_batches=num_batches)
        exp.plot_results()
        
        all_results[dataset] = results
    
    return all_results


if __name__ == "__main__":
    results = run_experiment_1(datasets=['ml-1m'], num_batches=20)
    print("\nExperiment 1 completed!")
