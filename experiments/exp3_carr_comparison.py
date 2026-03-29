"""
Experiment 3: Comparing CARR with Fixed Compression

This experiment compares CARR against fixed-depth compression baselines
using both standard recommendation metrics and collapse-sensitive metrics.

Key Outputs:
- Comprehensive comparison table
- Efficiency vs. quality plots
- Collapse severity comparison
"""

import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from config import get_experiment_config, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, ModelConfig
from data_loader import RecommendationDataModule
from models.carr_model import CARRModel, FixedCompressionModel, FullLLMModel
from models.baselines import SASRec, BERT4Rec, GRU4Rec
from metrics.collapse_metrics import CollapseMetricsComputer
from training.trainer import (
    Trainer, RecommendationMetrics, EfficiencyMetrics, 
    TrainingConfig, save_results
)


class Experiment3:
    """Experiment 3: CARR vs. Fixed Compression Comparison"""
    
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
        
        self.config = get_experiment_config("exp3_carr_comparison", dataset_name)
        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="recency"
        )
        
        self.collapse_metrics = CollapseMetricsComputer()
        self.rec_metrics = RecommendationMetrics()
        
        self.results = {
            'recommendation_metrics': {},
            'collapse_metrics': {},
            'efficiency_metrics': {},
            'summary_table': None
        }
    
    def create_all_models(self) -> Dict[str, torch.nn.Module]:
        """Create all models for comparison"""
        num_items = self.data_module.num_items
        model_config = self.config.model
        
        models = {
            # LLM-based models
            'Full-LLM': FullLLMModel(num_items, model_config),
            'Fixed-Early (k=3)': FixedCompressionModel(num_items, ModelConfig(compression_depth=3)),
            'Fixed-Mid (k=6)': FixedCompressionModel(num_items, ModelConfig(compression_depth=6)),
            'Fixed-Late (k=9)': FixedCompressionModel(num_items, ModelConfig(compression_depth=9)),
            'CARR': CARRModel(num_items, model_config),
            
            # Sequential baselines
            'SASRec': SASRec(num_items, hidden_dim=model_config.hidden_dim, num_layers=2),
            'BERT4Rec': BERT4Rec(num_items, hidden_dim=model_config.hidden_dim, num_layers=2),
            'GRU4Rec': GRU4Rec(num_items, hidden_dim=model_config.hidden_dim, num_layers=1),
        }
        
        for model in models.values():
            model.to(self.device)
        
        return models
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        num_batches: int = 100
    ) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_R_scores = []
        all_S_scores = []
        
        test_loader = self.data_module.test_dataloader()
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            if hasattr(model, 'use_adaptive_depth'):
                # CARR model
                outputs = model(
                    input_ids,
                    attention_mask,
                    return_hidden_states=True
                )
            else:
                outputs = model(input_ids, attention_mask, return_hidden_states=True)
            
            # Get predictions
            logits = outputs['logits'][:, -1, :]
            predictions = F.softmax(logits, dim=-1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            # Collapse metrics
            hidden_states = outputs.get('last_hidden_state')
            if hidden_states is not None:
                R_result = self.collapse_metrics.compute_reasoning_collapse_score(hidden_states)
                all_R_scores.append(R_result['reasoning_collapse_score'])
            
            # Evidence survival
            if 'ablated_input_ids' in batch:
                ablated_ids = batch['ablated_input_ids'].to(self.device)
                ablated_mask = batch['ablated_attention_mask'].to(self.device)
                
                if hasattr(model, 'use_adaptive_depth'):
                    outputs_ablated = model(ablated_ids, ablated_mask)
                else:
                    outputs_ablated = model(ablated_ids, ablated_mask)
                
                prob_ablated = F.softmax(outputs_ablated['logits'][:, -1, :], dim=-1)
                
                survival = self.collapse_metrics.compute_evidence_survival(
                    predictions.to(self.device), prob_ablated
                )
                all_S_scores.append(survival['evidence_survival'].mean().item())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Recommendation metrics
        rec_metrics = self.rec_metrics.compute_metrics(all_predictions, all_targets)
        
        # Collapse metrics
        collapse_metrics = {
            'avg_R_score': np.nanmean(all_R_scores) if all_R_scores else 0.0,
            'avg_S_score': np.nanmean(all_S_scores) if all_S_scores else 0.0,
        }
        
        # Efficiency metrics
        sample_batch = next(iter(test_loader))
        sample_input = sample_batch['input_ids'].to(self.device)
        sample_mask = sample_batch['attention_mask'].to(self.device)
        
        latency = EfficiencyMetrics.measure_latency(model, sample_input, sample_mask)
        memory = EfficiencyMetrics.measure_memory(model, sample_input, sample_mask)
        
        return {
            **rec_metrics,
            **collapse_metrics,
            **latency,
            **memory
        }
    
    def run(self, num_batches: int = 100) -> Dict:
        """Run Experiment 3"""
        
        print("=" * 60)
        print("Experiment 3: CARR vs. Fixed Compression Comparison")
        print("=" * 60)
        
        models = self.create_all_models()
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            metrics = self.evaluate_model(model, model_name, num_batches)
            
            self.results['recommendation_metrics'][model_name] = {
                k: v for k, v in metrics.items() 
                if k.startswith(('HR', 'NDCG', 'Recall', 'MRR'))
            }
            self.results['collapse_metrics'][model_name] = {
                k: v for k, v in metrics.items() 
                if 'R_score' in k or 'S_score' in k
            }
            self.results['efficiency_metrics'][model_name] = {
                k: v for k, v in metrics.items() 
                if 'latency' in k or 'memory' in k or 'throughput' in k
            }
        
        # Create summary table
        self.create_summary_table()
        
        # Save results
        save_results(self.results, f"exp3_{self.dataset_name}")
        
        return self.results
    
    def create_summary_table(self):
        """Create comprehensive summary table"""
        rows = []
        
        for model_name in self.results['recommendation_metrics'].keys():
            rec = self.results['recommendation_metrics'][model_name]
            col = self.results['collapse_metrics'][model_name]
            eff = self.results['efficiency_metrics'][model_name]
            
            row = {
                'Model': model_name,
                'HR@5': f"{rec.get('HR@5', 0):.4f}",
                'HR@10': f"{rec.get('HR@10', 0):.4f}",
                'NDCG@5': f"{rec.get('NDCG@5', 0):.4f}",
                'NDCG@10': f"{rec.get('NDCG@10', 0):.4f}",
                'MRR': f"{rec.get('MRR', 0):.4f}",
                'R Score': f"{col.get('avg_R_score', 0):.4f}",
                'S Score': f"{col.get('avg_S_score', 0):.4f}",
                'Latency (ms)': f"{eff.get('latency_mean', 0):.2f}",
                'Memory (MB)': f"{eff.get('gpu_memory_mb', 0):.1f}",
                'Throughput': f"{eff.get('throughput', 0):.1f}"
            }
            rows.append(row)
        
        self.results['summary_table'] = pd.DataFrame(rows)
        
        # Save as LaTeX table
        latex_path = TABLES_DIR / f'exp3_comparison_{self.dataset_name}.tex'
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        
        latex_table = self.results['summary_table'].to_latex(
            index=False,
            caption=f"Comparison of methods on {self.dataset_name}",
            label=f"tab:exp3_{self.dataset_name}"
        )
        
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        # Save as CSV
        csv_path = TABLES_DIR / f'exp3_comparison_{self.dataset_name}.csv'
        self.results['summary_table'].to_csv(csv_path, index=False)
        
        print(f"\nSummary Table:\n{self.results['summary_table'].to_string()}")
    
    def plot_results(self, save_dir: Path = FIGURES_DIR):
        """Generate comparison plots"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        models = list(self.results['recommendation_metrics'].keys())
        
        # Plot 1: Recommendation Quality Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics = ['HR@10', 'NDCG@10']
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [self.results['recommendation_metrics'][m].get(metric, 0) for m in models]
            
            bars = ax.bar(range(len(models)), values, color=colors)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Highlight CARR
            for i, m in enumerate(models):
                if 'CARR' in m:
                    bars[i].set_edgecolor('red')
                    bars[i].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp3_rec_quality_{self.dataset_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'exp3_rec_quality_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Collapse Metrics Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # R Score
        ax1 = axes[0]
        R_values = [self.results['collapse_metrics'][m].get('avg_R_score', 0) for m in models]
        bars = ax1.bar(range(len(models)), R_values, color=colors)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('R Score (Intent Preservation)')
        ax1.set_title('Reasoning Collapse Score')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # S Score
        ax2 = axes[1]
        S_values = [self.results['collapse_metrics'][m].get('avg_S_score', 0) for m in models]
        bars = ax2.bar(range(len(models)), S_values, color=colors)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('S Score (Evidence Survival)')
        ax2.set_title('Evidence Survival Score')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp3_collapse_{self.dataset_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'exp3_collapse_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Efficiency vs. Quality Trade-off
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Latency vs. NDCG
        ax1 = axes[0]
        latencies = [self.results['efficiency_metrics'][m].get('latency_mean', 0) for m in models]
        ndcg_values = [self.results['recommendation_metrics'][m].get('NDCG@10', 0) for m in models]
        
        for i, m in enumerate(models):
            marker = 's' if 'CARR' in m else 'o'
            size = 200 if 'CARR' in m else 100
            ax1.scatter(latencies[i], ndcg_values[i], s=size, marker=marker, 
                       color=colors[i], label=m, edgecolor='black' if 'CARR' in m else None)
        
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('NDCG@10')
        ax1.set_title('Efficiency-Quality Trade-off')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Memory vs. R Score
        ax2 = axes[1]
        memory = [self.results['efficiency_metrics'][m].get('gpu_memory_mb', 0) for m in models]
        R_values = [self.results['collapse_metrics'][m].get('avg_R_score', 0) for m in models]
        
        for i, m in enumerate(models):
            marker = 's' if 'CARR' in m else 'o'
            size = 200 if 'CARR' in m else 100
            ax2.scatter(memory[i], R_values[i], s=size, marker=marker,
                       color=colors[i], label=m, edgecolor='black' if 'CARR' in m else None)
        
        ax2.set_xlabel('GPU Memory (MB)')
        ax2.set_ylabel('R Score (Intent Preservation)')
        ax2.set_title('Efficiency vs. Reasoning Preservation')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp3_efficiency_{self.dataset_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'exp3_efficiency_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Radar Chart for Overall Comparison
        self._plot_radar_chart(save_dir)
        
        print(f"Plots saved to {save_dir}")
    
    def _plot_radar_chart(self, save_dir: Path):
        """Create radar chart for multi-dimensional comparison"""
        
        categories = ['NDCG@10', 'HR@10', 'MRR', 'R Score', 'S Score', 'Efficiency']
        
        # Select key models for radar
        key_models = ['Full-LLM', 'Fixed-Mid (k=6)', 'CARR', 'SASRec']
        key_models = [m for m in key_models if m in self.results['recommendation_metrics']]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(key_models)))
        
        for idx, model in enumerate(key_models):
            rec = self.results['recommendation_metrics'][model]
            col = self.results['collapse_metrics'][model]
            eff = self.results['efficiency_metrics'][model]
            
            # Normalize values to [0, 1] scale
            values = [
                rec.get('NDCG@10', 0),
                rec.get('HR@10', 0),
                rec.get('MRR', 0),
                col.get('avg_R_score', 0) / max(1, max(self.results['collapse_metrics'][m].get('avg_R_score', 1) for m in key_models)),
                col.get('avg_S_score', 0) / max(0.01, max(self.results['collapse_metrics'][m].get('avg_S_score', 0.01) for m in key_models)),
                1.0 / (1 + eff.get('latency_mean', 1) / 100)  # Inverse latency
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Multi-dimensional Model Comparison')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp3_radar_{self.dataset_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'exp3_radar_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_experiment_3(
    datasets: List[str] = ['ml-1m'],
    device: str = "cuda",
    num_batches: int = 100
):
    """Run Experiment 3"""
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)
        
        exp = Experiment3(dataset_name=dataset, device=device)
        results = exp.run(num_batches=num_batches)
        exp.plot_results()
        
        all_results[dataset] = results
    
    return all_results


if __name__ == "__main__":
    results = run_experiment_3(datasets=['ml-1m'], num_batches=50)
