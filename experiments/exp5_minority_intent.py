"""
Experiment 5: Minority-Intent Preservation

Validates the corollary on minority-intent erasure:
- Aggressive compression disproportionately erases minority interests
- CARR should preserve minority signals better through adaptive mechanisms

Key Outputs:
- Minority vs. majority intent recommendation accuracy
- Evidence survival for minority interests
- Case study visualizations
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from config import get_experiment_config, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, ModelConfig
from data_loader import RecommendationDataModule, identify_minority_items, compute_item_frequency
from models.carr_model import CARRModel, FixedCompressionModel, FullLLMModel
from metrics.collapse_metrics import CollapseMetricsComputer
from training.trainer import RecommendationMetrics, save_results


class Experiment5:
    """Experiment 5: Minority-Intent Preservation"""
    
    def __init__(
        self,
        dataset_name: str = "ml-1m",
        device: str = "cuda",
        minority_percentile: float = 20,  # Bottom 20% frequency items
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.minority_percentile = minority_percentile
        self.seed = seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.config = get_experiment_config("exp5_minority_intent", dataset_name)
        
        # Load data with minority τ strategy
        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="minority"
        )
        
        # Identify minority items
        self.item_frequency = compute_item_frequency(self.data_module.sequences)
        self.minority_items = identify_minority_items(
            self.item_frequency,
            percentile=minority_percentile
        )
        
        self.collapse_metrics = CollapseMetricsComputer()
        self.rec_metrics = RecommendationMetrics()
        
        self.results = {
            'overall_metrics': {},
            'minority_metrics': {},
            'majority_metrics': {},
            'evidence_survival': {},
            'disparity_analysis': {}
        }
    
    def split_by_target_type(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split predictions into minority and majority target groups"""
        
        minority_mask = torch.tensor([
            t.item() in self.minority_items 
            for t in targets
        ])
        
        minority_preds = predictions[minority_mask]
        minority_targets = targets[minority_mask]
        majority_preds = predictions[~minority_mask]
        majority_targets = targets[~minority_mask]
        
        return minority_preds, minority_targets, majority_preds, majority_targets
    
    def create_models(self) -> Dict[str, torch.nn.Module]:
        """Create models for comparison"""
        num_items = self.data_module.num_items
        
        models = {
            'Full-LLM': FullLLMModel(num_items, self.config.model),
            'Fixed-Early': FixedCompressionModel(num_items, ModelConfig(compression_depth=3)),
            'Fixed-Mid': FixedCompressionModel(num_items, ModelConfig(compression_depth=6)),
            'CARR': CARRModel(num_items, self.config.model)
        }
        
        for model in models.values():
            model.to(self.device)
        
        return models
    
    @torch.no_grad()
    def evaluate_minority_preservation(
        self,
        model: torch.nn.Module,
        model_name: str,
        num_batches: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model's preservation of minority intent signals"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        minority_S_scores = []
        majority_S_scores = []
        
        test_loader = self.data_module.test_dataloader()
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'][:, -1, :]
            predictions = F.softmax(logits, dim=-1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            # Evidence survival for minority interests
            if 'ablated_input_ids' in batch:
                ablated_ids = batch['ablated_input_ids'].to(self.device)
                ablated_mask = batch['ablated_attention_mask'].to(self.device)
                
                outputs_ablated = model(ablated_ids, ablated_mask)
                prob_ablated = F.softmax(outputs_ablated['logits'][:, -1, :], dim=-1)
                
                survival = self.collapse_metrics.compute_evidence_survival(
                    predictions, prob_ablated
                )['evidence_survival']
                
                # Split by target type
                for i, target in enumerate(targets):
                    if target.item() in self.minority_items:
                        minority_S_scores.append(survival[i].item())
                    else:
                        majority_S_scores.append(survival[i].item())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Split metrics
        minority_preds, minority_targets, majority_preds, majority_targets = \
            self.split_by_target_type(all_predictions, all_targets)
        
        results = {
            'overall': self.rec_metrics.compute_metrics(all_predictions, all_targets),
            'minority': self.rec_metrics.compute_metrics(minority_preds, minority_targets) if len(minority_targets) > 0 else {},
            'majority': self.rec_metrics.compute_metrics(majority_preds, majority_targets) if len(majority_targets) > 0 else {},
            'evidence_survival': {
                'minority_S': np.mean(minority_S_scores) if minority_S_scores else 0,
                'majority_S': np.mean(majority_S_scores) if majority_S_scores else 0,
                'S_disparity': (np.mean(majority_S_scores) - np.mean(minority_S_scores)) if minority_S_scores and majority_S_scores else 0
            }
        }
        
        # Compute disparity metrics
        if results['minority'] and results['majority']:
            results['disparity'] = {
                'NDCG_gap': results['majority'].get('NDCG@10', 0) - results['minority'].get('NDCG@10', 0),
                'HR_gap': results['majority'].get('HR@10', 0) - results['minority'].get('HR@10', 0)
            }
        else:
            results['disparity'] = {'NDCG_gap': 0, 'HR_gap': 0}
        
        return results
    
    def analyze_minority_users(self, num_samples: int = 100) -> Dict:
        """Analyze users with strong minority interests"""
        
        minority_user_stats = []
        
        for user_id, sequence in list(self.data_module.sequences.items())[:num_samples]:
            minority_count = sum(1 for item in sequence if item in self.minority_items)
            minority_ratio = minority_count / len(sequence) if sequence else 0
            
            minority_user_stats.append({
                'user_id': user_id,
                'seq_length': len(sequence),
                'minority_count': minority_count,
                'minority_ratio': minority_ratio
            })
        
        minority_users = [u for u in minority_user_stats if u['minority_ratio'] > 0.3]
        
        return {
            'total_users_analyzed': len(minority_user_stats),
            'users_with_strong_minority_interest': len(minority_users),
            'avg_minority_ratio': np.mean([u['minority_ratio'] for u in minority_user_stats])
        }
    
    def run(self, num_batches: int = 100) -> Dict:
        """Run Experiment 5"""
        
        print("=" * 60)
        print("Experiment 5: Minority-Intent Preservation")
        print("=" * 60)
        
        print(f"\nMinority items: {len(self.minority_items)} "
              f"(bottom {self.minority_percentile}% by frequency)")
        
        # Analyze minority distribution
        user_analysis = self.analyze_minority_users()
        print(f"User analysis: {user_analysis}")
        
        models = self.create_models()
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            results = self.evaluate_minority_preservation(model, model_name, num_batches)
            
            self.results['overall_metrics'][model_name] = results['overall']
            self.results['minority_metrics'][model_name] = results['minority']
            self.results['majority_metrics'][model_name] = results['majority']
            self.results['evidence_survival'][model_name] = results['evidence_survival']
            self.results['disparity_analysis'][model_name] = results['disparity']
        
        # Create summary table
        self.create_summary_table()
        
        # Save results
        save_results(self.results, f"exp5_{self.dataset_name}")
        
        return self.results
    
    def create_summary_table(self):
        """Create minority vs. majority comparison table"""
        rows = []
        
        for model_name in self.results['overall_metrics'].keys():
            overall = self.results['overall_metrics'][model_name]
            minority = self.results['minority_metrics'][model_name]
            majority = self.results['majority_metrics'][model_name]
            survival = self.results['evidence_survival'][model_name]
            disparity = self.results['disparity_analysis'][model_name]
            
            row = {
                'Model': model_name,
                'Overall NDCG@10': f"{overall.get('NDCG@10', 0):.4f}",
                'Minority NDCG@10': f"{minority.get('NDCG@10', 0):.4f}",
                'Majority NDCG@10': f"{majority.get('NDCG@10', 0):.4f}",
                'NDCG Gap': f"{disparity.get('NDCG_gap', 0):.4f}",
                'Minority S': f"{survival.get('minority_S', 0):.4f}",
                'Majority S': f"{survival.get('majority_S', 0):.4f}",
                'S Disparity': f"{survival.get('S_disparity', 0):.4f}"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save LaTeX
        latex_path = TABLES_DIR / f'exp5_minority_{self.dataset_name}.tex'
        df.to_latex(latex_path, index=False, caption="Minority Intent Preservation Analysis")
        
        # Save CSV
        csv_path = TABLES_DIR / f'exp5_minority_{self.dataset_name}.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\nMinority Preservation Analysis:\n{df.to_string()}")
        
        self.results['summary_table'] = df
    
    def plot_results(self, save_dir: Path = FIGURES_DIR):
        """Generate plots for minority intent analysis"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        models = list(self.results['overall_metrics'].keys())
        
        # Plot 1: Minority vs. Majority Performance Gap
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # NDCG comparison
        ax1 = axes[0]
        x = np.arange(len(models))
        width = 0.35
        
        minority_ndcg = [self.results['minority_metrics'][m].get('NDCG@10', 0) for m in models]
        majority_ndcg = [self.results['majority_metrics'][m].get('NDCG@10', 0) for m in models]
        
        bars1 = ax1.bar(x - width/2, minority_ndcg, width, label='Minority Items', color='coral', alpha=0.8)
        bars2 = ax1.bar(x + width/2, majority_ndcg, width, label='Majority Items', color='steelblue', alpha=0.8)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('NDCG@10')
        ax1.set_title('Minority vs. Majority Item Recommendation')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Evidence survival comparison
        ax2 = axes[1]
        
        minority_S = [self.results['evidence_survival'][m].get('minority_S', 0) for m in models]
        majority_S = [self.results['evidence_survival'][m].get('majority_S', 0) for m in models]
        
        bars3 = ax2.bar(x - width/2, minority_S, width, label='Minority Evidence', color='coral', alpha=0.8)
        bars4 = ax2.bar(x + width/2, majority_S, width, label='Majority Evidence', color='steelblue', alpha=0.8)
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Evidence Survival Score')
        ax2.set_title('Evidence Survival: Minority vs. Majority')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp5_minority_gap_{self.dataset_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'exp5_minority_gap_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Disparity Analysis
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ndcg_gaps = [self.results['disparity_analysis'][m].get('NDCG_gap', 0) for m in models]
        S_disparities = [self.results['evidence_survival'][m].get('S_disparity', 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ndcg_gaps, width, label='NDCG Gap (Majority - Minority)', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, S_disparities, width, label='S Disparity (Majority - Minority)', color='orange', alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Model')
        ax.set_ylabel('Disparity (higher = worse for minority)')
        ax.set_title('Performance Disparity Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight best (lowest disparity)
        best_ndcg_gap = np.argmin(np.abs(ndcg_gaps))
        bars1[best_ndcg_gap].set_edgecolor('green')
        bars1[best_ndcg_gap].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp5_disparity_{self.dataset_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'exp5_disparity_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {save_dir}")


def run_experiment_5(
    datasets: List[str] = ['ml-1m'],
    device: str = "cuda",
    num_batches: int = 100
):
    """Run Experiment 5"""
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)
        
        exp = Experiment5(dataset_name=dataset, device=device)
        results = exp.run(num_batches=num_batches)
        exp.plot_results()
        
        all_results[dataset] = results
    
    return all_results


if __name__ == "__main__":
    results = run_experiment_5(datasets=['ml-1m'], num_batches=50)
