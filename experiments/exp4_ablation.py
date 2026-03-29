"""
Experiment 4: Ablation Studies

This experiment isolates the contribution of each CARR component:
1. Collapse regularizer (L_collapse)
2. Evidence preservation regularizer (L_evidence)  
3. Adaptive depth selection
4. Adaptive register width

Key Outputs:
- Component-by-component ablation table
- Impact analysis plots
- Statistical significance tests
"""

import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    get_experiment_config, RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
    ModelConfig, ABLATION_CONFIGS
)
from data_loader import RecommendationDataModule
from models.carr_model import CARRModel
from metrics.collapse_metrics import CollapseMetricsComputer
from training.trainer import (
    Trainer, RecommendationMetrics, EfficiencyMetrics,
    TrainingConfig, save_results
)


class Experiment4:
    """Experiment 4: Ablation Studies"""
    
    def __init__(
        self,
        dataset_name: str = "ml-1m",
        device: str = "cuda",
        num_runs: int = 5,  # Multiple runs for statistical significance
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_runs = num_runs
        self.base_seed = seed
        
        self.config = get_experiment_config("exp4_ablation", dataset_name)
        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="recency"
        )
        
        self.collapse_metrics = CollapseMetricsComputer()
        self.rec_metrics = RecommendationMetrics()
        
        self.results = {
            'ablation_results': {},
            'statistical_tests': {},
            'summary_table': None
        }
    
    def create_ablation_model(
        self,
        ablation_config: Dict[str, bool],
        seed: int
    ) -> torch.nn.Module:
        """Create model with specific ablation configuration"""
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model_config = copy.deepcopy(self.config.model)
        
        # Apply ablation settings
        if not ablation_config['collapse_reg']:
            model_config.lambda_collapse = 0.0
        if not ablation_config['evidence_reg']:
            model_config.lambda_evidence = 0.0
        
        model = CARRModel(
            num_items=self.data_module.num_items,
            config=model_config
        )
        
        # Disable adaptive components if needed
        model.use_adaptive_depth = ablation_config['adaptive_depth']
        model.use_adaptive_width = ablation_config['adaptive_width']
        model.use_collapse_reg = ablation_config['collapse_reg']
        model.use_evidence_reg = ablation_config['evidence_reg']
        
        return model.to(self.device)
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model: torch.nn.Module,
        num_batches: int = 100
    ) -> Dict[str, float]:
        """Evaluate single model run"""
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
            
            outputs = model(
                input_ids,
                attention_mask,
                return_hidden_states=True
            )
            
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
                
                outputs_ablated = model(ablated_ids, ablated_mask)
                prob_ablated = F.softmax(outputs_ablated['logits'][:, -1, :], dim=-1)
                
                survival = self.collapse_metrics.compute_evidence_survival(
                    predictions.to(self.device), prob_ablated
                )
                all_S_scores.append(survival['evidence_survival'].mean().item())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        rec_metrics = self.rec_metrics.compute_metrics(all_predictions, all_targets)
        
        return {
            **rec_metrics,
            'R_score': np.nanmean(all_R_scores) if all_R_scores else 0.0,
            'S_score': np.nanmean(all_S_scores) if all_S_scores else 0.0
        }
    
    def run_ablation_variant(
        self,
        variant_name: str,
        ablation_config: Dict[str, bool],
        num_batches: int = 100
    ) -> Dict[str, List[float]]:
        """Run multiple trials of an ablation variant"""
        
        all_metrics = defaultdict(list)
        
        for run_idx in range(self.num_runs):
            seed = self.base_seed + run_idx
            print(f"  Run {run_idx + 1}/{self.num_runs} (seed={seed})")
            
            model = self.create_ablation_model(ablation_config, seed)
            metrics = self.evaluate_model(model, num_batches)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        return dict(all_metrics)
    
    def compute_statistical_tests(self) -> Dict:
        """Compute statistical significance between Full CARR and ablations"""
        
        full_carr = self.results['ablation_results'].get('full_carr')
        if full_carr is None:
            return {}
        
        tests = {}
        key_metrics = ['NDCG@10', 'HR@10', 'R_score', 'S_score']
        
        for variant_name, variant_results in self.results['ablation_results'].items():
            if variant_name == 'full_carr':
                continue
            
            variant_tests = {}
            for metric in key_metrics:
                full_values = full_carr.get(metric, [])
                variant_values = variant_results.get(metric, [])
                
                if len(full_values) >= 2 and len(variant_values) >= 2:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_ind(full_values, variant_values)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (np.std(full_values)**2 + np.std(variant_values)**2) / 2
                    )
                    cohens_d = (np.mean(full_values) - np.mean(variant_values)) / (pooled_std + 1e-10)
                    
                    variant_tests[metric] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05
                    }
            
            tests[variant_name] = variant_tests
        
        return tests
    
    def run(self, num_batches: int = 100) -> Dict:
        """Run full ablation study"""
        
        print("=" * 60)
        print("Experiment 4: Ablation Studies")
        print("=" * 60)
        
        # Run each ablation variant
        for variant_name, ablation_config in ABLATION_CONFIGS.items():
            print(f"\nVariant: {variant_name}")
            print(f"  Config: {ablation_config}")
            
            results = self.run_ablation_variant(
                variant_name,
                ablation_config,
                num_batches
            )
            self.results['ablation_results'][variant_name] = results
        
        # Statistical tests
        self.results['statistical_tests'] = self.compute_statistical_tests()
        
        # Create summary table
        self.create_summary_table()
        
        # Save results
        save_results(self.results, f"exp4_{self.dataset_name}")
        
        return self.results
    
    def create_summary_table(self):
        """Create ablation summary table with means and std"""
        rows = []
        
        for variant_name, metrics in self.results['ablation_results'].items():
            row = {'Variant': variant_name.replace('_', ' ').title()}
            
            for metric in ['HR@10', 'NDCG@10', 'MRR', 'R_score', 'S_score']:
                values = metrics.get(metric, [0])
                mean = np.mean(values)
                std = np.std(values)
                row[metric] = f"{mean:.4f} ± {std:.4f}"
            
            # Add significance markers
            if variant_name in self.results['statistical_tests']:
                tests = self.results['statistical_tests'][variant_name]
                sig_markers = []
                for metric, test_result in tests.items():
                    if test_result.get('significant'):
                        sig_markers.append(metric)
                row['Significant Diff'] = ', '.join(sig_markers) if sig_markers else '-'
            else:
                row['Significant Diff'] = '-'
            
            rows.append(row)
        
        self.results['summary_table'] = pd.DataFrame(rows)
        
        # Save
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        
        latex_path = TABLES_DIR / f'exp4_ablation_{self.dataset_name}.tex'
        latex_table = self.results['summary_table'].to_latex(
            index=False,
            caption=f"Ablation study results on {self.dataset_name}",
            label=f"tab:ablation_{self.dataset_name}",
            escape=False
        )
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        csv_path = TABLES_DIR / f'exp4_ablation_{self.dataset_name}.csv'
        self.results['summary_table'].to_csv(csv_path, index=False)
        
        print(f"\nAblation Summary:\n{self.results['summary_table'].to_string()}")
    
    def plot_results(self, save_dir: Path = FIGURES_DIR):
        """Generate ablation study plots"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        variants = list(self.results['ablation_results'].keys())
        
        # Plot 1: Component Impact Bar Charts
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['NDCG@10', 'HR@10', 'R_score', 'S_score']
        titles = ['NDCG@10', 'HR@10', 'R Score', 'S Score']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            means = []
            stds = []
            for v in variants:
                values = self.results['ablation_results'][v].get(metric, [0])
                means.append(np.mean(values))
                stds.append(np.std(values))
            
            x = np.arange(len(variants))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
            
            # Highlight full CARR
            for i, v in enumerate(variants):
                if v == 'full_carr':
                    bars[i].set_color('green')
                else:
                    bars[i].set_color('steelblue')
            
            ax.set_xticks(x)
            ax.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=8)
            ax.set_ylabel(title)
            ax.set_title(f'{title} by Ablation Variant')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp4_ablation_bars_{self.dataset_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'exp4_ablation_bars_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Component Contribution Analysis
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Calculate relative drop from full CARR
        full_carr_metrics = self.results['ablation_results']['full_carr']
        
        component_names = ['no_collapse_reg', 'no_evidence_reg', 'no_adaptive_depth', 'no_adaptive_width']
        display_names = ['w/o Collapse Reg', 'w/o Evidence Reg', 'w/o Adaptive Depth', 'w/o Adaptive Width']
        
        metrics_to_plot = ['NDCG@10', 'R_score', 'S_score']
        x = np.arange(len(component_names))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            full_value = np.mean(full_carr_metrics.get(metric, [0]))
            relative_drops = []
            
            for comp in component_names:
                ablated_value = np.mean(self.results['ablation_results'][comp].get(metric, [0]))
                drop = (full_value - ablated_value) / (full_value + 1e-10) * 100
                relative_drops.append(drop)
            
            ax.bar(x + i * width, relative_drops, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Ablated Component')
        ax.set_ylabel('Relative Performance Drop (%)')
        ax.set_title('Component Contribution Analysis')
        ax.set_xticks(x + width)
        ax.set_xticklabels(display_names, rotation=15)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp4_component_contribution_{self.dataset_name}.pdf', dpi=300)
        plt.savefig(save_dir / f'exp4_component_contribution_{self.dataset_name}.png', dpi=300)
        plt.close()
        
        # Plot 3: Box plots for multiple runs
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        
        for idx, metric in enumerate(['NDCG@10', 'HR@10', 'R_score', 'S_score']):
            ax = axes[idx]
            
            data = [self.results['ablation_results'][v].get(metric, [0]) for v in variants]
            bp = ax.boxplot(data, patch_artist=True)
            
            # Color boxes
            colors = ['green' if v == 'full_carr' else 'steelblue' for v in variants]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=7, rotation=45)
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp4_boxplots_{self.dataset_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'exp4_boxplots_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {save_dir}")


def run_experiment_4(
    datasets: List[str] = ['ml-1m'],
    device: str = "cuda",
    num_runs: int = 5,
    num_batches: int = 100
):
    """Run Experiment 4"""
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)
        
        exp = Experiment4(
            dataset_name=dataset,
            device=device,
            num_runs=num_runs
        )
        results = exp.run(num_batches=num_batches)
        exp.plot_results()
        
        all_results[dataset] = results
    
    return all_results


if __name__ == "__main__":
    results = run_experiment_4(datasets=['ml-1m'], num_runs=3, num_batches=30)
