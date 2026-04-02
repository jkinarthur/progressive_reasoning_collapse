"""
Experiment 6: Visualization of Layerwise Collapse

Provides interpretable visualizations of:
- Latent intent cluster contraction across layers
- Evidence-induced representation divergence
- Comparison between fixed compression and CARR

Uses PCA, t-SNE, and UMAP for dimensionality reduction
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings

# Optional UMAP import
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed. Using t-SNE as fallback.")

sys.path.append(str(Path(__file__).parent.parent))

from config import get_experiment_config, FIGURES_DIR, ModelConfig
from data_loader import RecommendationDataModule
from models.carr_model import CARRModel, FixedCompressionModel, FullLLMModel
from metrics.collapse_metrics import CollapseMetricsComputer
from training.trainer import save_results


class Experiment6:
    """Experiment 6: Visualization of Layerwise Collapse"""
    
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
        
        self.config = get_experiment_config("exp6_visualization", dataset_name)
        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="recency"
        )
        
        self.collapse_metrics = CollapseMetricsComputer()
        
        # Store extracted representations
        self.representations = {}
    
    def create_models(self) -> Dict[str, torch.nn.Module]:
        """Create models for visualization"""
        num_items = self.data_module.num_items
        
        models = {
            'Full-LLM': FullLLMModel(num_items, self.config.model),
            'Fixed-Mid': FixedCompressionModel(num_items, ModelConfig(compression_depth=6)),
            'CARR': CARRModel(num_items, self.config.model)
        }
        
        for model in models.values():
            model.to(self.device)
        
        return models
    
    @torch.no_grad()
    def extract_layerwise_representations(
        self,
        model: torch.nn.Module,
        num_samples: int = 500,
        layers_to_extract: List[int] = None
    ) -> Dict[int, np.ndarray]:
        """Extract hidden representations at specified layers"""
        model.eval()
        
        if layers_to_extract is None:
            layers_to_extract = [0, 3, 6, 9, 11]  # Key layers
        
        layer_representations = {l: [] for l in layers_to_extract}
        
        test_loader = self.data_module.test_dataloader()
        samples_collected = 0
        
        for batch in test_loader:
            if samples_collected >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = model(
                input_ids,
                attention_mask,
                return_hidden_states=True
            )
            
            hidden_states_list = outputs.get('all_hidden_states', [])
            
            for layer_idx in layers_to_extract:
                if layer_idx < len(hidden_states_list):
                    # Take mean over sequence dimension
                    h = hidden_states_list[layer_idx].mean(dim=1).cpu().numpy()
                    layer_representations[layer_idx].append(h)
            
            samples_collected += input_ids.size(0)
        
        # Concatenate
        for layer_idx in layer_representations:
            if layer_representations[layer_idx]:
                layer_representations[layer_idx] = np.concatenate(
                    layer_representations[layer_idx], axis=0
                )[:num_samples]
        
        return layer_representations
    
    @torch.no_grad()
    def extract_evidence_divergence(
        self,
        model: torch.nn.Module,
        num_samples: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract representations for full history and ablated history
        to visualize evidence-induced divergence
        """
        model.eval()
        
        full_reps = []
        ablated_reps = []
        divergences = []
        
        test_loader = self.data_module.test_dataloader()
        samples_collected = 0
        
        for batch in test_loader:
            if samples_collected >= num_samples:
                break
            
            if 'ablated_input_ids' not in batch:
                continue
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            ablated_ids = batch['ablated_input_ids'].to(self.device)
            ablated_mask = batch['ablated_attention_mask'].to(self.device)
            
            # Full history
            outputs_full = model(input_ids, attention_mask, return_hidden_states=True)
            h_full = outputs_full['last_hidden_state'].mean(dim=1).cpu().numpy()
            
            # Ablated history
            outputs_ablated = model(ablated_ids, ablated_mask, return_hidden_states=True)
            h_ablated = outputs_ablated['last_hidden_state'].mean(dim=1).cpu().numpy()
            
            # Compute divergence
            div = np.linalg.norm(h_full - h_ablated, axis=1)
            
            full_reps.append(h_full)
            ablated_reps.append(h_ablated)
            divergences.append(div)
            
            samples_collected += input_ids.size(0)
        
        return (
            np.concatenate(full_reps, axis=0)[:num_samples],
            np.concatenate(ablated_reps, axis=0)[:num_samples],
            np.concatenate(divergences, axis=0)[:num_samples]
        )
    
    def reduce_dimensions(
        self,
        X: np.ndarray,
        method: str = "pca",
        n_components: int = 2
    ) -> np.ndarray:
        """Reduce dimensionality for visualization"""
        
        if X.shape[0] < n_components:
            return X[:, :n_components] if X.shape[1] >= n_components else X
        
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=self.seed)
            return reducer.fit_transform(X)
        
        elif method == "tsne":
            perplexity = min(30, X.shape[0] - 1)
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=self.seed
            )
            return reducer.fit_transform(X)
        
        elif method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.seed
            )
            return reducer.fit_transform(X)
        
        else:
            # Fallback to PCA
            return PCA(n_components=n_components).fit_transform(X)
    
    def run(self, num_samples: int = 500) -> Dict:
        """Run visualization experiment"""
        
        print("=" * 60)
        print("Experiment 6: Visualization of Layerwise Collapse")
        print("=" * 60)
        
        models = self.create_models()
        
        results = {
            'layerwise_representations': {},
            'evidence_divergence': {}
        }
        
        for model_name, model in models.items():
            print(f"\nExtracting representations for {model_name}...")
            
            # Layerwise representations
            layer_reps = self.extract_layerwise_representations(
                model, num_samples=num_samples
            )
            results['layerwise_representations'][model_name] = layer_reps
            
            # Evidence divergence
            full_reps, ablated_reps, divergences = self.extract_evidence_divergence(
                model, num_samples=min(200, num_samples)
            )
            results['evidence_divergence'][model_name] = {
                'full': full_reps,
                'ablated': ablated_reps,
                'divergence': divergences
            }
        
        self.representations = results
        return results
    
    def plot_layerwise_collapse(self, save_dir: Path = FIGURES_DIR):
        """Visualize intent cluster contraction across layers"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        models = list(self.representations['layerwise_representations'].keys())
        layers = [0, 3, 6, 9]
        
        # Create figure with subplots for each model and layer
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(len(models), len(layers), figure=fig)
        
        for row, model_name in enumerate(models):
            layer_reps = self.representations['layerwise_representations'][model_name]
            
            for col, layer_idx in enumerate(layers):
                ax = fig.add_subplot(gs[row, col])
                
                if layer_idx not in layer_reps or len(layer_reps[layer_idx]) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    continue
                
                X = layer_reps[layer_idx]
                X_2d = self.reduce_dimensions(X, method="pca")
                
                # Cluster for coloring
                n_clusters = min(5, X.shape[0])
                if X.shape[0] > n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
                    clusters = kmeans.fit_predict(X)
                else:
                    clusters = np.zeros(X.shape[0])
                
                scatter = ax.scatter(
                    X_2d[:, 0], X_2d[:, 1],
                    c=clusters, cmap='tab10',
                    alpha=0.6, s=20
                )
                
                # Compute cluster spread
                R_metrics = self.collapse_metrics.compute_reasoning_collapse_score(
                    torch.from_numpy(X).float()
                )
                R_score = R_metrics['reasoning_collapse_score']
                
                ax.set_title(f"Layer {layer_idx}\nR={R_score:.3f}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                
                if col == 0:
                    ax.set_ylabel(model_name, fontsize=12, fontweight='bold')
        
        fig.suptitle('Layerwise Intent Cluster Visualization (PCA)', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f'exp6_layerwise_clusters_{self.dataset_name}.pdf', dpi=300)
        plt.savefig(save_dir / f'exp6_layerwise_clusters_{self.dataset_name}.png', dpi=300)
        plt.close()
    
    def plot_evidence_divergence(self, save_dir: Path = FIGURES_DIR):
        """Visualize evidence-induced representation divergence"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        models = list(self.representations['evidence_divergence'].keys())
        
        fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
        if len(models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(models):
            ax = axes[idx]
            data = self.representations['evidence_divergence'][model_name]
            
            full_reps = data['full']
            ablated_reps = data['ablated']
            divergences = data['divergence']
            
            # Combine and reduce
            combined = np.vstack([full_reps, ablated_reps])
            combined_2d = self.reduce_dimensions(combined, method="pca")
            
            n = full_reps.shape[0]
            full_2d = combined_2d[:n]
            ablated_2d = combined_2d[n:]
            
            # Plot
            ax.scatter(full_2d[:, 0], full_2d[:, 1], c='blue', alpha=0.5, 
                      label='Full History', s=30)
            ax.scatter(ablated_2d[:, 0], ablated_2d[:, 1], c='red', alpha=0.5,
                      label='Ablated History', s=30)
            
            # Draw lines connecting pairs
            for i in range(min(50, n)):  # Limit lines for clarity
                ax.plot(
                    [full_2d[i, 0], ablated_2d[i, 0]],
                    [full_2d[i, 1], ablated_2d[i, 1]],
                    'gray', alpha=0.2, linewidth=0.5
                )
            
            avg_div = np.mean(divergences)
            ax.set_title(f"{model_name}\nAvg Divergence: {avg_div:.3f}")
            ax.legend(fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.suptitle('Evidence-Induced Representation Divergence', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f'exp6_evidence_divergence_{self.dataset_name}.pdf', dpi=300)
        plt.savefig(save_dir / f'exp6_evidence_divergence_{self.dataset_name}.png', dpi=300)
        plt.close()
    
    def plot_collapse_progression(self, save_dir: Path = FIGURES_DIR):
        """Plot R score progression across layers"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = {'Full-LLM': 'blue', 'Fixed-Mid': 'red', 'CARR': 'green'}
        
        for model_name, layer_reps in self.representations['layerwise_representations'].items():
            layers = sorted(layer_reps.keys())
            R_scores = []
            
            for layer_idx in layers:
                X = layer_reps[layer_idx]
                if len(X) > 0:
                    R_metrics = self.collapse_metrics.compute_reasoning_collapse_score(
                        torch.from_numpy(X).float()
                    )
                    R_scores.append(R_metrics['reasoning_collapse_score'])
                else:
                    R_scores.append(np.nan)
            
            ax.plot(
                layers, R_scores,
                'o-', linewidth=2, markersize=8,
                label=model_name,
                color=colors.get(model_name, 'gray')
            )
        
        # Mark compression point
        ax.axvline(x=6, color='red', linestyle='--', alpha=0.5, label='Compression (k=6)')
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Reasoning Collapse Score R(l)', fontsize=12)
        ax.set_title('Progressive Reasoning Collapse Across Layers', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'exp6_collapse_progression_{self.dataset_name}.pdf', dpi=300)
        plt.savefig(save_dir / f'exp6_collapse_progression_{self.dataset_name}.png', dpi=300)
        plt.close()
    
    def plot_all(self, save_dir: Path = FIGURES_DIR):
        """Generate all visualizations"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating visualizations...")
        
        self.plot_layerwise_collapse(save_dir)
        print("  - Layerwise collapse visualization created")
        
        self.plot_evidence_divergence(save_dir)
        print("  - Evidence divergence visualization created")
        
        self.plot_collapse_progression(save_dir)
        print("  - Collapse progression plot created")
        
        print(f"All plots saved to {save_dir}")

    # =========================================================================
    # R11: Cluster Statistics Table
    # =========================================================================
    def compute_cluster_statistics(
        self,
        num_samples: int = 500,
        num_clusters: int = 5,
        layers_to_analyse: Optional[List[int]] = None,
    ) -> "pd.DataFrame":
        """
        Compute and save a cluster statistics table for key layers.

        For each model × layer combination, report:
          - Number of active clusters (non-empty after K-means)
          - Mean intra-cluster distance (compactness)
          - Inter-cluster distance (mean pairwise centroid distance)
          - Silhouette score (cluster quality in [-1, 1])

        Results are saved as:
          tables/exp6_cluster_statistics.{csv,tex}
        """
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import euclidean_distances

        if layers_to_analyse is None:
            layers_to_analyse = [0, 3, 6, 9, 11]

        rows = []

        for model_name, layer_reps in self.representations.items():
            for layer_idx in layers_to_analyse:
                if layer_idx not in layer_reps:
                    continue

                reps = layer_reps[layer_idx]    # (N, d)
                N    = reps.shape[0]
                if N < num_clusters + 1:
                    continue

                kmeans  = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                labels  = kmeans.fit_predict(reps)
                centers = kmeans.cluster_centers_          # (K, d)

                # Active clusters
                active = int(len(np.unique(labels)))

                # Mean intra-cluster distance
                intra_dists = []
                for k in range(num_clusters):
                    mask = labels == k
                    if mask.sum() > 1:
                        pts = reps[mask]
                        c   = centers[k]
                        intra_dists.append(float(np.mean(np.linalg.norm(pts - c, axis=1))))
                mean_intra = float(np.mean(intra_dists)) if intra_dists else float("nan")

                # Mean inter-cluster distance (pairwise centroid)
                pairwise = euclidean_distances(centers)
                mask_upper = np.triu(np.ones_like(pairwise, dtype=bool), k=1)
                mean_inter = float(pairwise[mask_upper].mean()) if mask_upper.any() else float("nan")

                # Silhouette score (subsample for speed)
                sil_idx = np.random.choice(N, size=min(1000, N), replace=False)
                try:
                    sil = float(silhouette_score(reps[sil_idx], labels[sil_idx]))
                except Exception:
                    sil = float("nan")

                rows.append({
                    "Model":            model_name,
                    "Layer":            layer_idx,
                    "Active Clusters":  active,
                    "Intra-Dist":       f"{mean_intra:.4f}",
                    "Inter-Dist":       f"{mean_inter:.4f}",
                    "Silhouette Score": f"{sil:.4f}",
                })

        import pandas as pd
        df = pd.DataFrame(rows)

        from config import TABLES_DIR
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TABLES_DIR / "exp6_cluster_statistics.csv", index=False)
        with open(TABLES_DIR / "exp6_cluster_statistics.tex", "w") as f:
            f.write(df.to_latex(index=False, escape=True))
        print(f"[Exp6] Cluster statistics table → {TABLES_DIR / 'exp6_cluster_statistics.csv'}")
        return df

    def plot_all(self, save_dir: Path = FIGURES_DIR):
        """Generate all visualizations including R11 cluster statistics."""
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating visualizations...")

        self.plot_layerwise_collapse(save_dir)
        print("  - Layerwise collapse visualization created")

        self.plot_evidence_divergence(save_dir)
        print("  - Evidence divergence visualization created")

        self.plot_collapse_progression(save_dir)
        print("  - Collapse progression plot created")

        # R11: Cluster statistics table
        if self.representations:
            self.compute_cluster_statistics()
            print("  - Cluster statistics table created (R11)")

        print(f"All plots saved to {save_dir}")
    datasets: List[str] = ['ml-1m'],
    device: str = "cuda",
    num_samples: int = 500
):
    """Run Experiment 6"""
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)
        
        exp = Experiment6(dataset_name=dataset, device=device)
        results = exp.run(num_samples=num_samples)
        exp.plot_all()
        
        all_results[dataset] = results
    
    return all_results


if __name__ == "__main__":
    results = run_experiment_6(datasets=['ml-1m'], num_samples=300)
