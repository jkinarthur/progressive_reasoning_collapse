"""
Collapse Metrics for Progressive Reasoning Collapse (PRC) Analysis

Implements:
1. Reasoning-collapse score R(l) = tr(Σ_W) / tr(Σ_B)
2. Evidence-survival function S_l(τ)
3. Intent diversity metrics
4. Counterfactual stability measures
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


class CollapseMetricsComputer:
    """
    Compute collapse-sensitive metrics for analyzing LLM-based recommendation
    
    Key metrics:
    - Reasoning-collapse score (R): Measures multi-intent geometry preservation
    - Evidence-survival score (S): Measures sensitivity to informative history
    - Intent diversity: Number/entropy of active intent clusters
    - Counterfactual stability: Divergence under history ablation
    """
    
    def __init__(
        self,
        num_clusters: int = 5,
        divergence_type: str = "js",
        device: str = "cuda"
    ):
        self.num_clusters = num_clusters
        self.divergence_type = divergence_type
        self.device = device
    
    def compute_reasoning_collapse_score(
        self,
        hidden_states: Tensor,
        cluster_labels: Optional[Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute the reasoning-collapse score R(l) = tr(Σ_W) / tr(Σ_B)
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim) or (N, hidden_dim)
            cluster_labels: Pre-computed cluster assignments (optional)
        
        Returns:
            Dictionary with R score and component traces
        """
        # Flatten if needed
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        
        hidden_np = hidden_states.detach().cpu().numpy()
        N, d = hidden_np.shape
        
        # Cluster if labels not provided
        if cluster_labels is None:
            if N < self.num_clusters:
                # Not enough points for clustering
                return {
                    'reasoning_collapse_score': float('nan'),
                    'trace_sigma_w': 0.0,
                    'trace_sigma_b': 0.0,
                    'num_active_clusters': 1
                }
            
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(hidden_np)
        else:
            cluster_labels = cluster_labels.cpu().numpy() if isinstance(cluster_labels, Tensor) else cluster_labels
        
        # Compute global mean
        global_mean = hidden_np.mean(axis=0)
        
        # Compute within-cluster and between-cluster covariances
        sigma_w = np.zeros((d, d))
        sigma_b = np.zeros((d, d))
        
        active_clusters = 0
        
        for k in range(self.num_clusters):
            mask = (cluster_labels == k)
            n_k = mask.sum()
            
            if n_k > 0:
                active_clusters += 1
                cluster_points = hidden_np[mask]
                cluster_mean = cluster_points.mean(axis=0)
                
                # Within-cluster covariance
                centered = cluster_points - cluster_mean
                sigma_w += (centered.T @ centered)
                
                # Between-cluster covariance
                mean_diff = cluster_mean - global_mean
                sigma_b += n_k * np.outer(mean_diff, mean_diff)
        
        sigma_w /= N
        sigma_b /= N
        
        trace_w = np.trace(sigma_w)
        trace_b = np.trace(sigma_b) + 1e-10  # Avoid division by zero
        
        R = trace_w / trace_b
        
        return {
            'reasoning_collapse_score': float(R),
            'trace_sigma_w': float(trace_w),
            'trace_sigma_b': float(trace_b),
            'num_active_clusters': active_clusters
        }
    
    def compute_evidence_survival(
        self,
        prob_full: Tensor,
        prob_ablated: Tensor,
        divergence_type: Optional[str] = None
    ) -> Dict[str, Tensor]:
        """
        Compute evidence-survival score S_l(τ) = D(p(·|H), p(·|H\τ))
        
        Args:
            prob_full: Probability distribution under full history (batch, num_items)
            prob_ablated: Probability distribution with τ removed (batch, num_items)
            divergence_type: "js", "tv", "kl", or "hellinger"
        
        Returns:
            Dictionary with evidence survival scores
        """
        if divergence_type is None:
            divergence_type = self.divergence_type
        
        # Ensure probabilities are valid
        prob_full = prob_full.clamp(min=1e-10)
        prob_ablated = prob_ablated.clamp(min=1e-10)
        
        # Normalize
        prob_full = prob_full / prob_full.sum(dim=-1, keepdim=True)
        prob_ablated = prob_ablated / prob_ablated.sum(dim=-1, keepdim=True)
        
        if divergence_type == "js":
            # Jensen-Shannon divergence (symmetric)
            m = 0.5 * (prob_full + prob_ablated)
            kl_pm = (prob_full * (prob_full.log() - m.log())).sum(dim=-1)
            kl_qm = (prob_ablated * (prob_ablated.log() - m.log())).sum(dim=-1)
            divergence = 0.5 * (kl_pm + kl_qm)
            
        elif divergence_type == "tv":
            # Total variation distance
            divergence = 0.5 * (prob_full - prob_ablated).abs().sum(dim=-1)
            
        elif divergence_type == "kl":
            # KL divergence (asymmetric)
            divergence = (prob_full * (prob_full.log() - prob_ablated.log())).sum(dim=-1)
            
        elif divergence_type == "hellinger":
            # Hellinger distance
            divergence = (1 / np.sqrt(2)) * ((prob_full.sqrt() - prob_ablated.sqrt())**2).sum(dim=-1).sqrt()
        
        else:
            raise ValueError(f"Unknown divergence type: {divergence_type}")
        
        return {
            'evidence_survival': divergence,
            f'{divergence_type}_divergence': divergence
        }
    
    def compute_intent_diversity(
        self,
        hidden_states: Tensor,
        return_clusters: bool = False
    ) -> Dict[str, float]:
        """
        Compute intent diversity metrics
        
        Measures:
        - Number of active intent clusters
        - Entropy of cluster assignments
        - Silhouette score (cluster quality)
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        
        hidden_np = hidden_states.detach().cpu().numpy()
        N, d = hidden_np.shape
        
        if N < self.num_clusters:
            return {
                'num_active_clusters': 1,
                'cluster_entropy': 0.0,
                'silhouette_score': 0.0
            }
        
        # Cluster
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(hidden_np)
        
        # Count active clusters
        unique_clusters = np.unique(cluster_labels)
        num_active = len(unique_clusters)
        
        # Cluster distribution entropy
        cluster_counts = np.bincount(cluster_labels, minlength=self.num_clusters)
        cluster_probs = cluster_counts / cluster_counts.sum()
        cluster_entropy = entropy(cluster_probs + 1e-10)
        
        # Silhouette score
        if num_active > 1 and N > self.num_clusters:
            sil_score = silhouette_score(hidden_np, cluster_labels)
        else:
            sil_score = 0.0
        
        result = {
            'num_active_clusters': num_active,
            'cluster_entropy': float(cluster_entropy),
            'silhouette_score': float(sil_score),
            'max_possible_entropy': float(np.log(self.num_clusters))
        }
        
        if return_clusters:
            result['cluster_labels'] = cluster_labels
            result['cluster_centers'] = kmeans.cluster_centers_
        
        return result
    
    def compute_counterfactual_stability(
        self,
        model,
        input_ids: Tensor,
        attention_mask: Tensor,
        ablated_input_ids: Tensor,
        ablated_attention_mask: Tensor,
        compression_depth: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute counterfactual stability - how much recommendations change
        when informative subsequence is removed
        """
        model.eval()
        
        with torch.no_grad():
            # Get predictions for full history
            outputs_full = model(
                input_ids,
                attention_mask,
                compression_depth=compression_depth
            )
            logits_full = outputs_full['logits'][:, -1, :]
            probs_full = F.softmax(logits_full, dim=-1)
            
            # Get predictions for ablated history
            outputs_ablated = model(
                ablated_input_ids,
                ablated_attention_mask,
                compression_depth=compression_depth
            )
            logits_ablated = outputs_ablated['logits'][:, -1, :]
            probs_ablated = F.softmax(logits_ablated, dim=-1)
        
        # Compute evidence survival metrics
        survival_metrics = self.compute_evidence_survival(probs_full, probs_ablated)
        
        # Additional stability metrics
        # Top-K overlap
        k_values = [5, 10, 20]
        overlaps = {}
        
        for k in k_values:
            topk_full = probs_full.topk(k, dim=-1).indices
            topk_ablated = probs_ablated.topk(k, dim=-1).indices
            
            # Compute overlap for each sample
            overlap_scores = []
            for i in range(topk_full.size(0)):
                set_full = set(topk_full[i].cpu().numpy())
                set_ablated = set(topk_ablated[i].cpu().numpy())
                overlap = len(set_full & set_ablated) / k
                overlap_scores.append(overlap)
            
            overlaps[f'topk_{k}_overlap'] = np.mean(overlap_scores)
        
        # Rank correlation
        ranks_full = probs_full.argsort(dim=-1, descending=True).argsort(dim=-1)
        ranks_ablated = probs_ablated.argsort(dim=-1, descending=True).argsort(dim=-1)
        
        # Use top items for rank correlation
        top_items = probs_full.topk(100, dim=-1).indices
        
        result = {
            'evidence_survival': survival_metrics['evidence_survival'].mean().item(),
            **overlaps
        }
        
        return result


class LayerwiseCollapseAnalyzer:
    """
    Analyze collapse progression across layers
    """
    
    def __init__(
        self,
        model,
        metrics_computer: CollapseMetricsComputer
    ):
        self.model = model
        self.metrics = metrics_computer
    
    @torch.no_grad()
    def analyze_layerwise_collapse(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        compression_depth: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Analyze reasoning collapse score across all layers
        
        Returns collapse metrics for each layer to verify monotonic decay
        """
        self.model.eval()
        
        # Get all hidden states
        outputs = self.model(
            input_ids,
            attention_mask,
            compression_depth=compression_depth,
            return_hidden_states=True,
            return_collapse_metrics=True
        )
        
        all_hidden_states = outputs.get('all_hidden_states', [])
        
        layerwise_R = []
        layerwise_trace_w = []
        layerwise_trace_b = []
        layerwise_diversity = []
        
        for layer_idx, hidden_states in enumerate(all_hidden_states):
            # Compute reasoning collapse score
            R_metrics = self.metrics.compute_reasoning_collapse_score(hidden_states)
            layerwise_R.append(R_metrics['reasoning_collapse_score'])
            layerwise_trace_w.append(R_metrics['trace_sigma_w'])
            layerwise_trace_b.append(R_metrics['trace_sigma_b'])
            
            # Compute intent diversity
            div_metrics = self.metrics.compute_intent_diversity(hidden_states)
            layerwise_diversity.append(div_metrics['cluster_entropy'])
        
        return {
            'layerwise_R': layerwise_R,
            'layerwise_trace_w': layerwise_trace_w,
            'layerwise_trace_b': layerwise_trace_b,
            'layerwise_diversity': layerwise_diversity,
            'compression_depth': compression_depth
        }
    
    @torch.no_grad()
    def analyze_evidence_survival_decay(
        self,
        model,
        input_ids_full: Tensor,
        input_ids_ablated: Tensor,
        attention_mask_full: Tensor,
        attention_mask_ablated: Tensor,
        compression_depths: List[Optional[int]] = None
    ) -> Dict[str, List[float]]:
        """
        Analyze evidence survival across different compression depths
        
        Validates Theorem 2: Evidence survival should decay after compression
        """
        if compression_depths is None:
            compression_depths = [None, 3, 6, 9]
        
        survival_by_depth = []
        
        for depth in compression_depths:
            # Get probabilities for full history
            outputs_full = model(
                input_ids_full,
                attention_mask_full,
                compression_depth=depth
            )
            probs_full = F.softmax(outputs_full['logits'][:, -1, :], dim=-1)
            
            # Get probabilities for ablated history
            outputs_ablated = model(
                input_ids_ablated,
                attention_mask_ablated,
                compression_depth=depth
            )
            probs_ablated = F.softmax(outputs_ablated['logits'][:, -1, :], dim=-1)
            
            # Compute evidence survival
            survival_metrics = self.metrics.compute_evidence_survival(probs_full, probs_ablated)
            survival_by_depth.append(survival_metrics['evidence_survival'].mean().item())
        
        return {
            'compression_depths': compression_depths,
            'evidence_survival': survival_by_depth
        }


class CriticalDepthFinder:
    """
    Find the critical compression depth k* that separates
    safe regime from collapse regime (Theorem 3)
    """
    
    def __init__(
        self,
        threshold_R: float = 0.1,
        threshold_S: float = 0.05
    ):
        self.threshold_R = threshold_R
        self.threshold_S = threshold_S
    
    def find_critical_depth(
        self,
        layerwise_R: List[float],
        layerwise_S: Optional[List[float]] = None
    ) -> Dict[str, int]:
        """
        Find critical compression depth
        
        Args:
            layerwise_R: Reasoning collapse scores by layer
            layerwise_S: Evidence survival scores by compression depth
        
        Returns:
            Critical depth and regime information
        """
        # Find first layer where R falls below threshold
        k_star_R = len(layerwise_R)  # Default: no compression needed
        
        for layer_idx, R in enumerate(layerwise_R):
            if R < self.threshold_R:
                k_star_R = layer_idx
                break
        
        # If evidence survival is provided
        if layerwise_S is not None:
            k_star_S = len(layerwise_S)
            for depth_idx, S in enumerate(layerwise_S):
                if S < self.threshold_S:
                    k_star_S = depth_idx
                    break
            
            # Critical depth is the minimum of both
            k_star = min(k_star_R, k_star_S)
        else:
            k_star = k_star_R
        
        return {
            'critical_depth': k_star,
            'critical_depth_R': k_star_R,
            'critical_depth_S': k_star_S if layerwise_S else None,
            'safe_depths': list(range(k_star, len(layerwise_R))),
            'collapse_depths': list(range(0, k_star))
        }


# =============================================================================
# Utility Functions
# =============================================================================
def compute_monotonicity_violation(values: List[float], expected_decrease: bool = True) -> float:
    """
    Compute how much a sequence violates expected monotonicity
    
    For Theorem 1/2 validation: R(l) and S_l should decrease monotonically
    """
    violations = 0
    total_pairs = 0
    
    for i in range(len(values) - 1):
        total_pairs += 1
        if expected_decrease:
            if values[i+1] > values[i]:  # Should decrease
                violations += 1
        else:
            if values[i+1] < values[i]:  # Should increase
                violations += 1
    
    return violations / max(total_pairs, 1)


def compute_exponential_decay_fit(
    values: List[float],
    layer_indices: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Fit exponential decay to validate theoretical bounds
    
    Theorems 1 & 2 predict: R(l) ≤ ρ^(l-k0) * R(k0)
    """
    import scipy.optimize as opt
    
    if layer_indices is None:
        layer_indices = list(range(len(values)))
    
    values = np.array(values)
    layers = np.array(layer_indices)
    
    # Filter valid values
    valid_mask = ~np.isnan(values) & (values > 0)
    if valid_mask.sum() < 2:
        return {'rho': float('nan'), 'initial': float('nan'), 'r_squared': 0.0}
    
    values = values[valid_mask]
    layers = layers[valid_mask]
    
    # Fit: y = A * rho^x
    try:
        def exp_decay(x, A, rho):
            return A * (rho ** x)
        
        popt, pcov = opt.curve_fit(
            exp_decay,
            layers - layers[0],
            values,
            p0=[values[0], 0.9],
            bounds=([0, 0], [np.inf, 1.0])
        )
        
        # R-squared
        predicted = exp_decay(layers - layers[0], *popt)
        ss_res = ((values - predicted) ** 2).sum()
        ss_tot = ((values - values.mean()) ** 2).sum()
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        
        return {
            'rho': float(popt[1]),
            'initial': float(popt[0]),
            'r_squared': float(r_squared)
        }
    except:
        return {'rho': float('nan'), 'initial': float('nan'), 'r_squared': 0.0}


if __name__ == "__main__":
    # Test metrics computation
    metrics = CollapseMetricsComputer(num_clusters=5)
    
    # Test data
    batch_size = 8
    seq_len = 50
    hidden_dim = 256
    num_items = 10000
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test reasoning collapse score
    R_metrics = metrics.compute_reasoning_collapse_score(hidden_states)
    print("Reasoning collapse score:", R_metrics)
    
    # Test evidence survival
    prob_full = F.softmax(torch.randn(batch_size, num_items), dim=-1)
    prob_ablated = F.softmax(torch.randn(batch_size, num_items), dim=-1)
    
    S_metrics = metrics.compute_evidence_survival(prob_full, prob_ablated)
    print("\nEvidence survival:", S_metrics['evidence_survival'].mean().item())
    
    # Test intent diversity
    div_metrics = metrics.compute_intent_diversity(hidden_states)
    print("\nIntent diversity:", div_metrics)
