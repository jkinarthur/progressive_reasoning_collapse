"""
Experiment 1b: Validation of Latent Intent Structure

Addresses Reviewer Comments R1 and R2:

R1 — Cluster Semantic Alignment + Intent Perturbation Test:
  - Cluster latent representations at the final transformer layer.
  - Compare cluster assignments to ground-truth item-category labels
    using Purity, Normalized Mutual Information (NMI), and Adjusted
    Rand Index (ARI).
  - Remove all tokens whose representation falls nearest to one intent
    cluster and measure resulting HR@10 / NDCG@10 degradation.

R2 — Jacobian Spectral Norm:
  - Wrap each transformer layer as an isolated function.
  - Estimate ||J_l||_2 via power iteration with automatic differentiation.
  - Verify contraction criterion: ||J_l||_2 < 1 across compressed layers,
    empirically justifying the linearized dynamics
      h^(l+1) = A_l h^(l) + b_l + epsilon.

Key Outputs:
  - tables/exp1b_cluster_alignment.{csv,tex}
  - tables/exp1b_intent_perturbation.{csv,tex}
  - figures/exp1b_jacobian_spectral_norm.pdf
  - results/exp1b_intent_validation_<dataset>.json
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config import get_experiment_config, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, ModelConfig
from data_loader import RecommendationDataModule
from models.carr_model import CARRModel, FixedCompressionModel, FullLLMModel
from training.trainer import RecommendationMetrics, save_results


# =============================================================================
# Utility: cluster purity
# =============================================================================
def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Fraction of samples correctly assigned to the dominant ground-truth label."""
    from scipy.stats import mode
    n = len(labels_true)
    purity = 0.0
    for k in np.unique(labels_pred):
        mask = labels_pred == k
        if mask.sum() > 0:
            dominant_count = mode(labels_true[mask], keepdims=False).count
            purity += dominant_count
    return float(purity / n)


# =============================================================================
# Utility: power-iteration spectral norm for a single layer
# =============================================================================
@torch.no_grad()
def _layer_spectral_norm_power_iter(
    layer: nn.Module,
    h_in: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_iters: int = 20,
    eps: float = 1e-3,
) -> float:
    """
    Estimate the spectral norm ||J_l||_2 of the Jacobian of `layer`
    at the point h_in using finite-difference power iteration.

    Algorithm:
        v_0 <- random unit vector (same shape as h_in)
        for i in range(num_iters):
            Jv  = (layer(h + eps*v) - layer(h)) / eps      [forward diff]
            u   = Jv / ||Jv||
            JTu via autograd (gradient of <layer(h), u> wrt h)
            v   = JTu / ||JTu||
        sigma = ||Jv||
    """
    h = h_in.detach()
    B, T, d = h.shape

    # Baseline output
    if isinstance(layer, type) and hasattr(layer, "forward"):
        def call_layer(x):
            out, _ = layer(x, attention_mask=attention_mask)
            return out
    else:
        def call_layer(x):
            out = layer(x, attention_mask=attention_mask)
            if isinstance(out, tuple):
                return out[0]
            return out

    h_out_base = call_layer(h).detach()

    v = torch.randn_like(h)
    v = v / (v.norm() + 1e-12)
    sigma = 0.0

    for _ in range(num_iters):
        # Forward difference: Jv
        h_perturbed = h + eps * v
        h_out_perturbed = call_layer(h_perturbed).detach()
        Jv = (h_out_perturbed - h_out_base) / eps        # (B, T, d)
        sigma = Jv.norm().item()
        if sigma < 1e-12:
            break
        u = Jv / sigma

        # Backward: J^T u = gradient of <layer(h), u> wrt h
        h_req = h.clone().requires_grad_(True)
        out_req = call_layer(h_req)
        loss = (out_req * u).sum()
        loss.backward()
        if h_req.grad is None:
            break
        JTu = h_req.grad.detach()
        v = JTu / (JTu.norm() + 1e-12)

    return sigma


# =============================================================================
# Main experiment class
# =============================================================================
class Experiment1b:
    """
    Experiment 1b: Latent Intent Validation (R1) + Jacobian Spectral Norm (R2)
    """

    def __init__(
        self,
        dataset_name: str = "ml-1m",
        device: str = "cuda",
        num_clusters: int = 5,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_clusters = num_clusters
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.config = get_experiment_config("exp1b_intent_validation", dataset_name)
        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="minority",
        )
        self.rec_metrics = RecommendationMetrics()

        self.results: Dict = {
            "cluster_alignment": {},
            "intent_perturbation": {},
            "jacobian_spectral_norm": {},
        }

    # =========================================================================
    # R1a: Cluster Semantic Alignment
    # =========================================================================
    @torch.no_grad()
    def run_cluster_semantic_alignment(
        self,
        model: nn.Module,
        model_name: str,
        layer_idx: int = -1,
        num_batches: int = 50,
    ) -> Dict[str, float]:
        """
        Cluster last-layer pooled representations and evaluate alignment
        against ground-truth item-category labels.

        Ground-truth categories are derived by quantile-binning item IDs
        (a monotone proxy for catalogue organisation).

        Returns dict with keys: purity, nmi, ari.
        """
        model.eval()
        all_pooled, all_categories = [], []

        test_loader = self.data_module.test_dataloader()
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc=f"[R1a] Cluster Alignment — {model_name}")
        ):
            if batch_idx >= num_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = model(input_ids, attention_mask, return_hidden_states=True)
            hs_list = outputs.get("all_hidden_states", [])
            if not hs_list:
                continue

            # Mean-pool over the sequence dimension
            hs = hs_list[layer_idx]                      # (B, T, d)
            pooled = hs.mean(dim=1).cpu().numpy()        # (B, d)
            all_pooled.append(pooled)

            # Target item IDs used as category proxies
            targets = batch.get("target", batch["input_ids"][:, -1])
            all_categories.append(targets.cpu().numpy())

        if not all_pooled:
            return {"purity": 0.0, "nmi": 0.0, "ari": 0.0}

        hidden_np = np.concatenate(all_pooled, axis=0)        # (N, d)
        cats_np = np.concatenate(all_categories, axis=0)      # (N,)

        # Bin item IDs into K buckets → ground-truth category labels
        bins = np.linspace(0, self.data_module.num_items, self.num_clusters + 1)
        true_labels = np.digitize(cats_np, bins) - 1
        true_labels = np.clip(true_labels, 0, self.num_clusters - 1)

        # K-means on latent representations → predicted cluster labels
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.seed, n_init=10)
        pred_labels = kmeans.fit_predict(hidden_np)

        purity = cluster_purity(true_labels, pred_labels)
        nmi    = normalized_mutual_info_score(true_labels, pred_labels)
        ari    = adjusted_rand_score(true_labels, pred_labels)

        return {"purity": float(purity), "nmi": float(nmi), "ari": float(ari)}

    # =========================================================================
    # R1b: Intent Perturbation Test
    # =========================================================================
    @torch.no_grad()
    def run_intent_perturbation_test(
        self,
        model: nn.Module,
        model_name: str,
        cluster_to_remove: int = 0,
        num_batches: int = 50,
    ) -> Dict[str, float]:
        """
        Mask out the 20% of tokens closest to cluster_to_remove in
        latent space and measure HR@10 / NDCG@10 degradation.

        Steps:
          1. Full-context baseline evaluation.
          2. Cluster last-layer token representations; identify cluster centre.
          3. Zero the attention mask for tokens nearest that centre.
          4. Re-evaluate with the perturbed mask.
          5. Report HR drop and NDCG drop.
        """
        model.eval()
        test_loader = self.data_module.test_dataloader()

        # ---------- Pass 1: baseline metrics + collect last-layer reps --------
        baseline_hits, baseline_ndcg = [], []
        saved_inputs: List[Dict] = []       # store inputs for Pass 2

        all_token_reps = []                 # (N_total, d)
        sample_split = []                   # (B_i,) per batch

        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc=f"[R1b] Perturbation Baseline — {model_name}")
        ):
            if batch_idx >= num_batches:
                break

            iids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            tgts = batch["target"].to(self.device)

            outputs = model(iids, mask, return_hidden_states=True)
            hs_list = outputs.get("all_hidden_states", [])

            # Collect token-level last-layer representations
            if hs_list:
                token_reps = hs_list[-1].cpu().numpy()   # (B, T, d)
                B, T, d = token_reps.shape
                all_token_reps.append(token_reps.reshape(B * T, d))
                sample_split.append(B)

            # Baseline recommendation quality
            logits = outputs.get("logits", outputs.get("scores"))
            if logits is not None:
                lgt = logits[:, -1, :] if logits.dim() == 3 else logits
                top10 = torch.topk(lgt, k=10, dim=-1).indices
                for pred, tgt in zip(top10, tgts):
                    hit  = int((pred == tgt).any().item())
                    rank = (pred == tgt).nonzero(as_tuple=True)[0]
                    ndcg = (1.0 / np.log2(rank[0].item() + 2)) if len(rank) > 0 else 0.0
                    baseline_hits.append(hit)
                    baseline_ndcg.append(ndcg)

            # Save for Pass 2
            saved_inputs.append({
                "input_ids": iids.cpu(),
                "attention_mask": mask.cpu(),
                "target": tgts.cpu(),
            })

        if not all_token_reps:
            return {
                "baseline_hr10": 0.0, "perturbed_hr10": 0.0, "hr_drop": 0.0,
                "baseline_ndcg10": 0.0, "perturbed_ndcg10": 0.0, "ndcg_drop": 0.0,
            }

        # Cluster all token representations globally
        token_reps_all = np.concatenate(all_token_reps, axis=0)   # (N_total, d)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.seed, n_init=10)
        kmeans.fit(token_reps_all)
        center = torch.tensor(
            kmeans.cluster_centers_[cluster_to_remove], dtype=torch.float32
        ).to(self.device)                                          # (d,)

        # ---------- Pass 2: perturbed evaluation ------------------------------
        perturbed_hits, perturbed_ndcg = [], []

        ptr = 0
        for saved in tqdm(saved_inputs, desc=f"[R1b] Perturbation Eval — {model_name}"):
            iids = saved["input_ids"].to(self.device)
            mask = saved["attention_mask"].to(self.device)
            tgts = saved["target"].to(self.device)
            B, T  = iids.shape

            # Recompute last-layer token reps for this batch
            with torch.no_grad():
                outputs_fresh = model(iids, mask, return_hidden_states=True)
            hs_list = outputs_fresh.get("all_hidden_states", [])
            if not hs_list:
                continue

            token_hs = hs_list[-1]                       # (B, T, d)
            # Squared distance of each token to the cluster centre
            dists = ((token_hs - center.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)  # (B, T)

            # Zero attention mask for the 20% of tokens closest to the centre
            threshold = torch.quantile(dists.view(B * T), 0.20)
            perturb_mask = mask.float()
            perturb_mask[dists <= threshold] = 0.0

            with torch.no_grad():
                perturbed_outputs = model(iids, perturb_mask.bool(), return_hidden_states=False)

            logits = perturbed_outputs.get("logits", perturbed_outputs.get("scores"))
            if logits is None:
                continue

            lgt = logits[:, -1, :] if logits.dim() == 3 else logits
            top10 = torch.topk(lgt, k=10, dim=-1).indices
            for pred, tgt in zip(top10, tgts):
                hit  = int((pred == tgt).any().item())
                rank = (pred == tgt).nonzero(as_tuple=True)[0]
                ndcg = (1.0 / np.log2(rank[0].item() + 2)) if len(rank) > 0 else 0.0
                perturbed_hits.append(hit)
                perturbed_ndcg.append(ndcg)

        baseline_hr  = float(np.mean(baseline_hits))  if baseline_hits  else 0.0
        baseline_nd  = float(np.mean(baseline_ndcg))  if baseline_ndcg  else 0.0
        perturbed_hr = float(np.mean(perturbed_hits)) if perturbed_hits else 0.0
        perturbed_nd = float(np.mean(perturbed_ndcg)) if perturbed_ndcg else 0.0

        return {
            "baseline_hr10":   baseline_hr,
            "perturbed_hr10":  perturbed_hr,
            "hr_drop":         baseline_hr - perturbed_hr,
            "baseline_ndcg10": baseline_nd,
            "perturbed_ndcg10": perturbed_nd,
            "ndcg_drop":        baseline_nd - perturbed_nd,
        }

    # =========================================================================
    # R2: Jacobian Spectral Norm
    # =========================================================================
    def run_jacobian_spectral_norm(
        self,
        model: nn.Module,
        model_name: str,
        compression_depth: int = 6,
        num_power_iters: int = 20,
        num_batches: int = 5,
    ) -> Dict[str, float]:
        """
        Estimate ||J_l||_2 for each transformer layer of `model`.

        For CARR / FullLLM models the layers are stored in model.layers
        (a nn.ModuleList of TransformerBlock instances).  For each layer l:

          layer_fn(h) = TransformerBlock_l(h, attention_mask)[0]

        Power iteration (finite-difference + autograd) is applied to estimate
        the largest singular value of the Jacobian at multiple input batches,
        then averaged.

        Returns: {layer_idx (str): average spectral norm}
        """
        # Retrieve the transformer layers (works for CARRModel, FullLLMModel,
        # FixedCompressionModel which all expose model.layers)
        if not hasattr(model, "layers"):
            print(f"  [R2] Model {model_name} has no .layers attribute — skipping.")
            return {}

        transformer_layers = model.layers          # nn.ModuleList

        model.eval()
        test_loader = self.data_module.test_dataloader()

        # Collect a few batches of input embeddings via forward hooks
        layer_input_cache: Dict[int, List[torch.Tensor]] = defaultdict(list)
        layer_mask_cache:  List[Optional[torch.Tensor]] = []
        batch_count = [0]

        def make_hook(layer_idx_: int):
            def hook(module, inp, out):
                if batch_count[0] < num_batches:
                    layer_input_cache[layer_idx_].append(inp[0].detach().cpu())
            return hook

        handles = [
            layer.register_forward_hook(make_hook(i))
            for i, layer in enumerate(transformer_layers)
        ]
        masks_seen: List[Optional[torch.Tensor]] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break
                iids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                masks_seen.append(mask.detach().cpu())
                _ = model(iids, mask, return_hidden_states=False)
                batch_count[0] += 1

        for h in handles:
            h.remove()

        # Power iteration per layer
        avg_norms: Dict[int, float] = {}
        num_layers = len(transformer_layers)

        for l_idx, layer in enumerate(
            tqdm(transformer_layers, desc=f"[R2] Jacobian norm — {model_name}")
        ):
            norms_for_layer = []
            cached_inputs = layer_input_cache.get(l_idx, [])

            for i, h_cpu in enumerate(cached_inputs):
                mask_cpu = masks_seen[i] if i < len(masks_seen) else None

                h = h_cpu.to(self.device)
                attn_mask = mask_cpu.to(self.device) if mask_cpu is not None else None

                # Build a self-contained layer function
                def layer_fn(x, _layer=layer, _mask=attn_mask):
                    out = _layer(x, attention_mask=_mask)
                    return out[0] if isinstance(out, tuple) else out

                try:
                    sigma = _layer_spectral_norm_power_iter(
                        layer_fn, h, attn_mask, num_iters=num_power_iters
                    )
                    norms_for_layer.append(sigma)
                except Exception as exc:
                    print(f"    Warning: spectral norm failed at layer {l_idx}: {exc}")

            if norms_for_layer:
                avg_norms[l_idx] = float(np.mean(norms_for_layer))

        return avg_norms

    # =========================================================================
    # Tables and plots
    # =========================================================================
    def _save_alignment_table(self, results: Dict) -> None:
        rows = [
            {
                "Model":  name,
                "Purity": f"{m.get('purity', 0):.4f}",
                "NMI":    f"{m.get('nmi', 0):.4f}",
                "ARI":    f"{m.get('ari', 0):.4f}",
            }
            for name, m in results.items()
        ]
        df = pd.DataFrame(rows)
        df.to_csv(TABLES_DIR / "exp1b_cluster_alignment.csv", index=False)
        with open(TABLES_DIR / "exp1b_cluster_alignment.tex", "w") as f:
            f.write(df.to_latex(index=False, escape=True))
        print(f"[Exp1b] Cluster alignment table → {TABLES_DIR / 'exp1b_cluster_alignment.csv'}")

    def _save_perturbation_table(self, results: Dict) -> None:
        rows = [
            {
                "Model":              name,
                "Baseline HR@10":     f"{m.get('baseline_hr10',   0):.4f}",
                "Perturbed HR@10":    f"{m.get('perturbed_hr10',  0):.4f}",
                "HR Drop":            f"{m.get('hr_drop',         0):.4f}",
                "Baseline NDCG@10":   f"{m.get('baseline_ndcg10', 0):.4f}",
                "Perturbed NDCG@10":  f"{m.get('perturbed_ndcg10',0):.4f}",
                "NDCG Drop":          f"{m.get('ndcg_drop',       0):.4f}",
            }
            for name, m in results.items()
        ]
        df = pd.DataFrame(rows)
        df.to_csv(TABLES_DIR / "exp1b_intent_perturbation.csv", index=False)
        with open(TABLES_DIR / "exp1b_intent_perturbation.tex", "w") as f:
            f.write(df.to_latex(index=False, escape=True))
        print(f"[Exp1b] Perturbation table → {TABLES_DIR / 'exp1b_intent_perturbation.csv'}")

    def _plot_jacobian_norms(
        self, all_results: Dict[str, Dict[int, float]], compression_depth: int
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        for model_name, layer_norms in all_results.items():
            if not layer_norms:
                continue
            layers = sorted(layer_norms.keys())
            norms  = [layer_norms[l] for l in layers]
            ax.plot(layers, norms, marker="o", label=model_name)

        ax.axvline(
            x=compression_depth, color="red", linestyle="--",
            label=f"Compression depth $k={compression_depth}$"
        )
        ax.axhline(
            y=1.0, color="grey", linestyle=":",
            label=r"$\|J_l\|_2 = 1$ (contraction boundary)"
        )
        ax.set_xlabel("Layer index $l$")
        ax.set_ylabel(r"$\|J_l\|_2$")
        ax.set_title("Jacobian Spectral Norm Across Transformer Layers")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        path = FIGURES_DIR / "exp1b_jacobian_spectral_norm.pdf"
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"[Exp1b] Jacobian plot → {path}")

    # =========================================================================
    # Main runner
    # =========================================================================
    def run(self) -> Dict:
        print("=" * 70)
        print("EXPERIMENT 1b: Latent Intent Validation + Jacobian Spectral Norm")
        print("=" * 70)

        num_items = self.data_module.num_items
        models = {
            "Full-LLM":         FullLLMModel(num_items, self.config.model),
            "Fixed-Mid (k=6)":  FixedCompressionModel(num_items, ModelConfig(compression_depth=6)),
            "CARR":             CARRModel(num_items, self.config.model),
        }
        for m in models.values():
            m.to(self.device)

        # --- R1a: Cluster Semantic Alignment ---
        print("\n[R1a] Cluster Semantic Alignment")
        alignment_results = {}
        for name, model in models.items():
            alignment_results[name] = self.run_cluster_semantic_alignment(model, name)
            print(f"  {name}: {alignment_results[name]}")
        self.results["cluster_alignment"] = alignment_results
        self._save_alignment_table(alignment_results)

        # --- R1b: Intent Perturbation Test ---
        print("\n[R1b] Intent Perturbation Test")
        perturbation_results = {}
        for name, model in models.items():
            perturbation_results[name] = self.run_intent_perturbation_test(model, name)
            print(f"  {name}: {perturbation_results[name]}")
        self.results["intent_perturbation"] = perturbation_results
        self._save_perturbation_table(perturbation_results)

        # --- R2: Jacobian Spectral Norm ---
        print("\n[R2] Jacobian Spectral Norm Estimation")
        jacobian_results = {}
        for name, model in models.items():
            jacobian_results[name] = self.run_jacobian_spectral_norm(
                model, name, compression_depth=6
            )
            if jacobian_results[name]:
                avg_norm = np.mean(list(jacobian_results[name].values()))
                print(f"  {name}: mean ||J_l||_2 = {avg_norm:.4f}")
        self.results["jacobian_spectral_norm"] = jacobian_results
        self._plot_jacobian_norms(jacobian_results, compression_depth=6)

        save_results(self.results, "exp1b_intent_validation", self.dataset_name)
        return self.results


# =============================================================================
# Entry point for main.py
# =============================================================================
def run_experiment_1b(datasets: List[str], device: str) -> Dict:
    all_results = {}
    for dataset in datasets:
        print(f"\n--- Dataset: {dataset} ---")
        exp = Experiment1b(dataset_name=dataset, device=device)
        all_results[dataset] = exp.run()
    return all_results
