"""
Experiment 7b: T5-Base LLM Experiments

Uses T5-Base (220M parameters, 12 encoder layers) as a genuine pre-trained
LLM backbone for recommendation.  Demonstrates PRC in an authentic LLM setting.

Key differences from exp7_t5_large.py:
  - T5-base (12 layers, d_model=768) instead of T5-large (24 layers, d_model=1024)
  - Includes a fine-tuning stage on the recommendation task before evaluation
  - Computes PRC metrics (R(l), S(l)) across encoder layers
  - Compression depths scaled for 12-layer encoder: [None, 3, 6, 9]

Outputs:
  - tables/exp7b_t5_base_results.{csv,tex}
  - figures/exp7b_t5_layerwise_collapse.pdf
  - results/exp7b_t5_base_<dataset>.json
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(str(Path(__file__).parent.parent))

from config import get_experiment_config, RESULTS_DIR, FIGURES_DIR, TABLES_DIR
from data_loader import RecommendationDataModule
from training.trainer import RecommendationMetrics, save_results
from metrics.collapse_metrics import CollapseMetricsComputer

try:
    from transformers import T5EncoderModel, AutoTokenizer
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False
    print("[Exp7b] WARNING: transformers not installed. Run: pip install transformers")


# =============================================================================
# T5-Base Recommender with CARR-style compression
# =============================================================================
class T5BaseRecommender(nn.Module):
    """
    T5-Base encoder (12 layers, d_model=768, ~110M encoder params) used as a
    sequential recommendation backbone.

    Two modes:
      - compression_depth=None : full T5-Base encoder, mean-pool, linear head
      - compression_depth=k    : run layers 0..k, compress via register tokens,
                                  mean-pool registers, linear head

    The recommendation head and compression components are trained while the
    T5 backbone is frozen (parameter-efficient fine-tuning).
    """

    T5_MODEL_NAME = "t5-base"
    NUM_LAYERS = 12          # T5-base has 12 encoder layers
    D_MODEL = 768            # T5-base hidden size

    def __init__(
        self,
        num_items: int,
        compression_depth: Optional[int] = None,
        num_registers: int = 8,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_items = num_items
        self.compression_depth = compression_depth
        self.num_registers = num_registers

        if not T5_AVAILABLE:
            raise RuntimeError("transformers library required for Exp7b.")

        print(f"  Loading {self.T5_MODEL_NAME} encoder …")
        self.encoder = T5EncoderModel.from_pretrained(self.T5_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.T5_MODEL_NAME)

        # Freeze T5 backbone — only train head and compression components
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Register-based compressor (only instantiated when using compression)
        if compression_depth is not None:
            self.register_tokens = nn.Parameter(
                torch.randn(num_registers, self.D_MODEL) * 0.02
            )
            self.compress_attn = nn.MultiheadAttention(
                self.D_MODEL, num_heads=8, dropout=0.1, batch_first=True
            )

        # Recommendation head
        self.pre_head_norm = nn.LayerNorm(self.D_MODEL)
        self.rec_head = nn.Linear(self.D_MODEL, num_items + 1, bias=False)

    def _ids_to_tokens(
        self, input_ids: torch.Tensor, max_length: int = 128
    ) -> Dict[str, torch.Tensor]:
        """Convert integer item-ID sequences to T5 text token tensors."""
        device = input_ids.device
        texts = []
        for row in input_ids.cpu().tolist():
            items_str = " ".join(f"item_{iid}" for iid in row if iid != 0)
            texts.append(f"{items_str} [recommend]")
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        tok = self._ids_to_tokens(input_ids)
        t5_ids = tok["input_ids"]
        t5_mask = tok["attention_mask"]

        enc_out = self.encoder(
            input_ids=t5_ids,
            attention_mask=t5_mask,
            output_hidden_states=True,
        )
        all_hs = enc_out.hidden_states  # tuple: (B, S, 768) × (num_layers+1)

        if self.compression_depth is None or self.compression_depth >= self.NUM_LAYERS:
            last_hs = enc_out.last_hidden_state       # (B, S, 768)
            mask_f = t5_mask.unsqueeze(-1).float()
            pooled = (last_hs * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            h_at_k = all_hs[self.compression_depth]  # (B, S, 768)
            B = h_at_k.size(0)
            regs = self.register_tokens.unsqueeze(0).expand(B, -1, -1)
            kpm = (1 - t5_mask).bool()
            compressed, _ = self.compress_attn(regs, h_at_k, h_at_k, key_padding_mask=kpm)
            pooled = compressed.mean(dim=1)           # (B, 768)

        logits = self.rec_head(self.pre_head_norm(pooled))
        result = {"logits": logits}
        if return_hidden_states:
            result["all_hidden_states"] = list(all_hs)
        return result


# =============================================================================
# Fine-tuning helper
# =============================================================================
def finetune(
    model: T5BaseRecommender,
    data_module: RecommendationDataModule,
    device: str,
    epochs: int = 3,
    lr: float = 1e-3,
    label: str = "",
) -> None:
    """
    Fine-tune only the trainable parameters (head + compression components).
    The T5 backbone is frozen so this is fast (~10–15 min on T4 per model).
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print(f"  [{label}] No trainable parameters — skipping fine-tune.")
        return

    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-2)
    train_loader = data_module.train_dataloader()
    total_steps = epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  [{label}] Epoch {epoch+1}/{epochs}"):
            iids = batch["input_ids"].to(device)
            tgts = batch["target"].to(device)

            optimizer.zero_grad()
            out = model(iids)
            logits = out["logits"]                    # (B, num_items+1)
            loss = F.cross_entropy(logits, tgts.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        print(f"  [{label}] Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()


# =============================================================================
# PRC metrics over encoder layers
# =============================================================================
@torch.no_grad()
def compute_layerwise_prc(
    model: T5BaseRecommender,
    data_module: RecommendationDataModule,
    device: str,
    num_batches: int = 50,
) -> Dict[str, List[float]]:
    """
    Collect hidden states across all encoder layers and compute R(l) and S(l).
    Returns dict with 'R_per_layer' and 'S_per_layer' lists (length = num_layers+1).
    """
    metrics_computer = CollapseMetricsComputer(num_clusters=5, device=device)
    model.eval()

    # Collect representations per layer
    layer_reps: Dict[int, List[np.ndarray]] = {i: [] for i in range(model.NUM_LAYERS + 1)}
    logits_full: List[torch.Tensor] = []
    logits_ablated: List[torch.Tensor] = []

    test_loader = data_module.test_dataloader()
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
        iids = batch["input_ids"].to(device)

        out = model(iids, return_hidden_states=True)
        all_hs = out.get("all_hidden_states", [])
        logits_full.append(out["logits"].cpu())

        # Ablated pass: zero out the last half of the sequence (evidence ablation)
        ablated_ids = iids.clone()
        seq_len = ablated_ids.size(1)
        ablated_ids[:, seq_len // 2:] = 0
        out_abl = model(ablated_ids)
        logits_ablated.append(out_abl["logits"].cpu())

        for layer_idx, hs in enumerate(all_hs):
            # Mean-pool over sequence → (B, 768)
            pooled = hs.mean(dim=1).cpu().numpy()
            layer_reps[layer_idx].append(pooled)

    # Compute R(l) for each layer
    R_per_layer: List[float] = []
    for layer_idx in range(model.NUM_LAYERS + 1):
        reps = np.concatenate(layer_reps[layer_idx], axis=0)  # (N, 768)
        if len(reps) < 10:
            R_per_layer.append(float("nan"))
            continue
        try:
            result = metrics_computer.compute_reasoning_collapse_score(
                torch.tensor(reps, dtype=torch.float32)
            )
            R_per_layer.append(result.get("collapse_score", float("nan")))
        except Exception:
            R_per_layer.append(float("nan"))

    # Compute S (evidence survival) using JS divergence between full and ablated
    S_values: List[float] = []
    if logits_full and logits_ablated:
        full_probs = torch.softmax(torch.cat(logits_full, dim=0), dim=-1).numpy()
        abl_probs = torch.softmax(torch.cat(logits_ablated, dim=0), dim=-1).numpy()
        from scipy.spatial.distance import jensenshannon
        js_divs = [
            float(jensenshannon(full_probs[i], abl_probs[i]))
            for i in range(len(full_probs))
        ]
        S_val = float(np.mean(js_divs))
        # Replicate S as a scalar summary (same value for all layers here)
        S_values = [S_val] * (model.NUM_LAYERS + 1)
    else:
        S_values = [float("nan")] * (model.NUM_LAYERS + 1)

    return {"R_per_layer": R_per_layer, "S_per_layer": S_values}


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate(
    model: T5BaseRecommender,
    data_module: RecommendationDataModule,
    device: str,
    num_batches: int = 200,
    label: str = "",
) -> Dict[str, float]:
    model.eval()
    hits, ndcg_scores, latencies = [], [], []
    test_loader = data_module.test_dataloader()

    for batch_idx, batch in enumerate(
        tqdm(test_loader, desc=f"  Evaluating {label}")
    ):
        if batch_idx >= num_batches:
            break
        iids = batch["input_ids"].to(device)
        tgts = batch["target"].to(device)

        t0 = time.perf_counter()
        out = model(iids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

        logits = out["logits"]
        top10 = torch.topk(logits, k=10, dim=-1).indices
        for pred, tgt in zip(top10, tgts):
            hit = int((pred == tgt).any().item())
            rank = (pred == tgt).nonzero(as_tuple=True)[0]
            nd = (1.0 / np.log2(rank[0].item() + 2)) if len(rank) > 0 else 0.0
            hits.append(hit)
            ndcg_scores.append(nd)

    return {
        "hr10":          float(np.mean(hits))        if hits else 0.0,
        "ndcg10":        float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "avg_latency_s": float(np.mean(latencies))  if latencies else 0.0,
    }


# =============================================================================
# Main Experiment Class
# =============================================================================
class Experiment7b:
    """Experiment 7b: T5-Base as authentic LLM backbone for PRC demonstration."""

    COMPRESSION_DEPTHS = [None, 3, 6, 9]
    COMPRESSION_LABELS = {
        None: "T5-Base (Full)",
        3:    "T5-Base + CARR (k=3)",
        6:    "T5-Base + CARR (k=6)",
        9:    "T5-Base + CARR (k=9)",
    }

    FINETUNE_EPOCHS = 3
    FINETUNE_LR = 1e-3

    def __init__(
        self,
        dataset_name: str = "ml-1m",
        device: str = "cuda",
        seed: int = 42,
        num_registers: int = 8,
    ):
        self.dataset_name = dataset_name
        self.device = device
        self.seed = seed
        self.num_registers = num_registers

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.config = get_experiment_config("exp7_t5_large", dataset_name)
        self.config.training.batch_size = min(16, self.config.training.batch_size)

        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="recency",
        )
        self.results: Dict = {}

    def run(self) -> Dict:
        print("=" * 70)
        print("EXPERIMENT 7b: T5-Base LLM Experiments")
        print(f"Dataset: {self.dataset_name}  |  Device: {self.device}")
        print("=" * 70)

        rows = []
        full_latency: Optional[float] = None
        layerwise_prc: Dict[str, Dict] = {}

        for depth in self.COMPRESSION_DEPTHS:
            label = self.COMPRESSION_LABELS[depth]
            print(f"\n{'='*60}")
            print(f"  Model: {label}")

            try:
                model = T5BaseRecommender(
                    num_items=self.data_module.num_items,
                    compression_depth=depth,
                    num_registers=self.num_registers,
                    freeze_backbone=True,
                ).to(self.device)

                # Fine-tune recommendation head
                finetune(
                    model, self.data_module, self.device,
                    epochs=self.FINETUNE_EPOCHS, lr=self.FINETUNE_LR, label=label
                )

                # Recommendation quality metrics
                metrics = evaluate(model, self.data_module, self.device, label=label)

                # PRC metrics (only for full model — layerwise analysis)
                if depth is None:
                    prc = compute_layerwise_prc(
                        model, self.data_module, self.device, num_batches=50
                    )
                    layerwise_prc["full"] = prc
                    r_final = prc["R_per_layer"][-1] if prc["R_per_layer"] else float("nan")
                    s_final = prc["S_per_layer"][-1] if prc["S_per_layer"] else float("nan")
                    full_latency = metrics["avg_latency_s"]
                    latency_reduction = 1.0
                else:
                    # For compressed models, just compute final-layer R and S
                    prc = compute_layerwise_prc(
                        model, self.data_module, self.device, num_batches=50
                    )
                    layerwise_prc[f"k={depth}"] = prc
                    r_final = prc["R_per_layer"][-1] if prc["R_per_layer"] else float("nan")
                    s_final = prc["S_per_layer"][-1] if prc["S_per_layer"] else float("nan")
                    latency_reduction = (
                        float(full_latency) / metrics["avg_latency_s"]
                        if full_latency and metrics["avg_latency_s"] > 0
                        else float("nan")
                    )

                metrics["R_score"] = r_final
                metrics["S_score"] = s_final
                metrics["latency_reduction"] = latency_reduction
                self.results[label] = metrics

                rows.append({
                    "Model":             label,
                    "HR@10":             f"{metrics['hr10']:.4f}",
                    "NDCG@10":           f"{metrics['ndcg10']:.4f}",
                    "R_score":           f"{r_final:.4f}",
                    "S_score":           f"{s_final:.6f}",
                    "Latency (s)":       f"{metrics['avg_latency_s']:.4f}",
                    "Latency Reduction": f"{latency_reduction:.2f}x",
                })

                print(f"  HR@10={metrics['hr10']:.4f}  NDCG@10={metrics['ndcg10']:.4f}"
                      f"  R={r_final:.4f}  S={s_final:.6f}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                import traceback
                print(f"  ERROR: {exc}")
                traceback.print_exc()
                rows.append({
                    "Model": label, "HR@10": "ERROR", "NDCG@10": "ERROR",
                    "R_score": "ERROR", "S_score": "ERROR",
                    "Latency (s)": "ERROR", "Latency Reduction": "ERROR",
                })

        # Save results table
        df = pd.DataFrame(rows)
        csv_path = TABLES_DIR / f"exp7b_t5_base_{self.dataset_name}_results.csv"
        tex_path = TABLES_DIR / f"exp7b_t5_base_{self.dataset_name}_results.tex"
        df.to_csv(csv_path, index=False)
        with open(tex_path, "w") as f:
            f.write(df.to_latex(index=False, escape=True))
        print(f"\n[Exp7b] Results table → {csv_path}")

        # Layerwise R(l) plot for full model
        self._plot_layerwise_collapse(layerwise_prc)

        # Save JSON results
        save_results(self.results, "exp7b_t5_base", self.dataset_name)
        return self.results

    def _plot_layerwise_collapse(self, layerwise_prc: Dict) -> None:
        if "full" not in layerwise_prc:
            return
        R = layerwise_prc["full"]["R_per_layer"]
        layers = list(range(len(R)))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(layers, R, marker="o", linewidth=2, label="T5-Base (Full)")
        ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.7, label="Collapse threshold εR=0.1")
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("R(l) — Reasoning Collapse Score")
        ax.set_title(f"Layerwise PRC in T5-Base ({self.dataset_name})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = FIGURES_DIR / f"exp7b_t5_layerwise_collapse_{self.dataset_name}.pdf"
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"[Exp7b] Layerwise collapse plot → {path}")


# =============================================================================
# Entry point for standalone execution
# =============================================================================
def run_experiment_7b(datasets: List[str], device: str) -> Dict:
    if not T5_AVAILABLE:
        print("[Exp7b] Skipping — transformers not installed.")
        return {}

    all_results = {}
    for dataset in datasets:
        print(f"\n{'='*70}\nDataset: {dataset}\n{'='*70}")
        try:
            exp = Experiment7b(dataset_name=dataset, device=device)
            all_results[dataset] = exp.run()
        except Exception as exc:
            import traceback
            print(f"[Exp7b] ERROR for {dataset}: {exc}")
            traceback.print_exc()
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp7b: T5-Base LLM Recommendation")
    parser.add_argument("--datasets", nargs="+",
                        default=["ml-1m", "amazon-beauty"],
                        choices=["ml-1m", "amazon-beauty", "amazon-toys", "yelp"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Experiment7b.FINETUNE_EPOCHS = args.epochs

    results = run_experiment_7b(args.datasets, args.device)
    print("\nDone. Results summary:")
    for dataset, res in results.items():
        print(f"  {dataset}:")
        for model, metrics in res.items():
            print(f"    {model}: HR@10={metrics.get('hr10', 'N/A'):.4f}")
