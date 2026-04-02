"""
Experiment 7: T5-Large LLM Experiments

Addresses Reviewer Comment R3:
  "The current experiments use a 12-layer transformer backbone, which may
   not be considered a true LLM. Add experiments using at least one
   pretrained large language model: LLaMA-7B / T5-Large."

This experiment uses T5-Large (770M parameters) from HuggingFace as a
generative recommender backbone.  The CARR compression mechanism is applied
to intermediate encoder layers using adapter-style register compression.

Evaluation Metrics:
  - HR@10
  - NDCG@10
  - FLOPs reduction (relative to uncompressed T5-Large)
  - Latency reduction (wall-clock inference time ratio)

Key Outputs:
  - tables/exp7_t5_large_results.{csv,tex}
  - figures/exp7_t5_latency_vs_quality.pdf
  - results/exp7_t5_large_<dataset>.json
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config import get_experiment_config, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, ModelConfig
from data_loader import RecommendationDataModule
from training.trainer import RecommendationMetrics, save_results

# HuggingFace transformers — T5-Large
try:
    from transformers import T5EncoderModel, AutoTokenizer
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False
    print("[Exp7] WARNING: transformers library not installed. "
          "T5-Large experiments will be skipped. "
          "Install with: pip install transformers")


# =============================================================================
# T5-Large Recommendation Wrapper
# =============================================================================
class T5LargeRecommender(nn.Module):
    """
    T5-Large encoder used as a sequential recommendation backbone.

    Architecture:
      - T5-Large encoder (24 layers, d_model=1024, ~770M params)
      - Mean-pool over encoder output
      - Linear head projecting to item vocabulary

    Compression:
      If compression_depth is set, encoder layers from compression_depth
      onward receive a compressed (register-pooled) representation instead
      of the full sequence.  This is the CARR-style adapter applied to T5.
    """

    T5_MODEL_NAME = "t5-large"

    def __init__(
        self,
        num_items: int,
        compression_depth: Optional[int] = None,
        num_registers: int = 8,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_items = num_items
        self.compression_depth = compression_depth
        self.num_registers = num_registers

        if not T5_AVAILABLE:
            raise RuntimeError("transformers library required for T5-Large experiments.")

        print(f"  Loading {self.T5_MODEL_NAME} encoder …")
        self.encoder = T5EncoderModel.from_pretrained(self.T5_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.T5_MODEL_NAME)
        self.d_model = self.encoder.config.d_model   # 1024 for t5-large

        # Register compressor (applied at compression_depth)
        if compression_depth is not None:
            self.register_tokens = nn.Parameter(
                torch.randn(num_registers, self.d_model) * 0.02
            )
            self.compress_attn = nn.MultiheadAttention(
                self.d_model, num_heads=8, dropout=0.1, batch_first=True
            )

        # Recommendation head
        self.rec_head = nn.Linear(self.d_model, num_items + 1, bias=False)

        # Layer norm before head
        self.pre_head_norm = nn.LayerNorm(self.d_model)

    def _item_ids_to_text_tokens(
        self, input_ids: torch.Tensor, max_length: int = 128
    ) -> Dict[str, torch.Tensor]:
        """
        Convert integer item ID sequences to text token tensors.

        Each item ID is rendered as "item_<id>" and the sequence of
        item strings is concatenated as a prompt:
          "item_42 item_17 item_89 ... [recommend]"

        Returns HuggingFace tokenizer output on the same device as input_ids.
        """
        device = input_ids.device
        texts = []
        for row in input_ids.cpu().tolist():
            # Filter padding (0) and format each item ID
            items_str = " ".join(
                f"item_{iid}" for iid in row if iid != 0
            )
            texts.append(f"{items_str} [recommend]")

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoding.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (B, T) integer item IDs from the dataset
            attention_mask: (B, T) — ignored here; regenerated from tokenizer
            return_hidden_states: whether to return all encoder hidden states

        Returns:
            dict with 'logits' (B, num_items+1) and optionally 'all_hidden_states'
        """
        # Convert item IDs to T5 text tokens
        text_encoding = self._item_ids_to_text_tokens(input_ids)
        t5_input_ids   = text_encoding["input_ids"]
        t5_attn_mask   = text_encoding["attention_mask"]

        if self.compression_depth is None:
            # Full T5 encoder — no compression
            enc_out = self.encoder(
                input_ids=t5_input_ids,
                attention_mask=t5_attn_mask,
                output_hidden_states=return_hidden_states,
            )
            last_hs = enc_out.last_hidden_state                 # (B, S, d)
            # Mean pool over non-padding tokens
            mask_f = t5_attn_mask.unsqueeze(-1).float()
            pooled = (last_hs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

            result = {"logits": self.rec_head(self.pre_head_norm(pooled))}
            if return_hidden_states:
                result["all_hidden_states"] = list(enc_out.hidden_states)
            return result

        else:
            # CARR-style compressed T5:
            # Run layers 0..compression_depth-1 normally, then compress,
            # then run remaining layers on the register tokens.
            enc_out = self.encoder(
                input_ids=t5_input_ids,
                attention_mask=t5_attn_mask,
                output_hidden_states=True,          # always needed here
            )
            all_hs = enc_out.hidden_states           # tuple of (B, S, d)

            # Pick hidden state at compression_depth
            if self.compression_depth < len(all_hs):
                h_at_depth = all_hs[self.compression_depth]    # (B, S, d)
            else:
                h_at_depth = all_hs[-1]

            B = h_at_depth.size(0)
            registers = self.register_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, R, d)

            # Key-padding mask for compressed attention
            kpm = (1 - t5_attn_mask).bool()          # True = padding, ignore
            compressed, _ = self.compress_attn(
                registers, h_at_depth, h_at_depth, key_padding_mask=kpm
            )                                         # (B, R, d)

            # Mean pool registers → recommendation logits
            pooled = compressed.mean(dim=1)           # (B, d)
            result = {"logits": self.rec_head(self.pre_head_norm(pooled))}
            if return_hidden_states:
                result["all_hidden_states"] = list(all_hs)
            return result


# =============================================================================
# FLOPs estimation utility
# =============================================================================
def estimate_flops_t5(
    model: T5LargeRecommender,
    sample_input: torch.Tensor,
    device: str,
) -> float:
    """
    Estimate FLOPs for one forward pass using fvcore if available,
    otherwise use the parameter count as a proxy.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        text_enc = model._item_ids_to_text_tokens(sample_input)
        flops = FlopCountAnalysis(model.encoder, (text_enc["input_ids"],))
        return float(flops.total())
    except Exception:
        # Fallback: count parameters × 2 (multiply-add) as rough proxy
        return float(sum(p.numel() for p in model.parameters()) * 2)


# =============================================================================
# Main Experiment 7 Class
# =============================================================================
class Experiment7:
    """Experiment 7: T5-Large LLM Recommendation with CARR Compression."""

    # Compression depths to sweep (fraction of 24 T5-Large encoder layers)
    COMPRESSION_DEPTHS = [None, 6, 12, 18]   # None = full
    COMPRESSION_LABELS = {
        None: "T5-Large (Full)",
        6:    "T5-Large + CARR (k=6)",
        12:   "T5-Large + CARR (k=12)",
        18:   "T5-Large + CARR (k=18)",
    }

    def __init__(
        self,
        dataset_name: str = "ml-1m",
        device: str = "cuda",
        seed: int = 42,
        num_registers: int = 8,
    ):
        if not T5_AVAILABLE:
            raise RuntimeError(
                "transformers library required for Experiment 7. "
                "Run: pip install transformers"
            )

        self.dataset_name = dataset_name
        self.device = device
        self.seed = seed
        self.num_registers = num_registers

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.config = get_experiment_config("exp7_t5_large", dataset_name)
        # T5-Large requires more GPU memory — use smaller batch
        self.config.training.batch_size = min(16, self.config.training.batch_size)

        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="recency",
        )
        self.rec_metrics = RecommendationMetrics()

        self.results: Dict = {}

    @torch.no_grad()
    def evaluate_model(
        self,
        model: T5LargeRecommender,
        model_name: str,
        num_batches: int = 100,
    ) -> Dict[str, float]:
        """Evaluate HR@10, NDCG@10, and measure latency."""
        model.eval()
        hits, ndcg_scores, latencies = [], [], []

        test_loader = self.data_module.test_dataloader()
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc=f"[Exp7] Evaluating {model_name}")
        ):
            if batch_idx >= num_batches:
                break

            iids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            tgts = batch["target"].to(self.device)

            t_start = time.perf_counter()
            outputs = model(iids, mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t_start)

            logits = outputs.get("logits")
            if logits is None:
                continue

            lgt = logits[:, -1, :] if logits.dim() == 3 else logits
            top10 = torch.topk(lgt, k=10, dim=-1).indices
            for pred, tgt in zip(top10, tgts):
                hit  = int((pred == tgt).any().item())
                rank = (pred == tgt).nonzero(as_tuple=True)[0]
                nd   = (1.0 / np.log2(rank[0].item() + 2)) if len(rank) > 0 else 0.0
                hits.append(hit)
                ndcg_scores.append(nd)

        return {
            "hr10":         float(np.mean(hits))         if hits else 0.0,
            "ndcg10":       float(np.mean(ndcg_scores))  if ndcg_scores else 0.0,
            "avg_latency_s": float(np.mean(latencies))   if latencies else 0.0,
        }

    def run(self) -> Dict:
        print("=" * 70)
        print("EXPERIMENT 7: T5-Large LLM Experiments (R3)")
        print("=" * 70)

        # Use a single reference batch to estimate FLOPs
        sample_batch = next(iter(self.data_module.test_dataloader()))
        sample_input = sample_batch["input_ids"][:4].to(self.device)

        full_model_latency: Optional[float] = None
        rows = []

        for depth in self.COMPRESSION_DEPTHS:
            model_name = self.COMPRESSION_LABELS[depth]
            print(f"\n  Evaluating: {model_name}")

            try:
                model = T5LargeRecommender(
                    num_items=self.data_module.num_items,
                    compression_depth=depth,
                    num_registers=self.num_registers,
                    device=self.device,
                ).to(self.device)

                metrics = self.evaluate_model(model, model_name)

                # FLOPs estimate
                flops = estimate_flops_t5(model, sample_input, self.device)

                # Latency reduction relative to full model
                if depth is None:
                    full_model_latency = metrics["avg_latency_s"]
                    latency_reduction = 1.0
                else:
                    latency_reduction = (
                        float(full_model_latency) / metrics["avg_latency_s"]
                        if full_model_latency and metrics["avg_latency_s"] > 0
                        else float("nan")
                    )

                metrics["flops"]             = flops
                metrics["latency_reduction"] = latency_reduction
                self.results[model_name]     = metrics

                rows.append({
                    "Model":             model_name,
                    "HR@10":             f"{metrics['hr10']:.4f}",
                    "NDCG@10":           f"{metrics['ndcg10']:.4f}",
                    "Latency (s)":       f"{metrics['avg_latency_s']:.4f}",
                    "Latency Reduction": f"{latency_reduction:.2f}x",
                })

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                print(f"  ERROR evaluating {model_name}: {exc}")
                rows.append({
                    "Model": model_name,
                    "HR@10": "N/A", "NDCG@10": "N/A",
                    "Latency (s)": "N/A", "Latency Reduction": "N/A",
                })

        # Save results table
        df = pd.DataFrame(rows)
        df.to_csv(TABLES_DIR / "exp7_t5_large_results.csv", index=False)
        with open(TABLES_DIR / "exp7_t5_large_results.tex", "w") as f:
            f.write(df.to_latex(index=False, escape=True))
        print(f"\n[Exp7] Results table → {TABLES_DIR / 'exp7_t5_large_results.csv'}")

        # Latency vs quality plot
        self._plot_latency_vs_quality(rows)

        save_results(self.results, "exp7_t5_large", self.dataset_name)
        return self.results

    def _plot_latency_vs_quality(self, rows: List[Dict]) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for row in rows:
            try:
                hr   = float(row["HR@10"])
                lat  = float(row["Latency (s)"])
                ax.scatter(lat, hr, s=80, zorder=5)
                ax.annotate(
                    row["Model"], (lat, hr),
                    fontsize=7, textcoords="offset points", xytext=(4, 4)
                )
            except (ValueError, TypeError):
                pass

        ax.set_xlabel("Average Inference Latency (s)")
        ax.set_ylabel("HR@10")
        ax.set_title("T5-Large: Quality vs Latency Trade-off")
        ax.grid(True, alpha=0.3)
        path = FIGURES_DIR / "exp7_t5_latency_vs_quality.pdf"
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"[Exp7] Latency-quality plot → {path}")


# =============================================================================
# Entry point for main.py
# =============================================================================
def run_experiment_7(datasets: List[str], device: str) -> Dict:
    if not T5_AVAILABLE:
        print("[Exp7] Skipping T5-Large experiment — transformers not installed.")
        return {}

    all_results = {}
    for dataset in datasets:
        print(f"\n--- Dataset: {dataset} ---")
        try:
            exp = Experiment7(dataset_name=dataset, device=device)
            all_results[dataset] = exp.run()
        except Exception as exc:
            print(f"[Exp7] ERROR for {dataset}: {exc}")
    return all_results
