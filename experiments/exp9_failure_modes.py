"""
Experiment 9: Failure Mode Analysis

Addresses Reviewer Comment R10:
  "Include a section 'Failure Modes of Collapse-Aware Compression'
   analysing cases where CARR fails:
     - extremely sparse users
     - single-intent users
     - cold-start items"

Three targeted evaluations are run:

  9a — Sparse Users:
        Isolate test users with fewer than 5 historical interactions.
        Compare CARR vs. Full-LLM on HR@10 / NDCG@10 for this group.
        A large gap indicates sensitivity to data sparsity.

  9b — Single-Intent Users:
        Identify users whose interaction history spans only one item
        category (approximated by item ID quartile).  Measure whether
        CARR's multi-intent preservation mechanism introduces any
        degradation for such users.

  9c — Cold-Start Items:
        Identify target items that appear fewer than 5 times in the
        training set.  Evaluate HR@10 / NDCG@10 for queries whose
        ground-truth target is a cold-start item.

Key Outputs:
  - tables/exp9_sparse_users.{csv,tex}
  - tables/exp9_single_intent.{csv,tex}
  - tables/exp9_cold_start.{csv,tex}
  - figures/exp9_failure_modes_summary.pdf
  - results/exp9_failure_modes_<dataset>.json
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config import get_experiment_config, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, ModelConfig
from data_loader import RecommendationDataModule
from models.carr_model import CARRModel, FixedCompressionModel, FullLLMModel
from training.trainer import RecommendationMetrics, save_results


# =============================================================================
# Helpers
# =============================================================================
def _topk_metrics(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 10
) -> List[Dict[str, float]]:
    """Return per-sample HR@k and NDCG@k."""
    lgt = logits[:, -1, :] if logits.dim() == 3 else logits
    topk_preds = torch.topk(lgt, k=k, dim=-1).indices
    metrics = []
    for pred, tgt in zip(topk_preds, targets):
        hit  = int((pred == tgt).any().item())
        rank = (pred == tgt).nonzero(as_tuple=True)[0]
        ndcg = (1.0 / np.log2(rank[0].item() + 2)) if len(rank) > 0 else 0.0
        metrics.append({"hr10": hit, "ndcg10": ndcg})
    return metrics


# =============================================================================
# Main Experiment 9 Class
# =============================================================================
class Experiment9:
    """Experiment 9: Failure Mode Analysis (R10)."""

    SPARSE_THRESHOLD       = 5    # users with fewer than N historical interactions
    COLD_START_THRESHOLD   = 5    # items seen fewer than N times in training

    def __init__(
        self,
        dataset_name: str = "ml-1m",
        device: str = "cuda",
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.device       = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed         = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.config      = get_experiment_config("exp9_failure_modes", dataset_name)
        self.data_module = RecommendationDataModule(
            self.config.dataset,
            batch_size=self.config.training.batch_size,
            tau_strategy="recency",
        )

        self.results: Dict = {
            "sparse_users":   {},
            "single_intent":  {},
            "cold_start":     {},
        }

        # Pre-compute item frequency from training sequences
        self._item_frequency: Dict[int, int] = self._compute_item_frequency()
        # Pre-compute per-user diversity from test sequences
        self._user_intent_span: Dict[int, int] = self._compute_user_intent_span()

    def _compute_item_frequency(self) -> Dict[int, int]:
        """Count how often each item_id appears across all training sequences."""
        freq: Dict[int, int] = defaultdict(int)
        for seq in self.data_module.train_seqs.values():
            for item in seq:
                freq[item] += 1
        return freq

    def _compute_user_intent_span(self) -> Dict[int, int]:
        """
        Approximate the number of distinct 'intents' per user by quantile-
        binning their item IDs: each quartile of the item vocabulary is
        treated as one intent category.  Return the number of distinct
        intent buckets present in each user's history.
        """
        num_items = self.data_module.num_items
        bucket_size = max(1, num_items // 4)   # 4 intent buckets
        span: Dict[int, int] = {}
        for uid, seq in self.data_module.test_seqs.items():
            buckets = set(item // bucket_size for item in seq if item > 0)
            span[uid] = len(buckets)
        return span

    # =========================================================================
    # 9a: Sparse Users
    # =========================================================================
    @torch.no_grad()
    def run_sparse_users(self, models: Dict[str, nn.Module]) -> Dict:
        """
        Evaluate models on the subset of test users whose history length
        is below self.SPARSE_THRESHOLD.
        """
        print("\n[9a] Sparse User Failure Mode")

        sparse_user_ids: Set = {
            uid
            for uid, seq in self.data_module.test_seqs.items()
            if len(seq) < self.SPARSE_THRESHOLD
        }
        print(f"  Sparse users: {len(sparse_user_ids)} / "
              f"{len(self.data_module.test_seqs)}")

        if not sparse_user_ids:
            print("  No sparse users found with current threshold.")
            return {}

        results: Dict = {}
        test_loader   = self.data_module.test_dataloader()

        for name, model in models.items():
            model.eval()
            sparse_hits, sparse_ndcg = [], []

            for batch in tqdm(test_loader, desc=f"  [9a] {name}", leave=False):
                iids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                tgts = batch["target"].to(self.device)
                uids = batch.get("user_id", torch.full((iids.size(0),), -1))

                outputs = model(iids, mask)
                logits  = outputs.get("logits", outputs.get("scores"))
                if logits is None:
                    continue

                sample_metrics = _topk_metrics(logits, tgts)
                for i, m in enumerate(sample_metrics):
                    uid = uids[i].item() if uids is not None else -1
                    if uid in sparse_user_ids or uid == -1:
                        sparse_hits.append(m["hr10"])
                        sparse_ndcg.append(m["ndcg10"])

            results[name] = {
                "hr10":   float(np.mean(sparse_hits))  if sparse_hits  else float("nan"),
                "ndcg10": float(np.mean(sparse_ndcg))  if sparse_ndcg  else float("nan"),
                "n_users": len(sparse_user_ids),
            }
            print(f"    {name}: HR@10={results[name]['hr10']:.4f}  "
                  f"NDCG@10={results[name]['ndcg10']:.4f}")

        return results

    # =========================================================================
    # 9b: Single-Intent Users
    # =========================================================================
    @torch.no_grad()
    def run_single_intent(self, models: Dict[str, nn.Module]) -> Dict:
        """
        Evaluate on users whose history touches only ONE intent bucket
        (approximated by item ID quartile).
        """
        print("\n[9b] Single-Intent User Failure Mode")

        single_intent_users: Set = {
            uid for uid, span in self._user_intent_span.items() if span <= 1
        }
        print(f"  Single-intent users: {len(single_intent_users)} / "
              f"{len(self.data_module.test_seqs)}")

        if not single_intent_users:
            print("  No single-intent users found.")
            return {}

        results: Dict = {}
        test_loader   = self.data_module.test_dataloader()

        for name, model in models.items():
            model.eval()
            si_hits, si_ndcg = [], []

            for batch in tqdm(test_loader, desc=f"  [9b] {name}", leave=False):
                iids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                tgts = batch["target"].to(self.device)
                uids = batch.get("user_id", torch.full((iids.size(0),), -1))

                outputs = model(iids, mask)
                logits  = outputs.get("logits", outputs.get("scores"))
                if logits is None:
                    continue

                sample_metrics = _topk_metrics(logits, tgts)
                for i, m in enumerate(sample_metrics):
                    uid = uids[i].item() if uids is not None else -1
                    if uid in single_intent_users or uid == -1:
                        si_hits.append(m["hr10"])
                        si_ndcg.append(m["ndcg10"])

            results[name] = {
                "hr10":   float(np.mean(si_hits))  if si_hits  else float("nan"),
                "ndcg10": float(np.mean(si_ndcg))  if si_ndcg  else float("nan"),
                "n_users": len(single_intent_users),
            }
            print(f"    {name}: HR@10={results[name]['hr10']:.4f}  "
                  f"NDCG@10={results[name]['ndcg10']:.4f}")

        return results

    # =========================================================================
    # 9c: Cold-Start Items
    # =========================================================================
    @torch.no_grad()
    def run_cold_start(self, models: Dict[str, nn.Module]) -> Dict:
        """
        Evaluate only on test queries whose ground-truth item appears
        fewer than self.COLD_START_THRESHOLD times in the training data.
        """
        print("\n[9c] Cold-Start Item Failure Mode")

        cold_items: Set[int] = {
            item for item, freq in self._item_frequency.items()
            if freq < self.COLD_START_THRESHOLD
        }
        print(f"  Cold-start items: {len(cold_items)} / {self.data_module.num_items}")

        results: Dict = {}
        test_loader   = self.data_module.test_dataloader()

        for name, model in models.items():
            model.eval()
            cs_hits, cs_ndcg = [], []

            for batch in tqdm(test_loader, desc=f"  [9c] {name}", leave=False):
                iids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                tgts = batch["target"].to(self.device)

                outputs = model(iids, mask)
                logits  = outputs.get("logits", outputs.get("scores"))
                if logits is None:
                    continue

                sample_metrics = _topk_metrics(logits, tgts)
                for i, m in enumerate(sample_metrics):
                    tgt_id = tgts[i].item()
                    if tgt_id in cold_items:
                        cs_hits.append(m["hr10"])
                        cs_ndcg.append(m["ndcg10"])

            results[name] = {
                "hr10":    float(np.mean(cs_hits))  if cs_hits  else float("nan"),
                "ndcg10":  float(np.mean(cs_ndcg))  if cs_ndcg  else float("nan"),
                "n_items": len(cold_items),
            }
            print(f"    {name}: HR@10={results[name]['hr10']:.4f}  "
                  f"NDCG@10={results[name]['ndcg10']:.4f}")

        return results

    # =========================================================================
    # Tables and plots
    # =========================================================================
    def _save_table(self, results: Dict, filename: str) -> None:
        rows = [
            {
                "Model":   name,
                "HR@10":   f"{m.get('hr10',   float('nan')):.4f}",
                "NDCG@10": f"{m.get('ndcg10', float('nan')):.4f}",
                "N":       str(m.get("n_users", m.get("n_items", "?"))),
            }
            for name, m in results.items()
        ]
        df = pd.DataFrame(rows)
        df.to_csv(TABLES_DIR / f"{filename}.csv", index=False)
        with open(TABLES_DIR / f"{filename}.tex", "w") as f:
            f.write(df.to_latex(index=False, escape=True))
        print(f"[Exp9] Table → {TABLES_DIR / filename}.csv")

    def _plot_summary(self) -> None:
        categories = ["Sparse Users", "Single-Intent", "Cold-Start"]
        data_keys  = ["sparse_users", "single_intent", "cold_start"]

        # Collect model names across all sub-experiments
        all_model_names: List[str] = []
        for key in data_keys:
            all_model_names.extend(self.results.get(key, {}).keys())
        model_names = list(dict.fromkeys(all_model_names))   # deduplicate, preserve order

        fig, axes = plt.subplots(1, len(categories), figsize=(14, 4), sharey=False)
        for ax, cat, key in zip(axes, categories, data_keys):
            sub = self.results.get(key, {})
            model_hr = [sub.get(n, {}).get("hr10", float("nan")) for n in model_names]
            bars = ax.bar(
                range(len(model_names)), model_hr,
                color=plt.cm.Set2(np.linspace(0, 1, len(model_names)))
            )
            ax.set_title(cat, fontsize=10)
            ax.set_ylabel("HR@10")
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Failure Mode Analysis: HR@10 Across User/Item Segments", y=1.02)
        plt.tight_layout()
        path = FIGURES_DIR / "exp9_failure_modes_summary.pdf"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"[Exp9] Failure mode plot → {path}")

    # =========================================================================
    # Main runner
    # =========================================================================
    def run(self) -> Dict:
        print("=" * 70)
        print("EXPERIMENT 9: Failure Mode Analysis (R10)")
        print("=" * 70)

        num_items = self.data_module.num_items
        models = {
            "Full-LLM":        FullLLMModel(num_items, self.config.model),
            "Fixed-Mid (k=6)": FixedCompressionModel(
                num_items, ModelConfig(compression_depth=6)
            ),
            "CARR":            CARRModel(num_items, self.config.model),
        }
        for m in models.values():
            m.to(self.device)

        self.results["sparse_users"]  = self.run_sparse_users(models)
        self.results["single_intent"] = self.run_single_intent(models)
        self.results["cold_start"]    = self.run_cold_start(models)

        self._save_table(self.results["sparse_users"],  "exp9_sparse_users")
        self._save_table(self.results["single_intent"], "exp9_single_intent")
        self._save_table(self.results["cold_start"],    "exp9_cold_start")
        self._plot_summary()

        save_results(self.results, "exp9_failure_modes", self.dataset_name)
        return self.results


# =============================================================================
# Entry point for main.py
# =============================================================================
def run_experiment_9(datasets: List[str], device: str) -> Dict:
    all_results = {}
    for dataset in datasets:
        print(f"\n--- Dataset: {dataset} ---")
        exp = Experiment9(dataset_name=dataset, device=device)
        all_results[dataset] = exp.run()
    return all_results
