# Progressive Reasoning Collapse (PRC) in LLM-Based Generative Recommendation

> **Paper under review — IEEE Transactions on Knowledge and Data Engineering (TKDE)**
> *Author: John Kingsley Arthur*

---

## Overview

This repository contains the full experimental framework for the **Progressive Reasoning Collapse (PRC)** theory and the **CARR (Collapse-Aware Register Recommendation)** method.

Modern LLM-based recommenders apply aggressive efficiency optimizations (context pruning, layer removal, KV-cache compression) that inadvertently destroy fine-grained, minority-intent signals in latent representations. We formalize this as **PRC** and propose CARR to detect and mitigate it adaptively.

---

## Key Theoretical Results

| Theorem | Statement |
|---|---|
| **Theorem 1** — Monotonic PRC | Reasoning collapse score R(l) is monotonically non-decreasing after compression depth k |
| **Theorem 2** — Evidence Survival Decay | Evidence survival S_l(τ) decays geometrically beyond k |
| **Theorem 3** — Critical Depth | A critical depth k* exists below which compression preserves multi-intent separability |

**Core metrics:**
- **R(l)** — Reasoning Collapse Score: ratio of within-class to between-class variance across layers
- **S_l(τ)** — Evidence Survival Function: distribution shift under history ablation

---

## Experiments

| # | Experiment | Validates |
|---|---|---|
| 1 | Progressive Collapse Validation | Theorems 1 & 2 |
| 2 | Critical Depth Analysis | Theorem 3 |
| 3 | CARR vs. Baselines (SASRec, BERT4Rec, GRU4Rec) | Full comparison |
| 4 | Ablation Studies | Component contributions |
| 5 | Minority-Intent Preservation | Fairness analysis |
| 6 | Layerwise Collapse Visualization | Interpretability |

**Datasets:** MovieLens-1M · Amazon Beauty · Amazon Toys · Yelp *(auto-downloaded at runtime)*

---

## Repository Structure

```
.
├── experiments/
│   ├── main.py                      # Master runner
│   ├── config.py                    # All hyperparameters
│   ├── data_loader.py               # Auto-downloads datasets
│   ├── requirements.txt
│   ├── exp1_progressive_collapse.py
│   ├── exp2_critical_depth.py
│   ├── exp3_carr_comparison.py
│   ├── exp4_ablation.py
│   ├── exp5_minority_intent.py
│   ├── exp6_visualization.py
│   ├── models/
│   │   ├── carr_model.py            # CARR with adaptive compression
│   │   └── baselines.py             # SASRec, BERT4Rec, GRU4Rec
│   ├── metrics/
│   │   └── collapse_metrics.py      # R(l) and S_l(τ) computation
│   └── training/
│       └── trainer.py               # Training + evaluation pipeline
├── paper/                           # LaTeX source
├── Dockerfile
├── .dockerignore
├── .gitignore
└── LICENSE
```

---

## Quickstart

### Option 1 — Docker (Recommended for AWS / reproducibility)

```bash
# Build image (one-time)
docker build -t carr-experiment .

# Run all experiments on MovieLens-1M
docker run --gpus all \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/data:/workspace/data \
  carr-experiment \
  python main.py --experiments all --datasets ml-1m

# Run specific experiments
docker run --gpus all \
  -v $(pwd)/results:/workspace/results \
  carr-experiment \
  python main.py --experiments 1 2 3 --datasets ml-1m amazon-beauty
```

### Option 2 — Local (Python 3.9+)

```bash
cd experiments
pip install -r requirements.txt
python main.py --experiments all --datasets ml-1m
```

> **Note:** Datasets are downloaded automatically on first run. No manual setup needed.

---

## Running on AWS EC2

Recommended instance: **`g4dn.xlarge`** (NVIDIA T4 GPU, ~$0.50/hr) with the **Ubuntu 22.04 Deep Learning AMI**.

```bash
# 1. Clone repo on EC2
git clone https://github.com/jkinarthur/progressive_reasoning_collapse.git
cd progressive_reasoning_collapse

# 2. Build and run
docker build -t carr-experiment .
docker run --gpus all \
  -v $(pwd)/results:/workspace/results \
  carr-experiment \
  python main.py --experiments all --datasets ml-1m amazon-beauty

# 3. Save results to S3
aws s3 sync results/ s3://your-bucket/carr-results/
```

Stop the instance when done to avoid charges.

---

## Requirements

- Python 3.9+
- PyTorch >= 2.0.0
- CUDA 11.8+ (for GPU)
- See `experiments/requirements.txt` for full list

---

## Citation

If you use this code, please cite:

```bibtex
@article{arthur2026prc,
  title   = {Progressive Reasoning Collapse in LLM-Based Generative Recommendation},
  author  = {Arthur, John Kingsley},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
