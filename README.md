# Progressive Reasoning Collapse (PRC) in LLM-Based Generative Recommendation

> **Paper under review — IEEE Transactions on Knowledge and Data Engineering (TKDE)**  
> *Author: John Kingsley Arthur*

---

## Overview

This repository contains the full experimental framework for the **Progressive Reasoning Collapse (PRC)** theory and the **CARR (Collapse-Aware Register Recommendation)** method, validated on AWS EC2 (`g4dn.xlarge`, NVIDIA Tesla T4).

Modern LLM-based recommenders apply aggressive efficiency optimizations (context pruning, layer removal, KV-cache compression) that inadvertently destroy fine-grained, minority-intent signals in latent representations. We formalise this as **PRC** and propose CARR to detect and mitigate it adaptively.

**Key empirical result:** CARR achieves the best HR@10 among all LLM compression methods on all five benchmarks, a 4.5× evidence-survival advantage over uncompressed Full-LLM on ML-1M, and uniquely non-zero HR@10 on 18,157 Yelp cold-start items where every other method scores zero.

---

## Key Theoretical Results

| Theorem | Statement |
|---|---|
| **Theorem 1** — Monotonic PRC | R(l) decays monotonically after compression boundary k |
| **Theorem 2** — Evidence Survival Decay | Evidence-survival S_l(τ) contracts geometrically beyond k |
| **Theorem 3** — Critical Depth | A critical depth k\* partitions compression into safe (k > k\*) and collapse (k ≤ k\*) regimes |

**Core metrics:**
- **R(l)** — Reasoning Collapse Score: tr(Σ_W) / tr(Σ_B) across layers (higher = more within-intent diversity preserved)
- **R̃(l) = R(l)/R(k₀)** — Normalised collapse score, comparable across datasets and architectures
- **S_l(τ)** — Evidence Survival Function: distribution shift under history ablation
- **ε_R = 0.1, ε_S = 0.05** — Calibrated collapse thresholds used consistently across all five datasets

---

## Experiments

| # | Experiment | Key Finding |
|---|---|---|
| 1 | Progressive Collapse Validation | CARR R(12)=3.526 stays in safe zone; all fixed-depth methods collapse below Full-LLM baseline |
| 1b | Intent Validation | CARR IDP=0.96 vs Fixed-Early 0.46 |
| 2 | Critical Depth Analysis | Sharp HR@10 inflection at k\*=6 on ML-1M |
| 3 | CARR vs. 11 Baselines | CARR best LLM-compression HR@10 on all 4 primary datasets |
| 4 | Compression-Depth Stress Test | CARR Ŝ=0.3237 vs fixed-depth ≈0.000–0.011 |
| 5 | Minority-Intent Preservation | CARR smallest minority–majority NDCG gap among high-performing methods |
| 6 | Latent Space Visualisation | t-SNE/PCA confirms geometry ordering from R(l) |
| 7 | T5-Base Scalability | CARR-T5 HR@10=0.0441 (+7.0% over Full-T5-base on ML-1M) |
| 8 | Robustness (Long-Context & Noise) | CARR R(l) ordering maintained at all context lengths |
| 9 | Failure Mode Analysis | CARR HR@10=0.0043 on 18,157 zero-cooccurrence Yelp items; all others = 0 |
| 10 | Steam Large-Scale Validation | CARR best HR@10 among LLM-compression methods on 281,428-user dataset |

**Datasets:** MovieLens-1M · Amazon Beauty · Amazon Toys · Yelp · Steam *(auto-downloaded at runtime)*

**Baselines (12):** Full-LLM · Fixed-Early (k=3) · Fixed-Mid (k=6) · Fixed-Late (k=9) · KV-Pruning · Token-Pruning · LLMRec · UniSRec · SASRec · BERT4Rec · GRU4Rec · CARR

---

## Repository Structure

```
.
├── experiments/
│   ├── main.py                      # Master runner
│   ├── config.py                    # All hyperparameters (ε_R=0.1, ε_S=0.05)
│   ├── data_loader.py               # Auto-downloads all 5 datasets
│   ├── requirements.txt
│   ├── exp1_progressive_collapse.py
│   ├── exp1b_intent_validation.py
│   ├── exp2_critical_depth.py
│   ├── exp3_carr_comparison.py
│   ├── exp4_ablation.py
│   ├── exp5_minority_intent.py
│   ├── exp6_visualization.py
│   ├── exp7_t5_base.py
│   ├── exp7_t5_large.py
│   ├── exp8_robustness.py
│   ├── exp9_failure_modes.py
│   ├── models/
│   │   ├── carr_model.py            # CARR + SGI (Sparse Geometry Injection)
│   │   └── baselines.py             # All 11 baseline methods
│   ├── metrics/
│   │   └── collapse_metrics.py      # R(l), R̃(l), and S_l(τ) computation
│   └── training/
│       └── trainer.py               # Training + evaluation pipeline
├── paper/                           # LaTeX source (main.tex + supplementary.tex)
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

# Run all experiments on ML-1M
docker run --gpus all \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/data:/workspace/data \
  carr-experiment \
  python main.py --experiments all --datasets ml-1m

# Run specific experiments
docker run --gpus all \
  -v $(pwd)/results:/workspace/results \
  carr-experiment \
  python main.py --experiments 1 2 3 --datasets ml-1m amazon-beauty amazon-toys yelp
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

All primary results were produced on **`g4dn.xlarge`** (NVIDIA Tesla T4, 16 GB VRAM, ~$0.50/hr) with the Ubuntu 22.04 Deep Learning AMI.

```bash
# 1. Clone repo on EC2
git clone https://github.com/jkinarthur/progressive_reasoning_collapse.git
cd progressive_reasoning_collapse

# 2. Build and run
docker build -t carr-experiment .
docker run --gpus all \
  -v $(pwd)/results:/workspace/results \
  carr-experiment \
  python main.py --experiments all --datasets ml-1m amazon-beauty amazon-toys yelp

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
