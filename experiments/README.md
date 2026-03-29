# Progressive Reasoning Collapse (PRC) in LLM-Based Generative Recommendation

## Overview

This repository contains the experimental framework for validating the **Progressive Reasoning Collapse (PRC)** theory and evaluating the **CARR (Collapse-Aware Register Recommendation)** method for IEEE TKDE submission.

### Paper Abstract

Modern generative recommendation systems leverage Large Language Models (LLMs) to produce personalized recommendations by reasoning over user interaction histories. However, deploying these models at scale requires aggressive efficiency optimizations—such as context pruning, layer removal, and KV‐cache compression—that inadvertently distort the latent intent structure captured in intermediate representations. We identify and formalize a phenomenon called **Progressive Reasoning Collapse (PRC)**, whereby layerwise compression disproportionately erases fine‐grained, minority‐intent signals while homogenizing user representations.

## Theoretical Framework

### Key Theorems

1. **Theorem 1 (Monotonic Progressive Reasoning Collapse)**: Under mild Lipschitz conditions on the LLM's residual stream, the reasoning collapse score R(l) is monotonically non-decreasing in layer index l after the compression point k.

2. **Theorem 2 (Evidence Survival Decay Under Compression)**: The evidence survival function S_l(τ) decays geometrically as l increases beyond k.

3. **Theorem 3 (Critical Compression Depth)**: There exists a critical depth k* below which compression preserves multi-intent separability.

### Metrics

- **R(l)**: Reasoning Collapse Score - ratio of within-class to between-class variance
- **S_l(τ)**: Evidence Survival Function - distribution shift between full and ablated histories

## Project Structure

```
experiments/
├── main.py                     # Main experiment runner
├── config.py                   # Configuration and hyperparameters
├── data_loader.py              # Dataset loading and preprocessing
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/
│   ├── __init__.py
│   ├── carr_model.py           # CARR model with adaptive compression
│   └── baselines.py            # Baseline models (SASRec, BERT4Rec, GRU4Rec)
│
├── metrics/
│   ├── __init__.py
│   └── collapse_metrics.py     # PRC metrics computation
│
├── training/
│   ├── __init__.py
│   └── trainer.py              # Training and evaluation pipeline
│
├── exp1_progressive_collapse.py  # Theorem 1 & 2 validation
├── exp2_critical_depth.py        # Theorem 3 validation
├── exp3_carr_comparison.py       # Comprehensive comparison
├── exp4_ablation.py              # Ablation studies
├── exp5_minority_intent.py       # Minority-intent preservation
├── exp6_visualization.py         # Layerwise collapse visualization
│
└── outputs/
    ├── results/                # JSON results
    ├── figures/                # Generated plots (PDF/PNG)
    └── tables/                 # LaTeX tables
```

## Experiments

### Experiment 1: Progressive Collapse Validation
Validates Theorems 1 and 2 by measuring:
- Layerwise reasoning collapse score R(l)
- Evidence survival function S_l(τ)
- Monotonicity of collapse progression

**Outputs**: Line plots showing R(l) and S_l(τ) across layers

### Experiment 2: Critical Depth Analysis
Validates Theorem 3 by:
- Sweeping compression depths k ∈ {2, 4, 6, 8, 10}
- Identifying phase transition in recommendation quality
- Computing critical depth k* for each dataset

**Outputs**: Phase transition diagrams, critical depth identification

### Experiment 3: CARR Comparative Evaluation
Compares CARR against:
- **LLM Variants**: Full-LLM, Fixed-Early, Fixed-Mid, Fixed-Late, Prompt Pruning, Layer Skipping
- **Neural Baselines**: SASRec, BERT4Rec, GRU4Rec

**Metrics**: HR@K, NDCG@K, R-score, Evidence Survival, FLOPs, Latency

**Outputs**: Comprehensive comparison tables, radar charts, efficiency plots

### Experiment 4: Ablation Studies
Isolates CARR component contributions:
- `full_carr`: Complete CARR model
- `no_collapse_reg`: Remove L_collapse loss
- `no_evidence_reg`: Remove L_evidence loss  
- `no_adaptive_depth`: Fixed compression depth
- `no_adaptive_width`: Fixed register count

**Outputs**: Ablation tables with statistical significance (t-tests, Cohen's d)

### Experiment 5: Minority-Intent Preservation
Validates the corollary on minority-intent erasure:
- Splits test set by target item popularity
- Measures performance gap between majority/minority items
- Compares preservation across methods

**Outputs**: Minority vs. majority performance analysis

### Experiment 6: Visualization of Layerwise Collapse
Provides interpretable visualizations:
- PCA/t-SNE/UMAP projections of latent representations
- Intent cluster contraction across layers
- Evidence-induced representation divergence

**Outputs**: Cluster visualizations, divergence plots

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install UMAP for advanced visualization
pip install umap-learn
```

## Usage

### Run All Experiments

```bash
python main.py --experiments all --datasets ml-1m amazon-beauty
```

### Run Specific Experiments

```bash
# Run experiments 1, 2, 3 on MovieLens
python main.py --experiments 1 2 3 --datasets ml-1m

# Run ablation study with 5 runs for significance
python main.py --experiments 4 --num-runs 5 --datasets ml-1m amazon-beauty

# Run visualization experiment
python main.py --experiments 6 --num-samples 500
```

### Quick Test Run

```bash
python main.py --experiments 1 --datasets ml-1m --quick
```

## Datasets

| Dataset | Users | Items | Interactions | Density |
|---------|-------|-------|--------------|---------|
| MovieLens-1M | 6,040 | 3,706 | 1,000,209 | 4.47% |
| Amazon Beauty | 22,363 | 12,101 | 198,502 | 0.07% |
| Amazon Toys | 19,412 | 11,924 | 167,597 | 0.07% |
| Yelp | 31,668 | 38,048 | 1,561,406 | 0.13% |

## Key Hyperparameters

```python
# CARR Model
num_registers = 16          # Register tokens for compression
compression_depth = 6       # Default compression depth k
lambda_collapse = 0.1       # Collapse regularization weight
lambda_evidence = 0.1       # Evidence preservation weight
lambda_compress = 0.01      # Compression penalty

# Training
learning_rate = 1e-4
batch_size = 32
max_epochs = 50
early_stopping_patience = 5
```

## Expected Results

### Table 1: Comprehensive Comparison (ML-1M)

| Method | HR@10 | NDCG@10 | R-score↓ | S-score↑ | FLOPs |
|--------|-------|---------|----------|----------|-------|
| Full-LLM | 0.42 | 0.28 | 0.15 | 0.82 | 1.00x |
| Fixed-Mid | 0.35 | 0.22 | 0.45 | 0.41 | 0.52x |
| **CARR** | **0.40** | **0.26** | **0.22** | **0.71** | **0.55x** |

### Figure 1: Progressive Collapse Visualization

The R(l) plot should show:
- Monotonic increase in collapse after compression point
- CARR maintaining lower R-scores than fixed compression
- Clear phase transition at critical depth k*

## Citation

```bibtex
@article{prc2024,
  title={Progressive Reasoning Collapse in LLM-Based Generative Recommendation},
  author={...},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024}
}
```

## License

This project is for academic research purposes.

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config.py
- Use `--device cpu` for testing
- Enable gradient checkpointing in model config

### Missing Dependencies
```bash
pip install torch transformers scikit-learn matplotlib
```

### Dataset Download Issues
The data loader will attempt to download datasets automatically. For manual download, place files in `data/` directory.
