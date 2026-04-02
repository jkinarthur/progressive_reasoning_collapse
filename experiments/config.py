"""
Configuration for Progressive Reasoning Collapse (PRC) Experiments
Targeting IEEE TKDE publication

Author: Based on theory by John Kingsley Arthur
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
FIGURES_DIR = ROOT_DIR / "figures"
TABLES_DIR = ROOT_DIR / "tables"

# Create directories
for dir_path in [DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    """Configuration for datasets"""
    name: str
    path: str
    min_interactions: int = 5
    max_seq_len: int = 50
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    # Transformer backbone
    hidden_dim: int = 256
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1
    ffn_dim: int = 1024
    
    # Compression settings
    compression_depth: int = 6  # Layer at which compression is applied
    num_registers: int = 8  # Number of register tokens
    max_registers: int = 32
    
    # CARR specific
    monitored_layers: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    collapse_threshold_R: float = 0.1  # δ_R
    collapse_threshold_S: float = 0.05  # δ_S  
    collapse_risk_threshold: float = 0.5  # η
    
    # Regularization weights
    lambda_collapse: float = 0.1  # λ_1
    lambda_evidence: float = 0.1  # λ_2
    lambda_compress: float = 0.01  # λ_3
    
    # Collapse detection
    num_intent_clusters: int = 5  # K latent intents


@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    
    # Evaluation
    eval_every: int = 1
    save_every: int = 5
    top_k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Master configuration for experiments"""
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    
    # Experiment specific
    experiment_name: str = "prc_experiment"
    num_runs: int = 5  # For statistical significance
    
    # Informative subsequence strategies for τ
    tau_strategies: List[str] = field(default_factory=lambda: [
        "recency",      # Recent history removal
        "minority",     # Minority interest removal
        "attention",    # High-attention token removal
        "random"        # Random ablation (control)
    ])


# =============================================================================
# Dataset Configurations
# =============================================================================
DATASET_CONFIGS = {
    "ml-1m": DatasetConfig(
        name="MovieLens-1M",
        path="data/ml-1m",
        min_interactions=20,
        max_seq_len=50
    ),
    "amazon-beauty": DatasetConfig(
        name="Amazon Beauty",
        path="data/amazon-beauty",
        min_interactions=5,
        max_seq_len=50
    ),
    "amazon-toys": DatasetConfig(
        name="Amazon Toys",
        path="data/amazon-toys",
        min_interactions=5,
        max_seq_len=50
    ),
    "yelp": DatasetConfig(
        name="Yelp",
        path="data/yelp",
        min_interactions=10,
        max_seq_len=50
    ),
}


# =============================================================================
# Experiment Configurations for Each RQ
# =============================================================================
def get_experiment_config(experiment_name: str, dataset_name: str = "ml-1m") -> ExperimentConfig:
    """Get experiment configuration by name"""
    
    dataset_config = DATASET_CONFIGS[dataset_name]
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    if experiment_name == "exp1_progressive_collapse":
        # Experiment 1: Observing Progressive Reasoning Collapse
        training_config.num_epochs = 50
        
    elif experiment_name == "exp2_critical_depth":
        # Experiment 2: Validating Critical Compression Depth
        # Sweep compression depth
        pass
        
    elif experiment_name == "exp3_carr_comparison":
        # Experiment 3: CARR vs Fixed Compression
        pass
        
    elif experiment_name == "exp4_ablation":
        # Experiment 4: Ablation Studies
        pass
        
    elif experiment_name == "exp5_minority_intent":
        # Experiment 5: Minority-Intent Preservation
        pass
        
    elif experiment_name == "exp6_visualization":
        # Experiment 6: Layerwise Collapse Visualization
        pass
    
    return ExperimentConfig(
        dataset=dataset_config,
        model=model_config,
        training=training_config,
        experiment_name=experiment_name
    )


# =============================================================================
# Compression Depth Sweep for Experiment 2
# =============================================================================
COMPRESSION_DEPTHS = list(range(1, 13))  # k from 1 to L

# =============================================================================
# Ablation Configurations for Experiment 4
# =============================================================================
ABLATION_CONFIGS = {
    "full_carr": {"collapse_reg": True, "evidence_reg": True, "adaptive_depth": True, "adaptive_width": True},
    "no_collapse_reg": {"collapse_reg": False, "evidence_reg": True, "adaptive_depth": True, "adaptive_width": True},
    "no_evidence_reg": {"collapse_reg": True, "evidence_reg": False, "adaptive_depth": True, "adaptive_width": True},
    "no_adaptive_depth": {"collapse_reg": True, "evidence_reg": True, "adaptive_depth": False, "adaptive_width": True},
    "no_adaptive_width": {"collapse_reg": True, "evidence_reg": True, "adaptive_depth": True, "adaptive_width": False},
}

# =============================================================================
# Default Run Settings
# =============================================================================
DEFAULT_DATASETS = ["ml-1m"]
ALL_EXPERIMENTS = [1, 2, 3, 4, 5, 6]
