"""
Training and Evaluation Pipeline for PRC Experiments

Includes:
- Multi-objective training with collapse-aware losses
- Standard recommendation evaluation (HR, NDCG, MRR)
- Collapse-sensitive evaluation metrics
- Efficiency benchmarking
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import TrainingConfig, ModelConfig, ExperimentConfig, RESULTS_DIR, CHECKPOINTS_DIR
from metrics.collapse_metrics import CollapseMetricsComputer, LayerwiseCollapseAnalyzer


# =============================================================================
# Logging Setup
# =============================================================================
def setup_logging(experiment_name: str, log_dir: Path = RESULTS_DIR):
    """Setup logging for experiment"""
    log_file = log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# =============================================================================
# Loss Functions
# =============================================================================
class CARRLoss(nn.Module):
    """
    Multi-objective loss for CARR training
    
    L = L_rec + λ₁·L_collapse + λ₂·L_evidence + λ₃·L_compress
    """
    
    def __init__(
        self,
        lambda_collapse: float = 0.1,
        lambda_evidence: float = 0.1,
        lambda_compress: float = 0.01,
        num_items: int = 10000
    ):
        super().__init__()
        self.lambda_collapse = lambda_collapse
        self.lambda_evidence = lambda_evidence
        self.lambda_compress = lambda_compress
        self.num_items = num_items
        
        # Main recommendation loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        collapse_metrics: Optional[Dict[str, torch.Tensor]] = None,
        evidence_survival: Optional[torch.Tensor] = None,
        compression_applied: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss
        
        Args:
            logits: (batch, seq_len, num_items+1) or (batch, num_registers, num_items+1)
            targets: (batch,) target item ids
            collapse_metrics: Dictionary of collapse scores by layer
            evidence_survival: Evidence survival scores
            compression_applied: Whether compression was used
        """
        batch_size = logits.size(0)
        
        # Recommendation loss (use last position)
        pred_logits = logits[:, -1, :]  # (batch, num_items+1)
        loss_rec = self.ce_loss(pred_logits, targets)
        
        losses = {'loss_rec': loss_rec}
        total_loss = loss_rec
        
        # Collapse regularization loss
        if collapse_metrics is not None and self.lambda_collapse > 0:
            # Encourage high reasoning collapse score (preserve multi-intent structure)
            R_scores = [v for k, v in collapse_metrics.items() if 'reasoning_collapse_score' in k]
            if R_scores:
                avg_R = torch.stack(R_scores).mean()
                # Penalize low R (encourage high R = good separation)
                loss_collapse = 1.0 / (avg_R + 1e-6)
                losses['loss_collapse'] = loss_collapse
                total_loss = total_loss + self.lambda_collapse * loss_collapse
        
        # Evidence preservation loss
        if evidence_survival is not None and self.lambda_evidence > 0:
            # Encourage high evidence survival (model should be sensitive to history)
            loss_evidence = -evidence_survival.mean()  # Maximize survival
            losses['loss_evidence'] = loss_evidence
            total_loss = total_loss + self.lambda_evidence * loss_evidence
        
        # Compression efficiency loss (encourage compression)
        if compression_applied and self.lambda_compress > 0:
            # Small bonus for using compression
            loss_compress = torch.tensor(0.0, device=logits.device)
            losses['loss_compress'] = loss_compress
            total_loss = total_loss - self.lambda_compress  # Reward compression
        
        losses['total_loss'] = total_loss
        
        return losses


# =============================================================================
# Recommendation Metrics
# =============================================================================
class RecommendationMetrics:
    """Standard recommendation evaluation metrics"""
    
    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Compute recommendation metrics
        
        Args:
            predictions: (batch, num_items) probability or logit scores
            targets: (batch,) target item ids
        """
        metrics = {}
        batch_size = predictions.size(0)
        
        # Get rankings
        _, indices = predictions.sort(dim=-1, descending=True)
        
        # Find target positions
        target_positions = (indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1]
        
        for k in k_values:
            # Hit Rate @ K
            hits = (target_positions < k).float()
            metrics[f'HR@{k}'] = hits.mean().item()
            
            # NDCG @ K
            dcg = torch.where(
                target_positions < k,
                1.0 / torch.log2(target_positions.float() + 2),
                torch.zeros_like(target_positions.float())
            )
            metrics[f'NDCG@{k}'] = dcg.mean().item()
            
            # Recall @ K (same as HR for single target)
            metrics[f'Recall@{k}'] = metrics[f'HR@{k}']
        
        # MRR
        mrr = (1.0 / (target_positions.float() + 1)).mean().item()
        metrics['MRR'] = mrr
        
        return metrics


# =============================================================================
# Efficiency Metrics
# =============================================================================
class EfficiencyMetrics:
    """Efficiency benchmarking utilities"""
    
    @staticmethod
    def measure_latency(
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """Measure inference latency"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_ids, attention_mask, **kwargs)
        
        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(input_ids, attention_mask, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append(end - start)
        
        return {
            'latency_mean': np.mean(latencies) * 1000,  # ms
            'latency_std': np.std(latencies) * 1000,
            'latency_p50': np.percentile(latencies, 50) * 1000,
            'latency_p95': np.percentile(latencies, 95) * 1000,
            'throughput': 1.0 / np.mean(latencies) * input_ids.size(0)
        }
    
    @staticmethod
    def measure_memory(
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Measure GPU memory usage"""
        if not torch.cuda.is_available():
            return {'gpu_memory_mb': 0}
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        model.eval()
        with torch.no_grad():
            _ = model(input_ids, attention_mask, **kwargs)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        return {
            'gpu_memory_mb': peak_memory,
            'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024
        }


# =============================================================================
# Trainer
# =============================================================================
class Trainer:
    """Training loop for CARR and baseline models"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        model_config: ModelConfig,
        train_loader,
        val_loader,
        test_loader,
        experiment_name: str = "experiment"
    ):
        self.model = model
        self.config = config
        self.model_config = model_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.experiment_name = experiment_name
        
        # Move to device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * len(train_loader) - config.warmup_steps
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.warmup_steps]
        )
        
        # Loss
        self.loss_fn = CARRLoss(
            lambda_collapse=model_config.lambda_collapse,
            lambda_evidence=model_config.lambda_evidence,
            lambda_compress=model_config.lambda_compress
        )
        
        # Metrics
        self.rec_metrics = RecommendationMetrics()
        self.collapse_metrics = CollapseMetricsComputer()
        
        # Mixed precision
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        # Tracking
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.history = defaultdict(list)
        
        # Logging
        self.logger = setup_logging(experiment_name)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = defaultdict(list)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Check if we have ablated inputs for evidence loss
            has_ablated = 'ablated_input_ids' in batch
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.scaler is not None):
                # Determine compression depth (adaptive for CARR)
                compression_depth = None
                if hasattr(self.model, 'use_adaptive_depth') and self.model.use_adaptive_depth:
                    compression_depth, _ = self.model.compute_adaptive_compression_depth(
                        input_ids, attention_mask
                    )
                elif hasattr(self.model, 'fixed_compression_depth'):
                    compression_depth = self.model.fixed_compression_depth
                
                # Main forward pass
                outputs = self.model(
                    input_ids,
                    attention_mask,
                    compression_depth=compression_depth,
                    return_collapse_metrics=True
                )
                
                # Evidence survival computation
                evidence_survival = None
                if has_ablated and self.loss_fn.lambda_evidence > 0:
                    ablated_ids = batch['ablated_input_ids'].to(self.device)
                    ablated_mask = batch['ablated_attention_mask'].to(self.device)
                    
                    survival_result = self.model.compute_evidence_survival(
                        input_ids, ablated_ids,
                        attention_mask, ablated_mask,
                        compression_depth=compression_depth
                    )
                    evidence_survival = survival_result['evidence_survival']
                
                # Compute loss
                losses = self.loss_fn(
                    outputs['logits'],
                    targets,
                    collapse_metrics=outputs.get('collapse_metrics'),
                    evidence_survival=evidence_survival,
                    compression_applied=outputs.get('compression_applied', False)
                )
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Record losses
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_losses[key].append(value.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Average losses
        return {key: np.mean(values) for key, values in epoch_losses.items()}
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        compute_collapse: bool = True,
        compression_depth: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_collapse_metrics = defaultdict(list)
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids,
                attention_mask,
                compression_depth=compression_depth,
                return_collapse_metrics=compute_collapse
            )
            
            # Get predictions
            logits = outputs['logits'][:, -1, :]
            predictions = F.softmax(logits, dim=-1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            # Collect collapse metrics
            if compute_collapse and 'collapse_metrics' in outputs:
                for key, value in outputs['collapse_metrics'].items():
                    if isinstance(value, torch.Tensor):
                        all_collapse_metrics[key].append(value.cpu().item())
        
        # Concatenate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute recommendation metrics
        metrics = self.rec_metrics.compute_metrics(
            all_predictions,
            all_targets,
            k_values=self.config.top_k_values
        )
        
        # Average collapse metrics
        for key, values in all_collapse_metrics.items():
            metrics[f'collapse_{key}'] = np.mean(values)
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Full training loop"""
        self.logger.info(f"Starting training: {self.experiment_name}")
        self.logger.info(f"Config: {self.config}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Log training losses
            self.logger.info(f"Epoch {epoch} - Train losses: {train_losses}")
            for key, value in train_losses.items():
                self.history[f'train_{key}'].append(value)
            
            # Evaluate
            if epoch % self.config.eval_every == 0:
                val_metrics = self.evaluate(self.val_loader)
                self.logger.info(f"Epoch {epoch} - Val metrics: {val_metrics}")
                
                for key, value in val_metrics.items():
                    self.history[f'val_{key}'].append(value)
                
                # Early stopping check
                current_metric = val_metrics.get('NDCG@10', 0)
                if current_metric > self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, 'best')
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save periodic checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
        
        # Final test evaluation
        self.load_checkpoint('best')
        test_metrics = self.evaluate(self.test_loader)
        self.logger.info(f"Test metrics: {test_metrics}")
        
        return {
            'history': dict(self.history),
            'test_metrics': test_metrics,
            'best_val_metric': self.best_val_metric
        }
    
    def save_checkpoint(self, epoch: int, name: str):
        """Save model checkpoint"""
        checkpoint_dir = CHECKPOINTS_DIR / self.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'history': dict(self.history)
        }
        
        torch.save(checkpoint, checkpoint_dir / f'{name}.pt')
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint"""
        checkpoint_path = CHECKPOINTS_DIR / self.experiment_name / f'{name}.pt'
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded checkpoint: {name}")
        else:
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")


# =============================================================================
# Evaluation Functions
# =============================================================================
def evaluate_model_comprehensive(
    model: nn.Module,
    test_loader,
    device: str = "cuda",
    compression_depths: List[Optional[int]] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation including:
    - Standard metrics across compression depths
    - Collapse metrics
    - Efficiency metrics
    """
    if compression_depths is None:
        compression_depths = [None, 3, 6, 9]
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = {}
    rec_metrics = RecommendationMetrics()
    collapse_metrics = CollapseMetricsComputer()
    
    for depth in compression_depths:
        depth_key = f"depth_{depth}" if depth is not None else "no_compression"
        
        all_predictions = []
        all_targets = []
        all_R_scores = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Eval {depth_key}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(
                    input_ids,
                    attention_mask,
                    compression_depth=depth,
                    return_hidden_states=True
                )
                
                logits = outputs['logits'][:, -1, :]
                predictions = F.softmax(logits, dim=-1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
                # Compute R score for last hidden state
                hidden_states = outputs.get('last_hidden_state')
                if hidden_states is not None:
                    R_result = collapse_metrics.compute_reasoning_collapse_score(hidden_states)
                    all_R_scores.append(R_result['reasoning_collapse_score'])
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Recommendation metrics
        metrics = rec_metrics.compute_metrics(all_predictions, all_targets)
        metrics['avg_R_score'] = np.mean([r for r in all_R_scores if not np.isnan(r)])
        
        results[depth_key] = metrics
    
    # Efficiency metrics (if sample batch exists)
    sample_batch = next(iter(test_loader))
    input_ids = sample_batch['input_ids'].to(device)
    attention_mask = sample_batch['attention_mask'].to(device)
    
    for depth in compression_depths:
        depth_key = f"depth_{depth}" if depth is not None else "no_compression"
        
        latency = EfficiencyMetrics.measure_latency(
            model, input_ids, attention_mask,
            compression_depth=depth
        )
        memory = EfficiencyMetrics.measure_memory(
            model, input_ids, attention_mask,
            compression_depth=depth
        )
        
        results[depth_key].update(latency)
        results[depth_key].update(memory)
    
    return results


def save_results(results: Dict[str, Any], experiment_name: str, results_dir: Path = RESULTS_DIR):
    """Save experiment results"""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f"{experiment_name}_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    serializable = json.loads(json.dumps(results, default=convert))
    
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    return results_path


if __name__ == "__main__":
    # Test training pipeline
    print("Training pipeline module loaded successfully")
