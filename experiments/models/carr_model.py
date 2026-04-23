"""
CARR: Collapse-Aware Register Recommendation
Model architecture for LLM-based recommendation with adaptive compression

Key Components:
1. Transformer backbone for sequential recommendation
2. Register-based compression mechanism
3. Adaptive depth selection based on collapse monitoring
4. Collapse-aware training objectives
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import ModelConfig


# =============================================================================
# Positional Encoding
# =============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        return self.dropout(x)


# =============================================================================
# Transformer Layer
# =============================================================================
class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_idx: int = 0
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_output, attn_weights = self.attention(
            normed, normed, normed,
            key_padding_mask=attention_mask,
            need_weights=return_attention
        )
        x = x + attn_output
        
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_weights


# =============================================================================
# Register Compression Module
# =============================================================================
class RegisterCompressor(nn.Module):
    """
    Compresses full sequence representation into register tokens
    Implements the compression operator C_{k,m} from the paper
    """
    
    def __init__(
        self,
        d_model: int,
        max_registers: int = 32,
        compression_type: str = "attention"
    ):
        super().__init__()
        self.d_model = d_model
        self.max_registers = max_registers
        self.compression_type = compression_type
        
        # Learnable register tokens
        self.register_tokens = nn.Parameter(torch.randn(max_registers, d_model) * 0.02)
        
        # Cross-attention for compression
        self.compress_attention = nn.MultiheadAttention(
            d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Projection for adaptive width
        self.width_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_registers),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_registers: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compress hidden states into register representation
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len)
            num_registers: Number of registers to use (if adaptive)
        
        Returns:
            register_states: (batch, num_registers, d_model)
            compression_weights: Attention weights for interpretability
        """
        batch_size = hidden_states.size(0)
        
        if num_registers is None:
            num_registers = self.max_registers
        
        # Get register queries
        registers = self.register_tokens[:num_registers].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention: registers attend to hidden states
        if self.compression_type == "attention":
            compressed, weights = self.compress_attention(
                registers,
                hidden_states,
                hidden_states,
                key_padding_mask=attention_mask
            )
        elif self.compression_type == "pooling":
            # Mean pooling with masking
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states * (1 - mask)
            compressed = hidden_states.mean(dim=1, keepdim=True).expand(-1, num_registers, -1)
            weights = None
        else:
            raise ValueError(f"Unknown compression type: {self.compression_type}")
        
        return compressed, weights
    
    def predict_width(self, hidden_states: Tensor) -> int:
        """Predict adaptive number of registers based on content"""
        pooled = hidden_states.mean(dim=1)
        width_dist = self.width_predictor(pooled)
        # Return expected width
        expected_width = (width_dist * torch.arange(1, self.max_registers + 1, device=width_dist.device)).sum(dim=-1)
        return int(expected_width.mean().item())


# =============================================================================
# CARR Model
# =============================================================================
class CARRModel(nn.Module):
    """
    Collapse-Aware Register Recommendation Model
    
    Implements Algorithm 1 from the paper with:
    - Adaptive compression depth selection
    - Collapse monitoring at specified layers
    - Evidence survival tracking
    - Multi-objective training
    """
    
    def __init__(
        self,
        num_items: int,
        config: ModelConfig
    ):
        super().__init__()
        self.config = config
        self.num_items = num_items
        
        # Item embedding
        self.item_embedding = nn.Embedding(
            num_items + 1,  # +1 for padding
            config.hidden_dim,
            padding_idx=0
        )
        
        # Positional encoding
        self.position_encoding = LearnedPositionalEncoding(
            config.hidden_dim,
            max_len=512,
            dropout=config.dropout
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim,
                config.num_heads,
                config.ffn_dim,
                config.dropout,
                layer_idx=i
            )
            for i in range(config.num_layers)
        ])
        
        # Register compressor
        self.compressor = RegisterCompressor(
            config.hidden_dim,
            config.max_registers,
            compression_type="attention"
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, num_items + 1, bias=False)
        
        # Collapse monitoring components
        self.collapse_threshold_R = config.collapse_threshold_R
        self.collapse_threshold_S = config.collapse_threshold_S
        self.collapse_risk_threshold = config.collapse_risk_threshold
        self.monitored_layers = set(config.monitored_layers)
        
        # Intent clustering for collapse detection
        self.num_intent_clusters = config.num_intent_clusters
        
        # Training mode flags
        self.use_adaptive_depth = True
        self.use_adaptive_width = True
        self.use_collapse_reg = True
        self.use_evidence_reg = True
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        compression_depth: Optional[int] = None,
        num_registers: Optional[int] = None,
        return_hidden_states: bool = False,
        return_collapse_metrics: bool = False
    ) -> Dict[str, Tensor]:
        """
        Forward pass with optional compression
        
        Args:
            input_ids: (batch, seq_len) - Item IDs
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
            compression_depth: Layer at which to apply compression (None = no compression)
            num_registers: Number of registers for compression
            return_hidden_states: Whether to return all hidden states
            return_collapse_metrics: Whether to compute collapse metrics
        
        Returns:
            Dictionary containing:
                - logits: (batch, seq_len, num_items+1) or (batch, num_registers, num_items+1)
                - hidden_states: List of hidden states if requested
                - collapse_metrics: Dict of collapse metrics if requested
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        hidden_states = self.item_embedding(input_ids)
        hidden_states = self.position_encoding(hidden_states)
        
        # Convert attention mask for transformer (True = ignore position)
        if attention_mask is not None:
            transformer_mask = ~attention_mask.bool()
        else:
            transformer_mask = None
        
        all_hidden_states = [hidden_states] if return_hidden_states else []
        collapse_metrics = {} if return_collapse_metrics else None
        
        compression_applied = False
        compression_weights = None
        
        # Layer-by-layer processing
        for layer_idx, layer in enumerate(self.layers):
            # Check if we should compress at this layer
            if compression_depth is not None and layer_idx == compression_depth and not compression_applied:
                # Determine number of registers
                if num_registers is None and self.use_adaptive_width:
                    num_registers = self.compressor.predict_width(hidden_states)
                elif num_registers is None:
                    num_registers = self.config.num_registers
                
                # Apply compression
                hidden_states, compression_weights = self.compressor(
                    hidden_states,
                    transformer_mask,
                    num_registers
                )
                
                # Update mask for compressed representation
                transformer_mask = None  # No masking needed for registers
                compression_applied = True
            
            # Apply transformer layer
            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=transformer_mask,
                return_attention=return_collapse_metrics
            )
            
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Compute collapse metrics at monitored layers
            if return_collapse_metrics and layer_idx in self.monitored_layers:
                layer_metrics = self._compute_layer_collapse_metrics(
                    hidden_states,
                    attn_weights,
                    layer_idx
                )
                for key, value in layer_metrics.items():
                    collapse_metrics[f"layer_{layer_idx}_{key}"] = value
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Output logits
        logits = self.output_projection(hidden_states)
        
        result = {
            'logits': logits,
            'last_hidden_state': hidden_states,
            'compression_applied': compression_applied,
            'compression_weights': compression_weights
        }
        
        if return_hidden_states:
            result['all_hidden_states'] = all_hidden_states
        
        if return_collapse_metrics:
            result['collapse_metrics'] = collapse_metrics
        
        return result
    
    def _compute_layer_collapse_metrics(
        self,
        hidden_states: Tensor,
        attention_weights: Optional[Tensor],
        layer_idx: int
    ) -> Dict[str, Tensor]:
        """Compute collapse-related metrics for a layer"""
        
        metrics = {}
        
        # Compute within-class and between-class covariances
        # This requires clustering the hidden states
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Flatten for clustering
        flat_states = hidden_states.contiguous().view(-1, hidden_dim)
        
        # Simple k-means-style clustering (differentiable approximation)
        # For efficiency, use random subset if too large
        if flat_states.size(0) > 1000:
            indices = torch.randperm(flat_states.size(0))[:1000]
            flat_states = flat_states[indices]
        
        # Initialize cluster centers
        cluster_centers = flat_states[torch.randperm(flat_states.size(0))[:self.num_intent_clusters]]
        
        # Assign to nearest cluster
        distances = torch.cdist(flat_states, cluster_centers)
        assignments = distances.argmin(dim=-1)
        
        # Compute within-cluster and between-cluster covariances
        global_mean = flat_states.mean(dim=0)
        
        sigma_w = torch.zeros(hidden_dim, hidden_dim, device=hidden_states.device)
        sigma_b = torch.zeros(hidden_dim, hidden_dim, device=hidden_states.device)
        
        for k in range(self.num_intent_clusters):
            mask = (assignments == k)
            if mask.sum() > 0:
                cluster_points = flat_states[mask]
                cluster_mean = cluster_points.mean(dim=0)
                
                # Within-cluster covariance
                centered = cluster_points - cluster_mean
                sigma_w += centered.T @ centered
                
                # Between-cluster covariance
                mean_diff = cluster_mean - global_mean
                sigma_b += mask.sum() * mean_diff.unsqueeze(1) @ mean_diff.unsqueeze(0)
        
        sigma_w /= flat_states.size(0)
        sigma_b /= flat_states.size(0)
        
        # Reasoning collapse score: tr(Σ_W) / tr(Σ_B)
        trace_w = sigma_w.trace()
        trace_b = sigma_b.trace() + 1e-8  # Avoid division by zero
        
        reasoning_collapse_score = trace_w / trace_b
        
        metrics['reasoning_collapse_score'] = reasoning_collapse_score
        metrics['trace_sigma_w'] = trace_w
        metrics['trace_sigma_b'] = trace_b
        
        return metrics
    
    def compute_adaptive_compression_depth(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tuple[int, Dict[str, float]]:
        """
        Determine optimal compression depth based on collapse risk
        Implements the adaptive depth selection from Algorithm 1
        """
        
        # Forward pass collecting collapse metrics at all monitored layers
        batch_size, seq_len = input_ids.shape
        hidden_states = self.item_embedding(input_ids)
        hidden_states = self.position_encoding(hidden_states)
        
        if attention_mask is not None:
            transformer_mask = ~attention_mask.bool()
        else:
            transformer_mask = None
        
        collapse_risks = {}
        selected_depth = self.config.num_layers  # Default: no compression
        
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=transformer_mask,
                return_attention=True
            )
            
            if layer_idx in self.monitored_layers:
                metrics = self._compute_layer_collapse_metrics(
                    hidden_states,
                    attn_weights,
                    layer_idx
                )
                
                # Compute collapse risk (Γ_l from paper)
                R_l = metrics['reasoning_collapse_score'].item()
                
                # Collapse risk increases with lower R (more collapse)
                # Safe to compress when R is still high enough
                collapse_risk = 1.0 / (R_l + 1e-8)
                collapse_risks[layer_idx] = collapse_risk
                
                # Select first layer where collapse risk is acceptable
                if self.use_adaptive_depth and collapse_risk <= self.collapse_risk_threshold:
                    if layer_idx < selected_depth:
                        selected_depth = layer_idx
        
        return selected_depth, collapse_risks
    
    def get_recommendation_distribution(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        compression_depth: Optional[int] = None
    ) -> Tensor:
        """Get probability distribution over items"""
        outputs = self.forward(
            input_ids,
            attention_mask,
            compression_depth=compression_depth
        )
        
        # Use last position for prediction
        logits = outputs['logits'][:, -1, :]  # (batch, num_items+1)
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def compute_evidence_survival(
        self,
        input_ids_full: Tensor,
        input_ids_ablated: Tensor,
        attention_mask_full: Optional[Tensor] = None,
        attention_mask_ablated: Optional[Tensor] = None,
        compression_depth: Optional[int] = None,
        divergence_type: str = "js"
    ) -> Dict[str, Tensor]:
        """
        Compute evidence survival score S_l(τ)
        
        Args:
            input_ids_full: Full history H
            input_ids_ablated: History with τ removed (H \ τ)
            divergence_type: "js" (Jensen-Shannon), "tv" (Total Variation), or "kl"
        
        Returns:
            Dictionary with layerwise evidence survival scores
        """
        
        # Get distributions for full and ablated history
        prob_full = self.get_recommendation_distribution(
            input_ids_full,
            attention_mask_full,
            compression_depth
        )
        
        prob_ablated = self.get_recommendation_distribution(
            input_ids_ablated,
            attention_mask_ablated,
            compression_depth
        )
        
        # Compute divergence
        if divergence_type == "js":
            # Jensen-Shannon divergence
            m = 0.5 * (prob_full + prob_ablated)
            kl_pm = F.kl_div(m.log(), prob_full, reduction='none').sum(dim=-1)
            kl_qm = F.kl_div(m.log(), prob_ablated, reduction='none').sum(dim=-1)
            divergence = 0.5 * (kl_pm + kl_qm)
            
        elif divergence_type == "tv":
            # Total variation distance
            divergence = 0.5 * (prob_full - prob_ablated).abs().sum(dim=-1)
            
        elif divergence_type == "kl":
            # KL divergence
            divergence = F.kl_div(
                prob_ablated.log(),
                prob_full,
                reduction='none'
            ).sum(dim=-1)
        
        return {
            'evidence_survival': divergence,
            'prob_full': prob_full,
            'prob_ablated': prob_ablated
        }


# =============================================================================
# Fixed Compression Baseline
# =============================================================================
class FixedCompressionModel(CARRModel):
    """
    Baseline model with fixed compression depth and width
    No adaptive mechanisms
    """
    
    def __init__(self, num_items: int, config: ModelConfig):
        super().__init__(num_items, config)
        self.use_adaptive_depth = False
        self.use_adaptive_width = False
        self.fixed_compression_depth = config.compression_depth
        self.fixed_num_registers = config.num_registers
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False,
        return_collapse_metrics: bool = False,
        **kwargs  # absorb any extra keyword args (e.g. externally passed compression_depth)
    ) -> Dict[str, Tensor]:
        return super().forward(
            input_ids,
            attention_mask,
            compression_depth=self.fixed_compression_depth,
            num_registers=self.fixed_num_registers,
            return_hidden_states=return_hidden_states,
            return_collapse_metrics=return_collapse_metrics
        )


# =============================================================================
# Full LLM Baseline (No Compression)
# =============================================================================
class FullLLMModel(CARRModel):
    """
    Full LLM without any compression
    Baseline for quality comparison
    """
    
    def __init__(self, num_items: int, config: ModelConfig):
        super().__init__(num_items, config)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False,
        return_collapse_metrics: bool = False,
        **kwargs  # absorb any extra keyword args (e.g. compression_depth)
    ) -> Dict[str, Tensor]:
        # No compression
        return super().forward(
            input_ids,
            attention_mask,
            compression_depth=None,
            num_registers=None,
            return_hidden_states=return_hidden_states,
            return_collapse_metrics=return_collapse_metrics
        )


# =============================================================================
# Model Factory
# =============================================================================
def create_model(
    model_type: str,
    num_items: int,
    config: ModelConfig
) -> nn.Module:
    """Create model by type"""
    
    if model_type == "carr":
        return CARRModel(num_items, config)
    elif model_type == "fixed_compression":
        return FixedCompressionModel(num_items, config)
    elif model_type == "full_llm":
        return FullLLMModel(num_items, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    config = ModelConfig()
    model = CARRModel(num_items=10000, config=config)
    
    # Test input
    batch_size = 4
    seq_len = 50
    input_ids = torch.randint(1, 10000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    outputs = model(
        input_ids,
        attention_mask,
        compression_depth=6,
        return_hidden_states=True,
        return_collapse_metrics=True
    )
    
    print("Output keys:", outputs.keys())
    print("Logits shape:", outputs['logits'].shape)
    print("Collapse metrics:", outputs.get('collapse_metrics', {}))
