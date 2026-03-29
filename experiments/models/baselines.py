"""
Baseline Sequential Recommendation Models

Implements:
1. SASRec - Self-Attentive Sequential Recommendation
2. BERT4Rec - BERT for Sequential Recommendation
3. GRU4Rec - GRU-based Sequential Recommendation

These baselines provide standard recommendation quality reference points
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# SASRec: Self-Attentive Sequential Recommendation
# =============================================================================
class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)
    
    Uses unidirectional (causal) self-attention for next-item prediction
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        
        # Self-attention blocks
        self.attention_layers = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.output_layer = nn.Linear(hidden_dim, num_items + 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        x = self.layer_norm(x)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1
        ).bool()
        
        # Attention layers
        all_hidden_states = [x] if return_hidden_states else []
        
        for layer in self.attention_layers:
            x = layer(x, causal_mask, attention_mask)
            if return_hidden_states:
                all_hidden_states.append(x)
        
        # Output
        logits = self.output_layer(x)
        
        result = {
            'logits': logits,
            'last_hidden_state': x
        }
        
        if return_hidden_states:
            result['all_hidden_states'] = all_hidden_states
        
        return result
    
    def get_recommendation_distribution(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs['logits'][:, -1, :]
        return F.softmax(logits, dim=-1)


class SASRecBlock(nn.Module):
    """Single SASRec attention block"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-8)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        causal_mask: Tensor,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Self-attention
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=(1 - padding_mask).bool() if padding_mask is not None else None
        )
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # FFN
        x = x + self.ffn(x)
        x = self.norm2(x)
        
        return x


# =============================================================================
# BERT4Rec: BERT for Sequential Recommendation
# =============================================================================
class BERT4Rec(nn.Module):
    """
    BERT-based Sequential Recommendation (Sun et al., 2019)
    
    Uses bidirectional self-attention with masked item prediction
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        mask_prob: float = 0.2
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_token = num_items  # Use special mask token
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)  # +2 for pad and mask
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.output_layer = nn.Linear(hidden_dim, num_items + 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def mask_sequence(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply random masking for training"""
        masked_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)  # -100 for ignore
        
        # Only mask non-padding tokens
        prob_matrix = torch.full(input_ids.shape, self.mask_prob, device=input_ids.device)
        prob_matrix[input_ids == 0] = 0  # Don't mask padding
        
        mask = torch.bernoulli(prob_matrix).bool()
        labels[mask] = input_ids[mask]
        masked_ids[mask] = self.mask_token
        
        return masked_ids, labels
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False,
        training: bool = False
    ) -> Dict[str, Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Apply masking during training
        if training:
            input_ids, labels = self.mask_sequence(input_ids)
        else:
            labels = None
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(self.layer_norm(x))
        
        # Padding mask (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = (1 - attention_mask).bool()
        else:
            src_key_padding_mask = None
        
        # Encoder
        hidden_states = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Output
        logits = self.output_layer(hidden_states)
        
        result = {
            'logits': logits,
            'last_hidden_state': hidden_states,
            'labels': labels
        }
        
        if return_hidden_states:
            result['all_hidden_states'] = [x, hidden_states]
        
        return result
    
    def get_recommendation_distribution(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        # For prediction, mask the last position
        masked_ids = input_ids.clone()
        masked_ids[:, -1] = self.mask_token
        
        outputs = self.forward(masked_ids, attention_mask, training=False)
        logits = outputs['logits'][:, -1, :]
        return F.softmax(logits, dim=-1)


# =============================================================================
# GRU4Rec: GRU-based Sequential Recommendation
# =============================================================================
class GRU4Rec(nn.Module):
    """
    GRU-based Sequential Recommendation (Hidasi et al., 2016)
    
    Uses GRU for session-based recommendation
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        
        embedding_dim = embedding_dim or hidden_dim
        
        # Item embedding
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output
        self.output_layer = nn.Linear(hidden_dim, num_items + 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, Tensor]:
        # Embedding
        x = self.item_embedding(input_ids)
        
        # Pack if we have attention mask (for variable length)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, hidden = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, hidden = self.gru(x)
        
        # Output logits
        logits = self.output_layer(output)
        
        result = {
            'logits': logits,
            'last_hidden_state': output
        }
        
        if return_hidden_states:
            result['all_hidden_states'] = [x, output]
        
        return result
    
    def get_recommendation_distribution(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs['logits'][:, -1, :]
        return F.softmax(logits, dim=-1)


# =============================================================================
# Prompt-Pruned LLM Baseline
# =============================================================================
class PromptPrunedLLM(nn.Module):
    """
    LLM-based recommender with prompt truncation/pruning
    Baseline for comparison with CARR
    """
    
    def __init__(
        self,
        base_model,
        pruning_ratio: float = 0.5,
        pruning_strategy: str = "recent"
    ):
        super().__init__()
        self.base_model = base_model
        self.pruning_ratio = pruning_ratio
        self.pruning_strategy = pruning_strategy
    
    def prune_input(
        self,
        input_ids: Tensor,
        attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Prune input sequence before passing to model"""
        batch_size, seq_len = input_ids.shape
        
        # Calculate target length
        target_len = max(1, int(seq_len * (1 - self.pruning_ratio)))
        
        if self.pruning_strategy == "recent":
            # Keep most recent items
            pruned_ids = input_ids[:, -target_len:]
            pruned_mask = attention_mask[:, -target_len:]
            
        elif self.pruning_strategy == "uniform":
            # Uniformly sample items
            indices = torch.linspace(0, seq_len - 1, target_len).long()
            pruned_ids = input_ids[:, indices]
            pruned_mask = attention_mask[:, indices]
            
        else:
            raise ValueError(f"Unknown pruning strategy: {self.pruning_strategy}")
        
        return pruned_ids, pruned_mask
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Prune input
        pruned_ids, pruned_mask = self.prune_input(input_ids, attention_mask)
        
        # Forward through base model
        return self.base_model(pruned_ids, pruned_mask, **kwargs)


# =============================================================================
# Layer-Skipping LLM Baseline
# =============================================================================
class LayerSkippingLLM(nn.Module):
    """
    LLM-based recommender with aggressive late-layer skipping
    Baseline for comparison with CARR
    """
    
    def __init__(
        self,
        base_model,
        skip_layers: List[int] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.skip_layers = set(skip_layers) if skip_layers else set()
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        # Modified forward that skips certain layers
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        hidden_states = self.base_model.item_embedding(input_ids)
        hidden_states = self.base_model.position_encoding(hidden_states)
        
        if attention_mask is not None:
            transformer_mask = (1 - attention_mask).bool()
        else:
            transformer_mask = None
        
        # Process layers, skipping specified ones
        for layer_idx, layer in enumerate(self.base_model.layers):
            if layer_idx not in self.skip_layers:
                hidden_states, _ = layer(hidden_states, attention_mask=transformer_mask)
        
        # Final output
        hidden_states = self.base_model.final_norm(hidden_states)
        logits = self.base_model.output_projection(hidden_states)
        
        return {
            'logits': logits,
            'last_hidden_state': hidden_states
        }


# =============================================================================
# Model Factory
# =============================================================================
def create_baseline_model(
    model_type: str,
    num_items: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    **kwargs
) -> nn.Module:
    """Create baseline model by type"""
    
    if model_type == "sasrec":
        return SASRec(
            num_items=num_items,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs
        )
    elif model_type == "bert4rec":
        return BERT4Rec(
            num_items=num_items,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs
        )
    elif model_type == "gru4rec":
        return GRU4Rec(
            num_items=num_items,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown baseline model: {model_type}")


if __name__ == "__main__":
    # Test baselines
    num_items = 10000
    batch_size = 4
    seq_len = 50
    
    input_ids = torch.randint(1, num_items, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Test SASRec
    print("Testing SASRec...")
    sasrec = create_baseline_model("sasrec", num_items)
    outputs = sasrec(input_ids, attention_mask)
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    # Test BERT4Rec
    print("\nTesting BERT4Rec...")
    bert4rec = create_baseline_model("bert4rec", num_items)
    outputs = bert4rec(input_ids, attention_mask, training=True)
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    # Test GRU4Rec
    print("\nTesting GRU4Rec...")
    gru4rec = create_baseline_model("gru4rec", num_items)
    outputs = gru4rec(input_ids, attention_mask)
    print(f"  Logits shape: {outputs['logits'].shape}")
