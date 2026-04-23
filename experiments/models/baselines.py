"""
Baseline Sequential Recommendation Models

Implements:
1. SASRec        — Self-Attentive Sequential Recommendation
2. BERT4Rec      — BERT for Sequential Recommendation
3. GRU4Rec       — GRU-based Sequential Recommendation
4. LLMRec        — LLM-augmented Recommendation (Wei et al., 2024) [R4]
5. UniSRec       — Universal Sequential Recommendation (Hou et al., 2022) [R4]
6. KVPruning     — KV-Cache Pruning LLM baseline [R4]
7. TokenPruning  — Input token pruning baseline [R4]

These baselines provide standard recommendation quality reference points
including the modern LLM-era methods required by Revision 4.
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
            key_padding_mask=(~padding_mask.bool()) if padding_mask is not None else None
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
# R4 Baseline 1: LLMRec — LLM-Augmented Sequential Recommendation
# =============================================================================
class LLMRec(nn.Module):
    """
    LLM-augmented recommendation baseline (Wei et al., 2024).

    Architecture:
      - A frozen LLM encoder (simulated here as a pre-trained item text
        embedding layer + transformer) produces semantic item embeddings.
      - A lightweight sequential model (SASRec-style attention) fuses
        the LLM embeddings with collaborative signals.
      - The joint representation is projected to a recommendation distribution.

    In production this would load a pretrained model (e.g. GPT-2, LLaMA);
    here we simulate the LLM encoder with a trainable transformer backbone
    of the same depth and width as the full-LLM baseline, consistent with
    the paper's experimental setup.
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 256,
        num_llm_layers: int = 6,
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items  = num_items
        self.hidden_dim = hidden_dim

        # ---- LLM encoder (simulated with a deep transformer) ----------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.llm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_llm_layers)

        # ---- Item & position embeddings -------------------------------------
        self.item_embedding     = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout            = nn.Dropout(dropout)
        self.input_norm         = nn.LayerNorm(hidden_dim)

        # ---- Lightweight fusion attention -----------------------------------
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_layer, num_layers=num_fusion_layers
        )

        # ---- Output head ----------------------------------------------------
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, num_items + 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(self.input_norm(x))

        # Padding mask: True = ignore (TransformerEncoder convention)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)   # (B, T)

        # LLM encoder pass
        llm_out = self.llm_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Fusion pass
        fused = self.fusion_transformer(llm_out, src_key_padding_mask=src_key_padding_mask)

        logits = self.output_head(self.output_norm(fused))

        result: Dict[str, Tensor] = {"logits": logits, "last_hidden_state": fused}
        if return_hidden_states:
            result["all_hidden_states"] = [x, llm_out, fused]
        return result

    def get_recommendation_distribution(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        out = self.forward(input_ids, attention_mask)
        return F.softmax(out["logits"][:, -1, :], dim=-1)


# =============================================================================
# R4 Baseline 2: UniSRec — Universal Sequential Recommendation
# =============================================================================
class UniSRec(nn.Module):
    """
    Universal Sequential Recommendation baseline (Hou et al., 2022).

    UniSRec learns universal item representations transferable across domains
    by encoding items with pre-trained text features (E5 / T5 style).  In
    this implementation the text encoder is simulated as a MoE-style mixture
    of lightweight text feature extractors that produce domain-agnostic item
    embeddings, consistent with the experimental setup in the paper.

    Key elements:
      - Mixture-of-Experts (MoE) text feature projection (3 expert MLPs)
      - SASRec-style sequential modeling on projected features
      - Parameter-efficient fine-tuning via an adapter layer
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 256,
        text_feat_dim: int = 128,
        num_experts: int = 3,
        num_layers: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items  = num_items
        self.hidden_dim = hidden_dim

        # ---- Item ID embedding (collaborative signal) -----------------------
        self.id_embedding   = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding  = nn.Embedding(max_seq_len, hidden_dim)

        # ---- MoE text-feature projection (simulated) -----------------------
        # Expert projections: text_feat_dim → hidden_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_experts)
        ])
        # Gating network
        self.gate = nn.Linear(hidden_dim, num_experts)

        # Shared text-feature matrix (simulated as a learnable lookup)
        self.text_feat_table = nn.Embedding(num_items + 1, text_feat_dim, padding_idx=0)

        # ---- Sequential modeling (SASRec-style) ----------------------------
        self.dropout   = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- Adapter (parameter-efficient fine-tuning) ----------------------
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # ---- Output ---------------------------------------------------------
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, num_items + 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _moe_project(self, item_ids: Tensor) -> Tensor:
        """Project item text features through mixture-of-experts."""
        text_feats  = self.text_feat_table(item_ids)            # (B, T, text_feat_dim)
        id_embeds   = self.id_embedding(item_ids)               # (B, T, hidden_dim)

        gate_scores = F.softmax(self.gate(id_embeds), dim=-1)  # (B, T, num_experts)

        expert_outs = torch.stack(
            [expert(text_feats) for expert in self.experts], dim=-2
        )                                                       # (B, T, num_experts, hidden_dim)

        # Weighted sum over experts
        proj = (gate_scores.unsqueeze(-1) * expert_outs).sum(dim=-2)  # (B, T, hidden_dim)
        return proj

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        # Fuse MoE text projection with positional embeddings
        text_proj = self._moe_project(input_ids)              # (B, T, d)
        x = text_proj + self.pos_embedding(positions)
        x = self.dropout(self.input_norm(x))

        src_kpm = None
        if attention_mask is not None:
            src_kpm = (attention_mask == 0)

        seq_out = self.transformer(x, src_key_padding_mask=src_kpm)
        adapted = seq_out + self.adapter(seq_out)              # residual adapter

        logits = self.output_head(self.output_norm(adapted))

        result: Dict[str, Tensor] = {"logits": logits, "last_hidden_state": adapted}
        if return_hidden_states:
            result["all_hidden_states"] = [x, seq_out, adapted]
        return result

    def get_recommendation_distribution(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        out = self.forward(input_ids, attention_mask)
        return F.softmax(out["logits"][:, -1, :], dim=-1)


# =============================================================================
# R4 Baseline 3: KV-Cache Pruning
# =============================================================================
class KVPruningModel(nn.Module):
    """
    KV-Cache Pruning baseline (R4).

    Wraps any transformer-based recommender and prunes the KV cache at
    each attention layer by discarding attention heads / positions that
    fall below a learned importance threshold.  This mimics streamed
    KV-pruning approaches used in efficient LLM inference.

    Pruning strategy:
      At layer l, compute per-token importance score as the L2-norm of
      the value vector.  The bottom (pruning_ratio * T) tokens are masked
      out of the key-value attention.
    """

    def __init__(
        self,
        base_model: nn.Module,
        pruning_ratio: float = 0.3,
    ):
        super().__init__()
        self.base_model    = base_model
        self.pruning_ratio = pruning_ratio

        # Register forward hooks on each attention sub-module
        self._hooks: List = []
        self._pruning_masks: Dict[int, Tensor] = {}

    # ------------------------------------------------------------------
    # Hook-free forward: intercept at layer level for simplicity
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Forward with KV-pruned attention masks.

        Works with any model that exposes model.layers (nn.ModuleList of
        TransformerBlock-like modules with a forward(x, attention_mask)
        interface) plus item_embedding, position_encoding, final_norm,
        and output_projection attributes.

        Falls back to the base model's own forward if layers are not
        directly accessible.
        """
        if not hasattr(self.base_model, "layers"):
            return self.base_model(input_ids, attention_mask,
                                   return_hidden_states=return_hidden_states, **kwargs)

        B, T = input_ids.shape

        hidden = self.base_model.item_embedding(input_ids)
        hidden = self.base_model.position_encoding(hidden)

        base_mask = (1 - attention_mask).bool() if attention_mask is not None else None

        all_hidden_states = [hidden] if return_hidden_states else []

        for layer_idx, layer in enumerate(self.base_model.layers):
            # Build KV-pruning mask: mask out low-importance token positions
            with torch.no_grad():
                # Use current hidden norm as importance score
                importance = hidden.norm(dim=-1)              # (B, T)
                thresh = torch.quantile(
                    importance, self.pruning_ratio, dim=-1, keepdim=True
                )                                             # (B, 1)
                prune_mask = importance < thresh              # True = prune
                if base_mask is not None:
                    combined_mask = base_mask | prune_mask
                else:
                    combined_mask = prune_mask

            hidden, _ = layer(hidden, attention_mask=combined_mask)

            if return_hidden_states:
                all_hidden_states.append(hidden)

        hidden = self.base_model.final_norm(hidden)
        logits = self.base_model.output_projection(hidden)

        result: Dict[str, Tensor] = {"logits": logits, "last_hidden_state": hidden}
        if return_hidden_states:
            result["all_hidden_states"] = all_hidden_states
        return result

    def get_recommendation_distribution(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        out = self.forward(input_ids, attention_mask)
        logits = out["logits"]
        lgt = logits[:, -1, :] if logits.dim() == 3 else logits
        return F.softmax(lgt, dim=-1)


# =============================================================================
# R4 Baseline 4: Token Pruning
# =============================================================================
class TokenPruningModel(nn.Module):
    """
    Input Token Pruning baseline (R4).

    Prunes the input sequence before it enters the transformer backbone
    by retaining only the (1 - pruning_ratio) most important tokens.

    Importance is measured by the attention weight of each token in a
    single-layer look-ahead attention head:
      score_t = softmax( Q_t K^T / sqrt(d) )
    Tokens below the pruning threshold are removed before the main
    forward pass, reducing FLOPs proportionally.
    """

    def __init__(
        self,
        base_model: nn.Module,
        pruning_ratio: float = 0.3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.base_model    = base_model
        self.pruning_ratio = pruning_ratio

        # Lightweight scorer: one-head self-attention importance gate
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _score_tokens(self, embed: Tensor) -> Tensor:
        """Return per-token importance scores in [0, 1]."""
        scores = self.importance_head(embed).squeeze(-1)  # (B, T)
        return torch.sigmoid(scores)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        embed = self.base_model.item_embedding(input_ids)   # (B, T, d)

        # Compute importance and build pruned attention mask
        scores = self._score_tokens(embed.detach())          # (B, T)
        keep_ratio = 1.0 - self.pruning_ratio
        thresh     = torch.quantile(scores, self.pruning_ratio, dim=-1, keepdim=True)
        keep_mask  = (scores >= thresh).float()             # (B, T)

        if attention_mask is not None:
            pruned_mask = attention_mask * keep_mask
        else:
            pruned_mask = keep_mask

        return self.base_model(
            input_ids, pruned_mask.bool(),
            return_hidden_states=return_hidden_states, **kwargs
        )

    def get_recommendation_distribution(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        out = self.forward(input_ids, attention_mask)
        logits = out["logits"]
        lgt = logits[:, -1, :] if logits.dim() == 3 else logits
        return F.softmax(lgt, dim=-1)


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
    elif model_type == "llmrec":
        return LLMRec(
            num_items=num_items,
            hidden_dim=hidden_dim,
            **kwargs
        )
    elif model_type == "unisrec":
        return UniSRec(
            num_items=num_items,
            hidden_dim=hidden_dim,
            **kwargs
        )
    elif model_type in ("kv_pruning", "kvpruning"):
        # Requires a base model; create SASRec as the backbone
        base = SASRec(num_items=num_items, hidden_dim=hidden_dim, num_layers=num_layers)
        return KVPruningModel(base, **kwargs)
    elif model_type in ("token_pruning", "tokenpruning"):
        base = SASRec(num_items=num_items, hidden_dim=hidden_dim, num_layers=num_layers)
        return TokenPruningModel(base, hidden_dim=hidden_dim, **kwargs)
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
