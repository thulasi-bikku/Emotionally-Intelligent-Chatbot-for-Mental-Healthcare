"""
Metaphor-Aware Transformer Block.

Implements gated fusion of transformer outputs with metaphor-aware features.
Augments a standard transformer block by incorporating metaphorical language
understanding through gating mechanisms.

Mathematical formulation:
  - Base output: y_base = T(x) [standard transformer block]
  - Metaphor projection: m' = m @ W_M, r = tanh([x; m'] @ W_r)
  - Gate: g_i = sigmoid([y_base; r] @ W_g + b_g)
  - Fused: y_i = g_i ⊙ y_base + (1 - g_i) ⊙ f_m(r)

See docs/Metaphor_Aware_Transformer.md for detailed equations and integration.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable


class MetaphorAwareBlock(nn.Module):
    """
    Gated fusion block that enriches transformer outputs with metaphor awareness.
    
    This module is typically inserted after a transformer's feed-forward layer
    or as a separate residual branch before layer normalization.
    
    Args:
        d_model: Model hidden dimension
        metaphor_dim: Dimension of metaphor embeddings
        dropout: Dropout rate for stability
    """
    
    def __init__(
        self,
        d_model: int,
        metaphor_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.metaphor_dim = metaphor_dim
        
        # Metaphor projection: m' = m @ W_M
        self.W_M = nn.Linear(metaphor_dim, d_model)
        
        # Residual projection: r = tanh([x; m'] @ W_r)
        self.W_r = nn.Linear(2 * d_model, d_model)
        
        # Gate computation: g = sigmoid([y; r] @ W_g + b_g)
        self.W_g = nn.Linear(2 * d_model, d_model)
        self.b_g = nn.Parameter(torch.zeros(d_model))
        
        # Metaphor MLP: maps r to same dimension as y
        self.f_m = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize gates near 1.0 to preserve base behavior
        nn.init.xavier_uniform_(self.W_g.weight)
        self.b_g.data.fill_(2.0)  # Softly bias toward gate ≈ 0.88
    
    def forward(
        self,
        y_base: torch.Tensor,
        x: torch.Tensor,
        metaphor_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_base: Transformer output [batch_size, seq_len, d_model]
            x: Original input tokens [batch_size, seq_len, d_model]
            metaphor_embedding: Metaphor context [batch_size, metaphor_dim] or [metaphor_dim]
                                Can be per-token or per-sequence.
            
        Returns:
            Fused output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = y_base.shape
        
        # Project metaphor embedding
        if metaphor_embedding.dim() == 1:
            # Per-sequence metaphor: broadcast to batch
            m_proj = self.W_M(metaphor_embedding.unsqueeze(0))  # [1, d_model]
            m_proj = m_proj.unsqueeze(1).expand(batch_size, seq_len, -1)  # [B, L, d_model]
        elif metaphor_embedding.dim() == 2:
            # Per-sequence: [batch_size, metaphor_dim]
            m_proj = self.W_M(metaphor_embedding)  # [batch_size, d_model]
            m_proj = m_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, d_model]
        else:
            # Assume per-token: [batch_size, seq_len, metaphor_dim]
            m_proj = self.W_M(metaphor_embedding)  # [batch_size, seq_len, d_model]
        
        # Compute metaphor-aware residual
        x_m_concat = torch.cat([x, m_proj], dim=-1)  # [batch_size, seq_len, 2*d_model]
        r = torch.tanh(self.W_r(x_m_concat))  # [batch_size, seq_len, d_model]
        
        # Compute gate
        y_r_concat = torch.cat([y_base, r], dim=-1)  # [batch_size, seq_len, 2*d_model]
        gate_logits = self.W_g(y_r_concat) + self.b_g  # [batch_size, seq_len, d_model]
        g = torch.sigmoid(gate_logits)  # [batch_size, seq_len, d_model]
        
        # Apply metaphor MLP
        f_m_r = self.f_m(r)  # [batch_size, seq_len, d_model]
        
        # Gated fusion: y_i = g_i ⊙ y_base + (1 - g_i) ⊙ f_m(r)
        fused = g * y_base + (1.0 - g) * f_m_r  # [batch_size, seq_len, d_model]
        
        # Layer normalization for stability
        output = self.layer_norm(fused)
        
        return output


class MetaphorAwareTransformer(nn.Module):
    """
    Transformer with integrated metaphor-aware blocks.
    
    Wraps a standard transformer and inserts metaphor-aware fusion blocks
    at specified locations.
    
    Args:
        transformer: Base transformer module (e.g., from transformers library)
        metaphor_dim: Dimension of metaphor embeddings
        insert_positions: List of layer indices where to insert fusion
                         (e.g., [0, 1, 2] for first 3 layers)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        transformer: nn.Module,
        metaphor_dim: int,
        insert_positions: Optional[list] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = transformer
        self.metaphor_dim = metaphor_dim
        
        # Infer d_model from transformer
        if hasattr(transformer, "config"):
            d_model = transformer.config.hidden_size
        elif hasattr(transformer, "d_model"):
            d_model = transformer.d_model
        else:
            raise ValueError("Cannot infer d_model from transformer")
        
        # Insert metaphor-aware blocks with fallback for different architectures
        self.metaphor_blocks = nn.ModuleDict()
        if insert_positions is None:
            # Try to detect transformer layers robustly
            try:
                num_layers = len(transformer.encoder.layer)
            except AttributeError:
                # Fallback for decoder-only or custom architectures
                try:
                    num_layers = getattr(transformer.config, 'num_hidden_layers', 12)
                except:
                    num_layers = 12
            insert_positions = list(range(min(6, num_layers)))
        
        for pos in insert_positions:
            block_name = f"metaphor_block_{pos}"
            self.metaphor_blocks[block_name] = MetaphorAwareBlock(
                d_model=d_model,
                metaphor_dim=metaphor_dim,
                dropout=dropout,
            )
        
        self.insert_positions = {int(k.split("_")[-1]): k for k in self.metaphor_blocks.keys()}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        metaphor_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            metaphor_embedding: Metaphor embeddings [batch_size, metaphor_dim]
            
        Returns:
            Enriched transformer output [batch_size, seq_len, d_model]
        """
        # Get original input embeddings
        input_embeds = self.transformer.embeddings(input_ids)
        x = input_embeds
        
        # Process through transformer layers with metaphor fusion
        for layer_idx, layer in enumerate(self.transformer.encoder.layer):
            # Standard transformer layer
            x = layer(x, attention_mask=attention_mask)[0]
            
            # Insert metaphor fusion if applicable
            # Pass current contextual x, not original input_embeds, to preserve layer-wise context
            if layer_idx in self.insert_positions and metaphor_embedding is not None:
                block_name = self.insert_positions[layer_idx]
                metaphor_block = self.metaphor_blocks[block_name]
                x = metaphor_block(x, x, metaphor_embedding)
        
        return x


# Example usage:
if __name__ == "__main__":
    # Minimal example: test the MetaphorAwareBlock in isolation
    batch_size, seq_len, d_model = 2, 10, 256
    metaphor_dim = 64
    
    # Random tensors
    y_base = torch.randn(batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    metaphor_embedding = torch.randn(batch_size, metaphor_dim)
    
    # Create block
    block = MetaphorAwareBlock(d_model=d_model, metaphor_dim=metaphor_dim)
    
    # Forward pass
    output = block(y_base, x, metaphor_embedding)
    print(f"Input transformer output shape: {y_base.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == y_base.shape
    print("✓ MetaphorAwareBlock test passed")
