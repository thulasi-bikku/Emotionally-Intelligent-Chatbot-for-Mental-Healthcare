"""
Cultural Attention Layer for transformer models.

Implements the Cultural Attention mechanism described in the manuscript.
Augments standard self-attention with cultural context vectors to bias
attention scores toward culturally-relevant tokens.

Mathematical formulation:
  - Base attention: A_base[i,j] = Q_i · K_j^T / √(d_k)
  - Cultural bias: B[i,j] = λ · (Q_i · (c @ W_C)^T)
  - Combined: A[i,j] = A_base[i,j] + B[i,j]
  - Softmax: α[i,j] = softmax_j(A[i,j])
  - Output: z_i = Σ_j α[i,j] · V_j

See docs/Cultural_Attention.md for detailed equations and integration guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CulturalAttention(nn.Module):
    """
    Single Cultural Attention Head.
    
    Modifies scaled-dot-product attention by adding a cultural bias term
    that's computed from a cultural embedding vector.
    
    Args:
        d_model: Model hidden dimension
        d_k: Dimension of keys/queries (typically d_model // num_heads)
        cultural_dim: Dimension of cultural embeddings (can equal d_k)
        lambda_type: "scalar" (learnable float) or "mlp" (small MLP)
    """
    
    def __init__(
        self,
        d_model: int,
        d_k: int,
        cultural_dim: int = None,
        lambda_type: str = "scalar",
    ):
        super().__init__()
        self.d_k = d_k
        self.d_model = d_model
        self.cultural_dim = cultural_dim or d_k
        
        # Standard attention projections
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        
        # Cultural projection
        self.W_C = nn.Linear(self.cultural_dim, d_k)
        
        # Learnable cultural influence parameter
        if lambda_type == "scalar":
            self.lambda_param = nn.Parameter(torch.tensor(1.0))
        elif lambda_type == "mlp":
            self.lambda_mlp = nn.Sequential(
                nn.Linear(d_k, d_k // 2),
                nn.ReLU(),
                nn.Linear(d_k // 2, 1),
            )
        else:
            raise ValueError(f"Unknown lambda_type: {lambda_type}")
        
        self.lambda_type = lambda_type
    
    def forward(
        self,
        X: torch.Tensor,
        cultural_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            X: Input embeddings [batch_size, seq_len, d_model]
            cultural_embedding: Cultural context [batch_size, cultural_dim] or [cultural_dim]
            mask: Optional attention mask
            
        Returns:
            Context vector [batch_size, seq_len, d_k]
        """
        batch_size, seq_len, _ = X.shape
        
        # Project to Q, K, V
        Q = self.W_Q(X)  # [batch_size, seq_len, d_k]
        K = self.W_K(X)  # [batch_size, seq_len, d_k]
        V = self.W_V(X)  # [batch_size, seq_len, d_k]
        
        # Project cultural embedding
        if cultural_embedding.dim() == 1:
            # Broadcast to batch dimension
            c_proj = self.W_C(cultural_embedding.unsqueeze(0))  # [1, d_k]
            c_proj = c_proj.expand(batch_size, -1)  # [batch_size, d_k]
        else:
            c_proj = self.W_C(cultural_embedding)  # [batch_size, d_k]
        
        # Base attention logits
        base_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # [batch_size, seq_len, seq_len]
        
        # Cultural bias term: λ · (Q_i · c')
        # Compute Q · c' for each sequence position
        cultural_bias = torch.einsum("bsd,bd->bs", Q, c_proj)  # [batch_size, seq_len]
        
        if self.lambda_type == "scalar":
            cultural_bias = self.lambda_param * cultural_bias
        else:
            # Per-position lambda (from MLP) with sigmoid for stability
            lambda_per_pos = torch.sigmoid(self.lambda_mlp(Q)).squeeze(-1)  # [batch_size, seq_len]
            cultural_bias = lambda_per_pos * cultural_bias
        
        # Add bias to logits (broadcast across key dimension)
        combined_logits = base_logits + cultural_bias.unsqueeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            combined_logits = combined_logits.masked_fill(mask == 0, -1e9)
        
        # Softmax attention weights
        attention_weights = F.softmax(combined_logits, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply to values
        output = torch.matmul(attention_weights, V)  # [batch_size, seq_len, d_k]
        
        return output


class CulturalMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Cultural Attention integration.
    
    Each head can optionally include cultural attention bias.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        cultural_dim: Dimension of cultural embeddings
        use_cultural: Whether to use cultural attention (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        cultural_dim: int = None,
        use_cultural: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_cultural = use_cultural
        
        # Initialize attention heads
        self.heads = nn.ModuleList([
            CulturalAttention(d_model, self.d_k, cultural_dim)
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        X: torch.Tensor,
        cultural_embedding: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            X: Input [batch_size, seq_len, d_model]
            cultural_embedding: Cultural context [batch_size, cultural_dim] or [cultural_dim]
            mask: Optional attention mask
            
        Returns:
            Output [batch_size, seq_len, d_model]
        """
        if cultural_embedding is None:
            # Fallback to standard attention (no cultural bias)
            cultural_embedding = torch.zeros(
                self.d_k, device=X.device, dtype=X.dtype
            )
        
        # Apply each head
        head_outputs = []
        for head in self.heads:
            head_out = head(X, cultural_embedding, mask)
            head_outputs.append(head_out)
        
        # Concatenate heads
        concatenated = torch.cat(head_outputs, dim=-1)  # [batch_size, seq_len, d_model]
        
        # Apply output projection
        output = self.W_O(concatenated)
        
        return output


# Example usage:
if __name__ == "__main__":
    # Create a sample batch
    batch_size, seq_len, d_model = 2, 10, 256
    X = torch.randn(batch_size, seq_len, d_model)
    
    # Cultural embedding (e.g., learned representation of cultural context)
    cultural_embedding = torch.randn(batch_size, 64)
    
    # Create multi-head cultural attention
    attention = CulturalMultiHeadAttention(
        d_model=d_model,
        num_heads=8,
        cultural_dim=64,
        use_cultural=True,
    )
    
    # Forward pass
    output = attention(X, cultural_embedding)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == X.shape
    print("✓ Cultural Multi-Head Attention test passed")
