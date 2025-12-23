## Cultural Attention Layer

This document describes the proposed Cultural Attention Layer, the mathematical formulation used, and integration guidance for reproducibility.

### Intuition
The Cultural Attention Layer augments standard self-attention with a learned cultural context vector $c$ that helps bias attention scores toward culturally-relevant tokens or features.

### Mathematical formulation
Let $X = [x_1, \dots, x_n]$ be token embeddings, and let $c$ be a cultural embedding (learned or provided) of dimension $d_c$.

- Compute queries, keys, and values as usual:
$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

- Compute standard scaled-dot-product attention logits:
$$A_{ij}^{(base)} = \frac{Q_i K_j^\top}{\sqrt{d_k}}$$

- Compute cultural bias term by projecting $c$ into the same key space:
$$c' = c W_C,\quad B_{ij} = \lambda \cdot (Q_i \cdot c')$$
where $\lambda$ is a learned scalar or small MLP that controls cultural influence.

- Combined attention logits:
$$A_{ij} = A_{ij}^{(base)} + B_{ij}$$

- Attention weights and context vector:
$$\alpha_{ij} = \text{softmax}_j(A_{ij})\\
z_i = \sum_j \alpha_{ij} V_j$$

### Fusion variants
- Additive fusion (above) — simple and stable.
- Gated fusion: compute a gate $g_i = \sigma(W_g[Q_i; c'])$ and fuse:
$$z_i = g_i \odot z_i^{(att)} + (1-g_i) \odot (C_f(c))$$
where $C_f$ is a lightweight cultural projection (MLP) returning a vector in value space.

### Pseudocode (PyTorch-style)
```
class CulturalAttention(nn.Module):
    def forward(self, X, c):
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        c_proj = c @ W_C
        base_logits = (Q @ K.transpose(-2, -1)) / sqrt(d_k)
        bias = lambda_param * (Q @ c_proj.unsqueeze(-1)).squeeze(-1)
        logits = base_logits + bias.unsqueeze(-1)
        weights = softmax(logits, dim=-1)
        return weights @ V
```

### Integration points
- Replace or augment the transformer's multi-head attention module. Apply cultural attention per head or as an extra attention head and then fuse into the final representation.
- Ensure the cultural embedding $c$ is available per example (e.g., metadata lookup, learned per demographic group, or computed from auxiliary features).

### Reproducibility notes
- Provide trained cultural embeddings and the exact configuration (projection sizes, whether $\lambda$ is scalar or MLP). See `REPRODUCIBILITY.md` for environment and commit hash.
