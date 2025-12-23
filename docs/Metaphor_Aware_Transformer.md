## Metaphor-Aware Transformer

This document describes the Metaphor-Aware Transformer module and provides mathematical details and integration guidance.

### Goal
Enhance token representations by incorporating metaphor-awareness through auxiliary metaphor embeddings $m$ and a gating mechanism that adapts transformer's representations when metaphoric language is detected.

### Mathematical formulation
Let $X$ be token embeddings, and $m$ a metaphor embedding (per-sequence or per-token). Let $T(\cdot)$ denote the standard transformer block (self-attention + feed-forward).

- Compute transformer output:
$$y_i^{(base)} = T(x_i)$$

- Compute metaphor projection:
$$m' = mW_M, \quad r_i = \tanh(W_r[x_i; m'])$$

- Compute gating scalar or vector:
$$g_i = \sigma(W_g [y_i^{(base)}; r_i] + b_g)$$

- Final fused representation:
$$y_i = g_i \odot y_i^{(base)} + (1 - g_i) \odot f_m(r_i)$$

where $f_m(\cdot)$ is a small MLP that maps metaphor-aware features into the transformer's hidden size.

### Metaphor detection and embeddings
- Metaphor embeddings can be derived from a dedicated metaphor classifier (returns a distribution) or learned jointly as embeddings indexed by metaphor types.

### Implementation notes
- Insert the gating and fusion after the transformer's feed-forward layer in each block, or optionally as an extra residual branch before layer normalization.
- Ensure stability with layer normalization and residual connections; consider scaling $g_i$ initialization toward 1.0 so the base transformer behaviour is preserved initially.

### Pseudocode
```
class MetaphorAwareBlock(nn.Module):
    def forward(self, x, m):
        y = TransformerBlock(x)
        m_proj = m @ W_M
        r = torch.tanh(cat([x, m_proj]) @ W_r)
        g = torch.sigmoid(cat([y, r]) @ W_g + b_g)
        fm = MLP_m(r)
        return g * y + (1 - g) * fm
```

### Reproducibility
- Record exact classifier configuration used to produce metaphor embeddings and whether embeddings are per-token or per-sequence. See `REPRODUCIBILITY.md` for environment and commit hash.
