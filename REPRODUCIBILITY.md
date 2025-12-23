## Reproducibility and Versioning

This file documents the exact environment and commit used when generating the reported results and provides instructions for verifying the `Cultural Attention` and `Metaphor-Aware Transformer` implementations.

- Repository commit hash: 4bbd93ce388bca6b6e97cc3c48272d993f4c6f93
- Python: 3.9+ recommended
- Key package versions (see `requirements.txt`):
  - `transformers==4.35.0`
  - `torch==1.13.1`
  - `sentence-transformers==2.2.2`

### How to verify Cultural Attention implementation
1. Review `docs/Cultural_Attention.md` for equations and pseudocode.
2. Implement `CulturalAttention` as an extra attention module or per-head modification in the transformer's `MultiHeadAttention`.
3. Run the training/evaluation script used in the paper and compare metrics. Record and share the exact model weights and cultural embedding files used.

### How to verify Metaphor-Aware Transformer implementation
1. Review `docs/Metaphor_Aware_Transformer.md` for gating and fusion equations.
2. Provide the metaphor classifier or embedding files used to produce $m$ in the experiments.
3. Run ablation studies with and without the metaphor components to reproduce reported gains.

### Notes for maintainers
- Keep the `requirements.txt` pinned to the versions above when reproducing results.
- Tag releases in git and include model checkpoints in a release or separate storage (Hugging Face Hub) with clear version mapping to this commit hash.
