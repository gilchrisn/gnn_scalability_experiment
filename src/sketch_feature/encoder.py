"""Embed + aggregate KMV bottom-k slots into a per-node feature vector.

Per node v with decoded slot ids [s_1, ..., s_k] in T_0:

    x_v = AGG_{j=1..k} ( E[s_j] )           # standard sketch-as-feature
    x_v = MLP(x_v0 || AGG(...))             # if base node features x_v0 exist

where E ∈ R^{|T_0| × d} is a learned embedding of T_0 entities and AGG is one
of {sum, mean, attention}. The default is ``mean`` because empty slots are
masked out — sum biases toward high-degree nodes which already saturate k.

This implements ``C_consume_feature`` from
``final_report/research_notes/25_amortization_cost_analysis.md`` §2:
    C_consume_feature = O(N · k · d) + C_GNN_backbone

The encoder is meta-path-agnostic: callers stack per-meta-path features
externally (the backbone HAN-style attention then handles M-way fusion).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import EMPTY_SLOT


class SketchFeatureEncoder(nn.Module):
    """Bottom-k slot embedding with masked aggregation.

    Args:
        n_t0: Size of the T_0 entity universe (== |T_0| in the paper).
        emb_dim: Embedding dimension d.
        agg: One of ``{"mean", "sum", "attention"}``. Default ``"mean"``.
        base_dim: If non-zero, expects an additional ``base_x`` of shape
            ``[N, base_dim]`` in ``forward``. The output is a learned mix of
            the base feature and the sketch-aggregated feature.
        dropout: Dropout applied to embeddings before aggregation.
    """

    def __init__(
        self,
        n_t0: int,
        emb_dim: int,
        agg: str = "mean",
        base_dim: int = 0,
        dropout: float = 0.0,
        shared_embedding: Optional[nn.Embedding] = None,
    ) -> None:
        super().__init__()
        if agg not in {"mean", "sum", "attention"}:
            raise ValueError(f"agg must be mean|sum|attention; got {agg!r}")
        self.agg = agg
        self.emb_dim = emb_dim
        self.base_dim = base_dim
        self.dropout = nn.Dropout(p=dropout)

        # +1 row for the EMPTY_SLOT padding index. Use padding_idx so its
        # gradients are zero — that row is never used after masking but the
        # safety belt avoids surprises if a downstream caller forgets the
        # mask.
        if shared_embedding is not None:
            if shared_embedding.num_embeddings != n_t0 + 1:
                raise ValueError(
                    f"shared_embedding has {shared_embedding.num_embeddings} rows "
                    f"but n_t0+1={n_t0 + 1} required (last row is padding)"
                )
            if shared_embedding.embedding_dim != emb_dim:
                raise ValueError(
                    f"shared_embedding dim {shared_embedding.embedding_dim} "
                    f"!= emb_dim {emb_dim}"
                )
            self.embedding = shared_embedding
        else:
            self.embedding = nn.Embedding(n_t0 + 1, emb_dim, padding_idx=n_t0)

        if agg == "attention":
            self.attn = nn.Linear(emb_dim, 1, bias=False)

        if base_dim > 0:
            self.fuse = nn.Linear(base_dim + emb_dim, emb_dim)

    def forward(
        self,
        decoded: torch.Tensor,
        base_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-node sketch features.

        Args:
            decoded: [N, k] int64. Output of
                :func:`src.sketch_feature.decoder.decode_sketches`. Empty
                slots use ``EMPTY_SLOT`` (= -1).
            base_x : [N, base_dim] float — optional native node features to
                fuse in. Required iff ``self.base_dim > 0``.

        Returns:
            x: [N, emb_dim] float.
        """
        N, k = decoded.shape
        mask = decoded != EMPTY_SLOT  # [N, k] bool

        # Map empty slots to padding index so the embedding layer is always
        # called with non-negative indices.
        pad_idx = self.embedding.num_embeddings - 1
        safe_ids = torch.where(mask, decoded, torch.full_like(decoded, pad_idx))
        emb = self.embedding(safe_ids)              # [N, k, d]
        emb = self.dropout(emb)

        # Mask out empty slots before aggregation.
        emb = emb * mask.unsqueeze(-1).to(emb.dtype)

        if self.agg == "sum":
            agg = emb.sum(dim=1)                     # [N, d]
        elif self.agg == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(emb.dtype)
            agg = emb.sum(dim=1) / denom             # [N, d]
        else:                                        # "attention"
            scores = self.attn(emb).squeeze(-1)      # [N, k]
            scores = scores.masked_fill(~mask, float("-inf"))
            # Rows with no valid slot would get all -inf; softmax produces
            # nan there. Replace with uniform-zero weights so output is 0.
            all_empty = ~mask.any(dim=1, keepdim=True)
            attn = torch.where(
                all_empty.expand_as(scores),
                torch.zeros_like(scores),
                F.softmax(scores, dim=1),
            )
            agg = (emb * attn.unsqueeze(-1)).sum(dim=1)

        if self.base_dim > 0:
            if base_x is None:
                raise ValueError("base_x required when base_dim > 0")
            agg = self.fuse(torch.cat([base_x, agg], dim=-1))

        return agg
