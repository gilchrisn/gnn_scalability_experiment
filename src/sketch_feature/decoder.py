"""Decode KMV bottom-k hash slots back to T_0 entity ids.

After KMV propagation, each target node v holds K[v] in R^k — the k smallest
random hash values seen along its meta-path neighborhood. To use those slots
as feature inputs we recover which T_0 entity each hash corresponds to.

Two operating modes:

1) Pre-built decode table (preferred for training / inference): the user
   passes ``sorted_hashes`` and ``sort_indices`` produced once at sketch
   build time, and ``decode_sketches`` returns an ``[N, k]`` int64 tensor of
   T_0 ids (with -1 for empty / non-decodable slots).

2) Ad-hoc decode for a single sketch: ``decode_one`` resolves a single bottom
   k-array against an existing table.

This is shared between sketch-as-feature consumption and the sketch-as
sparsifier path (which currently in-lines the decode in `kernels/kmv.py`).
The intent is for the sparsifier to migrate to use this helper too, but
that's not done in this scaffold.
"""
from __future__ import annotations

from typing import Tuple

import torch

INFINITY = torch.iinfo(torch.int64).max
EMPTY_SLOT = -1


def build_decode_table(t0_hashes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort the T_0 hash universe so future slots can be resolved by binsearch.

    Args:
        t0_hashes: [N_T0] int64. ``t0_hashes[i]`` is the random hash assigned
            to T_0 entity ``i``. Hashes must be unique (a collision would
            mean two T_0 entities share a slot, which the sketch can no
            longer separate).

    Returns:
        sorted_hashes: [N_T0] int64, ascending.
        sort_indices : [N_T0] int64. ``sort_indices[j]`` is the T_0 entity id
            whose hash sorts to position ``j``.
    """
    if t0_hashes.dtype != torch.int64:
        raise TypeError(f"t0_hashes must be int64; got {t0_hashes.dtype}")
    sorted_hashes, sort_indices = torch.sort(t0_hashes)
    return sorted_hashes, sort_indices


def decode_sketches(
    sketches: torch.Tensor,
    sorted_hashes: torch.Tensor,
    sort_indices: torch.Tensor,
) -> torch.Tensor:
    """Decode an ``[N, k]`` bottom-k hash matrix into T_0 ids.

    Args:
        sketches:     [N, k] int64. ``INFINITY`` marks an empty slot.
        sorted_hashes:[N_T0] int64. From :func:`build_decode_table`.
        sort_indices: [N_T0] int64. From :func:`build_decode_table`.

    Returns:
        ids: [N, k] int64. ``EMPTY_SLOT`` (= -1) marks slots that were either
        empty (INFINITY) or whose hash had no exact match in the universe
        (the latter shouldn't happen for self-consistent sketches but we
        defend against it so downstream code can ``mask = ids >= 0``).
    """
    if sketches.dtype != torch.int64:
        raise TypeError(f"sketches must be int64; got {sketches.dtype}")
    N, k = sketches.shape
    flat = sketches.reshape(-1)
    valid = flat != INFINITY

    out = torch.full_like(flat, EMPTY_SLOT)

    if valid.any():
        vals = flat[valid]
        idx = torch.searchsorted(sorted_hashes, vals)
        idx = idx.clamp(max=len(sorted_hashes) - 1)
        # Confirm exact match (binsearch returns insertion point even on miss).
        match = sorted_hashes[idx] == vals
        decoded = torch.where(match, sort_indices[idx], torch.full_like(idx, EMPTY_SLOT))
        out[valid] = decoded

    return out.view(N, k)


def slot_mask(decoded: torch.Tensor) -> torch.Tensor:
    """Boolean mask of valid (decoded) slots — convenience for aggregation.

    Args:
        decoded: [N, k] int64 from :func:`decode_sketches`.

    Returns:
        mask: [N, k] bool. True iff the slot resolved to a T_0 id.
    """
    return decoded != EMPTY_SLOT
