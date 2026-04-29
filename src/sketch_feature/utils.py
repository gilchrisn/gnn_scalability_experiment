"""Shared utilities for sketch-feature consumer-mode scripts.

Centralises three helpers that were previously duplicated across
``exp_sketch_feature_train.py``, ``exp_sketch_sparsifier_train.py``,
``exp_simple_hgn_baseline.py``, and ``exp_sketch_similarity.py``:

  * macro-F1 for single-label classification
  * macro-F1 for multi-label classification (per-class binary F1)
  * meta-path-induced edge construction from a decoded sketch tensor
"""
from __future__ import annotations

from typing import Optional

import torch


def macro_f1(y_pred: torch.Tensor, y_true: torch.Tensor, n_classes: int) -> float:
    """Macro-averaged F1 for single-label classification.

    Classes absent in ``y_true`` are skipped (they contribute neither to
    numerator nor denominator). Returns 0.0 if every class is skipped,
    which would only happen on degenerate splits.

    Args:
        y_pred: int tensor of predicted labels, shape [N].
        y_true: int tensor of ground-truth labels, shape [N].
        n_classes: total number of classes (ignored classes are detected
            empirically from y_true; this argument bounds the loop range).
    """
    yp = y_pred.detach().cpu().numpy()
    yt = y_true.detach().cpu().numpy()
    f1s = []
    for c in range(n_classes):
        tp = ((yp == c) & (yt == c)).sum()
        fp = ((yp == c) & (yt != c)).sum()
        fn = ((yp != c) & (yt == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            continue
        f1s.append(2 * prec * rec / (prec + rec))
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def macro_f1_multilabel(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Per-class binary F1, averaged across classes (multi-label setting).

    Args:
        logits: float tensor [N, C], pre-sigmoid model output.
        y_true: float tensor [N, C] with values in {0, 1}.
        threshold: threshold applied to ``sigmoid(logits)`` to get hard
            predictions (default 0.5).
    """
    pred = (torch.sigmoid(logits) >= threshold).int().detach().cpu().numpy()
    yt = y_true.int().detach().cpu().numpy()
    n_classes = pred.shape[1]
    f1s = []
    for c in range(n_classes):
        tp = ((pred[:, c] == 1) & (yt[:, c] == 1)).sum()
        fp = ((pred[:, c] == 1) & (yt[:, c] == 0)).sum()
        fn = ((pred[:, c] == 0) & (yt[:, c] == 1)).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            continue
        f1s.append(2 * prec * rec / (prec + rec))
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def build_mp_edges_from_decoded(
    decoded: torch.Tensor,
    n_target: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Build a meta-path-induced adjacency from a decoded sketch.

    Each row ``v`` of ``decoded`` lists the bottom-k T_0 ids that ``v``'s
    meta-path sketch retained (with ``-1`` marking empty slots). Treat
    each ``(v, decoded[v, j])`` for valid slots as an undirected edge in
    the meta-path-induced graph, optionally adding self-loops.

    This is used in two places:
      * ``exp_sketch_feature_train.py`` (han_sketch_edges backbone) — as
        the per-meta-path adjacency fed to HANConv.
      * ``exp_sketch_sparsifier_train.py`` — as the union adjacency for
        the sparsifier consumer mode.

    Args:
        decoded: ``[N, k]`` int64 tensor from
            :func:`src.sketch_feature.decoder.decode_sketches`.
        n_target: number of target-type nodes (= ``N``).
        add_self_loops: if True, append ``(v, v)`` for each v.

    Returns:
        ``[2, E]`` int64 edge_index tensor, deduplicated via
        ``torch_geometric.utils.coalesce``.
    """
    from torch_geometric.utils import (
        coalesce,
        add_self_loops as _add_sl,
    )

    n, k = decoded.shape
    src = torch.arange(n).unsqueeze(1).expand(-1, k)
    dst = decoded
    valid = dst >= 0
    src = src[valid]
    dst = dst[valid]
    ei = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ]).long()
    if add_self_loops:
        ei, _ = _add_sl(ei, num_nodes=n_target)
    ei = coalesce(ei, num_nodes=n_target)
    return ei
