"""
Chunked / subsampled-CKA pipeline for OGB-MAG (Approach A v2 — Directive 2).

Computes per-row cosine, per-row L2, Procrustes-Q=I, Procrustes-Q-orth,
unbiased Linear CKA, and Macro-F1 gap on OGB-MAG at the k=32 KMV
reconstruction, on a stratified 5,000-node V_test bootstrap.

Five stages, all resume-safe via on-disk artefacts:

  Stage A  Stratified 5,000-node V_test subsample (numpy RNG seed=42).
  Stage B  2-hop bipartite subgraph extraction from H_exact via two
           streaming passes over staging/OGB_MAG/mat_exact.adj.
  Stage C  Strict-SpMM SAGE forward on the localised exact subgraph,
           re-using inference_worker._sage_spmm_layerwise. Slice the
           5,000-node subsample out of the full Z_exact.
  Stage D  Same SpMM forward on the KMV-reconstructed adjacency
           (k=32, hash_seed=42). Re-uses cached z_kmv_*_layers.pt if
           a prior exp3_inference run produced it.
  Stage E  v2 metrics (row cosine / rel-L2 / Frobenius / Procrustes /
           unbiased CKA) via _compute_v2_metrics from exp3_inference.
  Stage F  Macro-F1 on the 5,000-row subsample for both Z_exact and
           Z_kmv. JSON written to:
           results/approach_a_2026_05_07/OGB_MAG/SAGE/chunked_cka_seed{seed}_k{k}.json

CLI
---
    python scripts/chunked_cka_ogb_mag.py
        [--force] [--n-subsample 5000] [--k 32] [--seed 42] [--dry-run]

--dry-run runs end-to-end on HGB_DBLP / APA at n=100 (local-machine smoke
test).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.bridge.engine import CppEngine
from src.analysis.cka import LinearCKA
from src.models import get_model

# Re-use canonical helpers from exp3_inference + inference_worker.
from scripts.exp3_inference import _compute_v2_metrics, _unbiased_cka
from scripts.inference_worker import (
    _build_normalized_adj_csr,
    _sage_spmm_layerwise,
    _sage_spmm_forward,
)
from scripts.bench_utils import (
    compile_rule_for_cpp,
    generate_qnodes,
    setup_global_res_dirs,
)


# ---------------------------------------------------------------------------
# Logger + memory helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return 0.0


def _log_rss(stage: str, log: logging.Logger, halt_gb: float = 100.0) -> None:
    rss_mb = _rss_mb()
    log.info("[%s] RSS = %.1f GB", stage, rss_mb / 1024.0)
    if rss_mb / 1024.0 > halt_gb:
        raise MemoryError(
            f"[{stage}] RSS {rss_mb/1024.0:.1f} GB exceeds halt threshold "
            f"{halt_gb:.1f} GB. Aborting.")


# ---------------------------------------------------------------------------
# Stage A — stratified subsample
# ---------------------------------------------------------------------------

def stratified_subsample(
    test_ids: np.ndarray,
    labels_test: np.ndarray,
    n_target: int,
    rng: np.random.RandomState,
    log: logging.Logger,
) -> np.ndarray:
    """Stratified pick of `n_target` ids from `test_ids`, by class proportion.

    Each class gets at least one node where possible. Adjusts up/down to hit
    exactly `n_target`. If multi-label (labels_test.ndim == 2), strafifies on
    argmax of label vector (best effort) or falls back to uniform random.
    """
    if labels_test.ndim == 2:
        log.info("[Stage A] multi-label labels detected; using argmax for stratification")
        # Best-effort: each row -> dominant label. Rows with all zeros -> -1.
        rs = labels_test.sum(axis=1)
        dom = labels_test.argmax(axis=1)
        dom = np.where(rs > 0, dom, -1)
        labels_test = dom

    # Filter out invalid (-1) labels — we cannot stratify on them. Sample
    # uniformly from the *valid* subset.
    valid_mask = labels_test >= 0
    if valid_mask.sum() < n_target:
        log.warning("[Stage A] only %d valid-label test nodes (< n_target=%d) — "
                    "falling back to taking all valid + uniform-random fill.",
                    int(valid_mask.sum()), n_target)
        valid_idx = np.where(valid_mask)[0]
        invalid_idx = np.where(~valid_mask)[0]
        n_extra = n_target - len(valid_idx)
        if n_extra > 0 and len(invalid_idx) > 0:
            extra = rng.choice(invalid_idx, size=min(n_extra, len(invalid_idx)),
                               replace=False)
            picks = np.concatenate([valid_idx, extra])
        else:
            picks = valid_idx[:n_target]
        return test_ids[picks]

    valid_test_ids = test_ids[valid_mask]
    valid_labels = labels_test[valid_mask]

    classes, class_counts = np.unique(valid_labels, return_counts=True)
    proportions = class_counts.astype(np.float64) / class_counts.sum()
    target_per_class = np.maximum(1, np.round(proportions * n_target).astype(int))
    target_per_class = np.minimum(target_per_class, class_counts)

    # Reconcile rounding: adjust the largest classes up/down to hit n_target.
    diff = n_target - target_per_class.sum()
    if diff != 0:
        order = np.argsort(-class_counts)  # largest classes first
        i = 0
        while diff != 0 and i < len(order) * 4:
            ci = order[i % len(order)]
            if diff > 0 and target_per_class[ci] < class_counts[ci]:
                target_per_class[ci] += 1
                diff -= 1
            elif diff < 0 and target_per_class[ci] > 1:
                target_per_class[ci] -= 1
                diff += 1
            i += 1

    log.info("[Stage A] %d classes, target_per_class min=%d max=%d sum=%d",
             len(classes), target_per_class.min(), target_per_class.max(),
             target_per_class.sum())

    picks: List[int] = []
    for cls, n_pick in zip(classes, target_per_class):
        idx_in_valid = np.where(valid_labels == cls)[0]
        chosen = rng.choice(idx_in_valid, size=n_pick, replace=False)
        picks.extend(valid_test_ids[chosen].tolist())

    picks_arr = np.array(picks, dtype=np.int64)
    rng.shuffle(picks_arr)

    if len(picks_arr) > n_target:
        picks_arr = picks_arr[:n_target]
    elif len(picks_arr) < n_target:
        log.warning("[Stage A] short by %d nodes — padding from remaining valid.",
                    n_target - len(picks_arr))
        remaining = np.setdiff1d(valid_test_ids, picks_arr, assume_unique=False)
        n_pad = n_target - len(picks_arr)
        pad = rng.choice(remaining, size=min(n_pad, len(remaining)), replace=False)
        picks_arr = np.concatenate([picks_arr, pad])

    return picks_arr


# ---------------------------------------------------------------------------
# Stage B — 2-hop subgraph extraction from on-disk .adj
# ---------------------------------------------------------------------------

def two_hop_extract(
    adj_file: str,
    seed_nodes_global: Set[int],
    node_offset: int,
    n_target: int,
    log: logging.Logger,
    max_neighbours_per_node: Optional[int] = None,
    hard_cap_h1_h2_total: int = 2_000_000,
) -> torch.Tensor:
    """Two streaming passes over `adj_file`. Returns `edge_index [2, E]` in
    *local* (target-type) ids (already offset-subtracted).

    Pass 1: collect 1-hop edges from seed_nodes_global -> H1.
    Pass 2: collect 1-hop edges from H1 -> H2 (so total reach = 2 hops).

    Final edge_index includes (a) seed -> H1 edges and (b) H1 -> H2 edges.
    Both src and dst columns store *local* target-type ids 0..n_target-1.
    Edges where either endpoint is outside the target type (after offset) are
    skipped. (Meta-path reconstruction guarantees both endpoints are paper.)

    `max_neighbours_per_node` caps neighbour-list length per row at read time
    (deterministic prefix). Use this to prevent |H1 union H2| from blowing up
    memory.
    """
    if not os.path.exists(adj_file):
        raise FileNotFoundError(f"adjacency file not found: {adj_file}")

    log.info("[Stage B] starting pass 1 (seeds -> H1) on %s", adj_file)
    t0 = time.perf_counter()

    src1: List[int] = []
    dst1: List[int] = []
    h1_set: Set[int] = set()
    line_count = 0

    with open(adj_file, "r") as f:
        for line in f:
            line_count += 1
            if line_count % 5_000_000 == 0:
                log.info("[Stage B p1] read %d lines, |H1| so far = %d",
                         line_count, len(h1_set))
            parts = line.split()
            if not parts:
                continue
            try:
                u_global = int(parts[0])
            except ValueError:
                continue
            if u_global not in seed_nodes_global:
                continue
            u_local = u_global - node_offset
            if u_local < 0 or u_local >= n_target:
                continue
            neigh = parts[1:]
            if max_neighbours_per_node is not None:
                neigh = neigh[:max_neighbours_per_node]
            for v in neigh:
                try:
                    v_global = int(v)
                except ValueError:
                    continue
                v_local = v_global - node_offset
                if v_local < 0 or v_local >= n_target:
                    continue
                src1.append(u_local)
                dst1.append(v_local)
                h1_set.add(v_global)
                # Self-loops handled by add_self_loops downstream.

    log.info("[Stage B p1] complete in %.1f s. lines=%d  |H1|=%d  edges_p1=%d",
             time.perf_counter() - t0, line_count, len(h1_set), len(src1))

    # Memory check between passes.
    if len(h1_set) > hard_cap_h1_h2_total // 2:
        log.warning("[Stage B] |H1|=%d exceeds soft cap %d — pass 2 will downsample.",
                    len(h1_set), hard_cap_h1_h2_total // 2)

    log.info("[Stage B] starting pass 2 (H1 -> H2)")
    t1 = time.perf_counter()

    src2: List[int] = []
    dst2: List[int] = []
    h2_set: Set[int] = set()
    line_count = 0

    with open(adj_file, "r") as f:
        for line in f:
            line_count += 1
            if line_count % 5_000_000 == 0:
                log.info("[Stage B p2] read %d lines, |H2| so far = %d",
                         line_count, len(h2_set))
            parts = line.split()
            if not parts:
                continue
            try:
                u_global = int(parts[0])
            except ValueError:
                continue
            # Pass 2 is for the layer-2 aggregate around H1 nodes.
            if u_global not in h1_set:
                continue
            u_local = u_global - node_offset
            if u_local < 0 or u_local >= n_target:
                continue
            neigh = parts[1:]
            if max_neighbours_per_node is not None:
                neigh = neigh[:max_neighbours_per_node]
            for v in neigh:
                try:
                    v_global = int(v)
                except ValueError:
                    continue
                v_local = v_global - node_offset
                if v_local < 0 or v_local >= n_target:
                    continue
                src2.append(u_local)
                dst2.append(v_local)
                h2_set.add(v_global)

    log.info("[Stage B p2] complete in %.1f s. lines=%d  |H2|=%d  edges_p2=%d",
             time.perf_counter() - t1, line_count, len(h2_set), len(src2))

    total_unique = len(seed_nodes_global | h1_set | h2_set)
    log.info("[Stage B] total reach |seed U H1 U H2| = %d   (n_target=%d)",
             total_unique, n_target)

    if total_unique > hard_cap_h1_h2_total:
        log.warning("[Stage B] reach %d > hard cap %d — downstream forward "
                    "will be heavy; consider --max-neighbours-per-node",
                    total_unique, hard_cap_h1_h2_total)

    src = src1 + src2
    dst = dst1 + dst2
    if not src:
        return torch.empty((2, 0), dtype=torch.long)

    ei = torch.tensor([src, dst], dtype=torch.long)
    from torch_geometric.utils import coalesce, add_self_loops
    ei = coalesce(ei, num_nodes=n_target)
    ei, _ = add_self_loops(ei, num_nodes=n_target)
    log.info("[Stage B] coalesced + self-looped edge_index: %d edges", ei.size(1))
    return ei


# ---------------------------------------------------------------------------
# Stage C/D — SAGE forward via strict SpMM, restricted to subsample rows
# ---------------------------------------------------------------------------

def run_sage_layerwise(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    weights_path: str,
    in_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_layers: int,
    log: logging.Logger,
    label: str,
) -> List[torch.Tensor]:
    """Build adj_csr, run model.eval() forward, return per-layer outputs (CPU)."""
    device = torch.device("cpu")
    n = x.size(0)

    log.info("[%s] building normalized adj CSR (n=%d, E=%d)",
             label, n, edge_index.size(1))
    adj_csr = _build_normalized_adj_csr(edge_index, n, device)

    log.info("[%s] loading SAGE weights from %s", label, weights_path)
    model = get_model("SAGE", in_dim, num_classes, hidden_dim,
                      num_layers=num_layers).to(device)
    state = torch.load(weights_path, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Pad x to in_dim (the worker pads; we mirror that contract).
    x_pad = F.pad(x, (0, max(0, in_dim - x.size(1))))

    log.info("[%s] running SpMM layerwise forward", label)
    t0 = time.perf_counter()
    with torch.no_grad():
        intermediates = _sage_spmm_layerwise(model, x_pad, adj_csr)
    log.info("[%s] forward done in %.2f s, %d layer outputs",
             label, time.perf_counter() - t0, len(intermediates))

    # Free model + adj early; intermediates are already CPU clones.
    del model, adj_csr, x_pad
    gc.collect()
    return intermediates


def slice_to_subsample(
    layers: List[torch.Tensor],
    subsample_local_ids: torch.Tensor,
) -> List[torch.Tensor]:
    """Restrict each layer output to the n_subsample rows."""
    return [Z[subsample_local_ids].contiguous() for Z in layers]


# ---------------------------------------------------------------------------
# C++ materialisation helpers (Stage B bootstrap + Stage D)
# ---------------------------------------------------------------------------

def ensure_exact_materialised(
    engine: CppEngine,
    folder: str,
    timeout: int,
    log: logging.Logger,
) -> str:
    rule_file   = os.path.join(engine.data_dir, f"cod-rules_{folder}.limit")
    output_file = os.path.join(engine.data_dir, "mat_exact.adj")
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        log.info("[exact-mat] reusing existing %s (%.1f GB)",
                 output_file, os.path.getsize(output_file) / 1e9)
        return output_file
    log.info("[exact-mat] running graph_prep materialize ...")
    t0 = time.perf_counter()
    engine.run_command("materialize", rule_file, output_file, timeout=timeout)
    log.info("[exact-mat] complete in %.1f s -> %s",
             time.perf_counter() - t0, output_file)
    return output_file


def ensure_kmv_materialised(
    engine: CppEngine,
    folder: str,
    k: int,
    seed: int,
    timeout: int,
    log: logging.Logger,
) -> str:
    rule_file   = os.path.join(engine.data_dir, f"cod-rules_{folder}.limit")
    output_base = os.path.join(engine.data_dir, "mat_sketch")
    output_file = output_base + "_0"
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        log.info("[kmv-mat] reusing existing %s (%.1f MB)",
                 output_file, os.path.getsize(output_file) / 1e6)
        return output_file
    log.info("[kmv-mat] running graph_prep sketch (k=%d, seed=%d) ...", k, seed)
    t0 = time.perf_counter()
    engine.run_command("sketch", rule_file, output_base,
                       k=k, seed=seed, timeout=timeout)
    log.info("[kmv-mat] complete in %.1f s -> %s",
             time.perf_counter() - t0, output_file)
    return output_file


# ---------------------------------------------------------------------------
# Macro-F1 helper (mirrors inference_worker, restricted to subsample rows)
# ---------------------------------------------------------------------------

def macro_f1(
    Z: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[float, float]:
    """Compute (macro_f1, micro_f1). labels is the subsample-restricted vector."""
    from torchmetrics.functional import f1_score as _f1
    if labels.dim() == 2:
        valid = labels.sum(dim=1) > 0
        if valid.sum() == 0:
            return 0.0, 0.0
        preds = (Z[valid] > 0).long()
        macro = _f1(preds, labels[valid].long(),
                    task="multilabel", num_labels=num_classes,
                    average="macro").item()
        micro = _f1(preds, labels[valid].long(),
                    task="multilabel", num_labels=num_classes,
                    average="micro").item()
        return macro, micro
    valid = labels >= 0
    if valid.sum() == 0:
        return 0.0, 0.0
    macro = _f1(Z[valid].argmax(1), labels[valid],
                task="multiclass", num_classes=num_classes,
                average="macro").item()
    micro = _f1(Z[valid].argmax(1), labels[valid],
                task="multiclass", num_classes=num_classes,
                average="micro").item()
    return macro, micro


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-subsample", type=int, default=5000)
    parser.add_argument("--k",          type=int, default=32,
                        help="KMV sketch size (must match an existing OGB-MAG "
                             "k-sweep run for cached z_kmv reuse).")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--force",      action="store_true",
                        help="Ignore cached intermediate artefacts and recompute.")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run end-to-end on HGB_DBLP / APA at n=100 instead of "
                             "OGB_MAG. Sanity-checks the pipeline locally.")
    parser.add_argument("--max-neighbours-per-node", type=int, default=None,
                        help="Cap neighbour-list length per node during the "
                             "two-hop streaming pass to bound |H1 U H2|.")
    parser.add_argument("--hard-cap-reach", type=int, default=2_000_000,
                        help="Soft halt if |seed U H1 U H2| exceeds this cap.")
    parser.add_argument("--timeout",    type=int, default=7200,
                        help="C++ subprocess timeout in seconds.")
    args = parser.parse_args()

    # ─── Reproducibility ────────────────────────────────────────────────
    np_rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # ─── Dataset selection (real vs dry-run) ────────────────────────────
    if args.dry_run:
        dataset = "HGB_DBLP"
        metapath = "author_to_paper,paper_to_author"
        n_subsample = min(100, args.n_subsample)
    else:
        dataset = "OGB_MAG"
        metapath = "rev_writes,writes"
        n_subsample = args.n_subsample

    out_dir = Path("results") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Logging ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("chunked_cka")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(ch)
    log_path = out_dir / f"chunked_cka_{ts}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    log.info("=" * 72)
    log.info("chunked_cka_ogb_mag.py — Approach A v2 Directive 2")
    log.info("dataset=%s  metapath=%s  n_subsample=%d  k=%d  seed=%d  L=%d",
             dataset, metapath, n_subsample, args.k, args.seed, args.num_layers)
    log.info("dry_run=%s  force=%s", args.dry_run, args.force)
    log.info("=" * 72)

    _log_rss("startup", log)

    # ─── Load dataset + partition ──────────────────────────────────────
    cfg = config.get_dataset_config(dataset)
    folder = config.get_folder_name(dataset)
    data_dir = config.get_staging_dir(dataset)
    target_ntype = cfg.target_node

    g_full, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, target_ntype)
    num_classes = info["num_classes"]
    labels_full = info["labels"]
    n_target = g_full[target_ntype].num_nodes
    x_full = g_full[target_ntype].x
    in_dim = x_full.size(1)

    # node-offset for global-id -> local-id
    s_ntypes = sorted(g_full.node_types)
    node_offset = sum(g_full[nt].num_nodes for nt in s_ntypes if nt < target_ntype)

    log.info("[load] target=%s  n_target=%d  in_dim=%d  num_classes=%d  node_offset=%d",
             target_ntype, n_target, in_dim, num_classes, node_offset)

    partition_path = out_dir / "partition.json"
    if not partition_path.exists():
        raise FileNotFoundError(
            f"partition.json missing for {dataset} — run exp1_partition.py first.")
    with open(partition_path) as f:
        part = json.load(f)
    test_ids = np.array(part["test_node_ids"], dtype=np.int64)
    log.info("[load] |V_test| = %d", len(test_ids))

    # Resolve weights path. SAGE legacy filename: <mp_safe>_L<L>.pt
    mp_safe = metapath.replace(",", "_").replace("/", "_")
    weights_path = out_dir / "weights" / f"{mp_safe}_L{args.num_layers}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"weights not found: {weights_path} — run exp2_train.py first.")
    log.info("[load] weights -> %s", weights_path)

    _log_rss("after dataset load", log)

    # ─── Stage A: stratified subsample ─────────────────────────────────
    sub_path = out_dir / f"chunked_cka_subsample_{n_subsample}.npy"
    if sub_path.exists() and not args.force:
        log.info("[Stage A] reusing cached subsample %s", sub_path)
        subsample_local = np.load(sub_path)
    else:
        if labels_full.dim() == 2:
            labels_test = labels_full[test_ids].cpu().numpy()
        else:
            labels_test = labels_full[test_ids].cpu().numpy()
        subsample_local = stratified_subsample(
            test_ids=test_ids, labels_test=labels_test,
            n_target=n_subsample, rng=np_rng, log=log)
        np.save(sub_path, subsample_local)
        log.info("[Stage A] saved %d-node subsample to %s",
                 len(subsample_local), sub_path)

    assert len(subsample_local) == n_subsample, (
        f"subsample size mismatch: got {len(subsample_local)}, expected {n_subsample}")
    subsample_local_t = torch.tensor(subsample_local, dtype=torch.long)
    seed_nodes_global = {int(v) + node_offset for v in subsample_local}

    _log_rss("after Stage A", log)

    # ─── C++ staging (idempotent) ──────────────────────────────────────
    setup_global_res_dirs(folder, project_root)
    engine = CppEngine(executable_path=config.CPP_EXECUTABLE, data_dir=data_dir)
    if not os.path.exists(os.path.join(data_dir, "meta.dat")):
        log.info("[stage] running PyGToCppAdapter on %s ...", dataset)
        PyGToCppAdapter(data_dir).convert(g_full)
    rule_file_path = os.path.join(data_dir, f"cod-rules_{folder}.limit")
    if not os.path.exists(rule_file_path) or args.force:
        compile_rule_for_cpp(metapath, g_full, data_dir, folder)
    qnodes_path = os.path.join(data_dir, f"qnodes_{folder}.dat")
    if not os.path.exists(qnodes_path) or args.force:
        generate_qnodes(data_dir, folder,
                        target_node_type=target_ntype, g_hetero=g_full)

    # ─── Stage B: 2-hop subgraph from H_exact ─────────────────────────
    subgraph_path = out_dir / "chunked_cka_subgraph_exact.pt"
    if subgraph_path.exists() and not args.force:
        log.info("[Stage B] reusing cached %s", subgraph_path)
        edge_index_exact = torch.load(subgraph_path, weights_only=True)
        log.info("[Stage B] loaded edge_index: %d edges", edge_index_exact.size(1))
    else:
        # mat_exact.adj must exist; trigger a materialise if not.
        adj_path = ensure_exact_materialised(engine, folder, args.timeout, log)
        edge_index_exact = two_hop_extract(
            adj_file=adj_path,
            seed_nodes_global=seed_nodes_global,
            node_offset=node_offset,
            n_target=n_target,
            log=log,
            max_neighbours_per_node=args.max_neighbours_per_node,
            hard_cap_h1_h2_total=args.hard_cap_reach,
        )
        torch.save(edge_index_exact, subgraph_path)
        log.info("[Stage B] saved subgraph to %s", subgraph_path)

    _log_rss("after Stage B", log)

    # ─── Stage C: SAGE on Z_exact (localised subgraph) ────────────────
    z_exact_path = out_dir / "chunked_cka_z_exact_layers.pt"
    if z_exact_path.exists() and not args.force:
        log.info("[Stage C] reusing cached %s", z_exact_path)
        z_exact_layers = torch.load(z_exact_path, weights_only=True)
    else:
        layers_full = run_sage_layerwise(
            edge_index=edge_index_exact,
            x=x_full,
            weights_path=str(weights_path),
            in_dim=in_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=num_classes,
            num_layers=args.num_layers,
            log=log,
            label="Stage C / exact",
        )
        z_exact_layers = slice_to_subsample(layers_full, subsample_local_t)
        del layers_full
        gc.collect()
        torch.save(z_exact_layers, z_exact_path)
        log.info("[Stage C] saved %d-layer Z_exact (subsample) to %s",
                 len(z_exact_layers), z_exact_path)
    log.info("[Stage C] z_exact_layers shapes: %s",
             [tuple(z.shape) for z in z_exact_layers])

    _log_rss("after Stage C", log)

    # ─── Stage D: KMV-side Z (reuse cached or regenerate) ─────────────
    z_kmv_path = out_dir / f"chunked_cka_z_kmv_k{args.k}_layers.pt"
    cached_kmv_layers = (out_dir / "inf_scratch" / mp_safe
                         / f"z_kmv_{args.k}_s{args.seed}_L{args.num_layers}_layers.pt")
    if z_kmv_path.exists() and not args.force:
        log.info("[Stage D] reusing %s", z_kmv_path)
        z_kmv_layers = torch.load(z_kmv_path, weights_only=True)
    elif cached_kmv_layers.exists() and not args.force:
        log.info("[Stage D] reusing exp3 cached KMV layers %s", cached_kmv_layers)
        full_kmv_layers = torch.load(cached_kmv_layers, weights_only=True)
        z_kmv_layers = slice_to_subsample(full_kmv_layers, subsample_local_t)
        del full_kmv_layers
        gc.collect()
        torch.save(z_kmv_layers, z_kmv_path)
        log.info("[Stage D] saved sliced %d-layer Z_kmv to %s",
                 len(z_kmv_layers), z_kmv_path)
    else:
        log.info("[Stage D] regenerating KMV via graph_prep sketch + SAGE forward")
        kmv_adj_file = ensure_kmv_materialised(
            engine, folder, args.k, args.seed, args.timeout, log)
        # Reload KMV adjacency and stream into 2-hop extract over the SAME
        # seed nodes — KMV reconstruction Ã has different topology so the
        # 2-hop reach is recomputed.
        edge_index_kmv = two_hop_extract(
            adj_file=kmv_adj_file,
            seed_nodes_global=seed_nodes_global,
            node_offset=node_offset,
            n_target=n_target,
            log=log,
            max_neighbours_per_node=args.max_neighbours_per_node,
            hard_cap_h1_h2_total=args.hard_cap_reach,
        )
        layers_full_kmv = run_sage_layerwise(
            edge_index=edge_index_kmv,
            x=x_full,
            weights_path=str(weights_path),
            in_dim=in_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=num_classes,
            num_layers=args.num_layers,
            log=log,
            label=f"Stage D / kmv_k{args.k}",
        )
        z_kmv_layers = slice_to_subsample(layers_full_kmv, subsample_local_t)
        del layers_full_kmv, edge_index_kmv
        gc.collect()
        torch.save(z_kmv_layers, z_kmv_path)
        log.info("[Stage D] saved %d-layer Z_kmv (subsample) to %s",
                 len(z_kmv_layers), z_kmv_path)

    log.info("[Stage D] z_kmv_layers shapes: %s",
             [tuple(z.shape) for z in z_kmv_layers])

    _log_rss("after Stage D", log)

    # Sanity: layer counts must match.
    assert len(z_exact_layers) == len(z_kmv_layers), (
        f"layer count mismatch: exact={len(z_exact_layers)}, "
        f"kmv={len(z_kmv_layers)}")
    for i, (le, lk) in enumerate(zip(z_exact_layers, z_kmv_layers)):
        assert le.shape == lk.shape, (
            f"shape mismatch at layer {i}: exact={le.shape}, kmv={lk.shape}")

    # ─── Stage E: v2 metrics via _compute_v2_metrics ──────────────────
    # _compute_v2_metrics reads layer files from disk, so we already saved
    # them above and now just call it with mask=ones (rows are pre-masked).
    device = torch.device("cpu")
    cka_calc = LinearCKA(device=device)
    n = z_exact_layers[0].size(0)
    mask = torch.ones(n, dtype=torch.bool)

    log.info("[Stage E] computing v2 metrics on %d-row subsample", n)
    v2 = _compute_v2_metrics(
        z_exact_path=None, layers_exact_path=str(z_exact_path),
        z_kmv_path=None,   layers_kmv_path=str(z_kmv_path),
        mask=mask, device=device, cka_calc=cka_calc)
    log.info("[Stage E] cka_unbiased_per_layer = %s", v2.get("cka_unbiased_per_layer"))
    log.info("[Stage E] row_cosine_per_layer_mean = %s",
             v2.get("row_cosine_per_layer_mean"))

    _log_rss("after Stage E", log)

    # ─── Stage F: Macro-F1 on subsample ───────────────────────────────
    labels_sub = labels_full[subsample_local_t]
    log.info("[Stage F] computing Macro-F1 on subsample (label_dim=%s)",
             tuple(labels_sub.shape))

    Z_exact_final = z_exact_layers[-1]
    Z_kmv_final   = z_kmv_layers[-1]
    f1_exact, micro_exact = macro_f1(Z_exact_final, labels_sub, num_classes)
    f1_kmv,   micro_kmv   = macro_f1(Z_kmv_final,   labels_sub, num_classes)
    f1_gap = f1_kmv - f1_exact
    micro_gap = micro_kmv - micro_exact
    log.info("[Stage F] f1_exact=%.4f f1_kmv=%.4f f1_gap=%+.4f",
             f1_exact, f1_kmv, f1_gap)

    # ─── JSON output ─────────────────────────────────────────────────
    if args.dry_run:
        spec_dir = (Path("results") / "approach_a_2026_05_07"
                    / dataset / "SAGE")
    else:
        spec_dir = (Path("results") / "approach_a_2026_05_07"
                    / "OGB_MAG" / "SAGE")
    spec_dir.mkdir(parents=True, exist_ok=True)
    json_path = spec_dir / f"chunked_cka_seed{args.seed}_k{args.k}.json"

    payload = {
        "spec_version":            "approach_a_2026_05_07",
        "directive":               "directive_2_chunked_cka",
        "chunked_cka_method":      "stratified_5000_2hop_subgraph",
        "dataset":                 dataset,
        "meta_path":               metapath,
        "target_type":             target_ntype,
        "arch":                    "SAGE",
        "n_layers":                args.num_layers,
        "hidden_dim":              config.HIDDEN_DIM,
        "seed":                    args.seed,
        "kmv_k":                   args.k,
        "n_subsample":             int(n),
        "n_target_nodes":          int(n_target),
        "n_test_nodes":            int(len(test_ids)),
        "n_edges_exact_localised": int(edge_index_exact.size(1)),
        # v2 metrics:
        "row_cosine_per_layer_mean":  v2.get("row_cosine_per_layer_mean", []),
        "row_cosine_per_layer_std":   v2.get("row_cosine_per_layer_std", []),
        "row_rel_l2_per_layer_mean":  v2.get("row_rel_l2_per_layer_mean", []),
        "row_rel_l2_per_layer_std":   v2.get("row_rel_l2_per_layer_std", []),
        "frob_recon_err_per_layer":   v2.get("frob_recon_err_per_layer", []),
        "procrustes_q_eq_i_per_layer":   v2.get("procrustes_q_eq_i_per_layer", []),
        "procrustes_q_orth_per_layer":  v2.get("procrustes_q_orth_per_layer", []),
        "cka_unbiased_per_layer":     v2.get("cka_unbiased_per_layer", []),
        # F1 on subsample:
        "macro_f1_exact":          f1_exact,
        "macro_f1_kmv":            f1_kmv,
        "macro_f1_gap":            f1_gap,
        "f1_gap":                  f1_gap,  # alias for legacy aggregators
        "micro_f1_exact":          micro_exact,
        "micro_f1_kmv":            micro_kmv,
        "micro_f1_gap":            micro_gap,
        "device_used":             "cpu",
        "log_path":                str(log_path),
        "subsample_path":          str(sub_path),
        "subgraph_path":           str(subgraph_path),
        "z_exact_layers_path":     str(z_exact_path),
        "z_kmv_layers_path":       str(z_kmv_path),
        "timestamp":               datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(json_path, "w", encoding="utf-8") as jfh:
        json.dump(payload, jfh, indent=2)
    log.info("=" * 72)
    log.info("DONE.  JSON -> %s", json_path)
    log.info("Final-layer unbiased CKA: %s",
             payload["cka_unbiased_per_layer"][-1]
             if payload["cka_unbiased_per_layer"] else "n/a")
    log.info("Final-layer row-cosine:   %s",
             payload["row_cosine_per_layer_mean"][-1]
             if payload["row_cosine_per_layer_mean"] else "n/a")
    log.info("Macro-F1 gap:             %+.4f", f1_gap)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
