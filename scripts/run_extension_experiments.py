"""
Multi-metapath extension experiment runner.

Mines (or loads cached) metapaths for a dataset, then for each valid metapath:
  Phase 1 — Materialize H_0 exactly (ExactD), train SAGE, freeze weights θ*
  Phase 2 — For each streaming snapshot, compare Exact vs KMV:
             Exact: C++ ExactD → H_t → frozen SAGE → Z_exact
             KMV:   C++ GloD   → H̃_t → frozen SAGE → Z_kmv
             Metrics: CKA(Z_exact, Z_kmv), F1 for both, materialization speedup

Output → results/<DATASET>/extension.csv
Resume-safe: metapaths already in extension.csv are skipped.

Usage:
    python scripts/run_extension_experiments.py HGB_DBLP
    python scripts/run_extension_experiments.py HGB_ACM --max-metapaths 5 --epochs 50
    python scripts/run_extension_experiments.py HGB_DBLP --fractions 0.6 0.8 1.0 --k 16
    python scripts/run_extension_experiments.py HGB_ACM --force-remine
"""
from __future__ import annotations

import argparse
import csv
import logging
import gc
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple


def _mem_mb() -> str:
    """Current RSS in MB (Linux only, 0 on Windows)."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
        return f"{rss / 1024:.0f}MB"
    except Exception:
        return "?MB"

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from torch_geometric.data import HeteroData

from src.config import config
from src.data import DatasetFactory
from src.data.streaming import TemporalPartitioner, Snapshot
from src.analysis.cka import LinearCKA
from src.bridge import PyGToCppAdapter, GraphPrepRunner, AnyBURLRunner, load_validated_metapaths
from src.bridge.engine import CppEngine
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, setup_global_res_dirs


DEFAULT_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_K         = 32
DEFAULT_EPOCHS    = 50
DEFAULT_TOPR      = "0.05"
DEFAULT_MIN_CONF  = 0.1

def _subgraph_by_year(
    g: HeteroData,
    fraction: float,
    time_node_type: str,
    time_attr: str = "year",
) -> HeteroData:
    """Keep only time-nodes with year in the bottom `fraction`, then prune
    all other node types to only those still referenced by surviving edges.

    This is a true induced subgraph: filtering papers also removes orphaned
    authors, fields, institutions, etc., dramatically reducing total node count
    for the C++ binary.
    """
    years = getattr(g[time_node_type], time_attr).squeeze()
    sorted_years, _ = years.sort()
    cutoff_idx = max(1, int(fraction * len(sorted_years))) - 1
    year_cutoff = sorted_years[cutoff_idx].item()

    keep_mask = years.squeeze() <= year_cutoff
    keep_ids = keep_mask.nonzero(as_tuple=False).squeeze(1)

    # Build old→new id mapping for the time node type
    id_maps = {}
    id_maps[time_node_type] = torch.full(
        (g[time_node_type].num_nodes,), -1, dtype=torch.long
    )
    id_maps[time_node_type][keep_ids] = torch.arange(
        keep_ids.size(0), dtype=torch.long
    )

    # --- Pass 1: filter edges, collect referenced non-time node IDs ----------
    filtered_edges = {}
    referenced = {nt: set() for nt in g.node_types if nt != time_node_type}

    for src_type, rel, dst_type in g.edge_types:
        ei = g[src_type, rel, dst_type].edge_index
        src_ids, dst_ids = ei[0].clone(), ei[1].clone()

        # Mark edges touching filtered-out time nodes as invalid
        valid = torch.ones(ei.size(1), dtype=torch.bool)
        if src_type == time_node_type:
            valid &= id_maps[time_node_type][src_ids] >= 0
        if dst_type == time_node_type:
            valid &= id_maps[time_node_type][dst_ids] >= 0

        src_ids, dst_ids = src_ids[valid], dst_ids[valid]
        filtered_edges[(src_type, rel, dst_type)] = (src_ids, dst_ids)

        # Track which non-time node IDs are still referenced
        if src_type != time_node_type and src_ids.numel() > 0:
            referenced[src_type].update(src_ids.tolist())
        if dst_type != time_node_type and dst_ids.numel() > 0:
            referenced[dst_type].update(dst_ids.tolist())

    # --- Build id maps for non-time node types (keep only referenced) --------
    for ntype in g.node_types:
        if ntype == time_node_type:
            continue
        if referenced[ntype]:
            old_ids = torch.tensor(sorted(referenced[ntype]), dtype=torch.long)
        else:
            old_ids = torch.zeros(0, dtype=torch.long)
        mapping = torch.full((g[ntype].num_nodes,), -1, dtype=torch.long)
        mapping[old_ids] = torch.arange(old_ids.size(0), dtype=torch.long)
        id_maps[ntype] = mapping

    # --- Pass 2: build the subgraph ------------------------------------------
    g_sub = HeteroData()

    # Copy node stores — filter all types to their surviving IDs
    for ntype in g.node_types:
        mapping = id_maps[ntype]
        surviving = (mapping >= 0).nonzero(as_tuple=False).squeeze(1)
        for key, val in g[ntype].items():
            if isinstance(val, torch.Tensor) and val.size(0) == g[ntype].num_nodes:
                g_sub[ntype][key] = val[surviving]
            else:
                g_sub[ntype][key] = val
        g_sub[ntype].num_nodes = surviving.size(0)

    # Remap edge endpoints and store
    for (src_type, rel, dst_type), (src_ids, dst_ids) in filtered_edges.items():
        if src_ids.numel() == 0:
            g_sub[src_type, rel, dst_type].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )
            continue
        new_src = id_maps[src_type][src_ids]
        new_dst = id_maps[dst_type][dst_ids]
        g_sub[src_type, rel, dst_type].edge_index = torch.stack(
            [new_src, new_dst], dim=0
        )

    return g_sub


_EXT_FIELDS = [
    "dataset", "metapath", "snapshot", "fraction", "k",
    "n_edges_exact", "n_edges_kmv",
    "adj_mb_exact", "adj_mb_kmv",
    "t_train",
    "t_exact_mat", "t_exact_infer",
    "f1_exact",
    "t_kmv_mat", "t_kmv_infer",
    "f1_kmv",
    "cka_kmv", "pred_agreement", "dirichlet_exact", "dirichlet_kmv",
    "depthwise_cka",
    "speedup_kmv",
]


def _setup_logging(out_dir: Path, dataset: str) -> logging.Logger:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"run_extension_{ts}.log"

    logger = logging.getLogger(f"extension.{dataset}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Log file: {log_file}")
    return logger


def _open_csv(path: Path, fields: List[str]) -> Tuple:
    is_new = not path.exists() or path.stat().st_size == 0
    fh     = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if is_new:
        writer.writeheader()
    return fh, writer


def _done_metapaths(path: Path, expected_snapshots: int) -> Set[str]:
    """Return metapaths that are complete (all snapshots written) or permanently failed."""
    if not path.exists():
        return set()
    counts: dict = {}
    failed: set = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mp = row["metapath"]
            if row.get("snapshot") == "FAILED":
                failed.add(mp)
            else:
                counts[mp] = counts.get(mp, 0) + 1
    complete = {mp for mp, n in counts.items() if n >= expected_snapshots}
    return complete | failed


def _train_sage(
    g_homo:      Data,
    in_dim:      int,
    num_classes: int,
    epochs:      int,
    device:      torch.device,
    log:         logging.Logger,
    patience:    int = 10,
) -> torch.nn.Module:
    """Train a 2-layer SAGE on exactly-materialized H_0 with early stopping.

    Returns frozen model (moved to *device* — caller may move to CPU afterwards).
    """
    from src.models import get_model
    model = get_model("SAGE", in_dim, num_classes, config.HIDDEN_DIM).to(device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    x          = g_homo.x.to(device)
    edge_index = g_homo.edge_index.to(device)
    labels     = g_homo.y.to(device)
    train_mask = g_homo.train_mask.to(device)
    val_mask   = g_homo.val_mask.to(device)

    # Filter out ignore-label nodes (-100) from masks — otherwise cross_entropy
    # returns nan and early stopping never triggers.
    valid_labels = labels >= 0
    train_mask = train_mask & valid_labels
    val_mask   = val_mask & valid_labels
    if val_mask.sum() == 0:
        log.warning("    [SAGE] No valid labels in val_mask — training %d epochs without early stop", epochs)
    if train_mask.sum() == 0:
        log.error("    [SAGE] No valid labels in train_mask — model will not learn!")

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        out  = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        opt.step()

        # Evaluate every 5 epochs for early stopping
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(model(x, edge_index)[val_mask],
                                           labels[val_mask]).item()
            if epoch % 20 == 0:
                log.debug("    [SAGE epoch %3d] loss=%.4f  val_loss=%.4f", epoch, loss.item(), val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    log.info("    [SAGE] Early stop at epoch %d (best val_loss=%.4f)", epoch, best_val_loss)
                    break
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _run_cpp_exact(engine, folder, timeout=600):
    """Run C++ exact materialization. Returns (algo_time_s, output_filepath).

    May raise MemoryError (C++ OOM) or RuntimeError (timeout / crash).
    """
    rule_file   = os.path.join(engine.data_dir, f"cod-rules_{folder}.limit")
    output_file = os.path.join(engine.data_dir, "mat_exact.adj")
    elapsed = engine.run_command("materialize", rule_file, output_file, timeout=timeout)
    return elapsed, output_file


def _run_cpp_sketch(engine, folder, k, timeout=600):
    """Run C++ KMV sketch materialization. Returns (algo_time_s, output_filepath).

    May raise MemoryError (C++ OOM) or RuntimeError (timeout / crash).
    """
    rule_file   = os.path.join(engine.data_dir, f"cod-rules_{folder}.limit")
    output_base = os.path.join(engine.data_dir, "mat_sketch")
    elapsed = engine.run_command("sketch", rule_file, output_base, k=k, timeout=timeout)
    return elapsed, output_base + "_0"


def _load_adj(engine, filepath, num_nodes, node_offset, max_adj_mb=None):
    """Load adjacency file into PyG Data.

    Raises MemoryError if file exceeds *max_adj_mb* or if the system runs out
    of RAM while building the tensor.
    """
    if max_adj_mb is not None and os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > max_adj_mb:
            raise MemoryError(
                f"Adjacency file is {size_mb:.0f} MB (limit {max_adj_mb:.0f} MB)")
    return engine.load_result(filepath, num_nodes, node_offset)


def _count_edges_from_file(filepath: str) -> int:
    """Count edges in a C++ adjacency file without loading into memory.

    File format: ``src_node  nbr1  nbr2 ...`` per line.
    Returns total number of (src, nbr) pairs.
    """
    n = 0
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                n += len(parts) - 1
    return n



def _infer(model, g_homo, in_dim, device):
    x          = F.pad(g_homo.x, (0, max(0, in_dim - g_homo.x.size(1)))).to(device)
    edge_index = g_homo.edge_index.to(device)
    with torch.no_grad():
        return model(x, edge_index)


def _infer_layerwise(model, g_homo, in_dim, device):
    """Run inference and return intermediate representations at each layer."""
    x          = F.pad(g_homo.x, (0, max(0, in_dim - g_homo.x.size(1)))).to(device)
    edge_index = g_homo.edge_index.to(device)
    intermediates = []
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            x = layer(x, edge_index)
            if i < model.num_layers - 1:
                x = F.relu(x)
            intermediates.append(x.clone())
    return intermediates  # list of tensors, one per layer


def _f1(logits, labels, mask):
    from torchmetrics.functional import f1_score
    # Filter out ignore-label nodes (e.g. -100) within the mask
    valid = mask.clone()
    valid[mask] &= (labels[mask] >= 0)
    if valid.sum() == 0:
        return 0.0
    preds = logits[valid].argmax(dim=1)
    return f1_score(
        preds, labels[valid],
        task="multiclass",
        num_classes=logits.size(1),
        average="macro",
    ).item()


def _prediction_agreement(z_exact, z_kmv, mask):
    """Fraction of nodes where argmax(z_exact) == argmax(z_kmv)."""
    pred_exact = z_exact[mask].argmax(dim=1)
    pred_kmv   = z_kmv[mask].argmax(dim=1)
    return (pred_exact == pred_kmv).float().mean().item()


def _dirichlet_energy(z, edge_index, num_nodes):
    """Dirichlet energy: (1/2|V|) * sum_{(i,j) in E} ||z_i - z_j||^2."""
    src, dst = edge_index[0], edge_index[1]
    diff = z[src] - z[dst]
    energy = 0.5 * (diff * diff).sum() / num_nodes
    return energy.item()


def _depthwise_cka(layers_exact, layers_kmv, mask, cka):
    """CKA at each GNN layer. Returns list of floats."""
    results = []
    for z_e, z_k in zip(layers_exact, layers_kmv):
        val = cka.calculate(z_e[mask], z_k[mask])
        results.append(round(val, 6))
    return results


def _run_one_metapath(
    metapath:    str,
    dataset:     str,
    folder:      str,
    data_dir:    str,
    g_full,
    info:        dict,
    runner:      GraphPrepRunner,
    engine:      CppEngine,
    cka:         LinearCKA,
    fractions:   List[float],
    k:           int,
    epochs:      int,
    topr:        str,
    max_adj_mb:  Optional[float],
    timeout:     int,
    ext_w:       csv.DictWriter,
    log:         logging.Logger,
) -> None:
    infer_device = config.DEVICE  # CPU when --cpu is set (fair timing for Phase 2)
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_ntype = config.get_dataset_config(dataset).target_node
    labels      = info["labels"]
    masks       = info["masks"]
    num_classes = info["num_classes"]

    # Build streaming snapshots
    time_node_type = next(
        (nt for nt in g_full.node_types if hasattr(g_full[nt], "year")), None
    )

    if time_node_type is not None:
        # Node-level temporal filtering: keep papers with year <= cutoff
        # Attach labels/masks to g_full so _subgraph_by_year slices them automatically
        g_full[target_ntype].y          = labels
        g_full[target_ntype].train_mask = masks["train"]
        g_full[target_ntype].val_mask   = masks["val"]
        g_full[target_ntype].test_mask  = masks["test"]
        log.info("    Using node-level temporal partitioning (year on '%s')", time_node_type)
        # Preview node counts without keeping all snapshots in memory
        for frac in fractions:
            g_snap = _subgraph_by_year(g_full, frac, time_node_type)
            log.info("      frac=%.0f%%  %s nodes=%d", frac*100, target_ntype,
                     g_snap[target_ntype].num_nodes)
            del g_snap
        import gc; gc.collect()

    # Helper: compute node_offset and num_target for a given snapshot graph
    def _snap_params(g_snap):
        s_ntypes = sorted(g_snap.node_types)
        offset = sum(g_snap[nt].num_nodes for nt in s_ntypes if nt < target_ntype)
        n_target = g_snap[target_ntype].num_nodes
        return offset, n_target

    node_level = time_node_type is not None

    def _make_snap(frac):
        """Lazily create a single snapshot (node-level or edge-level)."""
        if node_level:
            return _subgraph_by_year(g_full, frac, time_node_type)
        else:
            # Edge-level: build just this fraction via TemporalPartitioner
            rel_to_etype = {et[1]: et for et in g_full.edge_types}
            meta_path_tuples = [rel_to_etype[r] for r in metapath.split(",")]
            snaps = TemporalPartitioner.partition(
                g=g_full, meta_path=meta_path_tuples,
                fractions=[frac], time_node_type=None,
            )
            g_snap = snaps[0].graph
            g_snap[target_ntype].y          = labels
            g_snap[target_ntype].train_mask = masks["train"]
            g_snap[target_ntype].val_mask   = masks["val"]
            g_snap[target_ntype].test_mask  = masks["test"]
            return g_snap

    # Phase 1: Train on second-largest fraction (good label coverage,
    # feasible materialization). 100% can OOM/timeout on exact materialize.
    train_frac = fractions[-2] if len(fractions) >= 2 else fractions[-1]
    log.info("    [Phase 1] Staging training snapshot (%.0f%%) + training SAGE...  [RSS=%s]",
             train_frac * 100, _mem_mb())
    g_train = _make_snap(train_frac)
    PyGToCppAdapter(data_dir).convert(g_train)
    compile_rule_for_cpp(metapath, g_train, data_dir, folder)
    if node_level:
        generate_qnodes(data_dir, folder, target_node_type=target_ntype, g_hetero=g_train)

    offset0, n_target0 = _snap_params(g_train)
    _, exact_file = _run_cpp_exact(engine, folder, timeout)
    g_h0 = _load_adj(engine, exact_file, n_target0, offset0, max_adj_mb)

    g_h0.x          = g_train[target_ntype].x
    g_h0.y          = g_train[target_ntype].y if node_level else labels
    g_h0.train_mask = g_train[target_ntype].train_mask if node_level else masks["train"]
    g_h0.val_mask   = g_train[target_ntype].val_mask if node_level else masks["val"]
    g_h0.test_mask  = g_train[target_ntype].test_mask if node_level else masks["test"]
    del g_train; import gc; gc.collect()

    in_dim = g_h0.x.size(1)
    t0_train = time.perf_counter()
    try:
        sage_model = _train_sage(g_h0, in_dim, num_classes, epochs, train_device, log)
    except RuntimeError as e:
        if "CUDA" in str(e) and train_device.type == "cuda":
            log.warning("    [Phase 1] GPU OOM — retrying training on CPU")
            torch.cuda.empty_cache()
            train_device = torch.device("cpu")
            sage_model = _train_sage(g_h0, in_dim, num_classes, epochs, train_device, log)
        else:
            raise
    t_train = time.perf_counter() - t0_train
    del g_h0; gc.collect()
    sage_model = sage_model.to(infer_device)  # move to CPU for fair Phase 2 timing
    log.info("    [Phase 1] Done. Model frozen. (train=%.2fs, trained on %s)  [RSS=%s]",
             t_train, train_device, _mem_mb())

    # Phase 2: Compare exact vs KMV at each fraction (computed lazily)
    for i, frac in enumerate(fractions):
        label = f"t={i} ({int(frac*100)}%)"
        log.info("    [Phase 2] %s  [RSS=%s]", label, _mem_mb())
        g_snap = _make_snap(frac)

        # Per-snapshot params (node counts change with node-level filtering)
        snap_offset, snap_n_target = _snap_params(g_snap)
        snap_x      = g_snap[target_ntype].x
        snap_labels = g_snap[target_ntype].y if node_level else labels
        snap_test   = (g_snap[target_ntype].test_mask if node_level else masks["test"]).to(infer_device)
        snap_labels_d = snap_labels.to(infer_device)

        log.info("      [mem] after _make_snap:  [RSS=%s]", _mem_mb())
        PyGToCppAdapter(data_dir).convert(g_snap)
        compile_rule_for_cpp(metapath, g_snap, data_dir, folder)
        if node_level:
            generate_qnodes(data_dir, folder, target_node_type=target_ntype, g_hetero=g_snap)

        log.info("      [mem] after staging:     [RSS=%s]", _mem_mb())

        # --- Exact: run C++, then try loading into Python ---
        t_exact_mat      = None
        n_edges_exact    = None
        adj_mb_exact     = None
        z_exact          = None
        f1_exact         = None
        t_exact_infer    = None
        layers_exact     = None
        dirichlet_exact  = None

        try:
            t_exact_mat, exact_file = _run_cpp_exact(engine, folder, timeout)
            n_edges_exact = _count_edges_from_file(exact_file)
            if os.path.exists(exact_file):
                adj_mb_exact = os.path.getsize(exact_file) / (1024 * 1024)
            log.info("      [mem] after exact C++:   [RSS=%s]", _mem_mb())
            try:
                g_exact   = _load_adj(engine, exact_file, snap_n_target, snap_offset, max_adj_mb)
                g_exact.x = snap_x
                log.info("      [mem] after exact load:  [RSS=%s]", _mem_mb())
                t0        = time.perf_counter()
                z_exact   = _infer(sage_model, g_exact, in_dim, infer_device)
                t_exact_infer = time.perf_counter() - t0
                log.info("      [mem] after exact infer: [RSS=%s]", _mem_mb())
                f1_exact  = _f1(z_exact, snap_labels_d, snap_test)
                layers_exact = _infer_layerwise(sage_model, g_exact, in_dim, infer_device)
                log.info("      [mem] after layerwise:   [RSS=%s]", _mem_mb())
                dirichlet_exact = _dirichlet_energy(z_exact, g_exact.edge_index.to(infer_device), snap_n_target)
                del g_exact; gc.collect()  # free memory before KMV
                log.info("      [mem] after exact cleanup: [RSS=%s]", _mem_mb())
            except (MemoryError, RuntimeError) as e:
                log.warning("      Exact load/infer OOM: %s", e)
        except (MemoryError, RuntimeError) as e:
            log.warning("      Exact C++ OOM: %s", e)

        # --- KMV: always runs (capped at K × |peers| edges) ---
        log.info("      [mem] before KMV C++:    [RSS=%s]", _mem_mb())
        t_kmv_mat, kmv_file = _run_cpp_sketch(engine, folder, k, timeout)
        adj_mb_kmv = None
        if os.path.exists(kmv_file):
            adj_mb_kmv = os.path.getsize(kmv_file) / (1024 * 1024)
        g_kmv   = _load_adj(engine, kmv_file, snap_n_target, snap_offset)
        g_kmv.x = snap_x
        log.info("      [mem] after KMV load:    [RSS=%s]", _mem_mb())
        t0      = time.perf_counter()
        z_kmv   = _infer(sage_model, g_kmv, in_dim, infer_device)
        t_kmv_infer = time.perf_counter() - t0
        log.info("      [mem] after KMV infer:   [RSS=%s]", _mem_mb())
        f1_kmv  = _f1(z_kmv, snap_labels_d, snap_test)
        n_edges_kmv = _count_edges_from_file(kmv_file)
        layers_kmv = _infer_layerwise(sage_model, g_kmv, in_dim, infer_device)
        log.info("      [mem] after KMV layers:  [RSS=%s]", _mem_mb())
        dirichlet_kmv = _dirichlet_energy(z_kmv, g_kmv.edge_index.to(infer_device), snap_n_target)

        # --- Derived metrics (only when exact loaded successfully) ---
        cka_val        = None
        pred_agree     = None
        depthwise_vals = None
        if z_exact is not None:
            cka_val = cka.calculate(
                z_exact[snap_test].to(infer_device), z_kmv[snap_test].to(infer_device))
            pred_agree = _prediction_agreement(z_exact, z_kmv, snap_test)
        if layers_exact is not None:
            depthwise_vals = _depthwise_cka(layers_exact, layers_kmv, snap_test, cka)

        speedup = None
        if t_exact_mat is not None:
            speedup = t_exact_mat / max(t_kmv_mat, 1e-9)

        # --- Logging ---
        exact_str = (
            f"edges={n_edges_exact}  mat={t_exact_mat:.2f}s  "
            f"inf={t_exact_infer:.2f}s  F1={f1_exact:.4f}"
            if f1_exact is not None
            else f"edges={n_edges_exact or 'N/A'}  mat={t_exact_mat:.2f}s  LOAD_OOM"
            if t_exact_mat is not None
            else "C++_OOM"
        )
        kmv_str = (
            f"edges={n_edges_kmv}  mat={t_kmv_mat:.2f}s  "
            f"inf={t_kmv_infer:.2f}s  F1={f1_kmv:.4f}"
        )
        cka_str = f"  CKA={cka_val:.4f}" if cka_val is not None else ""
        spd_str = f"  speedup={speedup:.1f}x" if speedup is not None else ""
        log.info("      Exact: %s | KMV: %s%s%s", exact_str, kmv_str, cka_str, spd_str)

        # --- CSV row (empty string for unavailable fields) ---
        def _fmt(val, digits=6):
            return round(val, digits) if val is not None else ""

        ext_w.writerow({
            "dataset":          dataset,
            "metapath":         metapath,
            "snapshot":         label,
            "fraction":         frac,
            "k":                k,
            "n_edges_exact":    n_edges_exact if n_edges_exact is not None else "",
            "n_edges_kmv":      n_edges_kmv,
            "adj_mb_exact":     _fmt(adj_mb_exact, 2),
            "adj_mb_kmv":       _fmt(adj_mb_kmv, 2),
            "t_train":          round(t_train, 6),
            "t_exact_mat":      _fmt(t_exact_mat),
            "t_exact_infer":    _fmt(t_exact_infer),
            "f1_exact":         _fmt(f1_exact),
            "t_kmv_mat":        _fmt(t_kmv_mat),
            "t_kmv_infer":      _fmt(t_kmv_infer),
            "f1_kmv":           _fmt(f1_kmv),
            "cka_kmv":          _fmt(cka_val),
            "pred_agreement":   _fmt(pred_agree),
            "dirichlet_exact":  _fmt(dirichlet_exact),
            "dirichlet_kmv":    _fmt(dirichlet_kmv),
            "depthwise_cka":    str(depthwise_vals) if depthwise_vals is not None else "",
            "speedup_kmv":      _fmt(speedup, 4),
        })
        # Free per-snapshot tensors
        del g_snap, g_kmv, z_kmv, layers_kmv
        if z_exact is not None:
            del z_exact, layers_exact
        gc.collect()
        log.info("      [cleanup done]  [RSS=%s]", _mem_mb())

    # Free per-metapath tensors before next iteration
    import gc; gc.collect()
    # Re-stage the full graph for the next metapath
    PyGToCppAdapter(data_dir).convert(g_full)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run extension experiments (KMV vs Exact) across all metapaths."
    )
    parser.add_argument("dataset",
                        help="Dataset key (e.g. HGB_ACM, HGB_DBLP).")
    parser.add_argument("--max-metapaths",   type=int,   default=500)
    parser.add_argument("--min-conf",        type=float, default=DEFAULT_MIN_CONF)
    parser.add_argument("--force-remine",    action="store_true")
    parser.add_argument("--mining-timeout",  type=int,   default=10)
    parser.add_argument("--fractions",       type=float, nargs="+", default=DEFAULT_FRACTIONS,
                        help="Cumulative snapshot fractions (last must be 1.0).")
    parser.add_argument("--k",               type=int,   default=DEFAULT_K,
                        help="KMV sketch size k for GloD (default 32).")
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS,
                        help="SAGE training epochs (default 100).")
    parser.add_argument("--topr",            type=str,   default=DEFAULT_TOPR)
    parser.add_argument("--max-adj-mb",      type=float, default=50.0,
                        help="Skip metapaths whose EXACT adjacency file exceeds this size in MB. "
                             "Default: 50 (laptop-safe). Set 0 to disable (no limit, for servers).")
    parser.add_argument("--timeout",         type=int,   default=1800,
                        help="Per-C++-call subprocess timeout in seconds (default 1800 = 30 min). "
                             "Metapaths that exceed this are marked FAILED and skipped on resume.")
    parser.add_argument("--num-cpu-threads", type=int,   default=2,
                        help="Number of CPU threads for PyTorch (default 2). "
                             "Prevents SAGE training from saturating all cores.")
    parser.add_argument("--cpu",             action="store_true",
                        help="Force CPU for GNN training/inference (recommended for fair timing).")
    parser.add_argument("--output-name",     type=str, default="extension",
                        help="Base name for output CSV (default: extension → extension.csv).")
    args = parser.parse_args()

    # Normalise max_adj_mb: 0 means no limit
    if args.max_adj_mb == 0:
        args.max_adj_mb = None

    import torch as _torch
    _torch.set_num_threads(args.num_cpu_threads)

    if args.cpu:
        config.DEVICE = _torch.device("cpu")

    dataset  = args.dataset
    folder   = config.get_folder_name(dataset)
    data_dir = os.path.join(project_root, folder)
    out_dir  = Path(project_root) / "results" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logging(out_dir, dataset)

    log.info("")
    log.info("=" * 60)
    log.info("  Extension Experiments: %s", dataset)
    log.info("  fractions=%s  k=%d  epochs=%d  topr=%s",
             args.fractions, args.k, args.epochs, args.topr)
    log.info("  min_conf=%.3f  max_metapaths=%d  mining_timeout=%ds",
             args.min_conf, args.max_metapaths, args.mining_timeout)
    log.info("  timeout=%ds  max_adj_mb=%s  num_cpu_threads=%d",
             args.timeout,
             f"{args.max_adj_mb:.0f} MB" if args.max_adj_mb else "unlimited",
             args.num_cpu_threads)
    log.info("  Output → %s/", out_dir)
    log.info("=" * 60)

    # ---- Load graph --------------------------------------------------------
    log.info("")
    log.info("[1/4] Loading graph...")
    cfg           = config.get_dataset_config(dataset)
    g_full, info  = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    log.info("      node types  : %s", list(g_full.node_types))
    log.info("      target node : %s", cfg.target_node)

    # ---- Mine / load metapaths ---------------------------------------------
    log.info("")
    log.info("[2/4] Loading + validating metapaths...")

    # If the dataset has pre-configured metapaths in config, use them directly.
    # This avoids AnyBURL mining on large datasets (OGB_MAG etc.) where mining
    # is slow and the good metapaths are already known.
    if cfg.suggested_paths and not args.force_remine:
        metapaths = cfg.suggested_paths[:args.max_metapaths]
        log.info("      Using %d pre-configured metapaths from config (skip AnyBURL).", len(metapaths))
        for mp in metapaths:
            log.info("        %s", mp)
    else:
        work_dir   = os.path.join(config.DATA_DIR, f"mining_{dataset}")
        anyburl    = AnyBURLRunner(work_dir, config.ANYBURL_JAR)
        anyburl.export_for_mining(g_full)

        rules_file = os.path.join(work_dir, "anyburl_rules.txt")
        if args.force_remine or not os.path.exists(rules_file):
            log.info("      Mining via AnyBURL (snapshot at %ds)...", args.mining_timeout)
            anyburl.run_mining(timeout=args.mining_timeout, max_length=6, num_threads=4)
        else:
            log.info("      Using cached rules.")

        metapaths, stats = load_validated_metapaths(
            rules_file=rules_file,
            g_hetero=g_full,
            target_node=cfg.target_node,
            min_conf=args.min_conf,
            max_n=args.max_metapaths,
        )
        log.info("      valid metapaths: %d  (cap=%d)", stats["valid_mirrored"], args.max_metapaths)
        log.info("      returned:        %d", stats["returned"])

        if not metapaths:
            log.error("[ERROR] No valid metapaths. Try --force-remine or lower --min-conf.")
            sys.exit(1)

    # ---- Stage C++ data (once for full graph) ------------------------------
    log.info("")
    log.info("[3/4] Staging C++ data → %s/", data_dir)
    PyGToCppAdapter(data_dir).convert(g_full)
    generate_qnodes(data_dir, folder, target_node_type=cfg.target_node, g_hetero=g_full)
    setup_global_res_dirs(folder, project_root)

    runner = GraphPrepRunner(binary=config.CPP_EXECUTABLE, working_dir=project_root, verbose=False)
    engine = CppEngine(executable_path=config.CPP_EXECUTABLE, data_dir=data_dir)
    cka    = LinearCKA(device=config.DEVICE)

    # ---- Open CSV (resume-safe) --------------------------------------------
    log.info("")
    log.info("[4/4] Running extension experiments...")
    ext_path = out_dir / f"{args.output_name}.csv"
    expected_snapshots = len(args.fractions)  # Phase 2 runs all fractions
    done     = _done_metapaths(ext_path, expected_snapshots)
    total    = len(metapaths)
    n_done   = sum(1 for mp in metapaths if mp in done)
    log.info("      %d/%d already done, %d to run.", n_done, total, total - n_done)
    log.info("")

    ext_fh, ext_w = _open_csv(ext_path, _EXT_FIELDS)
    n_failed = 0

    try:
        for idx, metapath in enumerate(metapaths, start=1):
            if metapath in done:
                log.info("  [%3d/%d] skip  %s", idx, total, metapath[:70])
                continue

            log.info("  [%3d/%d] run   %s", idx, total, metapath[:70])
            log.debug("         full: %s", metapath)

            try:
                _run_one_metapath(
                    metapath=metapath, dataset=dataset, folder=folder,
                    data_dir=data_dir, g_full=g_full, info=info,
                    runner=runner, engine=engine, cka=cka,
                    fractions=args.fractions, k=args.k, epochs=args.epochs,
                    topr=args.topr, max_adj_mb=args.max_adj_mb,
                    timeout=args.timeout, ext_w=ext_w, log=log,
                )
                ext_fh.flush()
            except SystemExit as exc:
                n_failed += 1
                log.warning("  [%3d/%d] CRASH (C++ exit %s) — marking FAILED, will skip on resume: %s",
                             idx, total, exc.code, metapath[:70])
                log.debug("Full traceback:", exc_info=True)
                ext_w.writerow({"dataset": dataset, "metapath": metapath, "snapshot": "FAILED",
                                 "fraction": "", "k": "", "n_edges_exact": "", "n_edges_kmv": "",
                                 "adj_mb_exact": "", "adj_mb_kmv": "",
                                 "t_train": "", "t_exact_mat": "",
                                 "t_exact_infer": "", "f1_exact": "", "t_kmv_mat": "",
                                 "t_kmv_infer": "", "f1_kmv": "", "cka_kmv": "",
                                 "pred_agreement": "", "dirichlet_exact": "", "dirichlet_kmv": "",
                                 "depthwise_cka": "", "speedup_kmv": ""})
                ext_fh.flush()
                done.add(metapath)
            except Exception as exc:
                n_failed += 1
                log.warning("  [%3d/%d] ERROR (%s) — marking FAILED, will skip on resume: %s",
                             idx, total, exc, metapath[:70])
                log.debug("Full traceback:", exc_info=True)
                ext_w.writerow({"dataset": dataset, "metapath": metapath, "snapshot": "FAILED",
                                 "fraction": "", "k": "", "n_edges_exact": "", "n_edges_kmv": "",
                                 "adj_mb_exact": "", "adj_mb_kmv": "",
                                 "t_train": "", "t_exact_mat": "",
                                 "t_exact_infer": "", "f1_exact": "", "t_kmv_mat": "",
                                 "t_kmv_infer": "", "f1_kmv": "", "cka_kmv": "",
                                 "pred_agreement": "", "dirichlet_exact": "", "dirichlet_kmv": "",
                                 "depthwise_cka": "", "speedup_kmv": ""})
                ext_fh.flush()
                done.add(metapath)
    finally:
        ext_fh.close()

    # ---- Summary -----------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("  Done.  Results in %s/", out_dir)
    log.info("    extension.csv  (%d metapaths x %d snapshots, timing: train + mat + infer)",
             total, len(args.fractions) - 1)
    log.info("    columns: %s", ", ".join(_EXT_FIELDS))
    if n_failed:
        log.warning("  %d metapath(s) failed — check log for details.", n_failed)
    log.info("=" * 60)
    log.info("")


if __name__ == "__main__":
    main()
