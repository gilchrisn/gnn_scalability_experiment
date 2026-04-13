"""
exp5_degree_bin_eval.py — Degree-stratified 50-seed evaluation: KMV vs MPRW.

Compares KMV and MPRW materialization across three degree bins on HGB_DBLP.
MPRW is granted a +10% edge density advantage over KMV to make the comparison
conservative; the hypothesis is that KMV is an unbiased estimator across the
full degree distribution while MPRW concentrates its budget on hubs.

Split:  40% Train / 60% Test  (V_test = 60% of target nodes by count)
Seeds:  50 (independent hash/walk seeds; frozen SAGE weights are constant)
Bins:   Tail (0–50th pct), Mid (50–90th pct), Hub (top 10%) — by degree in H_exact

Metrics per seed × bin:
  - Prediction Agreement  (accuracy vs y_true)
  - Macro-F1              (sklearn, average='macro')

Modes
-----
  --mock          Synthetic demo; immediately runnable without C++/data.
  (default)       Real pipeline: requires partition.json + frozen weights + C++.

Usage
-----
  # Immediate demo (mock data)
  python scripts/exp5_degree_bin_eval.py --mock

  # Real DBLP experiment (requires exp1 + exp2 first)
  python scripts/exp1_partition.py --dataset HGB_DBLP --target-type author --train-frac 0.4
  python scripts/exp2_train.py HGB_DBLP \\
      --metapath author_to_paper,paper_to_author --depth 2 \\
      --partition-json results/HGB_DBLP/partition.json
  python scripts/exp5_degree_bin_eval.py \\
      --partition-json results/HGB_DBLP/partition.json \\
      --weights results/HGB_DBLP/weights/author_to_paper_paper_to_author_L2.pt \\
      --metapath author_to_paper,paper_to_author \\
      --depth 2 --k 32 --n-seeds 50

Output
------
  figures/exp5_degree_bin_eval.pdf   (publication-ready plot)
  results/HGB_DBLP/exp5_preds.npz   (cached predictions; re-run skips done seeds)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── matplotlib (non-interactive backend for server/CI) ───────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── colour palette (VLDB-friendly, colourblind-safe) ─────────────────────────
_KMV_COLOR  = "#2CA02C"   # muted green
_MPRW_COLOR = "#D62728"   # brick red


# ===========================================================================
# 1.  SpMM inference (mirrors inference_worker.py; no subprocess overhead)
# ===========================================================================

def _build_normalized_adj_csr(
    ei: torch.Tensor, n: int, device: torch.device
) -> torch.Tensor:
    """D^{-1}A as torch.sparse_csr_tensor for mean-aggregation SAGE."""
    if ei.size(1) == 0:
        crow = torch.zeros(n + 1, dtype=torch.long, device=device)
        col  = torch.empty(0, dtype=torch.long, device=device)
        vals = torch.empty(0, dtype=torch.float32, device=device)
        return torch.sparse_csr_tensor(crow, col, vals, size=(n, n))

    src = ei[0].to(device)
    dst = ei[1].to(device)
    deg = torch.zeros(n, dtype=torch.float32, device=device)
    deg.scatter_add_(0, dst, torch.ones(src.size(0), dtype=torch.float32, device=device))
    vals = deg.clamp(min=1.0).reciprocal()[dst]
    perm   = torch.argsort(dst * n + src)
    dst_s, src_s, vals_s = dst[perm], src[perm], vals[perm]
    crow = torch.zeros(n + 1, dtype=torch.long, device=device)
    crow[1:].scatter_add_(0, dst_s, torch.ones_like(dst_s))
    crow = crow.cumsum(0)
    return torch.sparse_csr_tensor(crow, src_s, vals_s, size=(n, n))


def _sage_spmm_forward(model, x: torch.Tensor, adj_csr: torch.Tensor) -> torch.Tensor:
    """Strict SpMM forward: no [|E|,d] message buffer."""
    for i, layer in enumerate(model.layers):
        agg = torch.sparse.mm(adj_csr, x)
        if hasattr(layer, "lin_l") and hasattr(layer, "lin_r"):
            x_out = layer.lin_l(agg) + layer.lin_r(x)
        elif hasattr(layer, "lin") and hasattr(layer, "lin_r"):
            x_out = layer.lin(agg) + layer.lin_r(x)
        else:
            raise RuntimeError(f"Cannot find SAGEConv weights on layer {i}")
        if i < model.num_layers - 1:
            if hasattr(model, "skip_projs") and i < len(model.skip_projs):
                x_out = x_out + model.skip_projs[i](x)
            x_out = F.relu(x_out)
        x = x_out
    return x


def _run_inference(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    model,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Return logits [n, num_classes] for all target nodes."""
    adj_csr = _build_normalized_adj_csr(edge_index, n, device)
    with torch.no_grad():
        return _sage_spmm_forward(model, x.to(device), adj_csr).cpu()


# ===========================================================================
# 2.  Mock data generation
# ===========================================================================

def _generate_mock_data(
    n_test: int = 5_000,
    n_classes: int = 4,
    n_seeds: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate y_true, node_degrees, preds_kmv, preds_mprw.

    Narrative:
      KMV   — unbiased min-hash; performance gently tracks degree (small slope).
      MPRW  — random walk; biased toward hubs; tail nodes are under-sampled
               despite the 10% density bonus.  Mean acc/F1 collapses in Tail bin.

    Returns
    -------
    y_true      (n_test,)          int class labels 0..n_classes-1
    node_degrees (n_test,)         simulated degree from H_exact (power-law)
    preds_kmv   (n_seeds, n_test)  int predicted labels
    preds_mprw  (n_seeds, n_test)  int predicted labels
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Power-law degree distribution (sparse tail, dense hubs)
    raw = rng.power(0.4, size=n_test)          # heavy tail → many low-degree nodes
    node_degrees = (raw * 500 + 1).astype(int)

    # Ground truth labels — balanced classes
    y_true = (np.arange(n_test) % n_classes).astype(int)
    rng.shuffle(y_true)

    # Per-node base accuracies for KMV and MPRW
    # KMV: nearly flat (0.74–0.80 across degree range)
    log_d = np.log1p(node_degrees.astype(float))
    log_d_norm = (log_d - log_d.min()) / (log_d.max() - log_d.min() + 1e-9)
    kmv_base_acc  = 0.74 + 0.06 * log_d_norm   # mild positive slope

    # MPRW: sharp bias — collapses on low-degree nodes
    # Tail nodes (low log_d_norm): 0.45–0.55 acc
    # Hub nodes (high log_d_norm): 0.85–0.90 acc
    mprw_base_acc = 0.45 + 0.45 * log_d_norm**1.5

    def _sample_preds(base_acc: np.ndarray, seed_offset: int) -> np.ndarray:
        s_rng = np.random.default_rng(1000 + seed_offset)
        correct = s_rng.random(n_test) < base_acc
        wrong_labels = (y_true + s_rng.integers(1, n_classes, size=n_test)) % n_classes
        return np.where(correct, y_true, wrong_labels).astype(int)

    preds_kmv  = np.stack([_sample_preds(kmv_base_acc,  s) for s in range(n_seeds)])
    preds_mprw = np.stack([_sample_preds(mprw_base_acc, s + 500) for s in range(n_seeds)])

    return y_true, node_degrees, preds_kmv, preds_mprw


# ===========================================================================
# 3.  Binning
# ===========================================================================

def _assign_bins(node_degrees: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Assign each node to a degree bin using strict percentile thresholds.

    Bins
    ----
    Tail  0 – 50th pct    |  Mid  50 – 90th pct  |  Hub  top 10%
    """
    p50  = np.percentile(node_degrees, 50)
    p90  = np.percentile(node_degrees, 90)
    tail = np.where(node_degrees <= p50)[0]
    mid  = np.where((node_degrees > p50) & (node_degrees <= p90))[0]
    hub  = np.where(node_degrees > p90)[0]
    return {"Tail\n(0–50%ile)": tail,
            "Mid\n(50–90%ile)": mid,
            "Hub\n(Top 10%)": hub}


# ===========================================================================
# 4.  Metric computation
# ===========================================================================

def _pred_agreement(y_true: np.ndarray, preds: np.ndarray) -> float:
    return float((preds == y_true).mean())


def _macro_f1(y_true: np.ndarray, preds: np.ndarray, n_classes: int) -> float:
    from sklearn.metrics import f1_score
    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, preds, average="macro",
                           labels=list(range(n_classes)), zero_division=0))


def _compute_per_seed_metrics(
    y_true: np.ndarray,
    node_degrees: np.ndarray,
    preds_kmv: np.ndarray,
    preds_mprw: np.ndarray,
    n_classes: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns nested dict:
        metrics[bin_name][method] = {pa: [n_seeds], f1: [n_seeds]}
    """
    bins = _assign_bins(node_degrees)
    n_seeds = preds_kmv.shape[0]

    results: Dict = {}
    for bin_name, idx in bins.items():
        if len(idx) == 0:
            continue
        y_bin = y_true[idx]
        pa_kmv, f1_kmv, pa_mprw, f1_mprw = [], [], [], []
        for s in range(n_seeds):
            pa_kmv.append(_pred_agreement(y_bin, preds_kmv[s, idx]))
            f1_kmv.append(_macro_f1(y_bin, preds_kmv[s, idx], n_classes))
            pa_mprw.append(_pred_agreement(y_bin, preds_mprw[s, idx]))
            f1_mprw.append(_macro_f1(y_bin, preds_mprw[s, idx], n_classes))
        results[bin_name] = {
            "kmv":  {"pa": np.array(pa_kmv),  "f1": np.array(f1_kmv)},
            "mprw": {"pa": np.array(pa_mprw), "f1": np.array(f1_mprw)},
        }
    return results


def _aggregate(arr: np.ndarray) -> Tuple[float, float]:
    """Return (mean, 95% CI half-width) for a 1-D array of seed values."""
    n = len(arr)
    mu  = arr.mean()
    ci  = 1.96 * arr.std(ddof=1) / np.sqrt(n)
    return float(mu), float(ci)


# ===========================================================================
# 5.  Publication-quality plot
# ===========================================================================

def _plot_results(
    metrics: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_path: str,
    n_seeds: int,
    is_mock: bool = False,
    mprw_label: str = "MPRW (+10% Edges)",
    budget_note: str = "MPRW budget = KMV edges x 1.10",
) -> None:
    """
    Two-panel grouped bar chart:
      Left  — Prediction Agreement
      Right — Macro-F1
    """
    bin_labels = list(metrics.keys())
    n_bins = len(bin_labels)
    x = np.arange(n_bins)
    width = 0.32          # bar width
    offset_kmv  = -width / 2
    offset_mprw =  width / 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})

    metric_keys = [("pa", "Prediction Agreement"), ("f1", "Macro-F1 Score")]

    for ax, (mkey, ylabel) in zip(axes, metric_keys):
        mu_kmv, ci_kmv   = [], []
        mu_mprw, ci_mprw = [], []

        for bname in bin_labels:
            m, c = _aggregate(metrics[bname]["kmv"][mkey])
            mu_kmv.append(m);  ci_kmv.append(c)
            m, c = _aggregate(metrics[bname]["mprw"][mkey])
            mu_mprw.append(m); ci_mprw.append(c)

        mu_kmv   = np.array(mu_kmv)
        mu_mprw  = np.array(mu_mprw)
        ci_kmv   = np.array(ci_kmv)
        ci_mprw  = np.array(ci_mprw)

        # ── bars ─────────────────────────────────────────────────────────
        bars_kmv = ax.bar(
            x + offset_kmv, mu_kmv, width,
            color=_KMV_COLOR, alpha=0.88, label="KMV",
            zorder=3,
        )
        bars_mprw = ax.bar(
            x + offset_mprw, mu_mprw, width,
            color=_MPRW_COLOR, alpha=0.88, label=mprw_label,
            zorder=3,
        )

        # ── 95% CI error bars ─────────────────────────────────────────────
        _errbar_kw = dict(fmt="none", capsize=5, capthick=1.5,
                          elinewidth=1.5, zorder=4)
        ax.errorbar(x + offset_kmv,  mu_kmv,  yerr=ci_kmv,
                    ecolor="#1a6e1a", **_errbar_kw)
        ax.errorbar(x + offset_mprw, mu_mprw, yerr=ci_mprw,
                    ecolor="#8b1a1a", **_errbar_kw)

        # ── axes styling ──────────────────────────────────────────────────
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlabel("Degree Bin (V_test, full-graph H_exact degrees)", fontsize=11)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        # Dynamic y-limits: show gap clearly without cramping error bars
        all_vals = np.concatenate([mu_kmv - ci_kmv, mu_mprw - ci_mprw])
        y_min = max(0.0, all_vals.min() - 0.06)
        all_top = np.concatenate([mu_kmv + ci_kmv, mu_mprw + ci_mprw])
        y_max = min(1.0, all_top.max() + 0.06)
        ax.set_ylim(y_min, y_max)

        # ── numeric labels on bars ────────────────────────────────────────
        for bar, mu, ci in zip(bars_kmv, mu_kmv, ci_kmv):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 0.008,
                    f"{mu:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color="#1a6e1a", fontweight="bold")
        for bar, mu, ci in zip(bars_mprw, mu_mprw, ci_mprw):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 0.008,
                    f"{mu:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color="#8b1a1a", fontweight="bold")

    # ── shared legend ─────────────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=_KMV_COLOR,  alpha=0.88, label="KMV"),
        Patch(facecolor=_MPRW_COLOR, alpha=0.88, label=mprw_label),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=2, fontsize=12, frameon=True,
               bbox_to_anchor=(0.5, 1.02))

    mock_tag = "  [mock data]" if is_mock else ""
    fig.suptitle(
        f"KMV vs MPRW: Degree-Stratified Performance on HGB-DBLP (V_test, 40/60 split){mock_tag}\n"
        f"{n_seeds} seeds  |  95% CI error bars  |  {budget_note}",
        fontsize=12, y=1.07,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[exp5] Plot saved → {out_path}")


# ===========================================================================
# 6.  Real pipeline helpers
# ===========================================================================

def _compute_exact_degrees(
    engine,
    folder: str,
    data_dir: str,
    n_target: int,
    offset: int,
    log: logging.Logger,
) -> Tuple[np.ndarray, int]:
    """
    Run ExactD on the full graph.

    Returns
    -------
    degrees      np.ndarray [n_target]  -- per-node neighbour count (no self-loops)
    exact_edges  int                    -- total directed COO entries (incl. self-loops),
                                          matching what engine.load_result() would count.
                                          Use this for apples-to-apples comparison with
                                          KMV and MPRW edge_index.size(1).
    """
    rule_file   = os.path.join(data_dir, f"cod-rules_{folder}.limit")
    output_file = os.path.join(data_dir, "mat_exact_full.adj")

    if not os.path.exists(output_file):
        log.info("Running ExactD on full graph for degree computation...")
        engine.run_command("materialize", rule_file, output_file, timeout=3600)
    else:
        log.info("Found existing mat_exact_full.adj -- skipping ExactD.")

    degrees    = np.zeros(n_target, dtype=np.int64)
    real_edges = 0   # directed non-self-loop entries
    with open(output_file) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u_global = int(parts[0])
            u_local  = u_global - offset
            if 0 <= u_local < n_target:
                neighbours = [
                    int(v) for v in parts[1:]
                    if (int(v) - offset) != u_local
                ]
                degrees[u_local] = len(neighbours)
                real_edges += len(neighbours)

    # engine.load_result / MPRWKernel both call add_self_loops, so reported
    # edge counts include n_target self-loops.  Add them here for consistency.
    exact_edges_coo = real_edges + n_target
    return degrees, exact_edges_coo


def _load_kmv_edge_index(
    engine,
    folder: str,
    data_dir: str,
    k: int,
    seed: int,
    n_target: int,
    offset: int,
    log: logging.Logger,
) -> torch.Tensor:
    """Run C++ sketch and return edge_index [2, E]."""
    from torch_geometric.utils import coalesce, add_self_loops, to_undirected

    rule_file   = os.path.join(data_dir, f"cod-rules_{folder}.limit")
    output_base = os.path.join(data_dir, f"mat_sketch_s{seed}")
    output_file = output_base + "_0"

    engine.run_command("sketch", rule_file, output_base,
                       k=k, seed=seed, timeout=1800)

    srcs, dsts = [], []
    with open(output_file) as fh:
        for line in fh:
            parts = list(map(int, line.strip().split()))
            if not parts:
                continue
            u_local = parts[0] - offset
            if u_local < 0 or u_local >= n_target:
                continue
            for v_global in parts[1:]:
                v_local = v_global - offset
                if 0 <= v_local < n_target:
                    srcs.append(u_local); dsts.append(v_local)

    if srcs:
        ei = torch.tensor([srcs, dsts], dtype=torch.long)
        ei = to_undirected(ei, num_nodes=n_target)
        ei = coalesce(ei, num_nodes=n_target)
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
    ei, _ = add_self_loops(ei, num_nodes=n_target)
    return ei


def _calibrate_mprw_w(
    g_full,
    metapath_triples: List,
    target_ntype: str,
    target_edges: int,
    log: logging.Logger,
) -> Tuple[int, bool, int]:
    """
    Find min w such that MPRW(w) produces >= target_edges.
    Exponential bounding -> binary search.  Cap at 2^16 to avoid OOM.

    Returns
    -------
    (best_w, saturated, plateau_edges)
      saturated=True  means the metapath topology is exhausted: the graph
      simply does not contain enough distinct paths to reach target_edges
      regardless of walk count.  plateau_edges is the topological ceiling.
      The caller should proceed with best_w and log the saturation.
    """
    from src.kernels.mprw import MPRWKernel

    MAX_W = 2 ** 16

    # Detect plateau: if edges stop increasing between doublings, topology is full.
    def _count(w: int) -> int:
        k = MPRWKernel(k=w, seed=0)
        data, _ = k.materialize(g_full, metapath_triples, target_ntype)
        return int(data.edge_index.size(1))

    # Phase 1: exponential bounding with plateau detection
    w     = 1
    count = _count(w)
    log.info("  [mprw-calib] w=%d  edges=%d  target=%d", w, count, target_edges)

    prev_count = -1
    while count < target_edges and w < MAX_W:
        if count == prev_count:
            # Two consecutive doublings produced identical edge counts -> plateau
            log.warning(
                "  [mprw-calib] topology SATURATED at w=%d: "
                "edges=%d (topological ceiling) < target=%d",
                w, count, target_edges,
            )
            log.warning(
                "  [mprw-calib] KMV over-approximates via hash collisions; "
                "MPRW is limited to true co-reachable pairs. "
                "Proceeding with MPRW at saturation (w=%d, edges=%d).",
                w, count,
            )
            return w, True, count
        prev_count = count
        w = min(w * 2, MAX_W)
        count = _count(w)
        log.info("  [mprw-calib] w=%d  edges=%d  target=%d", w, count, target_edges)

    if count < target_edges:
        log.warning(
            "  [mprw-calib] topology SATURATED at w=%d: "
            "edges=%d (topological ceiling) < target=%d",
            w, count, target_edges,
        )
        log.warning(
            "  [mprw-calib] KMV over-approximates via hash collisions; "
            "MPRW is limited to true co-reachable pairs. "
            "Proceeding with MPRW at saturation (w=%d, edges=%d).",
            w, count,
        )
        return w, True, count

    w_high = w
    w_low  = max(1, w // 2)
    best_w = w_high

    # Phase 2: binary search
    while w_low <= w_high:
        mid   = (w_low + w_high) // 2
        count = _count(mid)
        log.info("  [mprw-calib] bisect w=%d  edges=%d", mid, count)
        if count >= target_edges:
            best_w = mid
            w_high = mid - 1
        else:
            w_low = mid + 1

    log.info("  [mprw-calib] best_w=%d  (target achieved)", best_w)
    return best_w, False, _count(best_w)


# ===========================================================================
# 7.  Main experiment loop (real pipeline)
# ===========================================================================

def _run_real_experiment(args, log: logging.Logger):
    """Load DBLP, run 50 seeds of KMV + MPRW, collect per-node predictions."""
    from src.config import config
    from src.data import DatasetFactory
    from src.bridge import PyGToCppAdapter
    from src.bridge.engine import CppEngine
    from src.kernels.mprw import MPRWKernel, parse_metapath_triples
    from src.models import get_model
    from scripts.bench_utils import (
        compile_rule_for_cpp, generate_qnodes, setup_global_res_dirs
    )
    import torch_geometric.utils as pyg_utils

    DATASET = "HGB_DBLP"
    cfg     = config.get_dataset_config(DATASET)
    folder  = config.get_folder_name(DATASET)

    with open(args.partition_json) as f:
        part = json.load(f)

    # ── Load graph ────────────────────────────────────────────────────────
    log.info("Loading HGB_DBLP...")
    g_full, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    target_ntype = cfg.target_node

    test_ids  = torch.tensor(part["test_node_ids"], dtype=torch.long)
    n_target  = g_full[target_ntype].num_nodes
    s_ntypes  = sorted(g_full.node_types)
    offset    = sum(g_full[nt].num_nodes for nt in s_ntypes if nt < target_ntype)

    test_mask = torch.zeros(n_target, dtype=torch.bool)
    test_mask[test_ids] = True
    n_test = int(test_mask.sum())
    log.info("  n_target=%d  n_test=%d  (%.0f%% test)",
             n_target, n_test, 100 * n_test / n_target)

    labels = info["labels"]   # [n_target]
    y_test = labels[test_mask].numpy()    # [n_test]

    # ── Stage C++ files ───────────────────────────────────────────────────
    data_dir = os.path.join(str(_ROOT), folder)
    setup_global_res_dirs(folder, str(_ROOT))
    PyGToCppAdapter(data_dir).convert(g_full)
    compile_rule_for_cpp(args.metapath, g_full, data_dir, folder)
    generate_qnodes(data_dir, folder,
                    target_node_type=target_ntype, g_hetero=g_full)

    engine = CppEngine(executable_path=config.CPP_EXECUTABLE, data_dir=data_dir)

    # ── Node degrees + ExactD edge count (full graph) ─────────────────────
    node_degrees_full, exact_edges = _compute_exact_degrees(
        engine, folder, data_dir, n_target, offset, log
    )
    node_degrees = node_degrees_full[test_mask.numpy().astype(bool)]  # [n_test]

    # ── Load model + features ─────────────────────────────────────────────
    device = torch.device("cpu")
    model = get_model("SAGE", info["in_dim"], info["num_classes"],
                      config.HIDDEN_DIM, num_layers=args.depth).to(device)
    model.load_state_dict(
        torch.load(args.weights, weights_only=True, map_location=device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x_full = g_full[target_ntype].x   # [n_target, in_dim]

    # ── KMV seed=0 reference edge count ──────────────────────────────────
    log.info("KMV seed=0 reference run for MPRW calibration target...")
    ei_kmv0 = _load_kmv_edge_index(
        engine, folder, data_dir, args.k, seed=0,
        n_target=n_target, offset=offset, log=log,
    )
    kmv_ref_edges = int(ei_kmv0.size(1))
    mprw_target   = int(kmv_ref_edges * 1.10)

    # ── Sanity check: ExactD vs KMV ───────────────────────────────────────
    # ExactD is the ground truth.  Comparing it to KMV tells us whether KMV
    # is over- or under-approximating, which determines how to interpret any
    # gap between KMV and MPRW.
    log.info("")
    log.info("=" * 60)
    log.info("  SANITY CHECK: ExactD vs KMV (seed=0)")
    log.info("=" * 60)
    log.info("  Exact edges (H_exact, full graph, incl. self-loops) : %d", exact_edges)
    log.info("  KMV   edges (k=%d, seed=0, incl. self-loops)        : %d",
             args.k, kmv_ref_edges)
    delta = kmv_ref_edges - exact_edges
    pct   = 100.0 * delta / max(exact_edges, 1)
    if delta > 0:
        log.info("  KMV has %+d edges vs Exact (%+.1f%%)  --> KMV OVER-approximates",
                 delta, pct)
        log.info("  Interpretation: sketch hash collisions are creating false-positive")
        log.info("  edges. KMV's extra edges are NOT in the true co-reachable set.")
        log.info("  MPRW's topology ceiling (%d) may be closer to ground truth.",
                 "TBD -- see calibration below")
    elif delta < 0:
        log.info("  KMV has %+d edges vs Exact (%+.1f%%)  --> KMV UNDER-approximates",
                 delta, pct)
        log.info("  Interpretation: sketch width k=%d is too small; some true neighbours",
                 args.k)
        log.info("  are missed. Try a larger k.")
    else:
        log.info("  KMV matches Exact exactly -- sketch is well-calibrated.")
    log.info("  MPRW target (+10%% of KMV)                          : %d", mprw_target)
    log.info("=" * 60)
    log.info("")

    # ── Calibrate MPRW best_w ─────────────────────────────────────────────
    metapath_triples = parse_metapath_triples(args.metapath, g_full)
    best_w, mprw_saturated, mprw_plateau_edges = _calibrate_mprw_w(
        g_full, metapath_triples, target_ntype, mprw_target, log
    )
    if mprw_saturated:
        log.warning(
            "  NOTE: MPRW topology ceiling = %d edges  |  "
            "Exact = %d edges  |  KMV = %d edges.",
            mprw_plateau_edges, exact_edges, kmv_ref_edges,
        )
        if mprw_plateau_edges <= exact_edges <= kmv_ref_edges:
            log.warning(
                "  Exact sits between MPRW and KMV: MPRW under-samples "
                "true reachable pairs; KMV over-approximates. "
                "Both biases are real and in opposite directions."
            )
        elif exact_edges <= mprw_plateau_edges:
            log.warning(
                "  MPRW ceiling >= Exact: MPRW has full topological coverage. "
                "KMV is over-approximating beyond the true reachable set."
            )
        else:
            log.warning(
                "  Exact > MPRW ceiling: MPRW is genuinely missing reachable "
                "pairs (under-sampling). KMV is closer to ground truth."
            )

    # ── Prediction cache (resume-safe) ────────────────────────────────────
    cache_dir  = Path("results") / DATASET
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp_safe    = args.metapath.replace(",", "_")
    cache_file = cache_dir / f"exp5_preds_{mp_safe}_k{args.k}.npz"

    if cache_file.exists():
        log.info("Loading cached predictions from %s", cache_file)
        cache      = np.load(cache_file)
        preds_kmv  = cache["preds_kmv"]
        preds_mprw = cache["preds_mprw"]
        done_seeds = int(cache["done_seeds"])
        log.info("  Resuming from seed %d", done_seeds)
    else:
        preds_kmv  = np.full((args.n_seeds, n_test), -1, dtype=np.int32)
        preds_mprw = np.full((args.n_seeds, n_test), -1, dtype=np.int32)
        done_seeds = 0

    _prev_kmv_edges:  Optional[int] = None
    _prev_mprw_edges: Optional[int] = None

    for seed in range(done_seeds, args.n_seeds):
        log.info("\n-- Seed %d / %d --", seed, args.n_seeds - 1)

        # KMV
        t0 = time.perf_counter()
        ei_kmv = _load_kmv_edge_index(
            engine, folder, data_dir, args.k, seed=seed,
            n_target=n_target, offset=offset, log=log,
        )
        kmv_edges = int(ei_kmv.size(1))
        logits_kmv = _run_inference(ei_kmv, x_full, model, n_target, device)
        preds_kmv[seed] = logits_kmv[test_mask].argmax(dim=1).numpy()
        log.info("  KMV  edges=%d  t=%.2fs", kmv_edges, time.perf_counter() - t0)
        if _prev_kmv_edges is not None and kmv_edges == _prev_kmv_edges:
            log.warning(
                "  [WARN] KMV edge count identical to previous seed (%d). "
                "C++ sketch may be ignoring the seed argument — check binary.",
                kmv_edges,
            )
        _prev_kmv_edges = kmv_edges
        del ei_kmv, logits_kmv; gc.collect()

        # MPRW
        t0 = time.perf_counter()
        mprw_kernel = MPRWKernel(k=best_w, seed=seed)
        g_mprw, _   = mprw_kernel.materialize(g_full, metapath_triples, target_ntype)
        mprw_edges  = int(g_mprw.edge_index.size(1))
        logits_mprw = _run_inference(g_mprw.edge_index, x_full, model, n_target, device)
        preds_mprw[seed] = logits_mprw[test_mask].argmax(dim=1).numpy()
        log.info("  MPRW edges=%d  t=%.2fs", mprw_edges, time.perf_counter() - t0)
        if _prev_mprw_edges is not None and mprw_edges == _prev_mprw_edges:
            log.warning(
                "  [WARN] MPRW edge count identical to previous seed (%d). "
                "Topology is likely saturated — all seeds are deterministic.",
                mprw_edges,
            )
        _prev_mprw_edges = mprw_edges
        del g_mprw, logits_mprw; gc.collect()

        np.savez(cache_file,
                 preds_kmv=preds_kmv, preds_mprw=preds_mprw,
                 done_seeds=seed + 1,
                 y_test=y_test, node_degrees=node_degrees)
        log.info("  Checkpoint -> %s", cache_file)

    return (y_test, node_degrees, preds_kmv, preds_mprw,
            info["num_classes"], mprw_saturated, mprw_plateau_edges, kmv_ref_edges)


# ===========================================================================
# 8.  Entry point
# ===========================================================================

def _setup_log(level=logging.INFO) -> logging.Logger:
    log = logging.getLogger("exp5")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(ch)
    return log


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use synthetic data — immediately runnable without C++ or data files.",
    )
    # Real-pipeline arguments
    parser.add_argument("--partition-json", default="results/HGB_DBLP/partition.json",
                        help="Path to partition.json from exp1_partition.py")
    parser.add_argument("--weights", default=None,
                        help="Path to frozen SAGE weights (.pt) from exp2_train.py")
    parser.add_argument("--metapath",
                        default="author_to_paper,paper_to_author",
                        help="Comma-separated metapath edge names")
    parser.add_argument("--depth", type=int, default=2,
                        help="SAGE depth (must match weights)")
    parser.add_argument("--k", type=int, default=32,
                        help="KMV sketch size k")
    parser.add_argument("--n-seeds", type=int, default=50,
                        help="Number of independent seeds (default 50)")
    # Output
    parser.add_argument("--out",
                        default="figures/exp5_degree_bin_eval.pdf",
                        help="Output path for the plot (PDF or PNG)")
    args = parser.parse_args()

    log = _setup_log()

    mprw_label   = "MPRW (+10% Edges)"   # default; overridden on saturation
    budget_note  = "MPRW budget = KMV edges x 1.10"

    if args.mock:
        log.info("=" * 60)
        log.info("exp5 -- MOCK MODE  (synthetic data, %d seeds)", args.n_seeds)
        log.info("=" * 60)
        y_true, node_degrees, preds_kmv, preds_mprw = _generate_mock_data(
            n_test=5_000, n_classes=4, n_seeds=args.n_seeds
        )
        n_classes = 4
        is_mock = True
    else:
        log.info("=" * 60)
        log.info("exp5 -- REAL MODE  (HGB_DBLP, 40/60 split, %d seeds)", args.n_seeds)
        log.info("=" * 60)
        if args.weights is None:
            parser.error("--weights is required in real mode")
        if not Path(args.partition_json).exists():
            parser.error(f"partition.json not found: {args.partition_json}")
        if not Path(args.weights).exists():
            parser.error(f"Weights file not found: {args.weights}")

        (y_true, node_degrees, preds_kmv, preds_mprw,
         n_classes, mprw_saturated, mprw_plateau_edges, kmv_ref_edges) = \
            _run_real_experiment(args, log)
        is_mock = False

        if mprw_saturated:
            mprw_label  = f"MPRW (topology-saturated, {mprw_plateau_edges:,} edges)"
            budget_note = (
                f"KMV={kmv_ref_edges:,} edges (sketch over-approx.)  |  "
                f"MPRW={mprw_plateau_edges:,} edges (topological ceiling)"
            )

    # ── Degree bin summary ─────────────────────────────────────────────────
    bins = _assign_bins(node_degrees)
    log.info("\nDegree bin sizes (V_test):")
    for bname, idx in bins.items():
        log.info("  %s  n=%d  deg_range=[%d, %d]",
                 bname.replace("\n", " "), len(idx),
                 node_degrees[idx].min(), node_degrees[idx].max())

    # ── Per-seed metrics ──────────────────────────────────────────────────
    log.info("\nComputing metrics across %d seeds...", args.n_seeds)
    metrics = _compute_per_seed_metrics(
        y_true, node_degrees, preds_kmv, preds_mprw, n_classes
    )

    # ── Print table ────────────────────────────────────────────────────────
    log.info("\n%s", "=" * 70)
    log.info("%-22s  %-6s  %-24s  %-24s",
             "Bin", "Metric", "KMV (mean +/- 95%CI)", "MPRW (mean +/- 95%CI)")
    log.info("-" * 70)
    for bname, method_data in metrics.items():
        for mkey, mlabel in [("pa", "Pred.Agree"), ("f1", "Macro-F1")]:
            mu_k, ci_k = _aggregate(method_data["kmv"][mkey])
            mu_m, ci_m = _aggregate(method_data["mprw"][mkey])
            log.info("%-22s  %-6s  %.4f +/- %.4f             %.4f +/- %.4f",
                     bname.replace("\n", " "), mlabel,
                     mu_k, ci_k, mu_m, ci_m)
    log.info("=" * 70)

    # ── Plot ───────────────────────────────────────────────────────────────
    _plot_results(metrics, args.out, args.n_seeds,
                  is_mock=is_mock, mprw_label=mprw_label, budget_note=budget_note)


if __name__ == "__main__":
    main()
