"""
Experiment LP-Inference — Score link prediction on full H via frozen encoder,
sweeping Exact / KMV (k-sweep) / MPRW (w-sweep) materialization.

Stage II of the LP pipeline. Reuses exp3_inference.py's C++ materialization
helpers + RSS / time measurement pattern.

For each method:
  1. Materialize full H via method (Exact / KMV / MPRW)
  2. Forward pass SAGE(frozen) → Z ∈ R^{|V_0| x emb_dim}
  3. For each test positive edge (u, v), sample 50 2-hop negatives in H_exact
  4. Score positive + negatives via dot product; compute MRR, Hits@1/10, AP, ROC-AUC
  5. Stratify by source-node degree bin (Tail / Mid / Hub) — the key LP-specific analysis
  6. Append row to master_lp_results.csv

CSV Schema
----------
  Dataset, MetaPath, L, Method, k_value, w_value, Seed,
  Materialization_Time, Inference_Time, Mat_RAM_MB, Inf_RAM_MB,
  Edge_Count, Graph_Density,
  MRR, ROC_AUC, Hits_1, Hits_10, AP, Recall_10,
  MRR_tail, MRR_mid, MRR_hub,
  ROC_AUC_tail, ROC_AUC_mid, ROC_AUC_hub,
  exact_status

Usage
-----
    python scripts/exp_lp_inference.py HGB_DBLP \\
        --metapath author_to_paper,paper_to_author \\
        --depth 2 \\
        --k-values 8 32 \\
        --w-values 8 32 128 \\
        --run-id 0 \\
        --partition-json results/HGB_DBLP/partition.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.bridge.engine import CppEngine
from src.models import get_model
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, setup_global_res_dirs

# Reuse exp3_inference helpers
from scripts.exp3_inference import (
    _rss_mb, _PeakRSSMonitor,
    _run_mprw_exec, _run_exact, _run_sketch, _count_edges, _graph_density, _load_adj,
)
from scripts.exp_lp_train import _build_adj_dict, _sample_neg_2hop


# ---------------------------------------------------------------------------
# LP metric computation
# ---------------------------------------------------------------------------

def _compute_lp_metrics(
    z: torch.Tensor,
    pos_edges: torch.Tensor,        # [2, E] positive test edges
    adj_exact: list,                # 1-hop adjacency in H_exact (for 2-hop neg sampling)
    rng: np.random.Generator,
    n_negs_per_pos: int = 50,
    degree_bins: Optional[Dict[int, str]] = None,
) -> dict:
    """Compute MRR, Hits@1, Hits@10, ROC-AUC, AP, Recall@10.

    For each positive (u, v), sample n_negs 2-hop negatives in H_exact.
    Score all (n_negs + 1) candidates via dot(z_u, z_v'). Rank.

    If degree_bins is provided (maps source_node_id → 'tail'/'mid'/'hub'), also returns
    per-bin MRR and ROC-AUC for the key stratification analysis.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    device = z.device
    n_pos = pos_edges.size(1)
    if n_pos == 0:
        return {"n_pos": 0}

    z = z.detach()
    ranks = []                   # rank of positive among candidates (1 = best)
    all_pos_scores = []          # for ROC-AUC
    all_neg_scores = []

    per_bin_ranks: Dict[str, List[int]] = {"tail": [], "mid": [], "hub": []}
    per_bin_pos:   Dict[str, List[float]] = {"tail": [], "mid": [], "hub": []}
    per_bin_neg:   Dict[str, List[float]] = {"tail": [], "mid": [], "hub": []}

    n_nodes = z.size(0)

    # Dense-graph fast path: on dense graphs, 2-hop set enumeration is expensive.
    # Rejection-sample from V \ N1(u) \ {u} instead (on dense graphs, N^2(u) ⊇ V\N1)
    mean_deg = np.mean([len(a) for a in adj_exact])
    density = mean_deg / max(n_nodes, 1)
    dense_mode = density > 0.3
    saturated_mode = density > 0.8   # H near-complete → fall back to uniform random

    for i in range(n_pos):
        u = int(pos_edges[0, i])
        v = int(pos_edges[1, i])

        n1 = adj_exact[u]

        if saturated_mode:
            # Near-complete graph: 2-hop negatives essentially don't exist.
            # Fall back to uniform random (will be noted in paper as saturation fallback).
            n2_list = []
            while len(n2_list) < n_negs_per_pos:
                v_rand = int(rng.integers(n_nodes))
                if v_rand != u and v_rand != v:
                    n2_list.append(v_rand)
        elif dense_mode:
            # Rejection-sample candidates from V \ N1
            n2_list = []
            tries = 0
            max_tries = n_negs_per_pos * 20
            while len(n2_list) < n_negs_per_pos and tries < max_tries:
                v_rand = int(rng.integers(n_nodes))
                if v_rand != u and v_rand != v and v_rand not in n1:
                    n2_list.append(v_rand)
                tries += 1
            # If stuck, fall back to any v'
            while len(n2_list) < n_negs_per_pos:
                v_rand = int(rng.integers(n_nodes))
                if v_rand != u and v_rand != v:
                    n2_list.append(v_rand)
        else:
            n2 = set()
            for w in n1:
                n2 |= adj_exact[w]
            n2 -= n1
            n2.discard(u)

            if len(n2) < n_negs_per_pos:
                while len(n2) < n_negs_per_pos:
                    v_rand = int(rng.integers(n_nodes))
                    if v_rand != u and v_rand not in n1 and v_rand != v:
                        n2.add(v_rand)

            n2.discard(v)
            n2_list = list(n2)
            if len(n2_list) > n_negs_per_pos:
                n2_list = rng.choice(n2_list, size=n_negs_per_pos, replace=False).tolist()

        if len(n2_list) == 0:
            continue

        # Score positive + negatives
        cand_ids = torch.tensor([v] + n2_list, dtype=torch.long, device=device)
        scores = (z[u] * z[cand_ids]).sum(-1)  # [1 + n_negs]
        s_pos = float(scores[0].item())
        s_neg = scores[1:].cpu().numpy()

        # Rank = 1 + number of negatives >= positive score
        rank = 1 + int((s_neg >= s_pos).sum())
        ranks.append(rank)
        all_pos_scores.append(s_pos)
        all_neg_scores.extend(s_neg.tolist())

        if degree_bins is not None:
            b = degree_bins.get(u, "mid")
            per_bin_ranks[b].append(rank)
            per_bin_pos[b].append(s_pos)
            per_bin_neg[b].extend(s_neg.tolist())

    if not ranks:
        return {"n_pos": 0}

    ranks = np.array(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks))
    hits_1  = float(np.mean(ranks <= 1))
    hits_10 = float(np.mean(ranks <= 10))
    recall_10 = hits_10  # same quantity for K=1 positive / query

    labels_auc = np.concatenate([np.ones(len(all_pos_scores)),
                                 np.zeros(len(all_neg_scores))])
    scores_auc = np.concatenate([np.array(all_pos_scores),
                                 np.array(all_neg_scores)])
    try:
        roc_auc = float(roc_auc_score(labels_auc, scores_auc))
    except ValueError:
        roc_auc = float("nan")
    try:
        ap = float(average_precision_score(labels_auc, scores_auc))
    except ValueError:
        ap = float("nan")

    out = {
        "n_pos":    int(len(ranks)),
        "MRR":      mrr,
        "ROC_AUC":  roc_auc,
        "Hits_1":   hits_1,
        "Hits_10":  hits_10,
        "AP":       ap,
        "Recall_10": recall_10,
    }

    for b in ("tail", "mid", "hub"):
        rs = per_bin_ranks[b]
        if rs:
            out[f"MRR_{b}"] = float(np.mean(1.0 / np.array(rs, dtype=np.float64)))
            try:
                lbl_b = np.concatenate([np.ones(len(per_bin_pos[b])),
                                        np.zeros(len(per_bin_neg[b]))])
                scr_b = np.concatenate([np.array(per_bin_pos[b]),
                                        np.array(per_bin_neg[b])])
                out[f"ROC_AUC_{b}"] = float(roc_auc_score(lbl_b, scr_b))
            except ValueError:
                out[f"ROC_AUC_{b}"] = float("nan")
        else:
            out[f"MRR_{b}"] = float("nan")
            out[f"ROC_AUC_{b}"] = float("nan")

    return out


# ---------------------------------------------------------------------------
# Degree binning (Tail / Mid / Hub)
# ---------------------------------------------------------------------------

def _degree_bins(adj: list) -> Dict[int, str]:
    """Map node_id → 'tail' (0-50th pctile), 'mid' (50-90), 'hub' (top 10%)."""
    degs = np.array([len(a) for a in adj], dtype=np.float64)
    p50 = np.percentile(degs, 50)
    p90 = np.percentile(degs, 90)
    bins = {}
    for i, d in enumerate(degs):
        if d <= p50:
            bins[i] = "tail"
        elif d <= p90:
            bins[i] = "mid"
        else:
            bins[i] = "hub"
    return bins


# ---------------------------------------------------------------------------
# Test edge selection
# ---------------------------------------------------------------------------

def _select_test_edges(
    edge_index_full: torch.Tensor,
    v_test: torch.Tensor,
    max_edges: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Pick test edges: edges in full H involving at least one V_test node.

    Returns upper-triangular positives deduped, sampled to max_edges.
    """
    n_test = v_test.size(0)
    test_mask = torch.zeros(edge_index_full.max().item() + 1, dtype=torch.bool)
    test_mask[v_test] = True

    ei = edge_index_full
    # Upper-triangular for undirected dedup
    keep = (ei[0] < ei[1]) & (test_mask[ei[0]] | test_mask[ei[1]])
    pos = ei[:, keep]
    n_pos = pos.size(1)
    if n_pos == 0:
        return pos
    if n_pos > max_edges:
        perm = rng.permutation(n_pos)[:max_edges]
        pos = pos[:, torch.from_numpy(perm).long()]
    return pos


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference_and_score(
    g_homo: Data,
    model: torch.nn.Module,
    pos_test: torch.Tensor,
    adj_exact: list,
    degree_bins: Dict[int, str],
    rng: np.random.Generator,
    device: torch.device,
    log: logging.Logger,
    n_negs_per_pos: int = 50,
) -> Tuple[dict, float, float]:
    """Run inference on g_homo with model, score test edges, return (metrics, inf_time, peak_ram_mb)."""
    model.eval()

    def _forward_on(dev: torch.device):
        m = model.to(dev)
        x_d = g_homo.x.to(dev)
        ei_d = g_homo.edge_index.to(dev)
        with torch.no_grad():
            return m(x_d, ei_d)

    import tracemalloc
    tracemalloc.start()
    tracemalloc.clear_traces()

    t0 = time.perf_counter()
    try:
        z = _forward_on(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if device.type == "cuda" and ("out of memory" in str(e).lower() or
                                       isinstance(e, torch.cuda.OutOfMemoryError)):
            log.warning("    GPU OOM — falling back to CPU for this forward pass")
            torch.cuda.empty_cache()
            z = _forward_on(torch.device("cpu"))
        else:
            raise
    inf_time = time.perf_counter() - t0

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_bytes / (1024 * 1024)

    # Move z to CPU for metric computation (avoids GPU memory during bin iteration)
    z = z.detach().cpu()
    metrics = _compute_lp_metrics(
        z, pos_test, adj_exact, rng,
        n_negs_per_pos=n_negs_per_pos, degree_bins=degree_bins,
    )
    log.info("    inf_time=%.3fs peak_ram=%.1fMB MRR=%.4f AUC=%.4f Hits@10=%.4f",
             inf_time, peak_ram_mb,
             metrics.get("MRR", float("nan")),
             metrics.get("ROC_AUC", float("nan")),
             metrics.get("Hits_10", float("nan")))

    return metrics, inf_time, peak_ram_mb


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

_FIELDS = [
    "Dataset", "MetaPath", "L", "Method", "k_value", "w_value", "Seed",
    "Materialization_Time", "Inference_Time", "Mat_RAM_MB", "Inf_RAM_MB",
    "Edge_Count", "Graph_Density",
    "MRR", "ROC_AUC", "Hits_1", "Hits_10", "AP", "Recall_10",
    "MRR_tail", "MRR_mid", "MRR_hub",
    "ROC_AUC_tail", "ROC_AUC_mid", "ROC_AUC_hub",
    "n_test_pos", "exact_status",
]


def _open_csv(path: Path):
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    w  = csv.DictWriter(fh, fieldnames=_FIELDS)
    if is_new:
        w.writeheader()
    return fh, w


def _done_runs(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["MetaPath"], row["L"], row["Method"],
                   row.get("k_value", ""), row.get("w_value", ""),
                   row.get("Seed", ""))
            done.add(key)
    return done


def _write_row(w, d: dict):
    # Fill missing
    for k in _FIELDS:
        if k not in d:
            d[k] = ""
    w.writerow(d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset")
    parser.add_argument("--metapath", required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--partition-json", required=True)
    parser.add_argument("--weights-dir", default=None,
                        help="Default: results/<ds>/weights_lp/")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[8, 32])
    parser.add_argument("--w-values", type=int, nargs="+",
                        default=[8, 32, 128])
    parser.add_argument("--run-id", type=int, default=0,
                        help="Seed id for this run (0-indexed)")
    parser.add_argument("--hash-seed", type=int, default=None,
                        help="KMV/MPRW hash seed; default = run-id")
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--skip-kmv", action="store_true")
    parser.add_argument("--skip-mprw", action="store_true")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU inference (avoid GPU OOM on large graphs)")
    parser.add_argument("--max-test-edges", type=int, default=2000)
    parser.add_argument("--n-negs-per-pos", type=int, default=50)
    parser.add_argument("--mprw-bin", default=str(Path(project_root) / "bin" / "mprw_exec"))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    if args.hash_seed is None:
        args.hash_seed = args.run_id

    with open(args.partition_json) as f:
        part = json.load(f)

    cfg      = config.get_dataset_config(args.dataset)
    folder   = config.get_folder_name(args.dataset)
    out_dir  = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(args.weights_dir) if args.weights_dir else out_dir / "weights_lp"
    if not weights_dir.exists():
        raise SystemExit(f"Weights dir not found: {weights_dir}. Run exp_lp_train.py first.")

    mp_safe = args.metapath.replace(",", "_").replace("/", "_")
    weights_path = weights_dir / f"{mp_safe}_L{args.depth}.pt"
    if not weights_path.exists():
        raise SystemExit(f"Frozen weights not found: {weights_path}")

    csv_path = out_dir / "master_lp_results.csv"

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("exp_lp_inf")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    fh = logging.FileHandler(out_dir / f"run_lp_inf_{ts}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(ch)
    log.addHandler(fh)

    log.info("LP-Inf | dataset=%s metapath=%s L=%d run_id=%d hash_seed=%d",
             args.dataset, args.metapath, args.depth, args.run_id, args.hash_seed)

    random.seed(args.run_id)
    np.random.seed(args.run_id)
    torch.manual_seed(args.run_id)
    rng = np.random.default_rng(args.run_id)

    g_full, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    target_ntype = cfg.target_node
    n_target     = g_full[target_ntype].num_nodes
    in_dim       = g_full[target_ntype].x.size(1)
    x_full       = g_full[target_ntype].x
    v_test       = torch.tensor(part["test_node_ids"], dtype=torch.long)

    # Stage C++ on FULL graph
    data_dir = config.get_staging_dir(args.dataset)
    os.makedirs(data_dir, exist_ok=True)
    setup_global_res_dirs(folder, project_root)
    engine = CppEngine(executable_path=config.CPP_EXECUTABLE, data_dir=data_dir)

    PyGToCppAdapter(data_dir).convert(g_full)
    compile_rule_for_cpp(args.metapath, g_full, data_dir, folder)
    generate_qnodes(data_dir, folder, target_node_type=target_ntype, g_hetero=g_full)

    s_ntypes = sorted(g_full.node_types)
    offset   = sum(g_full[nt].num_nodes for nt in s_ntypes if nt < target_ntype)
    del g_full
    gc.collect()

    # Materialize Exact once to get ground-truth H structure for 2-hop sampling & test edges
    log.info("Running ExactD on FULL graph...")
    rule_file   = os.path.join(data_dir, f"cod-rules_{folder}.limit")
    exact_adj   = os.path.join(data_dir, "mat_exact.adj")
    with _PeakRSSMonitor() as mon_ex:
        mat_time_ex = engine.run_command("materialize", rule_file, exact_adj, timeout=args.timeout)
    mat_ram_ex = mon_ex.peak_delta_mb
    g_exact = engine.load_result(exact_adj, n_target, offset)
    g_exact.x = x_full
    n_edges_ex = g_exact.edge_index.size(1) if g_exact.edge_index is not None else 0
    log.info("  Exact: edges=%d mat_time=%.2fs peak_ram=%.1fMB", n_edges_ex, mat_time_ex, mat_ram_ex)

    adj_exact = _build_adj_dict(g_exact.edge_index, n_target)
    degree_bins = _degree_bins(adj_exact)

    pos_test = _select_test_edges(g_exact.edge_index, v_test, args.max_test_edges, rng)
    log.info("  Test edges (upper-tri, V_test-involving): %d", pos_test.size(1))

    # Load frozen model — force CPU for large graphs (GPU buffer scales with |E|*in_dim)
    force_cpu = (n_edges_ex > 1_000_000) or (in_dim > 2000 and n_target > 2000) \
                or args.force_cpu
    if force_cpu:
        log.info("  Using CPU (graph too large for GPU: |E|=%d in_dim=%d)",
                 n_edges_ex, in_dim)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("SAGE", in_dim, args.embedding_dim, config.HIDDEN_DIM,
                      num_layers=args.depth).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    log.info("  Loaded frozen encoder from %s (device=%s)", weights_path, device)

    done_keys = _done_runs(csv_path)
    csv_fh, csv_w = _open_csv(csv_path)

    try:
        # ------------------- Exact -------------------
        if not args.skip_exact:
            key = (args.metapath, str(args.depth), "Exact", "", "", "0")
            if key in done_keys:
                log.info("[Exact] already done — skipping")
            else:
                log.info("\n[Exact] scoring LP...")
                try:
                    m, inf_time, inf_ram = _run_inference_and_score(
                        g_exact, model, pos_test, adj_exact, degree_bins, rng, device, log,
                        n_negs_per_pos=args.n_negs_per_pos,
                    )
                    row = {
                        "Dataset": args.dataset, "MetaPath": args.metapath, "L": args.depth,
                        "Method": "Exact", "k_value": "", "w_value": "", "Seed": 0,
                        "Materialization_Time": round(mat_time_ex, 4),
                        "Inference_Time": round(inf_time, 4),
                        "Mat_RAM_MB": round(mat_ram_ex, 2),
                        "Inf_RAM_MB": round(inf_ram, 2),
                        "Edge_Count": n_edges_ex,
                        "Graph_Density": _graph_density(n_edges_ex, n_target),
                        "n_test_pos": m.get("n_pos", 0),
                        "exact_status": "OK",
                        **{k: round(v, 6) for k, v in m.items()
                           if k in _FIELDS and isinstance(v, float)},
                    }
                    _write_row(csv_w, row); csv_fh.flush()
                except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
                    err = str(e)[:80]
                    log.warning("[Exact] OOM / error on inference: %s", err)
                    row = {
                        "Dataset": args.dataset, "MetaPath": args.metapath, "L": args.depth,
                        "Method": "Exact", "k_value": "", "w_value": "", "Seed": 0,
                        "Materialization_Time": round(mat_time_ex, 4),
                        "Mat_RAM_MB": round(mat_ram_ex, 2),
                        "Edge_Count": n_edges_ex,
                        "Graph_Density": _graph_density(n_edges_ex, n_target),
                        "exact_status": f"INF_OOM({err})",
                    }
                    _write_row(csv_w, row); csv_fh.flush()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        # ------------------- KMV k-sweep -------------------
        if not args.skip_kmv:
            for k in args.k_values:
                key = (args.metapath, str(args.depth), "KMV", str(k), "", str(args.run_id))
                if key in done_keys:
                    log.info("[KMV k=%d seed=%d] already done — skipping", k, args.run_id)
                    continue

                log.info("\n[KMV k=%d seed=%d] materializing...", k, args.run_id)
                with _PeakRSSMonitor() as mon:
                    mat_time, sketch_file = _run_sketch(engine, folder, k, args.hash_seed,
                                                        timeout=args.timeout)
                mat_ram = mon.peak_delta_mb
                g_kmv = _load_adj(engine, sketch_file, n_target, offset)
                g_kmv.x = x_full
                n_edges = g_kmv.edge_index.size(1) if g_kmv.edge_index is not None else 0
                log.info("  KMV edges=%d mat_time=%.2fs peak_ram=%.1fMB",
                         n_edges, mat_time, mat_ram)

                m, inf_time, inf_ram = _run_inference_and_score(
                    g_kmv, model, pos_test, adj_exact, degree_bins, rng, device, log,
                    n_negs_per_pos=args.n_negs_per_pos,
                )
                row = {
                    "Dataset": args.dataset, "MetaPath": args.metapath, "L": args.depth,
                    "Method": "KMV", "k_value": k, "w_value": "", "Seed": args.run_id,
                    "Materialization_Time": round(mat_time, 4),
                    "Inference_Time": round(inf_time, 4),
                    "Mat_RAM_MB": round(mat_ram, 2),
                    "Inf_RAM_MB": round(inf_ram, 2),
                    "Edge_Count": n_edges,
                    "Graph_Density": _graph_density(n_edges, n_target),
                    "n_test_pos": m.get("n_pos", 0),
                    "exact_status": "OK",
                    **{k2: round(v, 6) for k2, v in m.items()
                       if k2 in _FIELDS and isinstance(v, float)},
                }
                _write_row(csv_w, row); csv_fh.flush()
                del g_kmv
                gc.collect()

        # ------------------- MPRW w-sweep -------------------
        if not args.skip_mprw:
            for w in args.w_values:
                key = (args.metapath, str(args.depth), "MPRW", "", str(w), str(args.run_id))
                if key in done_keys:
                    log.info("[MPRW w=%d seed=%d] already done — skipping", w, args.run_id)
                    continue

                log.info("\n[MPRW w=%d seed=%d] materializing...", w, args.run_id)
                mprw_out = os.path.join(data_dir, f"mat_mprw_w{w}.adj")
                try:
                    mat_time, mat_ram = _run_mprw_exec(
                        args.mprw_bin, data_dir, rule_file, mprw_out,
                        w=w, seed=args.hash_seed, timeout=args.timeout,
                    )
                except RuntimeError as e:
                    log.warning("  MPRW failed: %s", e)
                    row = {
                        "Dataset": args.dataset, "MetaPath": args.metapath, "L": args.depth,
                        "Method": "MPRW", "k_value": "", "w_value": w, "Seed": args.run_id,
                        "exact_status": f"MAT_FAIL({str(e)[:50]})",
                    }
                    _write_row(csv_w, row); csv_fh.flush()
                    continue

                g_mprw = _load_adj(engine, mprw_out, n_target, offset)
                g_mprw.x = x_full
                n_edges = g_mprw.edge_index.size(1) if g_mprw.edge_index is not None else 0
                log.info("  MPRW edges=%d mat_time=%.2fs peak_ram=%.1fMB",
                         n_edges, mat_time, mat_ram)

                m, inf_time, inf_ram = _run_inference_and_score(
                    g_mprw, model, pos_test, adj_exact, degree_bins, rng, device, log,
                    n_negs_per_pos=args.n_negs_per_pos,
                )
                row = {
                    "Dataset": args.dataset, "MetaPath": args.metapath, "L": args.depth,
                    "Method": "MPRW", "k_value": "", "w_value": w, "Seed": args.run_id,
                    "Materialization_Time": round(mat_time, 4),
                    "Inference_Time": round(inf_time, 4),
                    "Mat_RAM_MB": round(mat_ram, 2),
                    "Inf_RAM_MB": round(inf_ram, 2),
                    "Edge_Count": n_edges,
                    "Graph_Density": _graph_density(n_edges, n_target),
                    "n_test_pos": m.get("n_pos", 0),
                    "exact_status": "OK",
                    **{k2: round(v, 6) for k2, v in m.items()
                       if k2 in _FIELDS and isinstance(v, float)},
                }
                _write_row(csv_w, row); csv_fh.flush()
                del g_mprw
                gc.collect()

    finally:
        csv_fh.close()

    log.info("\nDone. Results → %s", csv_path)


if __name__ == "__main__":
    main()
