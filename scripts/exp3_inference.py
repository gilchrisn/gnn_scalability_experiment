"""
Experiment 3 — Inference sweep: Exact vs KMV across k values.

Reads partition.json (from exp1_partition.py) and frozen weights
(from exp2_train.py), then materializes the FULL graph with ExactD (once)
and with KMV sketch at each k, runs the frozen SAGE model, and evaluates
all metrics restricted to V_test nodes only.

The "test" set is derived from partition.json: all pivot nodes NOT in V_train.
For temporal mode this is all nodes with year > cutoff; for random_nodes mode
it is the (1 − train_frac) complement of the random permutation.

Output
------
  results/<dataset>/master_results.csv   (append)
      One row per (metapath, L, Method, k_value).
      Method="Exact"  — k-independent baseline (run once, written as k="")
      Method="KMV"    — k-specific sketch
      Method="MPRW"   — placeholder row (all metric cols empty)

CSV Schema
----------
  Dataset, MetaPath, L, Method, k_value,
  Materialization_Time, Inference_Time, Peak_RAM_MB, Edge_Count,
  CKA_L1, CKA_L2, CKA_L3, CKA_L4,
  Pred_Similarity, Macro_F1, exact_status

Usage
-----
    python scripts/exp3_inference.py HGB_DBLP \\
        --metapath author_to_paper,paper_to_author \\
        --depth 2 \\
        --weights-dir results/HGB_DBLP/weights \\
        --partition-json results/HGB_DBLP/partition.json

    python scripts/exp3_inference.py OGB_MAG \\
        --metapath paper_to_author,author_to_paper \\
        --depth 2 3 4 \\
        --k-values 8 16 32 \\
        --max-adj-mb 8000 --max-rss-gb 200 --timeout 3600
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.bridge.engine import CppEngine
from src.analysis.cka import LinearCKA
from src.analysis.metrics import DirichletEnergyMetric
from src.models import get_model
from src.kernels.mprw import MPRWKernel, parse_metapath_triples
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, setup_global_res_dirs


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return 0.0


def _rss_gb() -> Optional[float]:
    mb = _rss_mb()
    return mb / 1024 if mb else None


# ---------------------------------------------------------------------------
# V_test construction from partition.json
# ---------------------------------------------------------------------------

def _make_test_mask(g_full: HeteroData, part: dict) -> torch.Tensor:
    """
    Returns a boolean mask over g_full[target_type] where True = V_test.
    test_node_ids is read directly from partition.json (exp1_partition.py).
    """
    target_ntype = part["target_type"]   # exp1_partition.py key
    n_target     = g_full[target_ntype].num_nodes
    test_ids     = torch.tensor(part["test_node_ids"], dtype=torch.long)
    mask         = torch.zeros(n_target, dtype=torch.bool)
    mask[test_ids] = True
    return mask


# ---------------------------------------------------------------------------
# C++ materialization wrappers
# ---------------------------------------------------------------------------

def _run_exact(engine: CppEngine, folder: str, timeout: int) -> Tuple[float, str]:
    rule_file   = os.path.join(engine.data_dir, f"cod-rules_{folder}.limit")
    output_file = os.path.join(engine.data_dir, "mat_exact.adj")
    elapsed = engine.run_command("materialize", rule_file, output_file, timeout=timeout)
    return elapsed, output_file


def _run_sketch(engine: CppEngine, folder: str, k: int, kmv_seed: int,
                timeout: int) -> Tuple[float, str]:
    rule_file   = os.path.join(engine.data_dir, f"cod-rules_{folder}.limit")
    output_base = os.path.join(engine.data_dir, "mat_sketch")
    elapsed = engine.run_command("sketch", rule_file, output_base,
                                 k=k, seed=kmv_seed, timeout=timeout)
    return elapsed, output_base + "_0"


def _count_edges(filepath: str) -> int:
    n = 0
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                n += len(parts) - 1
    return n


def _load_adj(engine: CppEngine, filepath: str, num_nodes: int, node_offset: int,
              max_adj_mb: Optional[float] = None) -> Data:
    if max_adj_mb is not None and os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > max_adj_mb:
            raise MemoryError(
                f"Adjacency file {size_mb:.0f} MB exceeds limit {max_adj_mb:.0f} MB")
    return engine.load_result(filepath, num_nodes, node_offset)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer(model: torch.nn.Module, g_homo: Data, in_dim: int,
           device: torch.device) -> torch.Tensor:
    x          = F.pad(g_homo.x, (0, max(0, in_dim - g_homo.x.size(1)))).to(device)
    edge_index = g_homo.edge_index.to(device)
    with torch.no_grad():
        return model(x, edge_index)


def _infer_layerwise(model: torch.nn.Module, g_homo: Data, in_dim: int,
                     device: torch.device) -> List[torch.Tensor]:
    """Returns list of intermediate activations, one per layer."""
    x          = F.pad(g_homo.x, (0, max(0, in_dim - g_homo.x.size(1)))).to(device)
    edge_index = g_homo.edge_index.to(device)
    intermediates = []
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            x = layer(x, edge_index)
            if i < model.num_layers - 1:
                x = F.relu(x)
            intermediates.append(x.clone())
    return intermediates


def _f1_macro(logits: torch.Tensor, labels: torch.Tensor,
              mask: torch.Tensor) -> float:
    from torchmetrics.functional import f1_score
    valid = mask & (labels >= 0)
    if valid.sum() == 0:
        return 0.0
    return f1_score(
        logits[valid].argmax(1), labels[valid],
        task="multiclass", num_classes=logits.size(1), average="macro",
    ).item()


def _pred_agreement(z_exact: torch.Tensor, z_kmv: torch.Tensor,
                    mask: torch.Tensor) -> float:
    return (z_exact[mask].argmax(1) == z_kmv[mask].argmax(1)).float().mean().item()


_de_metric = DirichletEnergyMetric()


def _dirichlet_energy(z: torch.Tensor, edge_index: torch.Tensor,
                      num_nodes: int, device: torch.device) -> Optional[float]:
    """Dirichlet energy of embeddings z on the materialized graph."""
    try:
        return _de_metric.calculate(z.to(device), edge_index.to(device))
    except Exception:
        return None


_MPRW_CALIB_MAX_ITERS = 15
_MPRW_CALIB_TOL       = 0.05   # 5% edge-count tolerance


def _calibrate_mprw_w(
    g_full: "HeteroData",
    mprw_triples: list,
    target_ntype: str,
    target_edges: int,
    seed: int,
    device: torch.device,
    log: logging.Logger,
    tol: float = _MPRW_CALIB_TOL,
    w_max: int = 2000,
) -> Tuple[int, int, bool]:
    """
    Binary search for walk count w such that MPRW edge count ≈ target_edges ± tol.

    Termination guarantees (Gemini protocol):
      - Halts after MAX_ITERS=15 iterations regardless of convergence.
      - Halts early if |E_mprw - E_target| / E_target ≤ tol (5%).
      - Halts if w_hi ceiling is hit and MPRW still can't reach target density
        (topology exhausted — more walks only revisit the same hub nodes).
      - Always returns the best (w, count) seen, never oscillates indefinitely.

    Stochasticity note: E_mprw(w) is not perfectly monotone — random walk
    coverage has variance. Tracking best-so-far guards against bouncing.

    Returns:
        (calibrated_w, actual_edge_count, converged)
        converged=False means best-effort fallback was used.
    """
    w_lo, w_hi   = 1, w_max
    best_w       = w_lo
    best_count   = 0
    best_dist    = float("inf")
    ceiling_hit  = False

    for i in range(_MPRW_CALIB_MAX_ITERS):
        w_mid = (w_lo + w_hi) // 2
        kern  = MPRWKernel(k=w_mid, seed=seed, device=device)
        data, _ = kern.materialize(g_full, mprw_triples, target_ntype)
        count = data.edge_index.size(1)
        dist  = abs(count - target_edges)

        if dist < best_dist:
            best_w, best_count, best_dist = w_mid, count, dist

        log.debug("  [calib iter %d] w=%d  edges=%d  target=%d  dist=%d",
                  i + 1, w_mid, count, target_edges, dist)

        # Early exit: within tolerance
        if dist / max(target_edges, 1) <= tol:
            log.info("  [calib] converged at iter %d: w=%d  edges=%d  (%.1f%% of target)",
                     i + 1, w_mid, count, 100 * count / max(target_edges, 1))
            return w_mid, count, True

        if count > target_edges:
            w_hi = w_mid - 1
        else:
            # Check ceiling: if w_mid is already at w_max and still under target,
            # the graph topology is exhausted — more walks won't help.
            if w_mid >= w_max:
                ceiling_hit = True
                log.warning(
                    "  [calib] ceiling hit (w=%d): MPRW can only reach %d edges "
                    "(target=%d, gap=%.1f%%). Graph topology likely exhausted.",
                    w_mid, count, target_edges,
                    100 * dist / max(target_edges, 1),
                )
                break
            w_lo = w_mid + 1

        if w_lo > w_hi:
            break

    converged = best_dist / max(target_edges, 1) <= tol
    if not converged:
        log.warning(
            "  [calib] did not converge after %d iters%s. "
            "Using best-so-far: w=%d  edges=%d  (%.1f%% of target).",
            _MPRW_CALIB_MAX_ITERS,
            " (ceiling hit)" if ceiling_hit else "",
            best_w, best_count,
            100 * best_count / max(target_edges, 1),
        )
    return best_w, best_count, converged


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_FIELDS = [
    "Dataset", "MetaPath", "L", "Method", "k_value", "Density_Matched_w",
    "Materialization_Time", "Inference_Time", "Peak_RAM_MB", "Edge_Count",
    "CKA_L1", "CKA_L2", "CKA_L3", "CKA_L4",
    "Pred_Similarity", "Macro_F1", "Dirichlet_Energy", "exact_status",
]


def _open_csv(path: Path) -> Tuple:
    # If the file exists but was written with an old schema, back it up and
    # start fresh — new rows will be written with current _FIELDS.
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as _f:
            existing_fields = csv.DictReader(_f).fieldnames or []
        if set(existing_fields) != set(_FIELDS):
            backup = path.with_suffix(".csv.bak")
            path.rename(backup)
            logging.getLogger("exp3").warning(
                "CSV schema changed — old file backed up to %s", backup)
            is_new = True
        else:
            is_new = False
    else:
        is_new = not path.exists() or path.stat().st_size == 0

    fh = open(path, "a", newline="", encoding="utf-8")
    w  = csv.DictWriter(fh, fieldnames=_FIELDS)
    if is_new:
        w.writeheader()
    return fh, w


def _done_runs(path: Path) -> set:
    """Return set of (metapath, str(L), method, str(k)) already in CSV."""
    done = set()
    if not path.exists():
        return done
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            done.add((row.get("MetaPath", ""),
                      row.get("L", ""),
                      row.get("Method", ""),
                      row.get("k_value", "")))
    return done


def _fmt(val, digits: int = 6):
    return round(val, digits) if val is not None else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset")
    parser.add_argument("--metapath",       required=True)
    parser.add_argument("--depth",          type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--k-values",       type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--partition-json", required=True)
    parser.add_argument("--weights-dir",    type=str, default=None,
                        help="Directory with {mp_safe}_L{L}.pt files "
                             "(default: results/<dataset>/weights)")
    parser.add_argument("--output",         type=str, default=None,
                        help="Path to master_results.csv "
                             "(default: results/<dataset>/master_results.csv)")
    parser.add_argument("--max-adj-mb",     type=float, default=None)
    parser.add_argument("--max-rss-gb",     type=float, default=None)
    parser.add_argument("--timeout",        type=int,   default=600)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    out_dir     = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(args.weights_dir) if args.weights_dir else out_dir / "weights"
    csv_path    = Path(args.output) if args.output else out_dir / "master_results.csv"

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("exp3")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    fh_log = logging.FileHandler(out_dir / f"run_exp3_{ts}.log", encoding="utf-8")
    fh_log.setLevel(logging.DEBUG)
    fh_log.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(ch)
    log.addHandler(fh_log)

    log.info("Exp3 | dataset=%s  metapath=%s  depth=%s  k=%s",
             args.dataset, args.metapath, args.depth, args.k_values)

    # ------------------------------------------------------------------
    # Load partition + dataset
    # ------------------------------------------------------------------
    with open(args.partition_json) as f:
        part = json.load(f)

    cfg          = config.get_dataset_config(args.dataset)
    folder       = config.get_folder_name(args.dataset)
    data_dir     = os.path.join(project_root, folder)
    target_ntype = cfg.target_node

    assert part["target_type"] == target_ntype, (
        f"partition target_type '{part['target_type']}' != config target_node "
        f"'{target_ntype}'. Re-run exp1_partition.py."
    )

    g_full, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    num_classes  = info["num_classes"]
    labels_full  = info["labels"]    # [N_target]

    test_mask = _make_test_mask(g_full, part)   # True = V_test, shape [N_target]

    log.info("  V_test=%d / V_total=%d  (%.1f%%)",
             test_mask.sum().item(), len(test_mask),
             100 * test_mask.float().mean().item())

    # ------------------------------------------------------------------
    # Stage C++ files on the FULL graph (once)
    # ------------------------------------------------------------------
    setup_global_res_dirs(folder, project_root)
    engine = CppEngine(executable_path=config.CPP_EXECUTABLE, data_dir=data_dir)

    log.info("Staging full graph on disk...")
    PyGToCppAdapter(data_dir).convert(g_full)
    compile_rule_for_cpp(args.metapath, g_full, data_dir, folder)
    generate_qnodes(data_dir, folder, target_node_type=target_ntype, g_hetero=g_full)

    # Node offset for load_result
    s_ntypes    = sorted(g_full.node_types)
    node_offset = sum(g_full[nt].num_nodes for nt in s_ntypes if nt < target_ntype)
    n_target    = g_full[target_ntype].num_nodes
    x_full      = g_full[target_ntype].x     # feature matrix [N_target, D]
    in_dim      = x_full.size(1)

    kmv_seed    = part.get("hash_seed", 0)   # exp1_partition.py key
    device      = torch.device("cpu")   # inference always on CPU for fair timing
    cka_calc    = LinearCKA(device=device)

    # ------------------------------------------------------------------
    # CSV setup
    # ------------------------------------------------------------------
    csv_fh, csv_w = _open_csv(csv_path)
    done_runs     = _done_runs(csv_path)
    mp_safe       = args.metapath.replace(",", "_").replace("/", "_")

    # ------------------------------------------------------------------
    # Run ExactD once (k-independent)
    # ------------------------------------------------------------------
    z_exact_by_L: dict  = {}   # L → tensor
    layers_exact_by_L: dict = {}  # L → list[tensor]
    exact_edge_count   = None
    exact_status_flag  = "OK"
    t_exact_mat        = None

    log.info("\n--- Running ExactD on full graph ---")
    try:
        t0             = time.perf_counter()
        t_exact_mat, exact_file = _run_exact(engine, folder, args.timeout)
        exact_peak_mb = engine.last_peak_mb   # C++ child-process peak (Linux only)
        exact_edge_count = _count_edges(exact_file)
        log.info("  ExactD done: edges=%d  mat_time=%.2fs  peak_ram=%s",
                 exact_edge_count, t_exact_mat,
                 f"{exact_peak_mb:.0f}MB" if exact_peak_mb else "n/a")

        if args.max_rss_gb is not None:
            rss = _rss_gb()
            if rss is not None and rss > args.max_rss_gb:
                raise MemoryError(
                    f"RSS guard: {rss:.1f} GB > {args.max_rss_gb:.1f} GB before loading exact adj"
                )

        g_exact = _load_adj(engine, exact_file, n_target, node_offset, args.max_adj_mb)
        g_exact.x = x_full
        log.info("  Loaded exact adjacency  [RAM=%.0fMB]", _rss_mb())

        for L in args.depth:
            weights_path = weights_dir / f"{mp_safe}_L{L}.pt"
            if not weights_path.exists():
                log.warning("  [L=%d] weights not found at %s — skipping", L, weights_path)
                continue

            if (args.metapath, str(L), "Exact", "") in done_runs:
                log.info("  [L=%d] Exact row already in CSV — skipping", L)
                continue

            model = get_model("SAGE", in_dim, num_classes, config.HIDDEN_DIM,
                              num_layers=L).to(device)
            model.load_state_dict(torch.load(weights_path, weights_only=True,
                                             map_location=device))
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            t0_inf    = time.perf_counter()
            z_exact   = _infer(model, g_exact, in_dim, device)
            t_inf     = time.perf_counter() - t0_inf
            f1_exact  = _f1_macro(z_exact, labels_full.to(device), test_mask.to(device))

            # Layerwise for CKA computation against KMV later
            try:
                layers_exact = _infer_layerwise(model, g_exact, in_dim, device)
            except (MemoryError, RuntimeError):
                layers_exact = None
                log.warning("  [L=%d] layerwise OOM on exact", L)

            z_exact_by_L[L]      = z_exact.cpu()
            layers_exact_by_L[L] = [t.cpu() for t in layers_exact] if layers_exact else None

            de_exact = _dirichlet_energy(z_exact, g_exact.edge_index, n_target, device)

            # Write Exact row
            cka_cols = {f"CKA_L{i+1}": "" for i in range(4)}
            csv_w.writerow({
                "Dataset":              args.dataset,
                "MetaPath":             args.metapath,
                "L":                    L,
                "Method":               "Exact",
                "k_value":              "",
                "Density_Matched_w":    "",
                "Materialization_Time": _fmt(t_exact_mat),
                "Inference_Time":       _fmt(t_inf),
                "Peak_RAM_MB":          _fmt(exact_peak_mb, 1),
                "Edge_Count":           exact_edge_count,
                **cka_cols,
                "Pred_Similarity":      "",
                "Macro_F1":             _fmt(f1_exact),
                "Dirichlet_Energy":     _fmt(de_exact) if de_exact is not None else "",
                "exact_status":         "OK",
            })
            csv_fh.flush()
            log.info("  [L=%d] Exact  F1=%.4f  inf=%.2fs  peak_ram=%s",
                     L, f1_exact, t_inf,
                     f"{exact_peak_mb:.0f}MB" if exact_peak_mb else "n/a")

            del model
            gc.collect()

        del g_exact
        gc.collect()
        log.info("  Exact done for all depths.  [RAM=%.0fMB]", _rss_mb())

    except MemoryError as e:
        exact_status_flag = f"MAT_OOM"
        log.warning("  ExactD C++ OOM: %s", e)
    except RuntimeError as e:
        exact_status_flag = ("MAT_TIMEOUT" if "timed out" in str(e)
                             else f"MAT_ERR:{str(e)[:80]}")
        log.warning("  ExactD error: %s", e)

    # ------------------------------------------------------------------
    # KMV sweep over k values
    # ------------------------------------------------------------------
    kmv_edges_by_k: dict = {}   # k → edge count; used later for MPRW calibration
    for k in args.k_values:
        log.info("\n--- KMV k=%d ---", k)

        if args.max_rss_gb is not None:
            rss = _rss_gb()
            if rss is not None and rss > args.max_rss_gb:
                log.warning("  [k=%d] RSS guard: %.1f GB > %.1f GB — skipping",
                            k, rss, args.max_rss_gb)
                for L in args.depth:
                    csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                        "Dataset": args.dataset, "MetaPath": args.metapath,
                        "L": L, "Method": "KMV", "k_value": k,
                        "exact_status": f"RSS_OOM({rss:.0f}GB)",
                    }))
                csv_fh.flush()
                continue

        try:
            t_kmv_mat, kmv_file = _run_sketch(engine, folder, k, kmv_seed, args.timeout)
            kmv_peak_mb = engine.last_peak_mb   # C++ child-process peak (Linux only)
            kmv_edge_count = _count_edges(kmv_file)
            kmv_edges_by_k[k] = kmv_edge_count
            log.info("  KMV done: edges=%d  mat_time=%.2fs  peak_ram=%s",
                     kmv_edge_count, t_kmv_mat,
                     f"{kmv_peak_mb:.0f}MB" if kmv_peak_mb else "n/a")

            g_kmv = _load_adj(engine, kmv_file, n_target, node_offset, args.max_adj_mb)
            g_kmv.x = x_full

        except MemoryError as e:
            log.warning("  [k=%d] KMV OOM: %s", k, e)
            for L in args.depth:
                csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "KMV", "k_value": k,
                    "exact_status": "KMV_OOM",
                }))
            csv_fh.flush()
            continue
        except RuntimeError as e:
            log.warning("  [k=%d] KMV error: %s", k, e)
            for L in args.depth:
                csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "KMV", "k_value": k,
                    "exact_status": f"KMV_ERR:{str(e)[:60]}",
                }))
            csv_fh.flush()
            continue

        for L in args.depth:
            if (args.metapath, str(L), "KMV", str(k)) in done_runs:
                log.info("  [k=%d L=%d] already in CSV — skipping", k, L)
                continue

            weights_path = weights_dir / f"{mp_safe}_L{L}.pt"
            if not weights_path.exists():
                log.warning("  [k=%d L=%d] weights not found — skipping", k, L)
                continue

            model = get_model("SAGE", in_dim, num_classes, config.HIDDEN_DIM,
                              num_layers=L).to(device)
            model.load_state_dict(torch.load(weights_path, weights_only=True,
                                             map_location=device))
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            t0_inf = time.perf_counter()
            z_kmv  = _infer(model, g_kmv, in_dim, device)
            t_inf  = time.perf_counter() - t0_inf
            f1_kmv = _f1_macro(z_kmv, labels_full.to(device), test_mask.to(device))

            # CKA metrics against exact (if available)
            cka_cols    = {f"CKA_L{i+1}": "" for i in range(4)}
            pred_sim    = ""
            z_exact_cpu = z_exact_by_L.get(L)

            if z_exact_cpu is not None:
                mask_dev = test_mask.to(device)
                pred_sim = _fmt(_pred_agreement(
                    z_exact_cpu.to(device), z_kmv, mask_dev))

                # Layerwise CKA
                layers_exact = layers_exact_by_L.get(L)
                if layers_exact is not None:
                    try:
                        layers_kmv = _infer_layerwise(model, g_kmv, in_dim, device)
                        for i, (le, lk) in enumerate(zip(layers_exact, layers_kmv)):
                            if i >= 4:
                                break
                            val = cka_calc.calculate(
                                le.to(device)[mask_dev], lk[mask_dev])
                            cka_cols[f"CKA_L{i+1}"] = _fmt(val)
                    except (MemoryError, RuntimeError) as e:
                        log.warning("  [k=%d L=%d] layerwise CKA OOM: %s", k, L, e)

            de_kmv = _dirichlet_energy(z_kmv, g_kmv.edge_index, n_target, device)

            log.info("  [k=%d L=%d] KMV  F1=%.4f  DE=%.4f  inf=%.2fs  peak_ram=%s  pred_sim=%s",
                     k, L, f1_kmv, de_kmv or 0, t_inf,
                     f"{kmv_peak_mb:.0f}MB" if kmv_peak_mb else "n/a",
                     pred_sim if pred_sim != "" else "n/a")

            csv_w.writerow({
                "Dataset":              args.dataset,
                "MetaPath":             args.metapath,
                "L":                    L,
                "Method":               "KMV",
                "k_value":              k,
                "Density_Matched_w":    "",
                "Materialization_Time": _fmt(t_kmv_mat),
                "Inference_Time":       _fmt(t_inf),
                "Peak_RAM_MB":          _fmt(kmv_peak_mb, 1),
                "Edge_Count":           kmv_edge_count,
                **cka_cols,
                "Pred_Similarity":      pred_sim,
                "Macro_F1":             _fmt(f1_kmv),
                "Dirichlet_Energy":     _fmt(de_kmv) if de_kmv is not None else "",
                "exact_status":         exact_status_flag,
            })
            csv_fh.flush()

            del model
            gc.collect()

        del g_kmv
        gc.collect()

    # ------------------------------------------------------------------
    # MPRW sweep — subprocess-based so /usr/bin/time -v measures peak RSS
    # consistently with Exact/KMV C++ measurement.
    # ------------------------------------------------------------------

    # Parse metapath into triples once.
    try:
        mprw_triples = parse_metapath_triples(args.metapath, g_full)
    except (ValueError, RuntimeError) as e:
        log.warning("  MPRW: could not parse metapath — %s", e)
        mprw_triples = None

    # Serialize edge indices for each metapath step to .pt files (done once,
    # reused across all k values).  These are tiny — just int64 COO arrays —
    # so the subprocess never has to load the full HeteroData.
    mprw_work_dir = Path(data_dir) / "mprw_work"
    mprw_meta_file = mprw_work_dir / "meta.json"

    if mprw_triples is not None:
        mprw_work_dir.mkdir(parents=True, exist_ok=True)
        mprw_meta = {"n_target": n_target, "target_type": target_ntype, "steps": []}
        for i, (src_t, edge_name, dst_t) in enumerate(mprw_triples):
            edge_file = mprw_work_dir / f"edge_{i}.pt"
            torch.save(g_full[src_t, edge_name, dst_t].edge_index.cpu(), edge_file)
            mprw_meta["steps"].append({
                "src_type":  src_t,
                "edge_name": edge_name,
                "dst_type":  dst_t,
                "n_src":     g_full[src_t].num_nodes,
                "n_dst":     g_full[dst_t].num_nodes,
                "edge_file": str(edge_file),
            })
        with open(mprw_meta_file, "w") as f:
            json.dump(mprw_meta, f)
        log.info("\n--- MPRW sweep (subprocess) ---")

    # Reuse CppEngine's /usr/bin/time -v detection.
    time_bin = engine._time_binary()
    worker   = str(Path(project_root) / "scripts" / "mprw_worker.py")

    for k in args.k_values:
        log.info("\n--- MPRW k=%d (density-matched to KMV) ---", k)

        if mprw_triples is None:
            for L in args.depth:
                csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "MPRW", "k_value": k,
                    "exact_status": "MPRW_PARSE_ERR",
                }))
            csv_fh.flush()
            continue

        if args.max_rss_gb is not None:
            rss = _rss_gb()
            if rss is not None and rss > args.max_rss_gb:
                log.warning("  [MPRW k=%d] RSS guard: %.1f GB > %.1f GB — skipping",
                            k, rss, args.max_rss_gb)
                for L in args.depth:
                    csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                        "Dataset": args.dataset, "MetaPath": args.metapath,
                        "L": L, "Method": "MPRW", "k_value": k,
                        "exact_status": f"RSS_OOM({rss:.0f}GB)",
                    }))
                csv_fh.flush()
                continue

        # ------------------------------------------------------------------
        # Timer starts HERE — calibration time is part of materialization cost.
        # In a real system, an engineer using MPRW would have to run this same
        # search to hit a memory/density budget. KMV avoids this entirely.
        # ------------------------------------------------------------------
        t_mat_start = time.perf_counter()

        # Density calibration: binary-search for walk count w such that
        # MPRW global edge count ≈ KMV(k) global edge count (±5%).
        target_edges = kmv_edges_by_k.get(k)
        if target_edges is None:
            calibrated_w        = k
            calibrated_edge_count = None
            calib_converged     = False
            log.warning("  [MPRW k=%d] KMV edge count unavailable — using w=k", k)
        else:
            log.info("  [MPRW k=%d] calibrating w to match KMV edge count %d ...",
                     k, target_edges)
            calibrated_w, calibrated_edge_count, calib_converged = _calibrate_mprw_w(
                g_full, mprw_triples, target_ntype,
                target_edges=target_edges,
                seed=kmv_seed, device=device, log=log,
            )

        # ------------------------------------------------------------------
        # Subprocess call with calibrated_w — for fair peak-RSS measurement.
        # Time elapsed so far (calibration) is preserved in t_mat_start.
        # ------------------------------------------------------------------
        mprw_out  = mprw_work_dir / f"mat_mprw_{k}.pt"
        inner_cmd = [sys.executable, worker,
                     str(mprw_meta_file), str(mprw_out),
                     str(calibrated_w), str(kmv_seed)]
        cmd = ([time_bin, "-v"] + inner_cmd) if time_bin else inner_cmd

        log.info("  [MPRW] Subprocess (w=%d): %s", calibrated_w, " ".join(inner_cmd))

        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True,
                                 timeout=args.timeout)
        except subprocess.TimeoutExpired:
            log.warning("  [MPRW k=%d] subprocess timed out", k)
            for L in args.depth:
                csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "MPRW", "k_value": k,
                    "exact_status": "MPRW_TIMEOUT",
                }))
            csv_fh.flush()
            continue
        except subprocess.CalledProcessError as e:
            log.warning("  [MPRW k=%d] subprocess failed (exit %d):\n%s",
                        k, e.returncode, e.stderr[-400:])
            for L in args.depth:
                csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "MPRW", "k_value": k,
                    "exact_status": f"MPRW_ERR:{e.returncode}",
                }))
            csv_fh.flush()
            continue

        # Total materialization time = calibration + subprocess execution.
        t_mprw_mat = time.perf_counter() - t_mat_start

        # Parse net RSS from worker stdout (preferred: excludes Python/PyTorch
        # runtime overhead that inflates /usr/bin/time -v by ~300 MB and has no
        # equivalent in the C++ Exact/KMV subprocesses).
        # Falls back to /usr/bin/time -v peak if net_ram_mb line is absent.
        mprw_peak_mb: Optional[float] = None
        for line in res.stdout.split("\n"):
            if line.strip().lower().startswith("net_ram_mb:"):
                try:
                    mprw_peak_mb = float(line.split(":")[1].strip())
                except ValueError:
                    pass
                break
        if mprw_peak_mb is None:
            m = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", res.stderr)
            if m:
                mprw_peak_mb = int(m.group(1)) / 1024.0

        if not mprw_out.exists():
            log.warning("  [MPRW k=%d] output file missing", k)
            continue

        mprw_ei         = torch.load(mprw_out, weights_only=True)
        mprw_edge_count = mprw_ei.size(1)

        log.info("  MPRW done: edges=%d (calibrated_w=%d)  mat_time=%.2fs  peak_ram=%s",
                 mprw_edge_count, calibrated_w, t_mprw_mat,
                 f"{mprw_peak_mb:.0f}MB" if mprw_peak_mb else "n/a")

        from torch_geometric.data import Data as _Data
        g_mprw_data = _Data(edge_index=mprw_ei, num_nodes=n_target)
        g_mprw_data.x = x_full

        for L in args.depth:
            if (args.metapath, str(L), "MPRW", str(k)) in done_runs:
                log.info("  [MPRW k=%d L=%d] already in CSV — skipping", k, L)
                continue

            weights_path = weights_dir / f"{mp_safe}_L{L}.pt"
            if not weights_path.exists():
                log.warning("  [MPRW k=%d L=%d] weights not found — skipping", k, L)
                continue

            model = get_model("SAGE", in_dim, num_classes, config.HIDDEN_DIM,
                              num_layers=L).to(device)
            model.load_state_dict(torch.load(weights_path, weights_only=True,
                                             map_location=device))
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            t0_inf  = time.perf_counter()
            z_mprw  = _infer(model, g_mprw_data, in_dim, device)
            t_inf   = time.perf_counter() - t0_inf
            f1_mprw = _f1_macro(z_mprw, labels_full.to(device), test_mask.to(device))

            de_mprw  = _dirichlet_energy(z_mprw, mprw_ei, n_target, device)
            cka_cols = {f"CKA_L{i+1}": "" for i in range(4)}
            pred_sim = ""
            z_exact_cpu = z_exact_by_L.get(L)

            if z_exact_cpu is not None:
                mask_dev = test_mask.to(device)
                pred_sim = _fmt(_pred_agreement(
                    z_exact_cpu.to(device), z_mprw, mask_dev))

                layers_exact = layers_exact_by_L.get(L)
                if layers_exact is not None:
                    try:
                        layers_mprw = _infer_layerwise(model, g_mprw_data, in_dim, device)
                        for i, (le, lm) in enumerate(zip(layers_exact, layers_mprw)):
                            if i >= 4:
                                break
                            val = cka_calc.calculate(
                                le.to(device)[mask_dev], lm[mask_dev])
                            cka_cols[f"CKA_L{i+1}"] = _fmt(val)
                    except (MemoryError, RuntimeError) as e:
                        log.warning("  [MPRW k=%d L=%d] layerwise CKA OOM: %s", k, L, e)

            log.info("  [MPRW k=%d L=%d] F1=%.4f  DE=%.4f  inf=%.2fs  pred_sim=%s",
                     k, L, f1_mprw, de_mprw or 0, t_inf,
                     pred_sim if pred_sim != "" else "n/a")

            csv_w.writerow({
                "Dataset":              args.dataset,
                "MetaPath":             args.metapath,
                "L":                    L,
                "Method":               "MPRW",
                "k_value":              k,
                "Density_Matched_w":    calibrated_w,
                "Materialization_Time": _fmt(t_mprw_mat),
                "Inference_Time":       _fmt(t_inf),
                "Peak_RAM_MB":          _fmt(mprw_peak_mb, 1) if mprw_peak_mb else "",
                "Edge_Count":           mprw_edge_count,
                **cka_cols,
                "Pred_Similarity":      pred_sim,
                "Macro_F1":             _fmt(f1_mprw),
                "Dirichlet_Energy":     _fmt(de_mprw) if de_mprw is not None else "",
                "exact_status":         (exact_status_flag if calib_converged
                                         else f"MPRW_CALIB_APPROX|{exact_status_flag}"),
            })
            csv_fh.flush()

            del model
            gc.collect()

        del g_mprw_data, mprw_ei
        gc.collect()

    csv_fh.close()

    log.info("\nDone. Results → %s", csv_path)


if __name__ == "__main__":
    main()
