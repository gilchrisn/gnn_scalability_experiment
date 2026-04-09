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
from src.kernels.mprw import MPRWKernel, parse_metapath_triples
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, setup_global_res_dirs


# ---------------------------------------------------------------------------
# RSS helpers (parent process — used only for the RSS guard, not for peaking)
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
    target_ntype = part["target_type"]
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


def _graph_density(edge_count: Optional[int], n_nodes: int) -> str:
    """Directed graph density: edges / (n * (n-1)). Returns '' if inputs are invalid."""
    if edge_count is None or n_nodes <= 1:
        return ""
    return f"{edge_count / (n_nodes * (n_nodes - 1)):.6e}"


def _load_adj(engine: CppEngine, filepath: str, num_nodes: int, node_offset: int,
              max_adj_mb: Optional[float] = None) -> Data:
    if max_adj_mb is not None and os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > max_adj_mb:
            raise MemoryError(
                f"Adjacency file {size_mb:.0f} MB exceeds limit {max_adj_mb:.0f} MB")
    return engine.load_result(filepath, num_nodes, node_offset)


# ---------------------------------------------------------------------------
# Subprocess inference helper
# ---------------------------------------------------------------------------

_INF_WORKER = str(Path(__file__).resolve().parent / "inference_worker.py")


def _run_inference_worker(
    graph_file: str,
    graph_type: str,      # "adj" or "pt"
    feat_file: str,
    weights_path: str,
    z_out: str,
    labels_file: str,
    mask_file: str,
    n_target: int,
    node_offset: int,
    in_dim: int,
    num_classes: int,
    num_layers: int,
    timeout: int,
    log: logging.Logger,
    label: str = "",
) -> Optional[dict]:
    """
    Runs inference_worker.py in a fresh subprocess.
    Returns dict with keys: inf_peak_ram_mb, inf_time, inf_f1, inf_de.
    Returns None on failure.
    """
    cmd = [
        sys.executable, _INF_WORKER,
        "--graph-file",  graph_file,
        "--graph-type",  graph_type,
        "--feat-file",   feat_file,
        "--weights",     weights_path,
        "--z-out",       z_out,
        "--labels-file", labels_file,
        "--mask-file",   mask_file,
        "--n-target",    str(n_target),
        "--node-offset", str(node_offset),
        "--in-dim",      str(in_dim),
        "--num-classes", str(num_classes),
        "--num-layers",  str(num_layers),
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("  [inference %s] subprocess timed out", label)
        return {"inf_failed": True, "inf_timeout": True}
    except subprocess.CalledProcessError as e:
        # Python subprocess reports signal kills as negative signal numbers:
        # SIGKILL (9) → returncode = -9.
        # The shell represents this as 128+9=137, but Python gives -9 directly.
        oom = e.returncode in (-9, 137)
        log.warning("  [inference %s] subprocess failed (exit %d%s):\n%s",
                    label, e.returncode, " — OOM killer (SIGKILL)" if oom else "",
                    e.stderr[-400:])
        return {"inf_failed": True, "inf_oom": oom}

    out: dict = {}
    for line in res.stdout.split("\n"):
        line = line.strip()
        for key in ("inf_peak_ram_mb", "inf_time", "inf_f1", "inf_de"):
            if line.lower().startswith(f"{key}:"):
                try:
                    out[key] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        # SparseTensor-fallback OOM guard: worker exits 0 but sets this flag
        if line.lower().startswith("inf_oom:"):
            try:
                if int(line.split(":", 1)[1].strip()) == 1:
                    out["inf_oom"] = True
            except ValueError:
                pass
    return out


def _pred_agreement(z_a: torch.Tensor, z_b: torch.Tensor,
                    mask: torch.Tensor) -> float:
    return (z_a[mask].argmax(1) == z_b[mask].argmax(1)).float().mean().item()



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
    "Materialization_Time", "Inference_Time",
    "Mat_RAM_MB",   # subprocess-isolated materialization peak
    "Inf_RAM_MB",   # subprocess-isolated inference peak
    "Edge_Count", "Graph_Density",
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
    parser.add_argument("--max-adj-mb",      type=float, default=None)
    parser.add_argument("--max-rss-gb",      type=float, default=None)
    parser.add_argument("--timeout",         type=int,   default=600,
                        help="Timeout (s) for BOTH materialization and inference "
                             "subprocesses unless overridden by --inf-timeout")
    parser.add_argument("--inf-timeout",     type=int,   default=None,
                        help="Timeout (s) for inference subprocesses only "
                             "(default: same as --timeout)")
    args = parser.parse_args()
    # Resolve inference timeout: explicit --inf-timeout overrides --timeout
    args.inf_timeout = args.inf_timeout if args.inf_timeout is not None else args.timeout

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

    # Seed from partition.json so exp3 is bit-for-bit reproducible with exp2
    _master_seed = part.get("seed", 42)
    torch.manual_seed(_master_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    # Persist auxiliary tensors to disk once — inference_worker loads them.
    # These are written to a per-metapath scratch dir so the worker subprocess
    # starts with a completely fresh Python process (RSS = 0 base).
    # ------------------------------------------------------------------
    scratch_dir = out_dir / "inf_scratch" / mp_safe
    scratch_dir.mkdir(parents=True, exist_ok=True)
    feat_file   = str(scratch_dir / "x.pt")
    labels_file = str(scratch_dir / "labels.pt")
    mask_file   = str(scratch_dir / "mask.pt")
    torch.save(x_full.cpu(),           feat_file)
    torch.save(labels_full.cpu(),      labels_file)
    torch.save(test_mask.cpu(),        mask_file)

    # ------------------------------------------------------------------
    # Helper: run one inference subprocess, load z from disk for CKA.
    # Returns (inf_results_dict, z_path, layers_path_or_None).
    # ------------------------------------------------------------------
    def _inf_subprocess(graph_file, graph_type, label, L):
        z_out = str(scratch_dir / f"z_{label}_L{L}.pt")
        res = _run_inference_worker(
            graph_file=graph_file, graph_type=graph_type,
            feat_file=feat_file,
            weights_path=str(weights_dir / f"{mp_safe}_L{L}.pt"),
            z_out=z_out,
            labels_file=labels_file, mask_file=mask_file,
            n_target=n_target, node_offset=node_offset,
            in_dim=in_dim, num_classes=num_classes, num_layers=L,
            timeout=args.inf_timeout, log=log, label=f"{label}_L{L}",
        )
        layers_path = z_out.replace(".pt", "_layers.pt")
        if not os.path.exists(layers_path):
            layers_path = None
        return res, z_out, layers_path

    def _cka_from_disk(z_exact_path, layers_exact_path, z_approx_path,
                       layers_approx_path, L, label):
        """Load saved embeddings and compute CKA + pred_sim. No fresh inference."""
        cka_cols = {f"CKA_L{i+1}": "" for i in range(4)}
        pred_sim = ""
        if not (z_exact_path and os.path.exists(z_exact_path)):
            return cka_cols, pred_sim
        try:
            z_exact  = torch.load(z_exact_path,  weights_only=True).to(device)
            z_approx = torch.load(z_approx_path, weights_only=True).to(device)
            mask_dev = test_mask.to(device)
            pred_sim = _fmt(_pred_agreement(z_exact, z_approx, mask_dev))

            if layers_exact_path and os.path.exists(layers_exact_path) \
               and layers_approx_path and os.path.exists(layers_approx_path):
                try:
                    le_list = torch.load(layers_exact_path,  weights_only=True)
                    la_list = torch.load(layers_approx_path, weights_only=True)
                    for i, (le, la) in enumerate(zip(le_list, la_list)):
                        if i >= 4:
                            break
                        val = cka_calc.calculate(le.to(device)[mask_dev],
                                                 la.to(device)[mask_dev])
                        cka_cols[f"CKA_L{i+1}"] = _fmt(val)
                except (MemoryError, RuntimeError) as e:
                    log.warning("  [%s L=%d] layerwise CKA OOM: %s", label, L, e)
            del z_exact, z_approx
            gc.collect()
        except Exception as e:
            log.warning("  [%s L=%d] CKA disk load failed: %s", label, L, e)
        return cka_cols, pred_sim

    # ------------------------------------------------------------------
    # ExactD — materialization subprocess (C++ child), inference subprocess.
    # Mat_RAM_MB  = C++ child process peak via /usr/bin/time -v (isolated).
    # Inf_RAM_MB  = inference_worker.py fresh subprocess peak (isolated).
    # ------------------------------------------------------------------
    exact_edge_count  = None
    exact_status_flag = "OK"
    t_exact_mat       = None
    exact_mat_mb: Optional[float] = None
    z_exact_by_L: dict      = {}   # L → z_path (for comparison)
    layers_exact_by_L: dict = {}   # L → layers_path

    log.info("\n--- Running ExactD on full graph ---")
    try:
        t_exact_mat, exact_file = _run_exact(engine, folder, args.timeout)
        exact_mat_mb     = engine.last_peak_mb   # C++ child peak (Linux /usr/bin/time -v)
        exact_edge_count = _count_edges(exact_file)
        log.info("  ExactD done: edges=%d  mat_time=%.2fs  mat_ram=%s",
                 exact_edge_count, t_exact_mat,
                 f"{exact_mat_mb:.0f}MB" if exact_mat_mb else "n/a (Windows)")

        if args.max_rss_gb is not None:
            rss = _rss_gb()
            if rss is not None and rss > args.max_rss_gb:
                raise MemoryError(
                    f"RSS guard: {rss:.1f} GB > {args.max_rss_gb:.1f} GB after ExactD")

        # L-cascade flag: if inference fails for one L on this materialized graph,
        # it will fail for all larger L too (same adj_csr + features, similar peak RAM).
        # Write cascade rows immediately so mat data is preserved.
        exact_inf_cascade: Optional[str] = None  # set to status string on first failure

        for L in args.depth:
            weights_path = weights_dir / f"{mp_safe}_L{L}.pt"
            if not weights_path.exists():
                log.warning("  [Exact L=%d] weights not found — skipping", L)
                continue
            if (args.metapath, str(L), "Exact", "") in done_runs:
                log.info("  [Exact L=%d] already in CSV — skipping", L)
                continue

            if exact_inf_cascade is not None:
                log.warning("  [Exact L=%d] skipping — cascaded from earlier L failure (%s)",
                            L, exact_inf_cascade)
                csv_w.writerow({
                    **{f: "" for f in _FIELDS},
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "Exact", "k_value": "",
                    "Materialization_Time": _fmt(t_exact_mat),
                    "Mat_RAM_MB": _fmt(exact_mat_mb, 1) if exact_mat_mb else "",
                    "Edge_Count": exact_edge_count,
                    "Graph_Density": _graph_density(exact_edge_count, n_target),
                    "exact_status": f"INF_CASCADE({exact_inf_cascade})",
                })
                csv_fh.flush()
                continue

            inf_res, z_path, layers_path = _inf_subprocess(
                exact_file, "adj", "exact", L)
            if inf_res is not None and inf_res.get("inf_failed"):
                status = ("INF_OOM"     if inf_res.get("inf_oom")
                          else "INF_TIMEOUT" if inf_res.get("inf_timeout")
                          else "INF_FAIL")
                log.warning("  [Exact L=%d] inference failed: %s — cascading remaining depths",
                            L, status)
                exact_inf_cascade = status   # trigger cascade for remaining L
                csv_w.writerow({
                    **{f: "" for f in _FIELDS},
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "Exact", "k_value": "",
                    "Materialization_Time": _fmt(t_exact_mat),
                    "Mat_RAM_MB": _fmt(exact_mat_mb, 1) if exact_mat_mb else "",
                    "Edge_Count": exact_edge_count,
                    "Graph_Density": _graph_density(exact_edge_count, n_target),
                    "exact_status": status,
                })
                csv_fh.flush()
                continue

            z_exact_by_L[L]      = z_path
            layers_exact_by_L[L] = layers_path

            cka_cols = {f"CKA_L{i+1}": "" for i in range(4)}
            csv_w.writerow({
                "Dataset":              args.dataset,
                "MetaPath":             args.metapath,
                "L":                    L,
                "Method":               "Exact",
                "k_value":              "",
                "Density_Matched_w":    "",
                "Materialization_Time": _fmt(t_exact_mat),
                "Inference_Time":       _fmt(inf_res.get("inf_time")),
                "Mat_RAM_MB":           _fmt(exact_mat_mb, 1) if exact_mat_mb else "",
                "Inf_RAM_MB":           _fmt(inf_res.get("inf_peak_ram_mb"), 1),
                "Edge_Count":           exact_edge_count,
                "Graph_Density":        _graph_density(exact_edge_count, n_target),
                **cka_cols,
                "Pred_Similarity":      "",
                "Macro_F1":             _fmt(inf_res.get("inf_f1")),
                "Dirichlet_Energy":     _fmt(inf_res.get("inf_de")),
                "exact_status":         exact_status_flag,
            })
            csv_fh.flush()
            log.info("  [Exact L=%d] F1=%.4f  DE=%.4f  inf=%.2fs  inf_ram=%s  mat_ram=%s",
                     L, inf_res.get("inf_f1", 0), inf_res.get("inf_de", 0),
                     inf_res.get("inf_time", 0),
                     f"{inf_res.get('inf_peak_ram_mb', 0):.0f}MB",
                     f"{exact_mat_mb:.0f}MB" if exact_mat_mb else "n/a")

        log.info("  Exact done for all depths.  [parent RSS=%.0fMB]", _rss_mb())

    except MemoryError as e:
        exact_status_flag = "MAT_OOM"
        log.warning("  ExactD OOM: %s", e)
    except RuntimeError as e:
        exact_status_flag = ("MAT_TIMEOUT" if "timed out" in str(e)
                             else f"MAT_ERR:{str(e)[:80]}")
        log.warning("  ExactD error: %s", e)

    # ------------------------------------------------------------------
    # KMV sweep — materialization: C++ child peak. Inference: subprocess.
    # ------------------------------------------------------------------
    kmv_edges_by_k: dict = {}

    for k in args.k_values:
        log.info("\n--- KMV k=%d ---", k)

        if args.max_rss_gb is not None:
            rss = _rss_gb()
            if rss is not None and rss > args.max_rss_gb:
                log.warning("  [KMV k=%d] RSS guard: %.1f GB > %.1f GB — skipping",
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
            kmv_mat_mb     = engine.last_peak_mb   # C++ child peak
            kmv_edge_count = _count_edges(kmv_file)
            kmv_edges_by_k[k] = kmv_edge_count
            log.info("  KMV done: edges=%d  mat_time=%.2fs  mat_ram=%s",
                     kmv_edge_count, t_kmv_mat,
                     f"{kmv_mat_mb:.0f}MB" if kmv_mat_mb else "n/a (Windows)")
        except MemoryError as e:
            log.warning("  [KMV k=%d] OOM: %s", k, e)
            for L in args.depth:
                csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "KMV", "k_value": k,
                    "exact_status": "KMV_OOM",
                }))
            csv_fh.flush()
            continue
        except RuntimeError as e:
            log.warning("  [KMV k=%d] error: %s", k, e)
            for L in args.depth:
                csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "KMV", "k_value": k,
                    "exact_status": f"KMV_ERR:{str(e)[:60]}",
                }))
            csv_fh.flush()
            continue

        kmv_inf_cascade: Optional[str] = None  # L-cascade flag for this k

        for L in args.depth:
            if (args.metapath, str(L), "KMV", str(k)) in done_runs:
                log.info("  [KMV k=%d L=%d] already in CSV — skipping", k, L)
                continue
            weights_path = weights_dir / f"{mp_safe}_L{L}.pt"
            if not weights_path.exists():
                log.warning("  [KMV k=%d L=%d] weights not found — skipping", k, L)
                continue

            if kmv_inf_cascade is not None:
                log.warning("  [KMV k=%d L=%d] skipping — cascaded from earlier L failure (%s)",
                            k, L, kmv_inf_cascade)
                csv_w.writerow({
                    **{f: "" for f in _FIELDS},
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "KMV", "k_value": k,
                    "Materialization_Time": _fmt(t_kmv_mat),
                    "Mat_RAM_MB": _fmt(kmv_mat_mb, 1) if kmv_mat_mb else "",
                    "Edge_Count": kmv_edge_count,
                    "Graph_Density": _graph_density(kmv_edge_count, n_target),
                    "exact_status": f"INF_CASCADE({kmv_inf_cascade})",
                })
                csv_fh.flush()
                continue

            inf_res, z_kmv_path, layers_kmv_path = _inf_subprocess(
                kmv_file, "adj", f"kmv_{k}", L)
            if inf_res is not None and inf_res.get("inf_failed"):
                status = ("INF_OOM"     if inf_res.get("inf_oom")
                          else "INF_TIMEOUT" if inf_res.get("inf_timeout")
                          else "INF_FAIL")
                log.warning("  [KMV k=%d L=%d] inference failed: %s — cascading remaining depths",
                            k, L, status)
                kmv_inf_cascade = status
                csv_w.writerow({
                    **{f: "" for f in _FIELDS},
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "KMV", "k_value": k,
                    "Materialization_Time": _fmt(t_kmv_mat),
                    "Mat_RAM_MB": _fmt(kmv_mat_mb, 1) if kmv_mat_mb else "",
                    "Edge_Count": kmv_edge_count,
                    "Graph_Density": _graph_density(kmv_edge_count, n_target),
                    "exact_status": status,
                })
                csv_fh.flush()
                continue

            cka_cols, pred_sim = _cka_from_disk(
                z_exact_by_L.get(L), layers_exact_by_L.get(L),
                z_kmv_path, layers_kmv_path, L, f"KMV k={k}")

            log.info("  [KMV k=%d L=%d] F1=%.4f  DE=%.4f  inf=%.2fs  inf_ram=%s  mat_ram=%s  pred_sim=%s",
                     k, L, inf_res.get("inf_f1", 0), inf_res.get("inf_de", 0),
                     inf_res.get("inf_time", 0),
                     f"{inf_res.get('inf_peak_ram_mb', 0):.0f}MB",
                     f"{kmv_mat_mb:.0f}MB" if kmv_mat_mb else "n/a",
                     pred_sim or "n/a")

            csv_w.writerow({
                "Dataset":              args.dataset,
                "MetaPath":             args.metapath,
                "L":                    L,
                "Method":               "KMV",
                "k_value":              k,
                "Density_Matched_w":    "",
                "Materialization_Time": _fmt(t_kmv_mat),
                "Inference_Time":       _fmt(inf_res.get("inf_time")),
                "Mat_RAM_MB":           _fmt(kmv_mat_mb, 1) if kmv_mat_mb else "",
                "Inf_RAM_MB":           _fmt(inf_res.get("inf_peak_ram_mb"), 1),
                "Edge_Count":           kmv_edge_count,
                "Graph_Density":        _graph_density(kmv_edge_count, n_target),
                **cka_cols,
                "Pred_Similarity":      pred_sim,
                "Macro_F1":             _fmt(inf_res.get("inf_f1")),
                "Dirichlet_Energy":     _fmt(inf_res.get("inf_de")),
                "exact_status":         exact_status_flag,
            })
            csv_fh.flush()

    # ------------------------------------------------------------------
    # MPRW sweep — materialization: mprw_worker net_ram_mb. Inference: subprocess.
    # ------------------------------------------------------------------
    try:
        mprw_triples = parse_metapath_triples(args.metapath, g_full)
    except (ValueError, RuntimeError) as e:
        log.warning("  MPRW: could not parse metapath — %s", e)
        mprw_triples = None

    mprw_work_dir  = Path(data_dir) / "mprw_work"
    mprw_meta_file = mprw_work_dir / "meta.json"

    if mprw_triples is not None:
        mprw_work_dir.mkdir(parents=True, exist_ok=True)
        mprw_meta = {"n_target": n_target, "target_type": target_ntype, "steps": []}
        for i, (src_t, edge_name, dst_t) in enumerate(mprw_triples):
            edge_file = mprw_work_dir / f"edge_{i}.pt"
            torch.save(g_full[src_t, edge_name, dst_t].edge_index.cpu(), edge_file)
            mprw_meta["steps"].append({
                "src_type":  src_t, "edge_name": edge_name, "dst_type": dst_t,
                "n_src":     g_full[src_t].num_nodes,
                "n_dst":     g_full[dst_t].num_nodes,
                "edge_file": str(edge_file),
            })
        with open(mprw_meta_file, "w") as f:
            json.dump(mprw_meta, f)
        log.info("\n--- MPRW sweep ---")

    mprw_worker = str(Path(project_root) / "scripts" / "mprw_worker.py")

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

        # Timer starts before calibration — that IS materialization cost.
        t_mat_start = time.perf_counter()

        target_edges = kmv_edges_by_k.get(k)
        if target_edges is None:
            calibrated_w, calibrated_edge_count, calib_converged = k, None, False
            log.warning("  [MPRW k=%d] KMV edge count unavailable — using w=k", k)
        else:
            log.info("  [MPRW k=%d] calibrating w to match KMV edge count %d ...",
                     k, target_edges)
            calibrated_w, calibrated_edge_count, calib_converged = _calibrate_mprw_w(
                g_full, mprw_triples, target_ntype,
                target_edges=target_edges, seed=kmv_seed, device=device, log=log,
            )

        mprw_out = mprw_work_dir / f"mat_mprw_{k}.pt"
        mat_cmd  = [sys.executable, mprw_worker,
                    str(mprw_meta_file), str(mprw_out),
                    str(calibrated_w), str(kmv_seed)]
        log.info("  [MPRW] Materializing (w=%d) ...", calibrated_w)

        try:
            mat_res = subprocess.run(mat_cmd, check=True, capture_output=True,
                                     text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            log.warning("  [MPRW k=%d] materialization timed out", k)
            for L in args.depth:
                csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "MPRW", "k_value": k,
                    "exact_status": "MPRW_TIMEOUT",
                }))
            csv_fh.flush()
            continue
        except subprocess.CalledProcessError as e:
            log.warning("  [MPRW k=%d] materialization failed (exit %d):\n%s",
                        k, e.returncode, e.stderr[-400:])
            for L in args.depth:
                csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "MPRW", "k_value": k,
                    "exact_status": f"MPRW_ERR:{e.returncode}",
                }))
            csv_fh.flush()
            continue

        t_mprw_mat = time.perf_counter() - t_mat_start

        # Parse peak_ram_mb from worker stdout (true high-water mark above baseline)
        mprw_mat_mb: Optional[float] = None
        for line in mat_res.stdout.split("\n"):
            if line.strip().lower().startswith("peak_ram_mb:"):
                try:
                    mprw_mat_mb = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
                break

        if not mprw_out.exists():
            log.warning("  [MPRW k=%d] output file missing", k)
            continue

        # Edge count from saved .pt (load is small — just int64 COO)
        tmp_ei          = torch.load(mprw_out, weights_only=True)
        mprw_edge_count = tmp_ei.size(1)
        del tmp_ei

        log.info("  MPRW done: edges=%d (w=%d)  mat_time=%.2fs  mat_ram=%s",
                 mprw_edge_count, calibrated_w, t_mprw_mat,
                 f"{mprw_mat_mb:.0f}MB" if mprw_mat_mb is not None else "n/a")

        mprw_inf_cascade: Optional[str] = None  # L-cascade flag for this k

        for L in args.depth:
            if (args.metapath, str(L), "MPRW", str(k)) in done_runs:
                log.info("  [MPRW k=%d L=%d] already in CSV — skipping", k, L)
                continue
            weights_path = weights_dir / f"{mp_safe}_L{L}.pt"
            if not weights_path.exists():
                log.warning("  [MPRW k=%d L=%d] weights not found — skipping", k, L)
                continue

            calib_flag = "" if calib_converged else "MPRW_CALIB_APPROX|"

            if mprw_inf_cascade is not None:
                log.warning("  [MPRW k=%d L=%d] skipping — cascaded from earlier L failure (%s)",
                            k, L, mprw_inf_cascade)
                csv_w.writerow({
                    **{f: "" for f in _FIELDS},
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "MPRW", "k_value": k,
                    "Density_Matched_w": calibrated_w,
                    "Materialization_Time": _fmt(t_mprw_mat),
                    "Mat_RAM_MB": _fmt(mprw_mat_mb, 1) if mprw_mat_mb is not None else "",
                    "Edge_Count": mprw_edge_count,
                    "Graph_Density": _graph_density(mprw_edge_count, n_target),
                    "exact_status": f"{calib_flag}INF_CASCADE({mprw_inf_cascade})",
                })
                csv_fh.flush()
                continue

            inf_res, z_mprw_path, layers_mprw_path = _inf_subprocess(
                str(mprw_out), "pt", f"mprw_{k}", L)
            if inf_res is not None and inf_res.get("inf_failed"):
                status = ("INF_OOM"     if inf_res.get("inf_oom")
                          else "INF_TIMEOUT" if inf_res.get("inf_timeout")
                          else "INF_FAIL")
                log.warning("  [MPRW k=%d L=%d] inference failed: %s — cascading remaining depths",
                            k, L, status)
                mprw_inf_cascade = status
                csv_w.writerow({
                    **{f: "" for f in _FIELDS},
                    "Dataset": args.dataset, "MetaPath": args.metapath,
                    "L": L, "Method": "MPRW", "k_value": k,
                    "Density_Matched_w": calibrated_w,
                    "Materialization_Time": _fmt(t_mprw_mat),
                    "Mat_RAM_MB": _fmt(mprw_mat_mb, 1) if mprw_mat_mb is not None else "",
                    "Edge_Count": mprw_edge_count,
                    "Graph_Density": _graph_density(mprw_edge_count, n_target),
                    "exact_status": f"{calib_flag}{status}",
                })
                csv_fh.flush()
                continue

            cka_cols, pred_sim = _cka_from_disk(
                z_exact_by_L.get(L), layers_exact_by_L.get(L),
                z_mprw_path, layers_mprw_path, L, f"MPRW k={k}")

            log.info("  [MPRW k=%d L=%d] F1=%.4f  DE=%.4f  inf=%.2fs  inf_ram=%s  mat_ram=%s  pred_sim=%s",
                     k, L, inf_res.get("inf_f1", 0), inf_res.get("inf_de", 0),
                     inf_res.get("inf_time", 0),
                     f"{inf_res.get('inf_peak_ram_mb', 0):.0f}MB",
                     f"{mprw_mat_mb:.0f}MB" if mprw_mat_mb is not None else "n/a",
                     pred_sim or "n/a")

            csv_w.writerow({
                "Dataset":              args.dataset,
                "MetaPath":             args.metapath,
                "L":                    L,
                "Method":               "MPRW",
                "k_value":              k,
                "Density_Matched_w":    calibrated_w,
                "Materialization_Time": _fmt(t_mprw_mat),
                "Inference_Time":       _fmt(inf_res.get("inf_time")),
                "Mat_RAM_MB":           _fmt(mprw_mat_mb, 1) if mprw_mat_mb is not None else "",
                "Inf_RAM_MB":           _fmt(inf_res.get("inf_peak_ram_mb"), 1),
                "Edge_Count":           mprw_edge_count,
                "Graph_Density":        _graph_density(mprw_edge_count, n_target),
                **cka_cols,
                "Pred_Similarity":      pred_sim,
                "Macro_F1":             _fmt(inf_res.get("inf_f1")),
                "Dirichlet_Energy":     _fmt(inf_res.get("inf_de")),
                "exact_status":         (exact_status_flag if calib_converged
                                         else f"MPRW_CALIB_APPROX|{exact_status_flag}"),
            })
            csv_fh.flush()

    csv_fh.close()
    log.info("\nDone. Results -> %s", csv_path)


if __name__ == "__main__":
    main()
