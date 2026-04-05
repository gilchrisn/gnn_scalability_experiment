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
from src.models import get_model
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


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_FIELDS = [
    "Dataset", "MetaPath", "L", "Method", "k_value", "Density_Matched_w",
    "Materialization_Time", "Inference_Time", "Peak_RAM_MB", "Edge_Count",
    "CKA_L1", "CKA_L2", "CKA_L3", "CKA_L4",
    "Pred_Similarity", "Macro_F1", "exact_status",
]


def _open_csv(path: Path) -> Tuple:
    is_new = not path.exists() or path.stat().st_size == 0
    fh     = open(path, "a", newline="", encoding="utf-8")
    w      = csv.DictWriter(fh, fieldnames=_FIELDS)
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

            log.info("  [k=%d L=%d] KMV  F1=%.4f  inf=%.2fs  peak_ram=%s  pred_sim=%s",
                     k, L, f1_kmv, t_inf,
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
                "exact_status":         exact_status_flag,
            })
            csv_fh.flush()

            del model
            gc.collect()

        del g_kmv
        gc.collect()

    # ------------------------------------------------------------------
    # MPRW placeholder rows
    # ------------------------------------------------------------------
    for L in args.depth:
        for k in args.k_values:
            if (args.metapath, str(L), "MPRW", str(k)) in done_runs:
                continue
            csv_w.writerow(dict({f: "" for f in _FIELDS}, **{
                "Dataset":      args.dataset,
                "MetaPath":     args.metapath,
                "L":            L,
                "Method":       "MPRW",
                "k_value":      k,
                "exact_status": "MPRW_PENDING",
            }))
    csv_fh.flush()
    csv_fh.close()

    log.info("\nDone. Results → %s", csv_path)


if __name__ == "__main__":
    main()
