"""Sanity controls for the v2 fidelity metric stack (SPEC §7).

Tests whether the v2 metric stack catches known-bad cases:

  1. random_theta            — random model init (frozen weights replaced).
                                Expectation: all fidelity metrics drop to floor.
  2. edge_perturb             — replace fraction p of Exact edges with random
                                pairs at the same density.
                                Expectation: metrics degrade monotonically with p.
  3. density_matched_random   — uniformly drop Exact edges to match KMV edge
                                count at each k.
                                Expectation: KMV beats random on every metric.
  4. layer_permutation        — compare Z_exact_layer1 vs Z_kmv_layer2.
                                Expectation: CKA drops sharply (post-hoc; reads
                                existing per-layer tensors from inf_scratch).

Outputs JSON (one per run) to:
    results/approach_a_2026_05_07/sanity/<mode>/<run>.json

Usage
-----
    python scripts/exp_sanity_controls.py \\
        --mode random_theta \\
        --dataset HGB_DBLP \\
        --metapath author_to_paper,paper_to_author \\
        --target-type author \\
        --arch SAGE \\
        --depth 2 \\
        --k-values 8 16 32 64 128 \\
        --seeds 42 43 44 \\
        --partition-json results/HGB_DBLP/partition.json \\
        --weights-dir results/HGB_DBLP/weights
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.models import get_model
from src.bridge import PyGToCppAdapter
from src.bridge.engine import CppEngine
from src.analysis.cka import LinearCKA
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, setup_global_res_dirs
from scripts.exp3_inference import (
    _run_exact, _run_sketch, _run_inference_worker,
    _count_edges, _graph_density,
)


SPEC_VERSION = "approach_a_2026_05_07"
PERTURB_FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5]


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------

def _parse_adj_to_edge_index(filepath: str, n_target: int,
                             node_offset: int) -> torch.Tensor:
    """Parse C++ adjacency list ('u v1 v2 ...') into edge_index (local IDs)."""
    srcs, dsts = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                u_local = int(parts[0]) - node_offset
            except ValueError:
                continue
            if u_local < 0 or u_local >= n_target:
                continue
            for v in parts[1:]:
                try:
                    v_local = int(v) - node_offset
                except ValueError:
                    continue
                if 0 <= v_local < n_target:
                    srcs.append(u_local)
                    dsts.append(v_local)
    if not srcs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([srcs, dsts], dtype=torch.long)


def _save_edge_index_pt(edge_index: torch.Tensor, out_path: str) -> int:
    """Save edge_index to .pt. Returns number of directed edges."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(edge_index, out_path)
    return int(edge_index.size(1))


def _edge_index_perturb(edge_index: torch.Tensor, n: int, p: float,
                        rng: np.random.Generator) -> torch.Tensor:
    """Replace fraction p of edges with uniform random pairs at the same count.

    Maintains the same edge count (sampling without replacement, deduplicated
    against the kept edges and against itself)."""
    E = int(edge_index.size(1))
    if E == 0 or p <= 0.0:
        return edge_index.clone()

    n_replace = int(round(p * E))
    if n_replace == 0:
        return edge_index.clone()
    if n_replace >= E:
        n_replace = E

    keep_mask = np.ones(E, dtype=bool)
    drop_idx = rng.choice(E, size=n_replace, replace=False)
    keep_mask[drop_idx] = False
    kept = edge_index[:, torch.from_numpy(keep_mask)]

    # Build set of existing (after-drop) pairs to avoid duplicates.
    existing = set()
    if kept.size(1) > 0:
        for s, d in zip(kept[0].tolist(), kept[1].tolist()):
            existing.add((s, d))

    new_edges = []
    attempts = 0
    max_attempts = max(20 * n_replace, 1000)
    while len(new_edges) < n_replace and attempts < max_attempts:
        batch = max(n_replace - len(new_edges), 1)
        s = rng.integers(0, n, size=batch)
        d = rng.integers(0, n, size=batch)
        for si, di in zip(s.tolist(), d.tolist()):
            if si == di:
                continue
            if (si, di) in existing:
                continue
            existing.add((si, di))
            new_edges.append((si, di))
            if len(new_edges) >= n_replace:
                break
        attempts += batch

    if not new_edges:
        return kept

    new_arr = torch.tensor(new_edges, dtype=torch.long).T  # [2, M]
    return torch.cat([kept, new_arr], dim=1)


def _edge_index_random_subgraph(edge_index: torch.Tensor, target_edges: int,
                                rng: np.random.Generator) -> torch.Tensor:
    """Uniformly drop edges from edge_index until size == target_edges."""
    E = int(edge_index.size(1))
    if target_edges >= E:
        return edge_index.clone()
    if target_edges <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    keep = rng.choice(E, size=target_edges, replace=False)
    return edge_index[:, torch.from_numpy(np.sort(keep))]


# ---------------------------------------------------------------------------
# v2 metric computation
# ---------------------------------------------------------------------------

def _per_row_cosine(z_ref: torch.Tensor, z_app: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[float, float]:
    a = z_ref[mask].float()
    b = z_app[mask].float()
    eps = 1e-12
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1).clamp_min(eps) * b.norm(dim=1).clamp_min(eps)
    cos = num / den
    return float(cos.mean().item()), float(cos.std(unbiased=False).item())


def _per_row_rel_l2(z_ref: torch.Tensor, z_app: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[float, float]:
    a = z_ref[mask].float()
    b = z_app[mask].float()
    diff = (a - b).norm(dim=1)
    den = a.norm(dim=1).clamp_min(1e-12)
    rel = diff / den
    return float(rel.mean().item()), float(rel.std(unbiased=False).item())


def _frob_recon_err(z_ref: torch.Tensor, z_app: torch.Tensor,
                    mask: torch.Tensor) -> float:
    a = z_ref[mask].float()
    b = z_app[mask].float()
    num = (a - b).norm()
    den = a.norm().clamp_min(1e-12)
    return float((num / den).item())


def _procrustes_q_orth(z_ref: torch.Tensor, z_app: torch.Tensor,
                       mask: torch.Tensor) -> float:
    """Orthogonal-Procrustes residual: min_Q ||A - B Q||_F / ||A||_F."""
    a = z_ref[mask].float()
    b = z_app[mask].float()
    M = b.T @ a   # [d, d]
    try:
        u, _, vh = torch.linalg.svd(M, full_matrices=False)
        Q = u @ vh
        residual = (a - b @ Q).norm()
        return float((residual / a.norm().clamp_min(1e-12)).item())
    except Exception:
        return float("nan")


def _pred_agreement(z_a: torch.Tensor, z_b: torch.Tensor,
                    mask: torch.Tensor, multilabel: bool) -> float:
    if multilabel:
        a = (z_a[mask] > 0).long()
        b = (z_b[mask] > 0).long()
        return float((a == b).float().mean().item())
    return float((z_a[mask].argmax(1) == z_b[mask].argmax(1)).float().mean().item())


def _compute_v2_metrics(z_ref_path: str, layers_ref_path: Optional[str],
                        z_app_path: str, layers_app_path: Optional[str],
                        mask: torch.Tensor, multilabel: bool,
                        cka_calc: LinearCKA, device: torch.device) -> dict:
    """Compute v2 fidelity stack between reference and approximate embeddings."""
    out: dict = {}

    z_ref = torch.load(z_ref_path, weights_only=True).to(device)
    z_app = torch.load(z_app_path, weights_only=True).to(device)
    mask_dev = mask.to(device)

    out["pred_agreement"] = _pred_agreement(z_ref, z_app, mask_dev, multilabel)

    cos_means, cos_stds = [], []
    rel_l2_means, rel_l2_stds = [], []
    frob_errs = []
    proc_q_eq_i = []
    proc_q_orth = []
    cka_per_layer = []

    if (layers_ref_path and os.path.exists(layers_ref_path) and
            layers_app_path and os.path.exists(layers_app_path)):
        try:
            le_list = torch.load(layers_ref_path, weights_only=True)
            la_list = torch.load(layers_app_path, weights_only=True)
            for le, la in zip(le_list, la_list):
                le_d = le.to(device)
                la_d = la.to(device)
                cm, cs = _per_row_cosine(le_d, la_d, mask_dev)
                lm, ls = _per_row_rel_l2(le_d, la_d, mask_dev)
                cos_means.append(cm); cos_stds.append(cs)
                rel_l2_means.append(lm); rel_l2_stds.append(ls)
                frob_errs.append(_frob_recon_err(le_d, la_d, mask_dev))
                # Q=I procrustes is the relative Frobenius (no rotation).
                proc_q_eq_i.append(frob_errs[-1])
                proc_q_orth.append(_procrustes_q_orth(le_d, la_d, mask_dev))
                cka_per_layer.append(float(cka_calc.calculate(
                    le_d[mask_dev], la_d[mask_dev])))
        except Exception as e:
            logging.getLogger("sanity").warning(
                "  per-layer metrics failed: %s", e)

    out["row_cosine_per_layer_mean"]  = cos_means
    out["row_cosine_per_layer_std"]   = cos_stds
    out["row_rel_l2_per_layer_mean"]  = rel_l2_means
    out["row_rel_l2_per_layer_std"]   = rel_l2_stds
    out["frob_recon_err_per_layer"]   = frob_errs
    out["procrustes_q_eq_i_per_layer"] = proc_q_eq_i
    out["procrustes_q_orth_per_layer"] = proc_q_orth
    out["cka_per_layer"]              = cka_per_layer

    del z_ref, z_app
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _setup_dataset_and_engine(args, log: logging.Logger):
    """Stage C++ files, load dataset, build engine. Returns dict of fixtures."""
    with open(args.partition_json, "rb") as fh_raw:
        partition_bytes = fh_raw.read()
    partition_hash = "sha256:" + hashlib.sha256(partition_bytes).hexdigest()
    part = json.loads(partition_bytes.decode("utf-8"))

    cfg          = config.get_dataset_config(args.dataset)
    folder       = config.get_folder_name(args.dataset)
    data_dir     = config.get_staging_dir(args.dataset)
    os.makedirs(data_dir, exist_ok=True)
    target_ntype = args.target_type or cfg.target_node

    g_full, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    num_classes  = info["num_classes"]
    labels_full  = info["labels"]

    n_target  = g_full[target_ntype].num_nodes
    test_ids  = torch.tensor(part["test_node_ids"], dtype=torch.long)
    test_mask = torch.zeros(n_target, dtype=torch.bool)
    test_mask[test_ids] = True

    setup_global_res_dirs(folder, project_root)
    engine = CppEngine(executable_path=config.CPP_EXECUTABLE, data_dir=data_dir)

    log.info("Staging full graph...")
    PyGToCppAdapter(data_dir).convert(g_full)
    compile_rule_for_cpp(args.metapath, g_full, data_dir, folder)
    generate_qnodes(data_dir, folder, target_node_type=target_ntype, g_hetero=g_full)

    s_ntypes    = sorted(g_full.node_types)
    node_offset = sum(g_full[nt].num_nodes for nt in s_ntypes if nt < target_ntype)
    x_full      = g_full[target_ntype].x
    in_dim      = x_full.size(1)
    multilabel  = (labels_full.dim() == 2)
    kmv_seed    = (args.hash_seed if args.hash_seed is not None
                   else part.get("hash_seed", 0))

    mp_safe = args.metapath.replace(",", "_").replace("/", "_")
    out_dir = Path("results") / args.dataset
    scratch_dir = out_dir / "inf_scratch" / mp_safe
    scratch_dir.mkdir(parents=True, exist_ok=True)
    feat_file   = str(scratch_dir / "x.pt")
    labels_file = str(scratch_dir / "labels.pt")
    mask_file   = str(scratch_dir / "mask.pt")
    if not os.path.exists(feat_file):
        torch.save(x_full.cpu(),       feat_file)
    if not os.path.exists(labels_file):
        torch.save(labels_full.cpu(),  labels_file)
    if not os.path.exists(mask_file):
        torch.save(test_mask.cpu(),    mask_file)

    return {
        "engine":          engine,
        "data_dir":        data_dir,
        "folder":          folder,
        "g_full":          g_full,
        "n_target":        n_target,
        "node_offset":     node_offset,
        "in_dim":          in_dim,
        "num_classes":     num_classes,
        "x_full":          x_full,
        "labels_full":     labels_full,
        "test_mask":       test_mask,
        "multilabel":      multilabel,
        "kmv_seed":        kmv_seed,
        "target_ntype":    target_ntype,
        "mp_safe":         mp_safe,
        "scratch_dir":     scratch_dir,
        "feat_file":       feat_file,
        "labels_file":     labels_file,
        "mask_file":       mask_file,
        "partition_hash":  partition_hash,
        "partition":       part,
    }


def _weights_path_for(weights_dir: Path, mp_safe: str, L: int, arch: str) -> Path:
    suffix = "" if arch.upper() == "SAGE" else f"_{arch.upper()}"
    return weights_dir / f"{mp_safe}_L{L}{suffix}.pt"


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def _base_payload(args, fx: dict, sanity_mode: str, sanity_param: str) -> dict:
    return {
        "spec_version":     SPEC_VERSION,
        "sanity_mode":      sanity_mode,
        "sanity_param":     sanity_param,
        "dataset":          args.dataset,
        "meta_path":        args.metapath,
        "target_type":      fx["target_ntype"],
        "arch":             args.arch.upper(),
        "n_layers":         args.depth,
        "hidden_dim":       config.HIDDEN_DIM,
        "n_target_nodes":   int(fx["n_target"]),
        "n_test_nodes":     int(fx["test_mask"].sum().item()),
        "partition_hash":   fx["partition_hash"],
        "device_used":      "cpu",
        "timestamp":        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _materialize_exact_once(args, fx: dict, log: logging.Logger
                            ) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[int]]:
    """Run ExactD once. Returns (adj_path, mat_time, peak_ram_mb, edges)."""
    try:
        t_mat, exact_file = _run_exact(fx["engine"], fx["folder"], args.timeout)
        peak_mb = fx["engine"].last_peak_mb
        edges   = _count_edges(exact_file)
        log.info("ExactD: edges=%d  mat_time=%.2fs", edges, t_mat)
        return exact_file, t_mat, peak_mb, edges
    except Exception as e:
        log.error("ExactD failed: %s", e)
        return None, None, None, None


def _materialize_kmv_once(args, fx: dict, k: int, hash_seed: int,
                          log: logging.Logger
                          ) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[int]]:
    try:
        t_mat, kmv_file = _run_sketch(fx["engine"], fx["folder"], k,
                                      hash_seed, args.timeout)
        peak_mb = fx["engine"].last_peak_mb
        edges   = _count_edges(kmv_file)
        log.info("KMV k=%d: edges=%d  mat_time=%.2fs", k, edges, t_mat)
        return kmv_file, t_mat, peak_mb, edges
    except Exception as e:
        log.error("KMV k=%d failed: %s", k, e)
        return None, None, None, None


def _run_inference_with_random_weights(args, fx: dict, graph_file: str,
                                       graph_type: str, label: str,
                                       seed: int, log: logging.Logger
                                       ) -> Optional[Tuple[dict, str, Optional[str]]]:
    """Build random-init model, save weights, run inference, return result."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = get_model(args.arch.upper(), fx["in_dim"], fx["num_classes"],
                      config.HIDDEN_DIM, gat_heads=args.gat_heads,
                      num_layers=args.depth)
    rand_weights = fx["scratch_dir"] / f"random_theta_seed{seed}.pt"
    torch.save(model.state_dict(), rand_weights)

    z_out = str(fx["scratch_dir"] / f"z_{label}.pt")
    res = _run_inference_worker(
        graph_file=graph_file, graph_type=graph_type,
        feat_file=fx["feat_file"],
        weights_path=str(rand_weights),
        z_out=z_out,
        labels_file=fx["labels_file"], mask_file=fx["mask_file"],
        n_target=fx["n_target"], node_offset=fx["node_offset"],
        in_dim=fx["in_dim"], num_classes=fx["num_classes"],
        num_layers=args.depth, timeout=args.timeout, log=log,
        label=label, arch=args.arch.upper(), gat_heads=args.gat_heads,
    )
    if res is None or res.get("inf_failed"):
        return None
    layers_path = z_out.replace(".pt", "_layers.pt")
    if not os.path.exists(layers_path):
        layers_path = None
    return res, z_out, layers_path


def _run_inference_with_trained_weights(args, fx: dict, graph_file: str,
                                        graph_type: str, label: str,
                                        weights_path: Path,
                                        log: logging.Logger
                                        ) -> Optional[Tuple[dict, str, Optional[str]]]:
    z_out = str(fx["scratch_dir"] / f"z_{label}.pt")
    res = _run_inference_worker(
        graph_file=graph_file, graph_type=graph_type,
        feat_file=fx["feat_file"],
        weights_path=str(weights_path),
        z_out=z_out,
        labels_file=fx["labels_file"], mask_file=fx["mask_file"],
        n_target=fx["n_target"], node_offset=fx["node_offset"],
        in_dim=fx["in_dim"], num_classes=fx["num_classes"],
        num_layers=args.depth, timeout=args.timeout, log=log,
        label=label, arch=args.arch.upper(), gat_heads=args.gat_heads,
    )
    if res is None or res.get("inf_failed"):
        return None
    layers_path = z_out.replace(".pt", "_layers.pt")
    if not os.path.exists(layers_path):
        layers_path = None
    return res, z_out, layers_path


def mode_random_theta(args, fx: dict, sanity_dir: Path, cka_calc: LinearCKA,
                      device: torch.device, log: logging.Logger) -> None:
    """Random θ on Exact + KMV. All metrics should drop to floor."""
    out_dir = sanity_dir / "random_theta"
    out_dir.mkdir(parents=True, exist_ok=True)

    exact_file, exact_mat_time, exact_mat_mb, exact_edges = (
        _materialize_exact_once(args, fx, log))
    if exact_file is None:
        log.error("random_theta: ExactD failed — aborting mode.")
        return

    for seed in args.seeds:
        for k in args.k_values:
            out_path = out_dir / (f"{args.dataset}_{fx['mp_safe']}_"
                                  f"{args.arch.upper()}_seed{seed}_k{k}.json")
            if out_path.exists() and not args.force:
                log.info("[skip] %s", out_path.name)
                continue

            log.info("\n[random_theta] seed=%d k=%d", seed, k)

            # KMV at this k (one materialization per k)
            kmv_file, kmv_mat_time, kmv_mat_mb, kmv_edges = _materialize_kmv_once(
                args, fx, k, fx["kmv_seed"], log)
            if kmv_file is None:
                log.warning("  KMV k=%d failed — skipping", k)
                continue

            # Inference with random weights on Exact
            ex_label = f"sanity_randtheta_exact_seed{seed}"
            ex_res = _run_inference_with_random_weights(
                args, fx, exact_file, "adj", ex_label, seed, log)
            if ex_res is None:
                log.warning("  Exact inference failed (seed=%d)", seed)
                continue
            ex_inf, ex_z, ex_layers = ex_res

            # Inference with random weights on KMV
            kmv_label = f"sanity_randtheta_kmv_seed{seed}_k{k}"
            kmv_res = _run_inference_with_random_weights(
                args, fx, kmv_file, "adj", kmv_label, seed, log)
            if kmv_res is None:
                log.warning("  KMV inference failed (seed=%d k=%d)", seed, k)
                continue
            kmv_inf, kmv_z, kmv_layers = kmv_res

            metrics = _compute_v2_metrics(
                ex_z, ex_layers, kmv_z, kmv_layers,
                fx["test_mask"], fx["multilabel"], cka_calc, device)

            payload = _base_payload(args, fx, "random_theta", f"seed={seed},k={k}")
            payload.update({
                "seed":               seed,
                "kmv_k":              k,
                "kmv_hash_seed":      fx["kmv_seed"],
                "n_edges_exact":      exact_edges,
                "n_edges_kmv":        kmv_edges,
                "macro_f1_exact":     ex_inf.get("inf_f1"),
                "macro_f1_kmv":       kmv_inf.get("inf_f1"),
                "f1_gap":             (None
                                       if ex_inf.get("inf_f1") is None or kmv_inf.get("inf_f1") is None
                                       else float(kmv_inf["inf_f1"]) - float(ex_inf["inf_f1"])),
                "inference_time_exact_s": ex_inf.get("inf_time"),
                "inference_time_kmv_s":   kmv_inf.get("inf_time"),
                "mat_time_exact_s":       exact_mat_time,
                "mat_time_kmv_s":         kmv_mat_time,
                "inf_peak_rss_mb_exact":  ex_inf.get("inf_peak_ram_mb"),
                "inf_peak_rss_mb_kmv":    kmv_inf.get("inf_peak_ram_mb"),
                "mat_peak_rss_mb_exact":  exact_mat_mb,
                "mat_peak_rss_mb_kmv":    kmv_mat_mb,
                "arch_status":            "OK",
                **metrics,
            })

            with open(out_path, "w", encoding="utf-8") as jfh:
                json.dump(payload, jfh, indent=2)
            log.info("  -> %s", out_path)


def mode_edge_perturb(args, fx: dict, sanity_dir: Path, cka_calc: LinearCKA,
                      device: torch.device, log: logging.Logger) -> None:
    """Replace fraction p of Exact edges with random pairs at same density."""
    out_dir = sanity_dir / "edge_perturb"
    out_dir.mkdir(parents=True, exist_ok=True)

    exact_file, exact_mat_time, exact_mat_mb, exact_edges = (
        _materialize_exact_once(args, fx, log))
    if exact_file is None:
        log.error("edge_perturb: ExactD failed — aborting mode.")
        return

    # Reference Z_exact: trained-weights inference on Exact (one per seed-arch pair).
    # We use the existing trained weights in args.weights_dir.
    weights_path = _weights_path_for(Path(args.weights_dir), fx["mp_safe"],
                                     args.depth, args.arch.upper())
    if not weights_path.exists():
        log.error("edge_perturb: weights not found at %s — aborting mode.", weights_path)
        return

    ref_label = (f"sanity_perturb_ref_{args.arch.upper()}_L{args.depth}")
    ref_res = _run_inference_with_trained_weights(
        args, fx, exact_file, "adj", ref_label, weights_path, log)
    if ref_res is None:
        log.error("edge_perturb: reference Exact inference failed.")
        return
    ref_inf, ref_z, ref_layers = ref_res

    edge_index_exact = _parse_adj_to_edge_index(
        exact_file, fx["n_target"], fx["node_offset"])
    log.info("Loaded Exact edge_index: %d directed edges", edge_index_exact.size(1))

    for seed in args.seeds:
        for p in PERTURB_FRACTIONS:
            out_path = out_dir / (f"{args.dataset}_{fx['mp_safe']}_"
                                  f"{args.arch.upper()}_seed{seed}_p{p}.json")
            if out_path.exists() and not args.force:
                log.info("[skip] %s", out_path.name)
                continue

            log.info("\n[edge_perturb] seed=%d p=%.3f", seed, p)
            try:
                rng = np.random.default_rng(seed)
                ei_pert = _edge_index_perturb(
                    edge_index_exact, fx["n_target"], p, rng)
                pert_pt = str(fx["scratch_dir"]
                              / f"perturb_seed{seed}_p{p}.pt")
                pert_edges = _save_edge_index_pt(ei_pert, pert_pt)
                log.info("  perturbed graph: %d edges", pert_edges)
            except Exception as e:
                log.warning("  perturb construction failed: %s", e)
                continue

            pert_label = f"sanity_perturb_seed{seed}_p{p}"
            pert_res = _run_inference_with_trained_weights(
                args, fx, pert_pt, "pt", pert_label, weights_path, log)
            if pert_res is None:
                log.warning("  perturb inference failed (seed=%d p=%.3f)", seed, p)
                continue
            pert_inf, pert_z, pert_layers = pert_res

            metrics = _compute_v2_metrics(
                ref_z, ref_layers, pert_z, pert_layers,
                fx["test_mask"], fx["multilabel"], cka_calc, device)

            payload = _base_payload(args, fx, "edge_perturb", f"p={p},seed={seed}")
            payload.update({
                "seed":              seed,
                "perturb_p":         p,
                "n_edges_exact":     exact_edges,
                "n_edges_perturb":   pert_edges,
                "macro_f1_exact":    ref_inf.get("inf_f1"),
                "macro_f1_perturb":  pert_inf.get("inf_f1"),
                "f1_gap":            (None
                                      if ref_inf.get("inf_f1") is None or pert_inf.get("inf_f1") is None
                                      else float(pert_inf["inf_f1"]) - float(ref_inf["inf_f1"])),
                "inference_time_exact_s":   ref_inf.get("inf_time"),
                "inference_time_perturb_s": pert_inf.get("inf_time"),
                "arch_status":              "OK",
                **metrics,
            })

            with open(out_path, "w", encoding="utf-8") as jfh:
                json.dump(payload, jfh, indent=2)
            log.info("  -> %s", out_path)


def mode_density_matched_random(args, fx: dict, sanity_dir: Path,
                                 cka_calc: LinearCKA, device: torch.device,
                                 log: logging.Logger) -> None:
    """Uniformly-random subgraph at KMV's edge count vs. KMV at the same k."""
    out_dir = sanity_dir / "density_matched_random"
    out_dir.mkdir(parents=True, exist_ok=True)

    exact_file, exact_mat_time, exact_mat_mb, exact_edges = (
        _materialize_exact_once(args, fx, log))
    if exact_file is None:
        log.error("density_matched_random: ExactD failed — aborting mode.")
        return

    weights_path = _weights_path_for(Path(args.weights_dir), fx["mp_safe"],
                                     args.depth, args.arch.upper())
    if not weights_path.exists():
        log.error("density_matched_random: weights not found at %s — aborting.",
                  weights_path)
        return

    ref_label = (f"sanity_dmr_ref_{args.arch.upper()}_L{args.depth}")
    ref_res = _run_inference_with_trained_weights(
        args, fx, exact_file, "adj", ref_label, weights_path, log)
    if ref_res is None:
        log.error("density_matched_random: reference Exact inference failed.")
        return
    ref_inf, ref_z, ref_layers = ref_res

    edge_index_exact = _parse_adj_to_edge_index(
        exact_file, fx["n_target"], fx["node_offset"])

    for seed in args.seeds:
        for k in args.k_values:
            out_path = out_dir / (f"{args.dataset}_{fx['mp_safe']}_"
                                  f"{args.arch.upper()}_seed{seed}_k{k}.json")
            if out_path.exists() and not args.force:
                log.info("[skip] %s", out_path.name)
                continue

            log.info("\n[density_matched_random] seed=%d k=%d", seed, k)
            try:
                kmv_file, kmv_mat_time, kmv_mat_mb, kmv_edges = (
                    _materialize_kmv_once(args, fx, k, fx["kmv_seed"], log))
                if kmv_file is None:
                    continue
            except Exception as e:
                log.warning("  KMV materialize failed: %s", e)
                continue

            kmv_label = f"sanity_dmr_kmv_seed{seed}_k{k}"
            kmv_res = _run_inference_with_trained_weights(
                args, fx, kmv_file, "adj", kmv_label, weights_path, log)
            if kmv_res is None:
                log.warning("  KMV inference failed (k=%d)", k)
                continue
            kmv_inf, kmv_z, kmv_layers = kmv_res

            try:
                rng = np.random.default_rng(seed * 1000003 + k)
                ei_rand = _edge_index_random_subgraph(
                    edge_index_exact, kmv_edges, rng)
                rand_pt = str(fx["scratch_dir"]
                              / f"dmr_seed{seed}_k{k}.pt")
                rand_edges_actual = _save_edge_index_pt(ei_rand, rand_pt)
                log.info("  random subgraph: %d edges (target=%d)",
                         rand_edges_actual, kmv_edges)
            except Exception as e:
                log.warning("  random subgraph construction failed: %s", e)
                continue

            rand_label = f"sanity_dmr_rand_seed{seed}_k{k}"
            rand_res = _run_inference_with_trained_weights(
                args, fx, rand_pt, "pt", rand_label, weights_path, log)
            if rand_res is None:
                log.warning("  random inference failed (seed=%d k=%d)", seed, k)
                continue
            rand_inf, rand_z, rand_layers = rand_res

            kmv_metrics = _compute_v2_metrics(
                ref_z, ref_layers, kmv_z, kmv_layers,
                fx["test_mask"], fx["multilabel"], cka_calc, device)
            rand_metrics = _compute_v2_metrics(
                ref_z, ref_layers, rand_z, rand_layers,
                fx["test_mask"], fx["multilabel"], cka_calc, device)

            payload = _base_payload(args, fx, "density_matched_random",
                                    f"k={k},seed={seed}")
            payload.update({
                "seed":              seed,
                "kmv_k":             k,
                "kmv_hash_seed":     fx["kmv_seed"],
                "n_edges_exact":     exact_edges,
                "n_edges_kmv":       kmv_edges,
                "n_edges_random":    rand_edges_actual,
                "macro_f1_exact":    ref_inf.get("inf_f1"),
                "macro_f1_kmv":      kmv_inf.get("inf_f1"),
                "macro_f1_random":   rand_inf.get("inf_f1"),
                "kmv_metrics":       kmv_metrics,
                "random_metrics":    rand_metrics,
                "inference_time_exact_s":  ref_inf.get("inf_time"),
                "inference_time_kmv_s":    kmv_inf.get("inf_time"),
                "inference_time_random_s": rand_inf.get("inf_time"),
                "mat_time_exact_s":  exact_mat_time,
                "mat_time_kmv_s":    kmv_mat_time,
                "arch_status":       "OK",
            })

            with open(out_path, "w", encoding="utf-8") as jfh:
                json.dump(payload, jfh, indent=2)
            log.info("  -> %s", out_path)


def mode_layer_permutation(args, fx: dict, sanity_dir: Path,
                           cka_calc: LinearCKA, device: torch.device,
                           log: logging.Logger) -> None:
    """Cross-layer CKA mismatch: Z_exact_layer1 vs Z_kmv_layer2.

    Post-hoc: reads existing _layers.pt files saved by the v2 inference run."""
    out_dir = sanity_dir / "layer_permutation"
    out_dir.mkdir(parents=True, exist_ok=True)

    L = args.depth
    if L < 2:
        log.error("layer_permutation requires depth >= 2 (got %d)", L)
        return

    mask_dev = fx["test_mask"].to(device)

    for seed in args.seeds:
        for k in args.k_values:
            out_path = out_dir / (f"{args.dataset}_{fx['mp_safe']}_"
                                  f"{args.arch.upper()}_seed{seed}_k{k}.json")
            if out_path.exists() and not args.force:
                log.info("[skip] %s", out_path.name)
                continue

            ex_layers_path  = fx["scratch_dir"] / f"z_exact_L{L}_layers.pt"
            kmv_layers_path = (fx["scratch_dir"]
                               / f"z_kmv_{k}_s{seed}_L{L}_layers.pt")

            if not ex_layers_path.exists():
                log.warning("[layer_permutation] missing %s", ex_layers_path)
                continue
            if not kmv_layers_path.exists():
                log.warning("[layer_permutation] missing %s", kmv_layers_path)
                continue

            try:
                le_list = torch.load(str(ex_layers_path),  weights_only=True)
                la_list = torch.load(str(kmv_layers_path), weights_only=True)
            except Exception as e:
                log.warning("  load failed (seed=%d k=%d): %s", seed, k, e)
                continue

            if len(le_list) < 2 or len(la_list) < 2:
                log.warning("  not enough layers stored (have %d / %d)",
                            len(le_list), len(la_list))
                continue

            try:
                # Cross-layer mismatch: exact L1 vs KMV L2.
                z_ex_l1 = le_list[0].to(device)[mask_dev]
                z_km_l2 = la_list[1].to(device)[mask_dev]
                cka_cross = float(cka_calc.calculate(z_ex_l1, z_km_l2))

                # Aligned reference (layer-1 vs layer-1) for comparison.
                z_km_l1 = la_list[0].to(device)[mask_dev]
                cka_aligned_l1 = float(cka_calc.calculate(z_ex_l1, z_km_l1))

                # And layer-2 vs layer-2 aligned.
                z_ex_l2 = le_list[1].to(device)[mask_dev]
                cka_aligned_l2 = float(cka_calc.calculate(z_ex_l2, z_km_l2))
            except Exception as e:
                log.warning("  CKA failed (seed=%d k=%d): %s", seed, k, e)
                continue

            payload = _base_payload(args, fx, "layer_permutation",
                                    f"k={k},seed={seed}")
            payload.update({
                "seed":                       seed,
                "kmv_k":                      k,
                "cka_cross_exactL1_vs_kmvL2": cka_cross,
                "cka_aligned_L1":             cka_aligned_l1,
                "cka_aligned_L2":             cka_aligned_l2,
                "arch_status":                "OK",
            })

            with open(out_path, "w", encoding="utf-8") as jfh:
                json.dump(payload, jfh, indent=2)
            log.info("  -> %s  cross=%.4f  L1=%.4f  L2=%.4f",
                     out_path.name, cka_cross, cka_aligned_l1, cka_aligned_l2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True,
                        choices=["random_theta", "edge_perturb",
                                 "density_matched_random", "layer_permutation"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metapath", required=True)
    parser.add_argument("--target-type", default=None,
                        help="Target node type (default: from config).")
    parser.add_argument("--arch", type=str, default="SAGE",
                        choices=["SAGE", "GCN", "GAT", "GIN"])
    parser.add_argument("--depth", type=int, default=2,
                        help="Single L (sanity controls run at one depth).")
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--partition-json", required=True)
    parser.add_argument("--weights-dir", default=None,
                        help="Directory with frozen weights "
                             "(default: results/<dataset>/weights). Required for "
                             "edge_perturb and density_matched_random.")
    parser.add_argument("--hash-seed", type=int, default=None)
    parser.add_argument("--gat-heads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing JSONs instead of skipping.")
    args = parser.parse_args()

    if args.weights_dir is None:
        args.weights_dir = str(Path("results") / args.dataset / "weights")

    sanity_root = Path("results") / "approach_a_2026_05_07" / "sanity"
    sanity_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("sanity")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log_dir = Path("results") / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    fh_log = logging.FileHandler(
        log_dir / f"run_sanity_{args.mode}_{ts}.log", encoding="utf-8")
    fh_log.setLevel(logging.DEBUG)
    fh_log.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(ch)
    log.addHandler(fh_log)

    log.info("Sanity controls | mode=%s dataset=%s metapath=%s arch=%s depth=%d",
             args.mode, args.dataset, args.metapath, args.arch, args.depth)
    log.info("  k-values=%s  seeds=%s", args.k_values, args.seeds)

    device   = torch.device("cpu")
    cka_calc = LinearCKA(device=device)

    fx = _setup_dataset_and_engine(args, log)
    log.info("V_test=%d / V_total=%d  partition_hash=%s",
             int(fx["test_mask"].sum().item()), fx["n_target"],
             fx["partition_hash"])

    if args.mode == "random_theta":
        mode_random_theta(args, fx, sanity_root, cka_calc, device, log)
    elif args.mode == "edge_perturb":
        mode_edge_perturb(args, fx, sanity_root, cka_calc, device, log)
    elif args.mode == "density_matched_random":
        mode_density_matched_random(args, fx, sanity_root, cka_calc, device, log)
    elif args.mode == "layer_permutation":
        mode_layer_permutation(args, fx, sanity_root, cka_calc, device, log)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    log.info("\nDone. Sanity results -> %s", sanity_root / args.mode)


if __name__ == "__main__":
    main()
