"""
Experiment 2 — Train SAGE(L) on V_train.

Reads partition.json (from exp1_partition.py), reconstructs the V_train
induced subgraph, runs C++ ExactD once for strict isolation, then trains
SAGEConv with num_layers=L for each requested depth.

C++ materialization runs once and is reused across all L values.

Output
------
  results/<dataset>/weights/<metapath_safe>_L<L>.pt   frozen θ*
  results/<dataset>/training_log.csv
      One row per epoch: dataset, metapath, L, epoch, train_loss, val_loss, train_f1
      One DONE row per (metapath, L): convergence_epoch, t_train, weights_path

Usage
-----
    python scripts/exp1_partition.py --dataset HGB_DBLP --target-type author --train-frac 0.1
    python scripts/exp2_train.py HGB_DBLP \\
        --metapath author_to_paper,paper_to_author \\
        --depth 2 3 4 \\
        --epochs 100 \\
        --partition-json results/HGB_DBLP/partition.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
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
from src.models import get_model
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, setup_global_res_dirs


# ---------------------------------------------------------------------------
# V_train reconstruction
# ---------------------------------------------------------------------------

def _make_train_subgraph(g: HeteroData, part: dict, labels: torch.Tensor,
                         masks: dict, target_ntype: str) -> HeteroData:
    """
    Build the training subgraph by filtering ONLY the target node type to
    V_train (read directly from partition.json).  Every other node type is
    kept at 100%.  Only edges whose target-type endpoint falls outside V_train
    are dropped.
    """
    pivot_ntype = part["target_type"]   # exp1_partition.py key

    # V_train node IDs are pre-computed and stored in partition.json.
    keep_ids = torch.tensor(part["train_node_ids"], dtype=torch.long)
    keep_ids, _ = keep_ids.sort()  # deterministic new-ID assignment

    # old pivot ID → new pivot ID (non-kept → -1)
    id_map = torch.full((g[pivot_ntype].num_nodes,), -1, dtype=torch.long)
    id_map[keep_ids] = torch.arange(keep_ids.size(0))

    g_sub = HeteroData()

    # All non-pivot node types: copy everything unchanged
    for ntype in g.node_types:
        if ntype == pivot_ntype:
            continue
        g_sub[ntype].num_nodes = g[ntype].num_nodes
        for key, val in g[ntype].items():
            g_sub[ntype][key] = val

    # Pivot type: slice to kept nodes, attach labels/masks
    # NOTE: set num_nodes AFTER the copy loop — the loop's else-branch would
    # otherwise overwrite it with the full-graph count via PyG's attribute dict.
    for key, val in g[pivot_ntype].items():
        if key == 'num_nodes':
            continue  # will be set explicitly below
        if isinstance(val, torch.Tensor) and val.size(0) == g[pivot_ntype].num_nodes:
            g_sub[pivot_ntype][key] = val[keep_ids]
        else:
            g_sub[pivot_ntype][key] = val
    g_sub[pivot_ntype].num_nodes = keep_ids.size(0)

    # Attach labels / masks onto the (already sliced) target type
    g_sub[target_ntype].y          = labels[keep_ids] if target_ntype == pivot_ntype else labels
    g_sub[target_ntype].train_mask = (masks["train"][keep_ids]
                                      if target_ntype == pivot_ntype else masks["train"])
    g_sub[target_ntype].val_mask   = (masks["val"][keep_ids]
                                      if target_ntype == pivot_ntype else masks["val"])
    g_sub[target_ntype].test_mask  = (masks["test"][keep_ids]
                                      if target_ntype == pivot_ntype else masks["test"])

    # Edges: drop any edge whose pivot-type endpoint is outside keep_ids;
    # remap pivot-type endpoints to new local IDs; leave all other IDs unchanged.
    for src_type, rel, dst_type in g.edge_types:
        ei    = g[src_type, rel, dst_type].edge_index
        valid = torch.ones(ei.size(1), dtype=torch.bool)
        if src_type == pivot_ntype:
            valid &= id_map[ei[0]] >= 0
        if dst_type == pivot_ntype:
            valid &= id_map[ei[1]] >= 0
        new_src = id_map[ei[0][valid]] if src_type == pivot_ntype else ei[0][valid]
        new_dst = id_map[ei[1][valid]] if dst_type == pivot_ntype else ei[1][valid]
        g_sub[src_type, rel, dst_type].edge_index = torch.stack([new_src, new_dst])

    return g_sub


def _fix_masks(g_snap: HeteroData, target_ntype: str) -> None:
    """Recreate 80/10/10 masks if the sliced subgraph has no valid val entries."""
    labels  = g_snap[target_ntype].y
    # Multi-label: labeled = row has at least one active class; single-label: label >= 0
    valid   = (labels.sum(dim=1) > 0) if labels.dim() == 2 else (labels >= 0)
    val_mask = getattr(g_snap[target_ntype], "val_mask",
                       torch.zeros(labels.size(0), dtype=torch.bool))
    if valid.sum() > 0 and (val_mask & valid).sum() == 0:
        idx  = valid.nonzero(as_tuple=False).squeeze(1)
        perm = idx[torch.randperm(idx.size(0))]
        n    = idx.size(0)
        n_tr, n_va = int(0.8 * n), int(0.1 * n)
        tr = torch.zeros(labels.size(0), dtype=torch.bool)
        va = torch.zeros(labels.size(0), dtype=torch.bool)
        te = torch.zeros(labels.size(0), dtype=torch.bool)
        tr[perm[:n_tr]]            = True
        va[perm[n_tr:n_tr + n_va]] = True
        te[perm[n_tr + n_va:]]     = True
        g_snap[target_ntype].train_mask = tr
        g_snap[target_ntype].val_mask   = va
        g_snap[target_ntype].test_mask  = te


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    from torchmetrics.functional import f1_score
    if labels.dim() == 2:  # multi-label
        valid = mask & (labels.sum(dim=1) > 0)
        if valid.sum() == 0:
            return 0.0
        preds = (logits[valid] > 0).long()
        return f1_score(
            preds, labels[valid].long(),
            task="multilabel", num_labels=labels.size(1), average="macro",
        ).item()
    valid = mask & (labels >= 0)
    if valid.sum() == 0:
        return 0.0
    return f1_score(
        logits[valid].argmax(1), labels[valid],
        task="multiclass", num_classes=logits.size(1), average="macro",
    ).item()


def _train(
    g_homo: Data,
    in_dim: int,
    num_classes: int,
    num_layers: int,
    epochs: int,
    device: torch.device,
    log: logging.Logger,
    arch: str = "SAGE",
    gat_heads: int = 8,
    no_early_stop: bool = False,
) -> Tuple[torch.nn.Module, List[dict], int]:
    """Train ``arch`` with early stopping (disabled if no_early_stop=True).
    Returns (model, history, conv_epoch)."""
    model = get_model(arch, in_dim, num_classes, config.HIDDEN_DIM,
                      gat_heads=gat_heads, num_layers=num_layers).to(device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    x          = g_homo.x.to(device)
    edge_index = g_homo.edge_index.to(device)
    labels     = g_homo.y.to(device)
    is_multilabel = labels.dim() == 2

    if is_multilabel:
        # Restrict masks to rows that have at least one active label
        labeled     = labels.sum(dim=1) > 0
        train_mask  = (g_homo.train_mask.to(device) & labeled)
        val_mask    = (g_homo.val_mask.to(device)   & labeled)
    else:
        train_mask  = (g_homo.train_mask & (g_homo.y >= 0)).to(device)
        val_mask    = (g_homo.val_mask   & (g_homo.y >= 0)).to(device)

    best_val_f1  = -1.0
    best_val_loss = float("inf")
    best_state   = None
    conv_epoch   = 1
    wait         = 0
    patience     = 30
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out  = model(x, edge_index)
        if is_multilabel:
            loss = F.binary_cross_entropy_with_logits(out[train_mask], labels[train_mask])
        else:
            loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_e    = model(x, edge_index)
            if is_multilabel:
                val_loss = (F.binary_cross_entropy_with_logits(
                                out_e[val_mask], labels[val_mask]).item()
                            if val_mask.sum() > 0 else float("nan"))
            else:
                val_loss = (F.cross_entropy(out_e[val_mask], labels[val_mask]).item()
                            if val_mask.sum() > 0 else float("nan"))
            train_f1 = _f1(out_e, labels, train_mask)
            val_f1   = _f1(out_e, labels, val_mask)

        history.append({
            "epoch":      epoch,
            "train_loss": round(loss.item(), 6),
            "val_loss":   round(val_loss, 6) if val_loss == val_loss else "",
            "train_f1":   round(train_f1, 6),
            "val_f1":     round(val_f1, 6),
        })

        if epoch % 10 == 0:
            log.debug("  [L=%d ep %3d] loss=%.4f val=%.4f train_f1=%.4f val_f1=%.4f",
                      num_layers, epoch, loss.item(), val_loss, train_f1, val_f1)

        if val_mask.sum() > 0:
            # Save checkpoint at best val F1 (what we care about)
            if val_f1 > best_val_f1 + 1e-4:
                best_val_f1 = val_f1
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                conv_epoch  = epoch
            # Early-stop on val_loss (reliable signal training is still useful)
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience and not no_early_stop:
                    log.info("  [L=%d] Early stop at ep %d", num_layers, epoch)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, history, conv_epoch


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_LOG_FIELDS = [
    "dataset", "metapath", "L", "arch",
    "epoch", "train_loss", "val_loss", "train_f1", "val_f1",
    "convergence_epoch", "t_train", "weights_path", "note",
]


def _open_log(path: Path) -> Tuple:
    """Open the training log CSV with backward-compat for legacy schema.

    Legacy logs lack the 'arch' column (SAGE-only era).  If detected, rewrite
    the file in place defaulting old rows to arch='SAGE' so resume logic
    works correctly under the new arch-aware scheme.
    """
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as _f:
            existing_fields = csv.DictReader(_f).fieldnames or []
        existing_set = set(existing_fields)
        current_set  = set(_LOG_FIELDS)
        if existing_set != current_set and existing_set.issubset(current_set):
            # Additive change — rewrite in place defaulting arch=SAGE.
            with open(path, newline="", encoding="utf-8") as _f:
                rows = list(csv.DictReader(_f))
            with open(path, "w", newline="", encoding="utf-8") as _f:
                _w = csv.DictWriter(_f, fieldnames=_LOG_FIELDS)
                _w.writeheader()
                for r in rows:
                    if "arch" not in r or not r.get("arch"):
                        r["arch"] = "SAGE"
                    _w.writerow({fld: r.get(fld, "") for fld in _LOG_FIELDS})
        is_new = False
    else:
        is_new = True

    fh     = open(path, "a", newline="", encoding="utf-8")
    w      = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
    if is_new:
        w.writeheader()
    return fh, w


def _done_runs(log_path: Path) -> set:
    """Return set of (metapath, L, arch) tuples that have a DONE row.

    Legacy DONE rows without the 'arch' column default to 'SAGE' so a
    pre-existing SAGE training does NOT block GCN/GAT for the same (mp, L)."""
    done = set()
    if not log_path.exists():
        return done
    with open(log_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("note", "").startswith("DONE"):
                arch = (row.get("arch") or "SAGE").upper()
                done.add((row["metapath"], row["L"], arch))
    return done


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
    parser.add_argument("--depth", type=int, nargs="+", default=[2, 3, 4],
                        help="SAGE depth(s) to train, e.g. --depth 2 3 4")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--partition-json", required=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for weight init + training (default 42)")
    parser.add_argument("--arch", type=str, default="SAGE",
                        choices=["SAGE", "GCN", "GAT", "GIN"],
                        help="GNN architecture (default SAGE — legacy default)")
    parser.add_argument("--gat-heads", type=int, default=8,
                        help="Attention heads for --arch GAT (default 8)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable patience-based early stopping; train full --epochs.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load partition + dataset
    # ------------------------------------------------------------------
    with open(args.partition_json, "rb") as fh_raw:
        partition_bytes = fh_raw.read()
    partition_hash = "sha256:" + hashlib.sha256(partition_bytes).hexdigest()
    part = json.loads(partition_bytes.decode("utf-8"))

    cfg          = config.get_dataset_config(args.dataset)
    folder       = config.get_folder_name(args.dataset)
    out_dir      = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir  = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    log_path     = out_dir / "training_log.csv"

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    log    = logging.getLogger("exp2")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    fh = logging.FileHandler(out_dir / f"run_exp2_{ts}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(ch)
    log.addHandler(fh)

    log.info("Exp2 | dataset=%s  metapath=%s  depth=%s  seed=%d  arch=%s",
             args.dataset, args.metapath, args.depth, args.seed, args.arch)
    log.info("       partition_hash=%s", partition_hash)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g_full, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    target_ntype = cfg.target_node

    # ------------------------------------------------------------------
    # Reconstruct V_train (once)
    # ------------------------------------------------------------------
    log.info("Reconstructing V_train (train_frac=%.0f%%)...", part["train_frac"] * 100)
    g_train = _make_train_subgraph(g_full, part, info["labels"], info["masks"], target_ntype)
    _fix_masks(g_train, target_ntype)
    del g_full
    gc.collect()
    log.info("  %s=%d nodes", target_ntype, g_train[target_ntype].num_nodes)

    # ------------------------------------------------------------------
    # Stage C++ files + run ExactD (once, reused across all L)
    # ------------------------------------------------------------------
    data_dir = config.get_staging_dir(args.dataset)
    os.makedirs(data_dir, exist_ok=True)
    setup_global_res_dirs(folder, project_root)
    engine   = CppEngine(executable_path=config.CPP_EXECUTABLE, data_dir=data_dir)

    PyGToCppAdapter(data_dir).convert(g_train)
    compile_rule_for_cpp(args.metapath, g_train, data_dir, folder)
    generate_qnodes(data_dir, folder, target_node_type=target_ntype, g_hetero=g_train)

    s_ntypes = sorted(g_train.node_types)
    offset   = sum(g_train[nt].num_nodes for nt in s_ntypes if nt < target_ntype)
    n_target = g_train[target_ntype].num_nodes

    log.info("Running ExactD on V_train...")
    rule_file   = os.path.join(data_dir, f"cod-rules_{folder}.limit")
    output_file = os.path.join(data_dir, "mat_exact.adj")
    engine.run_command("materialize", rule_file, output_file)

    g_h0             = engine.load_result(output_file, n_target, offset)
    g_h0.x           = g_train[target_ntype].x
    g_h0.y           = g_train[target_ntype].y
    g_h0.train_mask  = g_train[target_ntype].train_mask
    g_h0.val_mask    = g_train[target_ntype].val_mask
    g_h0.test_mask   = g_train[target_ntype].test_mask
    del g_train
    gc.collect()

    in_dim = g_h0.x.size(1)
    log.info("  H_train edges=%d  in_dim=%d",
             g_h0.edge_index.size(1) if g_h0.edge_index is not None else 0, in_dim)

    # ------------------------------------------------------------------
    # Train SAGE for each depth
    # ------------------------------------------------------------------
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp_safe   = args.metapath.replace(",", "_").replace("/", "_")
    done_runs = _done_runs(log_path)
    log_fh, log_w = _open_log(log_path)

    arch_key = args.arch.upper()
    try:
        for L in args.depth:
            if (args.metapath, str(L), arch_key) in done_runs:
                log.info("[L=%d arch=%s] already done — skipping", L, arch_key)
                continue

            log.info("\n[L=%d] Training %s(%d)...", L, args.arch, L)
            t0 = time.perf_counter()
            try:
                model, history, conv_epoch = _train(
                    g_h0, in_dim, info["num_classes"], L, args.epochs, device, log,
                    arch=args.arch, gat_heads=args.gat_heads,
                    no_early_stop=args.no_early_stop,
                )
            except RuntimeError as e:
                if ("CUDA" in str(e) or "out of memory" in str(e).lower()) \
                   and device.type == "cuda":
                    log.warning("  [L=%d] GPU OOM — retrying on CPU", L)
                    torch.cuda.empty_cache()
                    device = torch.device("cpu")
                    model, history, conv_epoch = _train(
                        g_h0, in_dim, info["num_classes"], L, args.epochs, device, log,
                        arch=args.arch, gat_heads=args.gat_heads,
                    )
                else:
                    raise
            t_train = time.perf_counter() - t0

            # Arch-specific weight filename so SAGE/GCN/GAT runs don't clobber
            # one another.  SAGE retains the legacy `<mp>_L<L>.pt` name to
            # remain backward-compatible with the existing weights/results.
            if args.arch.upper() == "SAGE":
                weights_path = str(weights_dir / f"{mp_safe}_L{L}.pt")
            else:
                weights_path = str(weights_dir / f"{mp_safe}_L{L}_{args.arch.upper()}.pt")
            torch.save(model.state_dict(), weights_path)

            # Sidecar metadata JSON for exp3 to validate partition_hash + arch
            meta_path = weights_path.replace(".pt", ".meta.json")
            with open(meta_path, "w", encoding="utf-8") as mfh:
                json.dump({
                    "spec_version":      "approach_a_2026_05_05",
                    "dataset":           args.dataset,
                    "metapath":          args.metapath,
                    "arch":              args.arch.upper(),
                    "n_layers":          L,
                    "hidden_dim":        config.HIDDEN_DIM,
                    "gat_heads":         args.gat_heads if args.arch.upper() == "GAT" else None,
                    "seed":              args.seed,
                    "partition_hash":    partition_hash,
                    "convergence_epoch": conv_epoch,
                    "training_time_s":   round(t_train, 4),
                    "timestamp":         datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }, mfh, indent=2)

            log.info("  Saved θ* → %s  (conv_epoch=%d  t=%.2fs)",
                     weights_path, conv_epoch, t_train)

            for h in history:
                log_w.writerow({
                    "dataset": args.dataset, "metapath": args.metapath, "L": L,
                    "arch": arch_key,
                    "epoch": h["epoch"], "train_loss": h["train_loss"],
                    "val_loss": h["val_loss"], "train_f1": h["train_f1"],
                    "val_f1": h.get("val_f1", ""),
                    "convergence_epoch": "", "t_train": "", "weights_path": "", "note": "",
                })
            done_h = history[conv_epoch - 1] if history else {}
            log_w.writerow({
                "dataset": args.dataset, "metapath": args.metapath, "L": L,
                "arch": arch_key,
                "epoch": "DONE", "train_loss": "", "val_loss": "",
                "train_f1": done_h.get("train_f1", ""),
                "val_f1":   done_h.get("val_f1", ""),
                "convergence_epoch": conv_epoch,
                "t_train": round(t_train, 4),
                "weights_path": weights_path,
                "note": "DONE",
            })
            log_fh.flush()

            del model
            gc.collect()
    finally:
        log_fh.close()

    log.info("\nDone. Logs → %s  Weights → %s", log_path, weights_dir)


if __name__ == "__main__":
    main()
