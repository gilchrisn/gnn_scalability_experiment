"""
Experiment 2 — Train SAGE(L) on V_train.

Reads partition.json (from exp1_inductive_split.py), reconstructs the V_train
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
    python scripts/exp1_inductive_split.py HGB_DBLP --train-frac 0.1
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
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

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
    Build the training subgraph by filtering ONLY the pivot node type to
    train_frac.  Every other node type is kept at 100% — authors, subjects,
    institutions etc. are not pruned just because some of their papers are
    held out.  Only edges whose pivot-type endpoint falls outside V_train
    are dropped.
    """
    pivot_ntype    = part["pivot_ntype"]
    partition_mode = part["partition_mode"]
    train_frac     = part["train_frac"]

    # Determine which pivot nodes to keep (sorted for deterministic remapping)
    if partition_mode == "temporal_year":
        years = g[pivot_ntype].year.squeeze()
        sorted_years, _ = years.sort()
        cutoff   = sorted_years[max(1, int(train_frac * len(sorted_years))) - 1].item()
        keep_ids = (years <= cutoff).nonzero(as_tuple=False).squeeze(1)
    else:
        gen = torch.Generator()
        gen.manual_seed(part["seed"])
        perm     = torch.randperm(g[pivot_ntype].num_nodes, generator=gen)
        keep_ids = perm[:max(1, int(train_frac * g[pivot_ntype].num_nodes))]

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
    g_sub[pivot_ntype].num_nodes = keep_ids.size(0)
    for key, val in g[pivot_ntype].items():
        if isinstance(val, torch.Tensor) and val.size(0) == g[pivot_ntype].num_nodes:
            g_sub[pivot_ntype][key] = val[keep_ids]
        else:
            g_sub[pivot_ntype][key] = val

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
    valid   = labels >= 0
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
) -> Tuple[torch.nn.Module, List[dict], int]:
    """Train SAGE(num_layers) with early stopping. Returns (model, history, conv_epoch)."""
    model = get_model("SAGE", in_dim, num_classes, config.HIDDEN_DIM,
                      num_layers=num_layers).to(device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    x          = g_homo.x.to(device)
    edge_index = g_homo.edge_index.to(device)
    labels     = g_homo.y.to(device)
    train_mask = (g_homo.train_mask & (g_homo.y >= 0)).to(device)
    val_mask   = (g_homo.val_mask   & (g_homo.y >= 0)).to(device)

    best_val   = float("inf")
    best_state = None
    conv_epoch = 1
    wait       = 0
    patience   = 15
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out  = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_e    = model(x, edge_index)
            val_loss = (F.cross_entropy(out_e[val_mask], labels[val_mask]).item()
                        if val_mask.sum() > 0 else float("nan"))
            train_f1 = _f1(out_e, labels, train_mask)

        history.append({
            "epoch":      epoch,
            "train_loss": round(loss.item(), 6),
            "val_loss":   round(val_loss, 6) if val_loss == val_loss else "",
            "train_f1":   round(train_f1, 6),
        })

        if epoch % 10 == 0:
            log.debug("  [L=%d ep %3d] loss=%.4f val=%.4f f1=%.4f",
                      num_layers, epoch, loss.item(), val_loss, train_f1)

        if val_mask.sum() > 0:
            if val_loss < best_val - 1e-4:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                conv_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= patience:
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
    "dataset", "metapath", "L",
    "epoch", "train_loss", "val_loss", "train_f1",
    "convergence_epoch", "t_train", "weights_path", "note",
]


def _open_log(path: Path) -> Tuple:
    is_new = not path.exists() or path.stat().st_size == 0
    fh     = open(path, "a", newline="", encoding="utf-8")
    w      = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
    if is_new:
        w.writeheader()
    return fh, w


def _done_runs(log_path: Path) -> set:
    done = set()
    if not log_path.exists():
        return done
    with open(log_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("note", "").startswith("DONE"):
                done.add((row["metapath"], row["L"]))
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
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load partition + dataset
    # ------------------------------------------------------------------
    with open(args.partition_json) as f:
        part = json.load(f)

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

    log.info("Exp2 | dataset=%s  metapath=%s  depth=%s  seed=%d",
             args.dataset, args.metapath, args.depth, args.seed)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

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
    data_dir = os.path.join(project_root, folder)
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

    try:
        for L in args.depth:
            if (args.metapath, str(L)) in done_runs:
                log.info("[L=%d] already done — skipping", L)
                continue

            log.info("\n[L=%d] Training SAGE(%d)...", L, L)
            t0 = time.perf_counter()
            model, history, conv_epoch = _train(
                g_h0, in_dim, info["num_classes"], L, args.epochs, device, log
            )
            t_train = time.perf_counter() - t0

            weights_path = str(weights_dir / f"{mp_safe}_L{L}.pt")
            torch.save(model.state_dict(), weights_path)
            log.info("  Saved θ* → %s  (conv_epoch=%d  t=%.2fs)",
                     weights_path, conv_epoch, t_train)

            for h in history:
                log_w.writerow({
                    "dataset": args.dataset, "metapath": args.metapath, "L": L,
                    "epoch": h["epoch"], "train_loss": h["train_loss"],
                    "val_loss": h["val_loss"], "train_f1": h["train_f1"],
                    "convergence_epoch": "", "t_train": "", "weights_path": "", "note": "",
                })
            log_w.writerow({
                "dataset": args.dataset, "metapath": args.metapath, "L": L,
                "epoch": "DONE", "train_loss": "", "val_loss": "",
                "train_f1": history[-1]["train_f1"] if history else "",
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
