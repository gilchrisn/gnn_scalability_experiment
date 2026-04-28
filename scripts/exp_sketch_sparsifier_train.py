"""Sketch-as-sparsifier consumer mode for Path 1.

Same KMV propagation as exp_sketch_feature_train.py, different consumption:
decode each bottom-k slot into a (target -> decoded_target) edge, union
those edges across meta-paths into one homogeneous adjacency, and run
plain GraphSAGE on it for node classification.

This is the "Algorithm 2" variant from earlier paper drafts — the sketch
gives us an approximate meta-path-induced adjacency Ã, and message
passing happens on Ã with native target-type features as input. There's
no per-T_0 entity embedding lookup here; the sketch is consumed entirely
as edge structure, not as features.

Together with exp_sketch_feature_train.py --backbone mlp this script
covers the two consumer modes promised in CURRENT_STATE.md §"What's IN
scope" #2.

Usage
-----
    python scripts/exp_sketch_sparsifier_train.py HGB_DBLP --k 32 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, coalesce

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.models import get_model
from src.sketch_feature import (
    SketchBundle,
    decode_sketches,
    extract_sketches,
)


# Reuse the meta-path defaults from exp_sketch_feature_train so both
# consumer modes are evaluated on the same propagation pass.
from scripts.exp_sketch_feature_train import _DEFAULT_META_PATHS  # noqa: E402


def _macro_f1(y_pred: torch.Tensor, y_true: torch.Tensor, n_classes: int) -> float:
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    f1s = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            continue
        f1s.append(2 * prec * rec / (prec + rec))
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def _macro_f1_multilabel(logits: torch.Tensor, y_true: torch.Tensor,
                         threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) >= threshold).int().cpu().numpy()
    yt = y_true.int().cpu().numpy()
    n_classes = pred.shape[1]
    f1s = []
    for c in range(n_classes):
        tp = ((pred[:, c] == 1) & (yt[:, c] == 1)).sum()
        fp = ((pred[:, c] == 1) & (yt[:, c] == 0)).sum()
        fn = ((pred[:, c] == 0) & (yt[:, c] == 1)).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            continue
        f1s.append(2 * prec * rec / (prec + rec))
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def _build_edges_from_decoded(
    decoded: torch.Tensor, n_target: int, add_self_loops_flag: bool = True
) -> torch.Tensor:
    """Same construction as exp_sketch_feature_train but isolated here so
    the sparsifier script doesn't import the feature script (fewer
    cross-script dependencies)."""
    n, k = decoded.shape
    src = torch.arange(n).unsqueeze(1).expand(-1, k)
    dst = decoded
    valid = dst >= 0
    src = src[valid]
    dst = dst[valid]
    ei = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ]).long()
    if add_self_loops_flag:
        ei, _ = add_self_loops(ei, num_nodes=n_target)
    ei = coalesce(ei, num_nodes=n_target)
    return ei


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--meta-paths", nargs="+")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--depth", type=int, default=2,
                   help="Number of SAGEConv layers")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patience", type=int, default=30)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    cfg = config.get_dataset_config(args.dataset)
    target_type = cfg.target_node
    out_dir = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = args.meta_paths or _DEFAULT_META_PATHS.get(args.dataset)
    if not meta_paths:
        print(f"No default meta-paths for {args.dataset}", file=sys.stderr)
        return 2

    print(f"[load] dataset={args.dataset} target={target_type} seed={args.seed}")
    g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, target_type)
    n_target = g[target_type].num_nodes
    n_classes = int(info["num_classes"])
    labels = info["labels"]
    masks = info["masks"]
    train_mask = masks["train"].bool()
    val_mask = masks["val"].bool()
    test_mask = masks["test"].bool()

    multi_label = labels.dim() == 2 and labels.dtype in (torch.float32, torch.float64)
    if not multi_label:
        valid = labels != -100
        train_mask &= valid
        val_mask &= valid
        test_mask &= valid

    # Cached sketch bundle (compatible with exp_sketch_feature_train.py).
    cache_path = out_dir / f"sketch_bundle_k{args.k}_seed{args.seed}.pt"
    if cache_path.exists():
        bundle = SketchBundle.load(cache_path)
        if bundle.k != args.k or sorted(bundle.meta_paths) != sorted(meta_paths):
            cache_path.unlink()
            bundle = None
    else:
        bundle = None

    extract_time_s = 0.0
    if bundle is None:
        t0 = time.perf_counter()
        bundle = extract_sketches(
            g, meta_paths, target_type=target_type,
            k=args.k, seed=args.seed, device="cpu",
        )
        extract_time_s = time.perf_counter() - t0
        bundle.save(cache_path)
        print(f"[extract] propagated in {extract_time_s:.2f}s")
    else:
        print(f"[extract] reusing cached bundle: {cache_path}")

    # Union all per-meta-path decoded edges into a single homogeneous
    # adjacency. This is the "sketch-as-sparsifier" — message passing
    # happens on the union, not on the original heterogeneous graph.
    union_src = []
    union_dst = []
    edge_counts = {}
    for mp in meta_paths:
        decoded = decode_sketches(
            bundle.sketches_by_mp[mp], bundle.sorted_hashes, bundle.sort_indices
        )
        ei = _build_edges_from_decoded(decoded, n_target, add_self_loops_flag=False)
        edge_counts[mp] = int(ei.size(1))
        union_src.append(ei[0])
        union_dst.append(ei[1])
    full_ei = torch.stack([torch.cat(union_src), torch.cat(union_dst)])
    full_ei, _ = add_self_loops(full_ei, num_nodes=n_target)
    full_ei = coalesce(full_ei, num_nodes=n_target)
    n_edges_union = int(full_ei.size(1))
    print(f"[edges] per-mp counts: {edge_counts}")
    print(f"[edges] union after dedup + self-loops: {n_edges_union}")

    # Native target features as input to SAGE.
    x = g[target_type].x
    if x is None:
        print(f"[features] target {target_type!r} has no native features; "
              f"falling back to identity matrix (one-hot).")
        x = torch.eye(n_target)
    in_dim = int(x.size(1))
    print(f"[features] in_dim={in_dim}")

    # PyG SAGE (from src.models).
    out_dim = n_classes if not multi_label else int(labels.size(1))
    model = get_model("SAGE", in_dim, out_dim, args.hidden_dim,
                      num_layers=args.depth)

    device = torch.device(args.device)
    model = model.to(device)
    x = x.to(device)
    full_ei = full_ei.to(device)
    labels_d = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    wait = 0

    print(f"[train] depth={args.depth} hidden={args.hidden_dim} "
          f"lr={args.lr} dropout={args.dropout}")
    train_t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, full_ei)
        if multi_label:
            loss = F.binary_cross_entropy_with_logits(
                logits[train_mask], labels_d[train_mask].float()
            )
        else:
            loss = F.cross_entropy(logits[train_mask], labels_d[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, full_ei)
            if multi_label:
                val_f1 = _macro_f1_multilabel(logits[val_mask], labels_d[val_mask])
            else:
                pred = logits.argmax(dim=-1)
                val_f1 = _macro_f1(pred[val_mask], labels_d[val_mask], n_classes)

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"  ep {epoch:3d}  loss={loss.item():.4f}  val_f1={val_f1:.4f}  "
                  f"(best={best_val_f1:.4f}@{best_epoch})")
        if wait >= args.patience:
            print(f"  early stop at ep {epoch}")
            break

    train_time_s = time.perf_counter() - train_t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, full_ei)
        if multi_label:
            test_f1 = _macro_f1_multilabel(logits[test_mask], labels_d[test_mask])
        else:
            pred = logits.argmax(dim=-1)
            test_f1 = _macro_f1(pred[test_mask], labels_d[test_mask], n_classes)
    print(f"[result] val_f1={best_val_f1:.4f} (ep {best_epoch})  "
          f"test_f1={test_f1:.4f}  train_time={train_time_s:.1f}s "
          f"extract_time={extract_time_s:.2f}s")

    out = {
        "dataset": args.dataset,
        "target_type": target_type,
        "method": "sketch_sparsifier_SAGE",
        "meta_paths": meta_paths,
        "k": args.k,
        "depth": args.depth,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "edges_per_mp": edge_counts,
        "edges_union": n_edges_union,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "test_f1": test_f1,
        "train_time_s": train_time_s,
        "extract_time_s": extract_time_s,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    res_path = out_dir / f"sketch_sparsifier_pilot_k{args.k}_seed{args.seed}.json"
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[save] {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
