"""Multi-task Simple-HGN baseline — fair multi-query comparison.

The single-task ``exp_simple_hgn_baseline.py`` script trains one Simple-HGN
encoder + classifier per task. Comparing KMV's "one sketch, multiple
queries" against "Q independent SHGN trainings" is unfair, because a
SHGN deployment serving Q queries would in practice train ONE encoder
with Q heads (multi-task learning). This script implements that fair
baseline.

Tasks supported in this version: **NC + LP**, jointly trained.
- Shared encoder: same Simple-HGN port from ``exp_simple_hgn_baseline.py``.
- NC head: per-class softmax / BCE-with-logits.
- LP head: dot product on encoder outputs.
- Loss: weighted sum of NC and LP losses, jointly optimised.

What we measure
---------------
- Single end-to-end wall-clock to train both heads on a shared encoder.
- Test-time NC F1 and LP MRR, comparable to the per-task numbers.
- Compare against KMV {sketch precompute + per-task consume × 2}.

Usage
-----
    python scripts/exp_simple_hgn_multitask.py HGB_DBLP \\
        --partition-json results/HGB_DBLP/partition.json \\
        --num-seeds 3 --epochs 100
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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.sketch_feature import macro_f1, macro_f1_multilabel
from scripts.exp_simple_hgn_baseline import SimpleHGN
from scripts.exp_lp_train import _build_adj_dict, _sample_neg_2hop
from scripts.exp_sketch_lp_train import _ranking_metrics


def _make_lp_train_test_edges(g, target: str, partition: dict, rng,
                              max_test_edges: int = 2000):
    """Build LP train / test edge sets from the heterogeneous graph.

    Computes the canonical 2-hop meta-path adjacency (target -> X -> target)
    via sparse matrix multiplication: A @ B where A is the target->X
    edge_index and B is the X->target edge_index, both as scipy CSR matrices.
    The previous implementation used a triple-nested Python loop and was
    intractable on PubMed/ACM/IMDB; sparse matmul handles those in seconds.
    """
    from scipy.sparse import csr_matrix

    # Pick a canonical 2-hop meta-path: T_0 -> X -> T_0 where X is the
    # most-connected adjacent type via target_to_X edges.
    out_rels = [(et, g[et].edge_index) for et in g.edge_types
                if et[0] == target and et[2] != target]
    if not out_rels:
        raise RuntimeError(f"No outgoing edges from {target} found")
    et_out, ei_out = max(out_rels, key=lambda r: r[1].size(1))
    intermediate = et_out[2]
    in_rels = [(et, g[et].edge_index) for et in g.edge_types
               if et[0] == intermediate and et[2] == target]
    if not in_rels:
        raise RuntimeError(f"No return edges {intermediate}->{target}")
    et_in, ei_in = max(in_rels, key=lambda r: r[1].size(1))
    print(f"[lp_edges] using {et_out} -> {et_in} as canonical meta-path adjacency")

    n_target = g[target].num_nodes
    n_inter = g[intermediate].num_nodes

    # Sparse A (n_target × n_inter), B (n_inter × n_target). C = A @ B is
    # the meta-path adjacency. Boolean dtype so multiplicities don't blow
    # up RAM on dense graphs.
    a_np = ei_out.cpu().numpy()
    b_np = ei_in.cpu().numpy()
    A = csr_matrix(
        (np.ones(a_np.shape[1], dtype=bool), (a_np[0], a_np[1])),
        shape=(n_target, n_inter),
    )
    B = csr_matrix(
        (np.ones(b_np.shape[1], dtype=bool), (b_np[0], b_np[1])),
        shape=(n_inter, n_target),
    )
    C = (A @ B).tocoo()

    # Keep upper triangle only (u < v) to dedup undirected pairs and drop
    # self-loops in one step.
    keep = C.row < C.col
    src_np = C.row[keep].astype(np.int64)
    dst_np = C.col[keep].astype(np.int64)
    full_ei = torch.from_numpy(np.stack([src_np, dst_np]))

    # Vectorised train/test mask via np.isin against the partition arrays.
    train_arr = np.asarray(partition["train_node_ids"], dtype=np.int64)
    test_arr  = np.asarray(partition["test_node_ids"],  dtype=np.int64)
    in_train = np.isin(src_np, train_arr) & np.isin(dst_np, train_arr)
    in_test  = np.isin(src_np, test_arr)  | np.isin(dst_np, test_arr)

    pos_train = full_ei[:, torch.from_numpy(in_train)]
    pos_test  = full_ei[:, torch.from_numpy(in_test)]
    if pos_test.size(1) > max_test_edges:
        idx = rng.choice(pos_test.size(1), size=max_test_edges, replace=False)
        pos_test = pos_test[:, torch.from_numpy(idx).long()]

    adj = _build_adj_dict(full_ei, n_target)
    return pos_train, pos_test, adj


def run_one_seed(args, seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = config.get_dataset_config(args.dataset)
    target = cfg.target_node

    print(f"\n=== {args.dataset} seed={seed} multi-task NC+LP ===")
    g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, target)
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

    in_dim_by_type = {nt: int(g[nt].x.size(1))
                      for nt in g.node_types
                      if "x" in g[nt] and g[nt].x is not None}
    n_by_type = {nt: g[nt].num_nodes for nt in g.node_types}
    x_dict = {nt: g[nt].x for nt in g.node_types if nt in in_dim_by_type}
    edge_index_dict = {et: g[et].edge_index for et in g.edge_types}

    device = torch.device(args.device)
    encoder = SimpleHGN(
        in_dim_by_type=in_dim_by_type,
        hidden_dim=args.hidden_dim,
        out_dim=args.hidden_dim,  # we want embeddings, not logits
        meta_node_types=list(g.node_types),
        meta_edge_types=list(g.edge_types),
        target_type=target,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    # NC head: linear classifier on top.
    nc_head = nn.Linear(args.hidden_dim, n_classes).to(device)
    # LP head: implicit (dot product on normalised embeddings).
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    labels_d = labels.to(device)
    train_mask_d = train_mask.to(device)
    val_mask_d = val_mask.to(device)
    test_mask_d = test_mask.to(device)

    # LP edges (loaded from partition).
    if args.partition_json:
        with open(args.partition_json) as f:
            partition = json.load(f)
        pos_train, pos_test_lp, adj_lp = _make_lp_train_test_edges(
            g, target, partition, rng, max_test_edges=args.max_test_edges)
        n_pos_train = pos_train.size(1)
        n_val = max(1, int(0.1 * n_pos_train))
        perm = rng.permutation(n_pos_train)
        pos_val_lp = pos_train[:, torch.from_numpy(perm[:n_val]).long()]
        pos_train_lp = pos_train[:, torch.from_numpy(perm[n_val:]).long()]
        print(f"[lp_edges] train={pos_train_lp.size(1)} val={pos_val_lp.size(1)} test={pos_test_lp.size(1)}")
    else:
        print("[skip] no partition.json provided; LP head won't be trained")
        pos_train_lp = pos_test_lp = pos_val_lp = None
        adj_lp = None

    params = list(encoder.parameters()) + list(nc_head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val_combined = -1.0
    best_state = None
    best_epoch = 0
    wait = 0

    print(f"[train] epochs={args.epochs} lr={args.lr} dropout={args.dropout}")
    train_t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        nc_head.train()
        opt.zero_grad()
        h = encoder(x_dict, edge_index_dict, n_by_type)  # [N_target, hidden_dim]

        # NC loss.
        nc_logits = nc_head(h)
        if multi_label:
            nc_loss = F.binary_cross_entropy_with_logits(
                nc_logits[train_mask_d], labels_d[train_mask_d].float())
        else:
            nc_loss = F.cross_entropy(nc_logits[train_mask_d], labels_d[train_mask_d])

        # LP loss (jointly with NC).
        lp_loss = torch.tensor(0.0, device=device)
        if pos_train_lp is not None:
            if pos_train_lp.size(1) > args.max_pos_per_epoch:
                idx = rng.choice(pos_train_lp.size(1),
                                 size=args.max_pos_per_epoch, replace=False)
                pos_ep = pos_train_lp[:, torch.from_numpy(idx).long()]
            else:
                pos_ep = pos_train_lp
            neg_ep = _sample_neg_2hop(pos_ep, adj_lp, h.size(0), rng)
            pos_ep = pos_ep.to(device)
            neg_ep = neg_ep.to(device)
            h_norm = F.normalize(h, dim=-1)
            s_pos = (h_norm[pos_ep[0]] * h_norm[pos_ep[1]]).sum(-1)
            s_neg = (h_norm[neg_ep[0]] * h_norm[neg_ep[1]]).sum(-1)
            lp_loss = F.binary_cross_entropy_with_logits(
                torch.cat([s_pos, s_neg]),
                torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)]))

        loss = nc_loss + args.lp_weight * lp_loss
        loss.backward()
        opt.step()

        # Eval.
        encoder.eval()
        nc_head.eval()
        with torch.no_grad():
            h = encoder(x_dict, edge_index_dict, n_by_type)
            nc_logits = nc_head(h)
            if multi_label:
                val_nc_f1 = macro_f1_multilabel(nc_logits[val_mask_d],
                                                labels_d[val_mask_d])
            else:
                pred = nc_logits.argmax(dim=-1)
                val_nc_f1 = macro_f1(pred[val_mask_d], labels_d[val_mask_d], n_classes)

            val_lp_mrr = float("nan")
            if pos_val_lp is not None:
                h_norm = F.normalize(h, dim=-1).cpu()
                val_metrics = _ranking_metrics(
                    h_norm, pos_val_lp, adj_lp, rng, n_negs_per_pos=20)
                val_lp_mrr = val_metrics["MRR"]

        combined = val_nc_f1 + (val_lp_mrr if val_lp_mrr == val_lp_mrr else 0.0)
        if combined > best_val_combined + 1e-4:
            best_val_combined = combined
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                "nc_head": {k: v.detach().clone() for k, v in nc_head.state_dict().items()},
            }
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"  ep {epoch:3d}  loss={loss.item():.4f}  "
                  f"NC_loss={nc_loss.item():.4f}  LP_loss={lp_loss.item():.4f}  "
                  f"val_NC={val_nc_f1:.4f}  val_LP_MRR={val_lp_mrr:.4f}")
        if wait >= args.patience:
            print(f"  early stop at ep {epoch}")
            break

    train_time_s = time.perf_counter() - train_t0
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        nc_head.load_state_dict(best_state["nc_head"])
    encoder.eval()
    nc_head.eval()
    with torch.no_grad():
        h = encoder(x_dict, edge_index_dict, n_by_type)
        nc_logits = nc_head(h)
        if multi_label:
            test_nc_f1 = macro_f1_multilabel(nc_logits[test_mask_d],
                                             labels_d[test_mask_d])
        else:
            pred = nc_logits.argmax(dim=-1)
            test_nc_f1 = macro_f1(pred[test_mask_d], labels_d[test_mask_d], n_classes)
        test_lp_metrics = None
        if pos_test_lp is not None:
            h_norm = F.normalize(h, dim=-1).cpu()
            test_lp_metrics = _ranking_metrics(
                h_norm, pos_test_lp, adj_lp, rng,
                n_negs_per_pos=args.n_negs_per_pos)

    print(f"[result] seed={seed}  test_NC_F1={test_nc_f1:.4f}  "
          f"test_LP={test_lp_metrics}  train_time={train_time_s:.1f}s")

    out = {
        "dataset": args.dataset,
        "method": "simple_hgn_multitask_nc_lp",
        "target": target,
        "seed": seed,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lp_weight": args.lp_weight,
        "best_val_combined": best_val_combined,
        "best_epoch": best_epoch,
        "test_nc_f1": test_nc_f1,
        "test_lp_metrics": test_lp_metrics,
        "train_time_s": train_time_s,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    res_dir = Path("results") / args.dataset
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / f"simple_hgn_multitask_seed{seed}.json"
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--partition-json", required=True,
                   help="Path to partition.json from exp1_partition.py.")
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lp-weight", type=float, default=1.0,
                   help="Weight on LP loss when summing with NC loss.")
    p.add_argument("--max-pos-per-epoch", type=int, default=4000)
    p.add_argument("--max-test-edges", type=int, default=2000)
    p.add_argument("--n-negs-per-pos", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.num_seeds = 1
        args.epochs = 30

    seeds = [args.seed_base + i for i in range(args.num_seeds)]
    results = [run_one_seed(args, s) for s in seeds]
    test_nc = [r["test_nc_f1"] for r in results]
    times = [r["train_time_s"] for r in results]
    print(f"\n=== Summary | {args.dataset} | n={len(results)} seeds ===")
    print(f"  test_NC_F1   = {statistics.fmean(test_nc):.4f} ± "
          f"{statistics.pstdev(test_nc) if len(test_nc) > 1 else 0.0:.4f}")
    if results[0].get("test_lp_metrics"):
        test_mrr = [r["test_lp_metrics"]["MRR"] for r in results
                    if r.get("test_lp_metrics")]
        if test_mrr:
            print(f"  test_LP_MRR  = {statistics.fmean(test_mrr):.4f} ± "
                  f"{statistics.pstdev(test_mrr) if len(test_mrr) > 1 else 0.0:.4f}")
    print(f"  train_time   = {statistics.fmean(times):.1f}s ± "
          f"{statistics.pstdev(times) if len(times) > 1 else 0.0:.1f}s "
          f"(JOINT NC+LP training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
