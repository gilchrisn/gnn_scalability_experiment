"""
Experiment LP-Train — Train SAGE(L) encoder + dot-product decoder for
link prediction on V_train subgraph of the materialized meta-path graph H.

Parallel to exp2_train.py but for LP instead of NC. Reuses the V_train
subgraph construction, then uses all edges in H_train as positive examples.
Negative samples are drawn via 2-hop neighbor sampling (HGB protocol, Lv 2021).

Pipeline
--------
1. Load partition.json (from exp1_partition.py)
2. Build V_train subgraph (same as exp2)
3. Materialize H_train via C++ ExactD
4. 2-hop negative sampling: for each positive (u,v), sample v' ∈ N^H_2(u) \\ N^H_1(u)
5. Train SAGE encoder (output dim = embedding dim) with BCE loss on positive/negative scores
6. Save frozen weights

Output
------
  results/<dataset>/weights_lp/<metapath_safe>_L<L>.pt     frozen θ*
  results/<dataset>/lp_training_log.csv

Usage
-----
    python scripts/exp_lp_train.py HGB_DBLP \\
        --metapath author_to_paper,paper_to_author \\
        --depth 2 \\
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
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

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
from scripts.exp2_train import _make_train_subgraph


# ---------------------------------------------------------------------------
# 2-hop negative sampling
# ---------------------------------------------------------------------------

def _build_adj_dict(edge_index: torch.Tensor, n_nodes: int) -> list:
    """Dict-of-lists adjacency for fast 1-hop lookup. Undirected."""
    adj = [set() for _ in range(n_nodes)]
    ei = edge_index.numpy()
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _sample_neg_2hop(
    pos_edges: torch.Tensor,
    adj: list,
    n_nodes: int,
    rng: np.random.Generator,
    max_tries: int = 40,
    dense_threshold: float = 0.3,
) -> torch.Tensor:
    """For each positive edge (u, v), sample negative v' ∈ N^2(u) \\ N^1(u) \\ {u}.

    Fast path: if graph is dense (mean_deg/n_nodes > threshold), use rejection
    sampling from V \\ N^1(u) \\ {u} — on dense graphs the 2-hop reach essentially
    covers all of V \\ N^1(u), making explicit 2-hop enumeration redundant and
    O(deg^2) expensive.

    Slow path (sparse): explicitly compute 2-hop set union, sample from it.
    """
    src = pos_edges[0].numpy()
    n_pos = len(src)
    neg_dst = np.full(n_pos, -1, dtype=np.int64)

    mean_deg = np.mean([len(a) for a in adj])
    dense_mode = (mean_deg / max(n_nodes, 1)) > dense_threshold

    for i, u in enumerate(src):
        n1 = adj[u]
        if not n1:
            v_prime = int(rng.integers(n_nodes))
            while v_prime == u:
                v_prime = int(rng.integers(n_nodes))
            neg_dst[i] = v_prime
            continue

        if dense_mode:
            # Rejection sampling: v' uniform in V \ N1 \ {u}
            # On dense graphs, N2(u) ⊇ V \ N1(u) almost surely, so this is fine.
            for _ in range(max_tries):
                v_prime = int(rng.integers(n_nodes))
                if v_prime != u and v_prime not in n1:
                    neg_dst[i] = v_prime
                    break
            if neg_dst[i] == -1:
                # Give up, pick any v'
                neg_dst[i] = (int(u) + 1) % n_nodes
        else:
            # Sparse path: explicit 2-hop set
            n2 = set()
            for w in n1:
                n2 |= adj[w]
            n2 -= n1
            n2.discard(int(u))

            if not n2:
                for _ in range(max_tries):
                    v_prime = int(rng.integers(n_nodes))
                    if v_prime != u and v_prime not in n1:
                        neg_dst[i] = v_prime
                        break
                if neg_dst[i] == -1:
                    neg_dst[i] = int(rng.integers(n_nodes))
            else:
                n2_list = list(n2)
                neg_dst[i] = n2_list[int(rng.integers(len(n2_list)))]

    return torch.stack([torch.from_numpy(src).long(), torch.from_numpy(neg_dst).long()], dim=0)


# ---------------------------------------------------------------------------
# LP training loop
# ---------------------------------------------------------------------------

def _train_lp(
    g_homo: Data,
    in_dim: int,
    embedding_dim: int,
    num_layers: int,
    epochs: int,
    device: torch.device,
    rng: np.random.Generator,
    log: logging.Logger,
    max_pos_per_epoch: int = 10000,
) -> Tuple[torch.nn.Module, list, int]:
    """Train SAGE encoder for LP on H_train.

    Out-of-architecture choice: model's output dim is embedding_dim (not num_classes).
    Score = dot(z_u, z_v). Loss = BCE(sigmoid(score), label).

    Returns (model, history, convergence_epoch).
    """
    # SAGE with output = embedding_dim
    model = get_model("SAGE", in_dim, embedding_dim, config.HIDDEN_DIM,
                      num_layers=num_layers).to(device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    x          = g_homo.x.to(device)
    edge_index = g_homo.edge_index.to(device)
    n_nodes    = x.size(0)

    # Build undirected adj for negative sampling (on CPU, once)
    adj_dict = _build_adj_dict(g_homo.edge_index, n_nodes)
    log.info("  Built 2-hop adjacency index (%d nodes, mean_deg=%.2f)",
             n_nodes, np.mean([len(a) for a in adj_dict]))

    # Positive edges = all training edges (upper triangular to dedup undirected)
    ei = g_homo.edge_index
    mask = ei[0] < ei[1]
    pos_all = ei[:, mask]  # [2, E_pos]
    n_pos_all = pos_all.size(1)
    log.info("  Positive edges (upper-tri): %d", n_pos_all)

    if n_pos_all == 0:
        log.warning("  H_train has no edges! Aborting.")
        return model, [], 0

    # Train / val edge split (90 / 10) — monitors convergence
    n_val = max(1, int(0.1 * n_pos_all))
    perm  = torch.from_numpy(rng.permutation(n_pos_all).astype(np.int64))
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    pos_train = pos_all[:, train_idx]
    pos_val   = pos_all[:, val_idx]

    best_val_auc = -1.0
    best_state   = None
    conv_epoch   = 1
    wait = 0
    patience = 30
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z = model(x, edge_index)  # [N, embedding_dim]

        # Subsample positive edges per epoch for tractability on dense graphs
        if pos_train.size(1) > max_pos_per_epoch:
            idx_sub = torch.from_numpy(
                rng.choice(pos_train.size(1), size=max_pos_per_epoch, replace=False)
            ).long()
            pos_ep = pos_train[:, idx_sub].to(device)
        else:
            pos_ep = pos_train.to(device)

        # Sample fresh negatives every epoch (matched count)
        neg_train = _sample_neg_2hop(pos_ep.cpu(), adj_dict, n_nodes, rng)
        neg_train = neg_train.to(device)

        # Scores
        s_pos = (z[pos_ep[0]] * z[pos_ep[1]]).sum(-1)
        s_neg = (z[neg_train[0]] * z[neg_train[1]]).sum(-1)

        # BCE loss
        pos_target = torch.ones_like(s_pos)
        neg_target = torch.zeros_like(s_neg)
        logits = torch.cat([s_pos, s_neg])
        labels = torch.cat([pos_target, neg_target])
        loss   = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        # Val AUC (capped size for dense graphs)
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index)
            val_cap = min(1000, pos_val.size(1))
            if pos_val.size(1) > val_cap:
                idx_v = torch.from_numpy(
                    rng.choice(pos_val.size(1), size=val_cap, replace=False)
                ).long()
                pos_val_ep = pos_val[:, idx_v]
            else:
                pos_val_ep = pos_val
            neg_val = _sample_neg_2hop(pos_val_ep.cpu(), adj_dict, n_nodes, rng)
            pos_val_ep = pos_val_ep.to(device)
            neg_val = neg_val.to(device)
            s_pos_v = (z[pos_val_ep[0]] * z[pos_val_ep[1]]).sum(-1).cpu().numpy()
            s_neg_v = (z[neg_val[0]]    * z[neg_val[1]]).sum(-1).cpu().numpy()

            from sklearn.metrics import roc_auc_score
            scores_v = np.concatenate([s_pos_v, s_neg_v])
            labels_v = np.concatenate([np.ones(len(s_pos_v)), np.zeros(len(s_neg_v))])
            try:
                val_auc = roc_auc_score(labels_v, scores_v)
            except ValueError:
                val_auc = float("nan")

        history.append({
            "epoch":     epoch,
            "loss":      round(loss.item(), 6),
            "val_auc":   round(val_auc, 6) if val_auc == val_auc else "",
        })

        if epoch % 10 == 0:
            log.debug("  [L=%d ep %3d] loss=%.4f val_auc=%.4f",
                      num_layers, epoch, loss.item(), val_auc)

        if val_auc == val_auc and val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            conv_epoch   = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info("  [L=%d] Early stop at ep %d (best val_auc=%.4f @ ep %d)",
                         num_layers, epoch, best_val_auc, conv_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, history, conv_epoch


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

_LOG_FIELDS = [
    "dataset", "metapath", "L", "embedding_dim",
    "epoch", "loss", "val_auc",
    "convergence_epoch", "best_val_auc", "t_train", "weights_path", "note",
]


def _open_log(path: Path):
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    w  = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
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
    parser.add_argument("--depth", type=int, nargs="+", default=[2],
                        help="SAGE depth(s) to train, e.g. --depth 2 3 4")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--partition-json", required=True)
    parser.add_argument("--embedding-dim", type=int, default=64,
                        help="Output embedding dim for LP decoder")
    parser.add_argument("--max-pos-per-epoch", type=int, default=10000,
                        help="Subsample positives per epoch (dense graphs otherwise hang)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.partition_json) as f:
        part = json.load(f)

    cfg      = config.get_dataset_config(args.dataset)
    folder   = config.get_folder_name(args.dataset)
    out_dir  = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights_lp"
    weights_dir.mkdir(exist_ok=True)
    log_path = out_dir / "lp_training_log.csv"

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("exp_lp_train")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    fh = logging.FileHandler(out_dir / f"run_lp_train_{ts}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(ch)
    log.addHandler(fh)

    log.info("LP-Train | dataset=%s  metapath=%s  depth=%s  seed=%d  emb_dim=%d",
             args.dataset, args.metapath, args.depth, args.seed, args.embedding_dim)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(args.seed)

    g_full, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    target_ntype = cfg.target_node

    log.info("Reconstructing V_train (train_frac=%.0f%%)...", part["train_frac"] * 100)
    g_train = _make_train_subgraph(g_full, part, info["labels"], info["masks"], target_ntype)
    del g_full
    gc.collect()
    log.info("  %s=%d nodes", target_ntype, g_train[target_ntype].num_nodes)

    # C++ ExactD on V_train
    data_dir = config.get_staging_dir(args.dataset)
    os.makedirs(data_dir, exist_ok=True)
    setup_global_res_dirs(folder, project_root)
    engine = CppEngine(executable_path=config.CPP_EXECUTABLE, data_dir=data_dir)

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

    g_h0 = engine.load_result(output_file, n_target, offset)
    g_h0.x = g_train[target_ntype].x
    del g_train
    gc.collect()

    in_dim = g_h0.x.size(1)
    log.info("  H_train edges=%d  in_dim=%d",
             g_h0.edge_index.size(1) if g_h0.edge_index is not None else 0, in_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp_safe = args.metapath.replace(",", "_").replace("/", "_")
    done_runs = _done_runs(log_path)
    log_fh, log_w = _open_log(log_path)

    try:
        for L in args.depth:
            if (args.metapath, str(L)) in done_runs:
                log.info("[L=%d] already done — skipping", L)
                continue

            log.info("\n[L=%d] Training LP-SAGE(%d, emb=%d)...",
                     L, L, args.embedding_dim)
            t0 = time.perf_counter()
            try:
                model, history, conv_epoch = _train_lp(
                    g_h0, in_dim, args.embedding_dim, L, args.epochs,
                    device, rng, log, max_pos_per_epoch=args.max_pos_per_epoch,
                )
            except RuntimeError as e:
                if "CUDA" in str(e) and device.type == "cuda":
                    log.warning("  [L=%d] GPU OOM — retrying on CPU", L)
                    torch.cuda.empty_cache()
                    device = torch.device("cpu")
                    model, history, conv_epoch = _train_lp(
                        g_h0, in_dim, args.embedding_dim, L, args.epochs,
                        device, rng, log, max_pos_per_epoch=args.max_pos_per_epoch,
                    )
                else:
                    raise
            t_train = time.perf_counter() - t0

            weights_path = str(weights_dir / f"{mp_safe}_L{L}.pt")
            torch.save(model.state_dict(), weights_path)

            best_val_auc = max((h["val_auc"] for h in history
                                if isinstance(h.get("val_auc"), float)), default=float("nan"))
            log.info("  Saved θ* → %s  (conv_epoch=%d best_val_auc=%.4f t=%.2fs)",
                     weights_path, conv_epoch, best_val_auc, t_train)

            for h in history:
                log_w.writerow({
                    "dataset": args.dataset, "metapath": args.metapath, "L": L,
                    "embedding_dim": args.embedding_dim,
                    "epoch": h["epoch"], "loss": h["loss"], "val_auc": h["val_auc"],
                    "convergence_epoch": "", "best_val_auc": "",
                    "t_train": "", "weights_path": "", "note": "",
                })
            log_w.writerow({
                "dataset": args.dataset, "metapath": args.metapath, "L": L,
                "embedding_dim": args.embedding_dim,
                "epoch": "DONE", "loss": "", "val_auc": "",
                "convergence_epoch": conv_epoch,
                "best_val_auc": round(best_val_auc, 6) if best_val_auc == best_val_auc else "",
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
