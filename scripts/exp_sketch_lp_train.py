"""Link prediction via the new sketch-feature pipeline.

Mirrors the role of ``exp_sketch_feature_train.py`` for LP instead of NC:
load a (cached) ``SketchBundle``, encode each target node's bottom-k slots
into a learned embedding, train a dot-product LP decoder on V_train edges
with 2-hop negative sampling (HGB protocol, Lv 2021), evaluate on
V_test-involving edges with MRR / ROC-AUC / Hits@10.

Why this matters for the journal extension
-------------------------------------------

The multi-query amortisation argument (CURRENT_STATE.md, Path 1) depends on
showing that one KMV propagation supports multiple downstream queries.
``exp_multi_query_amortization.py`` currently times {NC, similarity}; this
script lets us extend it to Q=3 by adding {LP}.

Quality framing (per ``19_pilot_results_dblp.md``)
--------------------------------------------------

LP from sketches on DBLP-APA was previously roughly tied with MPRW; we do
not claim a per-task LP win. The contribution here is operational —
demonstrating that the same propagation pass supports LP queries — not
a quality improvement.

Usage
-----
    python scripts/exp_sketch_lp_train.py HGB_DBLP \\
        --partition-json results/HGB_DBLP/partition.json \\
        --k 32 --emb-dim 128 --epochs 100 --seed 42

    python scripts/exp_sketch_lp_train.py HGB_DBLP --quick   # smoke test
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
from src.sketch_feature import (
    SketchBundle,
    SketchFeatureEncoder,
    SketchHAN,
    decode_sketches,
    extract_sketches,
)
from scripts.exp_lp_train import _build_adj_dict, _sample_neg_2hop
from scripts.exp_sketch_feature_train import _DEFAULT_META_PATHS  # noqa: E402


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class SketchLPEncoder(nn.Module):
    """Same architecture as LoneTypedMLP for NC, minus the classifier head.

    Outputs an embedding of dimension ``emb_dim`` per target node, suitable
    for dot-product LP scoring.
    """

    def __init__(self, n_t0: int, emb_dim: int, meta_paths: list,
                 target_base_dim: int, agg: str, dropout: float):
        super().__init__()
        self.meta_paths = list(meta_paths)
        self.shared_t0 = nn.Embedding(n_t0 + 1, emb_dim, padding_idx=n_t0)
        self.encoders = nn.ModuleDict({
            SketchHAN._sanitize(mp): SketchFeatureEncoder(
                n_t0=n_t0, emb_dim=emb_dim, agg=agg,
                base_dim=target_base_dim, dropout=dropout,
                shared_embedding=self.shared_t0,
            ) for mp in self.meta_paths
        })
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, sketch_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        per_mp = []
        for mp in self.meta_paths:
            decoded, base_x = sketch_inputs[mp]
            per_mp.append(self.encoders[SketchHAN._sanitize(mp)](decoded, base_x))
        x = torch.stack(per_mp, dim=0).mean(dim=0)  # [N, emb_dim]
        return F.normalize(self.proj(x), dim=-1)    # L2-normalised — typical LP


# ---------------------------------------------------------------------------
# LP metrics (mirrors exp_lp_inference._compute_lp_metrics for honesty)
# ---------------------------------------------------------------------------


def _ranking_metrics(
    z: torch.Tensor,
    pos_edges: torch.Tensor,
    adj_exact: list,
    rng: np.random.Generator,
    n_negs_per_pos: int = 50,
) -> Dict[str, float]:
    """For each positive edge (u, v), score against (u, v_neg) for n_negs
    randomly-sampled non-neighbours of u. Compute MRR, Hits@1, Hits@10,
    and ROC-AUC (treating positives vs negatives as a binary task).
    """
    n_pos = pos_edges.size(1)
    if n_pos == 0:
        return {"MRR": 0.0, "Hits_1": 0.0, "Hits_10": 0.0, "ROC_AUC": 0.0}
    n_nodes = z.size(0)

    mrrs = []
    hits1 = 0
    hits10 = 0
    pos_scores_all = []
    neg_scores_all = []

    for i in range(n_pos):
        u, v = int(pos_edges[0, i]), int(pos_edges[1, i])
        forbidden = adj_exact[u] | {u, v}
        candidates = []
        tries = 0
        while len(candidates) < n_negs_per_pos and tries < 4 * n_negs_per_pos:
            cand = int(rng.integers(n_nodes))
            if cand not in forbidden:
                candidates.append(cand)
                forbidden.add(cand)  # avoid duplicates in this draw
            tries += 1
        if not candidates:
            continue

        z_u = z[u]
        z_v = z[v]
        z_negs = z[candidates]
        s_pos = (z_u * z_v).sum().item()
        s_neg = (z_negs @ z_u).cpu().numpy()
        rank = 1 + int(np.sum(s_neg >= s_pos))
        mrrs.append(1.0 / rank)
        hits1 += int(rank <= 1)
        hits10 += int(rank <= 10)
        pos_scores_all.append(s_pos)
        neg_scores_all.extend(s_neg.tolist())

    n = len(mrrs)
    if n == 0:
        return {"MRR": 0.0, "Hits_1": 0.0, "Hits_10": 0.0, "ROC_AUC": 0.0}

    # ROC-AUC via sklearn if available, else NaN.
    try:
        from sklearn.metrics import roc_auc_score
        scores = np.array(pos_scores_all + neg_scores_all)
        labels = np.array([1] * len(pos_scores_all) + [0] * len(neg_scores_all))
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = float("nan")

    return {
        "MRR":     float(statistics.fmean(mrrs)),
        "Hits_1":  hits1 / n,
        "Hits_10": hits10 / n,
        "ROC_AUC": auc,
        "n_pos":   n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--meta-paths", nargs="+",
                   help="Override default meta-paths.")
    p.add_argument("--partition-json",
                   help="Optional partition.json from exp1_partition.py. "
                        "If absent, all target nodes are eligible (transductive).")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--agg", choices=["mean", "sum", "attention"], default="attention")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--max-pos-per-epoch", type=int, default=10000)
    p.add_argument("--max-test-edges", type=int, default=2000)
    p.add_argument("--n-negs-per-pos", type=int, default=50)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: epochs=20, k=8")
    args = p.parse_args()

    if args.quick:
        args.epochs = 20
        args.k = 8

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = config.get_dataset_config(args.dataset)
    target_type = cfg.target_node
    out_dir = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = args.meta_paths or _DEFAULT_META_PATHS.get(args.dataset)
    if not meta_paths:
        print(f"No default meta-paths for {args.dataset}", file=sys.stderr)
        return 2

    print(f"[load] {args.dataset} target={target_type} seed={args.seed}")
    g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, target_type)
    n_target = g[target_type].num_nodes

    # Cached sketch bundle — shares cache with NC scripts so amortisation
    # accounting stays honest (precompute is paid once across NC + LP).
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

    decoded_by_mp = {
        mp: decode_sketches(sk, bundle.sorted_hashes, bundle.sort_indices)
        for mp, sk in bundle.sketches_by_mp.items()
    }

    # Native target features for fusion.
    base_x = None
    target_base_dim = 0
    if "x" in g[target_type] and g[target_type].x is not None:
        base_x = g[target_type].x
        target_base_dim = int(base_x.size(1))
        print(f"[features] using native {target_type} features: {tuple(base_x.shape)}")

    # ------------------------------------------------------------------
    # Build LP edges from a meta-path-induced adjacency. We use the union
    # of sketch-decoded edges across meta-paths as the positive-edge set
    # — same convention as the sparsifier consumer mode.
    # The held-out test edges come from the partition file (V_test).
    # ------------------------------------------------------------------
    from src.sketch_feature import build_mp_edges_from_decoded
    union_src, union_dst = [], []
    for mp in meta_paths:
        ei = build_mp_edges_from_decoded(decoded_by_mp[mp], n_target,
                                         add_self_loops=False)
        union_src.append(ei[0])
        union_dst.append(ei[1])
    full_ei = torch.stack([torch.cat(union_src), torch.cat(union_dst)]).long()
    n_edges_total = full_ei.size(1)
    print(f"[edges] {n_edges_total} sketch-decoded edges across {len(meta_paths)} mps")

    if args.partition_json:
        with open(args.partition_json) as f:
            part = json.load(f)
        v_train = set(int(x) for x in part["train_node_ids"])
        v_test = set(int(x) for x in part["test_node_ids"])
        print(f"[partition] V_train={len(v_train)}  V_test={len(v_test)}")
    else:
        # Fall back to a 80/20 random split over target nodes.
        perm = rng.permutation(n_target)
        n_train = int(0.8 * n_target)
        v_train = set(int(x) for x in perm[:n_train])
        v_test = set(int(x) for x in perm[n_train:])
        print(f"[partition] random 80/20 — V_train={len(v_train)}  V_test={len(v_test)}")

    # Train edges: both endpoints in V_train; upper-triangular for dedup.
    src_np = full_ei[0].numpy()
    dst_np = full_ei[1].numpy()
    train_mask = np.array([
        (s in v_train) and (d in v_train) and (s < d)
        for s, d in zip(src_np, dst_np)
    ])
    test_mask = np.array([
        ((s in v_test) or (d in v_test)) and (s != d) and (s < d)
        for s, d in zip(src_np, dst_np)
    ])
    pos_train = full_ei[:, train_mask]
    pos_test = full_ei[:, test_mask]
    if pos_test.size(1) > args.max_test_edges:
        idx = rng.choice(pos_test.size(1), size=args.max_test_edges, replace=False)
        pos_test = pos_test[:, torch.from_numpy(idx).long()]
    print(f"[edges] positive train edges = {pos_train.size(1)}")
    print(f"[edges] positive test edges (capped at {args.max_test_edges}) = {pos_test.size(1)}")

    # Adjacency for negative sampling (uses the FULL sketch-decoded graph).
    adj_exact = _build_adj_dict(full_ei, n_target)

    # ------------------------------------------------------------------
    # Model + training loop.
    # ------------------------------------------------------------------
    device = torch.device(args.device)
    print(f"[device] {device}")
    model = SketchLPEncoder(
        n_t0=n_target, emb_dim=args.emb_dim, meta_paths=meta_paths,
        target_base_dim=target_base_dim, agg=args.agg, dropout=args.dropout,
    ).to(device)

    sketch_inputs = {
        mp: (
            decoded_by_mp[mp].to(device),
            base_x.to(device) if base_x is not None else None,
        )
        for mp in meta_paths
    }

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    # Train / val split inside V_train edges — 90/10.
    n_pos = pos_train.size(1)
    n_val = max(1, int(0.1 * n_pos))
    perm_t = torch.from_numpy(rng.permutation(n_pos)).long()
    pos_val = pos_train[:, perm_t[:n_val]]
    pos_train_only = pos_train[:, perm_t[n_val:]]

    print(f"[train] epochs={args.epochs} lr={args.lr} dropout={args.dropout}")
    train_t0 = time.perf_counter()
    best_val_mrr = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        z = model(sketch_inputs)

        if pos_train_only.size(1) > args.max_pos_per_epoch:
            idx = rng.choice(pos_train_only.size(1), size=args.max_pos_per_epoch,
                             replace=False)
            pos_ep = pos_train_only[:, torch.from_numpy(idx).long()]
        else:
            pos_ep = pos_train_only

        neg_ep = _sample_neg_2hop(pos_ep, adj_exact, n_target, rng)
        pos_ep = pos_ep.to(device)
        neg_ep = neg_ep.to(device)

        s_pos = (z[pos_ep[0]] * z[pos_ep[1]]).sum(-1)
        s_neg = (z[neg_ep[0]] * z[neg_ep[1]]).sum(-1)
        logits = torch.cat([s_pos, s_neg])
        labels = torch.cat([
            torch.ones_like(s_pos), torch.zeros_like(s_neg),
        ])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        # Val MRR.
        model.eval()
        with torch.no_grad():
            z = model(sketch_inputs).cpu()
            val_metrics = _ranking_metrics(
                z, pos_val, adj_exact, rng,
                n_negs_per_pos=min(args.n_negs_per_pos, 20),
            )
        val_mrr = val_metrics["MRR"]
        if val_mrr > best_val_mrr + 1e-4:
            best_val_mrr = val_mrr
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"  ep {epoch:3d}  loss={loss.item():.4f}  val_MRR={val_mrr:.4f}  "
                  f"(best={best_val_mrr:.4f}@{best_epoch})")
        if wait >= args.patience:
            print(f"  early stop at ep {epoch}")
            break

    train_time_s = time.perf_counter() - train_t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model(sketch_inputs).cpu()
    test_metrics = _ranking_metrics(
        z, pos_test, adj_exact, rng, n_negs_per_pos=args.n_negs_per_pos,
    )
    print(f"[result] val_MRR={best_val_mrr:.4f} (ep {best_epoch})")
    print(f"[result] test {test_metrics}  train_time={train_time_s:.1f}s "
          f"extract_time={extract_time_s:.2f}s")

    out = {
        "dataset": args.dataset,
        "target_type": target_type,
        "method": "sketch_lp_dot",
        "meta_paths": meta_paths,
        "k": args.k,
        "emb_dim": args.emb_dim,
        "agg": args.agg,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_val_mrr": best_val_mrr,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "train_time_s": train_time_s,
        "extract_time_s": extract_time_s,
        "n_train_edges": int(pos_train_only.size(1)),
        "n_test_edges": int(pos_test.size(1)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    res_path = out_dir / f"sketch_lp_pilot_k{args.k}_seed{args.seed}.json"
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[save] {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
