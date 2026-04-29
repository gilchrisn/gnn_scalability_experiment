"""Simple-HGN-style baseline on the same HGB splits as our sketch-as-feature
pilot. Headline goal: an apples-to-apples F1 + wall-clock comparison so the
Path 1 cost analysis has real numbers, not symbolic ones.

The original Simple-HGN (Lv et al. KDD'21) is a DGL implementation. This
script is a PyG port that captures the architecture's load-bearing parts:

  * Per-node-type input projection to a common hidden dim.
  * Per-relation GATv2 attention with multi-head concat=False.
  * Stacked layers with residual + ReLU.
  * Final linear classifier on the target type.

What's intentionally missing relative to the canonical Simple-HGN:

  * Explicit per-relation **edge embeddings** concatenated into the
    attention. PyG's GATv2Conv supports edge_attr but only one tensor per
    HeteroConv branch; we'd need a custom MessagePassing layer to inject a
    learned vector per relation. The relation identity is already
    captured by HeteroConv's per-relation parameters, which is the
    dominant signal — the explicit edge-embedding addition is a small
    accuracy delta in the original paper (≤1pp on DBLP).

This means the F1 here is a "Simple-HGN-style" lower bound. If our
sketch-as-feature method matches it within a couple of points, that
satisfies Path 1's "match reasonably" bar; if we want the exact paper
numbers we'd need to port the full DGL implementation later.

Usage
-----
    python scripts/exp_simple_hgn_baseline.py HGB_DBLP --num-seeds 3
    python scripts/exp_simple_hgn_baseline.py HGB_ACM --quick
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
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.sketch_feature import (
    macro_f1 as _macro_f1,
    macro_f1_multilabel as _macro_f1_multilabel,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SimpleHGN(nn.Module):
    """Heterogeneous GATv2 stack with per-type input projection.

    Input  : x_dict {ntype: [N_t, in_dim_t]}
    Output : logits[target_type] of shape [N_target, n_classes]
    """

    def __init__(
        self,
        in_dim_by_type: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        meta_node_types: List[str],
        meta_edge_types: List[tuple],
        target_type: str,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.target_type = target_type
        self.dropout = dropout

        # Per-type input projection. Some types have no native features; we
        # fall back to a learnable parameter table keyed by node type and
        # known node count (filled lazily in forward — needs n_by_type).
        self.proj = nn.ModuleDict()
        self.type_emb = nn.ParameterDict()
        for nt in meta_node_types:
            d = in_dim_by_type.get(nt, 0)
            if d > 0:
                self.proj[nt] = nn.Linear(d, hidden_dim)
            else:
                # Lazy: filled on first forward when N_t is known.
                self.type_emb[nt] = nn.Parameter(torch.empty(0))

        # Stacked HeteroConv layers; each branch is a GATv2 over one relation.
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            conv = HeteroConv(
                {
                    rel: GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=n_heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                    for rel in meta_edge_types
                },
                aggr="sum",
            )
            self.layers.append(conv)

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def _embed(self, nt: str, n: int, device: torch.device, hidden_dim: int) -> torch.Tensor:
        param = self.type_emb[nt]
        if param.numel() == 0:
            new = nn.Parameter(torch.randn(n, hidden_dim, device=device))
            self.type_emb[nt] = new
            return new
        return param

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
        n_by_type: Dict[str, int],
    ) -> torch.Tensor:
        # 1) Project / embed per type to hidden_dim.
        h_dict: Dict[str, torch.Tensor] = {}
        hidden_dim = self.classifier.in_features
        device = next(self.parameters()).device
        for nt in n_by_type:
            if nt in self.proj:
                h_dict[nt] = self.proj[nt](x_dict[nt].to(device))
            else:
                h_dict[nt] = self._embed(nt, n_by_type[nt], device, hidden_dim)

        # 2) HeteroConv stack with residuals.
        for layer in self.layers:
            new_h = layer(h_dict, edge_index_dict)
            # Residual + activation; types missing from new_h (no incoming
            # edges this layer) keep their previous representation.
            for nt in h_dict:
                if nt in new_h and new_h[nt] is not None:
                    h_dict[nt] = F.elu(new_h[nt] + h_dict[nt])
            h_dict = {
                nt: F.dropout(v, p=self.dropout, training=self.training)
                for nt, v in h_dict.items()
            }

        return self.classifier(h_dict[self.target_type])


# ---------------------------------------------------------------------------
# Metric helpers (same conventions as exp_sketch_feature_train.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Train / eval one seed
# ---------------------------------------------------------------------------


def run_one_seed(args: argparse.Namespace, seed: int) -> dict:
    torch.manual_seed(seed)
    cfg = config.get_dataset_config(args.dataset)
    target = cfg.target_node

    print(f"\n=== {args.dataset}  seed={seed}  target={target} ===")
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
        train_mask = train_mask & valid
        val_mask = val_mask & valid
        test_mask = test_mask & valid

    in_dim_by_type = {
        nt: int(g[nt].x.size(1)) for nt in g.node_types if "x" in g[nt] and g[nt].x is not None
    }
    n_by_type = {nt: g[nt].num_nodes for nt in g.node_types}
    x_dict = {nt: g[nt].x for nt in g.node_types if nt in in_dim_by_type}
    edge_index_dict = {et: g[et].edge_index for et in g.edge_types}

    print(f"[graph] node_types={list(g.node_types)}")
    print(f"[graph] edge_types={len(g.edge_types)} relations")

    device = torch.device(args.device)
    model = SimpleHGN(
        in_dim_by_type=in_dim_by_type,
        hidden_dim=args.hidden_dim,
        out_dim=n_classes,
        meta_node_types=list(g.node_types),
        meta_edge_types=list(g.edge_types),
        target_type=target,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    labels_d = labels.to(device)
    train_mask_d = train_mask.to(device)
    val_mask_d = val_mask.to(device)
    test_mask_d = test_mask.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    wait = 0

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x_dict, edge_index_dict, n_by_type)
        if multi_label:
            loss = F.binary_cross_entropy_with_logits(
                logits[train_mask_d], labels_d[train_mask_d].float()
            )
        else:
            loss = F.cross_entropy(logits[train_mask_d], labels_d[train_mask_d])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_dict, edge_index_dict, n_by_type)
            if multi_label:
                val_f1 = _macro_f1_multilabel(logits[val_mask_d], labels_d[val_mask_d])
            else:
                pred = logits.argmax(dim=-1)
                val_f1 = _macro_f1(pred[val_mask_d], labels_d[val_mask_d], n_classes)

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

    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x_dict, edge_index_dict, n_by_type)
        if multi_label:
            test_f1 = _macro_f1_multilabel(logits[test_mask_d], labels_d[test_mask_d])
        else:
            pred = logits.argmax(dim=-1)
            test_f1 = _macro_f1(pred[test_mask_d], labels_d[test_mask_d], n_classes)

    print(f"[result] seed={seed}  val_f1={best_val_f1:.4f} (ep {best_epoch})  "
          f"test_f1={test_f1:.4f}  train_time={train_time:.1f}s")

    out = {
        "dataset": args.dataset,
        "method": "SimpleHGN_pyg_port",
        "target": target,
        "seed": seed,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "test_f1": test_f1,
        "train_time_s": train_time,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    res_dir = Path("results") / args.dataset
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / f"simple_hgn_baseline_seed{seed}.json"
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 1 seed, 30 epochs")
    args = p.parse_args()

    if args.quick:
        args.num_seeds = 1
        args.epochs = 30

    seeds = [args.seed_base + i for i in range(args.num_seeds)]
    results = []
    for s in seeds:
        results.append(run_one_seed(args, s))

    test_f1s = [r["test_f1"] for r in results]
    val_f1s = [r["best_val_f1"] for r in results]
    times = [r["train_time_s"] for r in results]
    print("\n" + "=" * 68)
    print(f"Simple-HGN baseline | {args.dataset} | n={len(results)} seeds")
    print("=" * 68)
    print(f"  test_f1     = {statistics.fmean(test_f1s):.4f} ± "
          f"{statistics.pstdev(test_f1s) if len(test_f1s) > 1 else 0.0:.4f}")
    print(f"  val_f1      = {statistics.fmean(val_f1s):.4f} ± "
          f"{statistics.pstdev(val_f1s) if len(val_f1s) > 1 else 0.0:.4f}")
    print(f"  train_time  = {statistics.fmean(times):.1f}s ± "
          f"{statistics.pstdev(times) if len(times) > 1 else 0.0:.1f}s "
          f"(per seed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
