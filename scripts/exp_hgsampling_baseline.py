"""HGSampling-style baseline using PyG's HGTLoader.

The point is *not* to claim HGSampling underperforms our method on quality
— it usually doesn't. The point is to **measure the per-epoch sampler
overhead** that motivates KMV's amortisation argument: every epoch,
HGSampling re-walks the graph to build mini-batches, while KMV pays its
graph-walk cost ONCE at sketch precompute.

What we measure
---------------
- NC test macro-F1 (3 seeds), comparable to single-task SHGN.
- Per-epoch wall-clock decomposition: pure sampler time vs forward+backward
  time. The sampler-only fraction is the term that doesn't amortise across
  tasks the way precomputed sketches do.

Architecture: HGTConv stacked layers on the sampled mini-batches, NC head
on the target type. Hyperparameters mirror the single-task SHGN baseline
to keep comparisons honest.

Usage
-----
    python scripts/exp_hgsampling_baseline.py HGB_DBLP --num-seeds 3
    python scripts/exp_hgsampling_baseline.py HGB_ACM --num-seeds 1 --quick
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
from torch_geometric.loader import HGTLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.sketch_feature import macro_f1, macro_f1_multilabel


# ---------------------------------------------------------------------------
# Model — HGTConv stack
# ---------------------------------------------------------------------------


class HGTNet(nn.Module):
    """HGT-style heterogeneous network for batched sampler training.

    Input: x_dict (per-type node features), edge_index_dict (per-relation
    edges) on a mini-batch produced by HGTLoader. Output: logits on the
    target type.
    """

    def __init__(self, in_dim_by_type: Dict[str, int], hidden_dim: int,
                 out_dim: int, meta_node_types: List[str],
                 meta_edge_types: List[tuple], target_type: str,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        from torch_geometric.nn import HGTConv
        self.target_type = target_type
        self.dropout = dropout
        self.proj = nn.ModuleDict()
        self.type_emb = nn.ParameterDict()
        for nt in meta_node_types:
            d = in_dim_by_type.get(nt, 0)
            if d > 0:
                self.proj[nt] = nn.Linear(d, hidden_dim)
            else:
                self.type_emb[nt] = nn.Parameter(torch.empty(0))
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(meta_node_types, meta_edge_types),
                heads=n_heads,
            ))
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def _embed(self, nt, n, device):
        param = self.type_emb[nt]
        if param.numel() == 0:
            new = nn.Parameter(torch.randn(n, self.classifier.in_features,
                                           device=device))
            self.type_emb[nt] = new
            return new
        return param

    def forward(self, x_dict, edge_index_dict, n_by_type=None,
                batch_n_by_type=None):
        device = next(iter(edge_index_dict.values())).device
        h_dict = {}
        # Iterate over ALL metadata types — HGTConv needs every node type
        # in x_dict, including featureless ones. For featureless types,
        # use the (lazily-initialised) type embedding indexed by the
        # batch's node ids.
        for nt in self.proj.keys():
            if nt in x_dict:
                h_dict[nt] = self.proj[nt](x_dict[nt].to(device))
        for nt in self.type_emb.keys():
            # batch_n_by_type tells us how many of this type appear in the
            # mini-batch, when called from a HGTLoader-batched context.
            n = (batch_n_by_type or n_by_type or {}).get(nt, 0)
            if n == 0:
                # No nodes of this type in this batch — give an empty tensor.
                h_dict[nt] = torch.empty(0, self.classifier.in_features,
                                         device=device)
            else:
                emb_full = self._embed(nt, n_by_type[nt] if n_by_type else n,
                                       device)
                # In a sampled mini-batch, n usually < n_by_type[nt]; we
                # still pass the full table and rely on edge_index to
                # gather only the needed rows. Truncate to first n if needed
                # — HGTLoader returns nodes contiguously.
                h_dict[nt] = emb_full[:n] if emb_full.size(0) >= n else emb_full
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
            h_dict = {k: F.elu(v) if v is not None else v for k, v in h_dict.items()}
            h_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                      for k, v in h_dict.items()}
        return self.classifier(h_dict[self.target_type])


# ---------------------------------------------------------------------------
# Train one seed
# ---------------------------------------------------------------------------


def run_one_seed(args, seed: int) -> dict:
    torch.manual_seed(seed)
    cfg = config.get_dataset_config(args.dataset)
    target = cfg.target_node

    print(f"\n=== {args.dataset} seed={seed} HGSampling ===")
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

    # Attach labels + masks to the HeteroData target type so HGTLoader can
    # carry them into mini-batches.
    g[target].y = labels
    g[target].train_mask = train_mask
    g[target].val_mask = val_mask
    g[target].test_mask = test_mask

    in_dim_by_type = {nt: int(g[nt].x.size(1))
                      for nt in g.node_types
                      if "x" in g[nt] and g[nt].x is not None}
    n_by_type = {nt: g[nt].num_nodes for nt in g.node_types}

    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    test_idx = torch.where(test_mask)[0]

    # HGTLoader: per-epoch heterogeneous sampler.
    train_loader = HGTLoader(
        g,
        num_samples={nt: [args.fanout] * args.n_layers for nt in g.node_types},
        batch_size=args.batch_size,
        input_nodes=(target, train_idx),
    )
    eval_loader = HGTLoader(
        g,
        num_samples={nt: [args.fanout * 2] * args.n_layers for nt in g.node_types},
        batch_size=args.batch_size,
        input_nodes=(target, torch.cat([val_idx, test_idx])),
        shuffle=False,
    )

    device = torch.device(args.device)
    model = HGTNet(
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
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best_val = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    sampler_time_total = 0.0
    fwbw_time_total = 0.0

    print(f"[train] epochs={args.epochs} batch={args.batch_size} fanout={args.fanout}")
    train_t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for sample_t0 in [time.perf_counter()]:
            for batch in train_loader:
                sample_dt = time.perf_counter() - sample_t0
                sampler_time_total += sample_dt

                fwbw_t0 = time.perf_counter()
                batch = batch.to(device)
                opt.zero_grad()
                x_dict = {nt: batch[nt].x for nt in batch.node_types
                          if "x" in batch[nt] and batch[nt].x is not None}
                edge_index_dict = {et: batch[et].edge_index
                                   for et in batch.edge_types}
                batch_n = {nt: batch[nt].num_nodes for nt in batch.node_types}
                logits = model(x_dict, edge_index_dict, n_by_type, batch_n)
                # Only the seed nodes (input nodes) have labels in this batch.
                tgt_size = batch[target].batch_size
                y = batch[target].y[:tgt_size]
                if multi_label:
                    loss = F.binary_cross_entropy_with_logits(
                        logits[:tgt_size], y.float().to(device))
                else:
                    loss = F.cross_entropy(logits[:tgt_size], y.to(device))
                loss.backward()
                opt.step()
                fwbw_dt = time.perf_counter() - fwbw_t0
                fwbw_time_total += fwbw_dt
                epoch_loss += loss.item()
                n_batches += 1
                sample_t0 = time.perf_counter()  # reset for next batch sample

        avg_loss = epoch_loss / max(n_batches, 1)

        # Val.
        model.eval()
        val_preds, val_labels_l = [], []
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                x_dict = {nt: batch[nt].x for nt in batch.node_types
                          if "x" in batch[nt] and batch[nt].x is not None}
                edge_index_dict = {et: batch[et].edge_index
                                   for et in batch.edge_types}
                batch_n = {nt: batch[nt].num_nodes for nt in batch.node_types}
                logits = model(x_dict, edge_index_dict, n_by_type, batch_n)
                tgt_size = batch[target].batch_size
                val_preds.append(logits[:tgt_size].cpu())
                val_labels_l.append(batch[target].y[:tgt_size].cpu())
        val_preds = torch.cat(val_preds, dim=0)
        val_labels = torch.cat(val_labels_l, dim=0)

        # Split val_preds back into val and test based on input order.
        n_val_eval = val_idx.size(0)
        v_logits, v_labels = val_preds[:n_val_eval], val_labels[:n_val_eval]
        if multi_label:
            val_score = macro_f1_multilabel(v_logits, v_labels)
        else:
            val_score = macro_f1(v_logits.argmax(dim=-1), v_labels, n_classes)

        if val_score > best_val + 1e-4:
            best_val = val_score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"  ep {epoch:3d}  loss={avg_loss:.4f}  val={val_score:.4f}  "
                  f"(best={best_val:.4f}@{best_epoch})  "
                  f"sampler_ratio={sampler_time_total/(sampler_time_total+fwbw_time_total):.2f}")
        if wait >= args.patience:
            print(f"  early stop at ep {epoch}")
            break

    train_time_s = time.perf_counter() - train_t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels_l = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            x_dict = {nt: batch[nt].x for nt in batch.node_types
                      if "x" in batch[nt] and batch[nt].x is not None}
            edge_index_dict = {et: batch[et].edge_index
                               for et in batch.edge_types}
            logits = model(x_dict, edge_index_dict, n_by_type)
            tgt_size = batch[target].batch_size
            test_preds.append(logits[:tgt_size].cpu())
            test_labels_l.append(batch[target].y[:tgt_size].cpu())
    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels_l, dim=0)
    n_val_eval = val_idx.size(0)
    t_logits, t_labels = test_preds[n_val_eval:], test_labels[n_val_eval:]
    if multi_label:
        test_score = macro_f1_multilabel(t_logits, t_labels)
    else:
        test_score = macro_f1(t_logits.argmax(dim=-1), t_labels, n_classes)

    print(f"[result] seed={seed}  val={best_val:.4f} (ep {best_epoch})  "
          f"test={test_score:.4f}  train_time={train_time_s:.1f}s  "
          f"sampler={sampler_time_total:.1f}s  fwbw={fwbw_time_total:.1f}s")

    out = {
        "dataset": args.dataset,
        "method": "hgsampling_HGT",
        "target": target,
        "seed": seed,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "fanout": args.fanout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "best_val_f1": best_val,
        "best_epoch": best_epoch,
        "test_f1": test_score,
        "train_time_s": train_time_s,
        "sampler_time_s": sampler_time_total,
        "fwbw_time_s": fwbw_time_total,
        "sampler_fraction": sampler_time_total / max(train_time_s, 1e-9),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    res_dir = Path("results") / args.dataset
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / f"hgsampling_baseline_seed{seed}.json"
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=50,
                   help="Sampler-based training is more expensive per epoch; "
                        "default reduced relative to single-task SHGN.")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fanout", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.num_seeds = 1
        args.epochs = 10

    seeds = [args.seed_base + i for i in range(args.num_seeds)]
    results = [run_one_seed(args, s) for s in seeds]
    test_f1 = [r["test_f1"] for r in results]
    times = [r["train_time_s"] for r in results]
    sampler_frac = [r["sampler_fraction"] for r in results]
    print(f"\n=== HGSampling | {args.dataset} | n={len(results)} ===")
    print(f"  test_f1     = {statistics.fmean(test_f1):.4f} ± "
          f"{statistics.pstdev(test_f1) if len(test_f1) > 1 else 0.0:.4f}")
    print(f"  train_time  = {statistics.fmean(times):.1f}s ± "
          f"{statistics.pstdev(times) if len(times) > 1 else 0.0:.1f}s")
    print(f"  sampler_frac= {statistics.fmean(sampler_frac):.2f} ± "
          f"{statistics.pstdev(sampler_frac) if len(sampler_frac) > 1 else 0.0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
