"""End-to-end sketch-as-feature pilot for Path 1.

For one HGB dataset and a list of mirrored meta-paths:

  1. Load the heterogeneous graph + labels + masks.
  2. Extract a coordinated KMV sketch per meta-path
     (:func:`src.sketch_feature.extract_sketches`).
  3. Decode bottom-k slots to T_0 entity ids
     (:func:`src.sketch_feature.decode_sketches`).
  4. Train :class:`src.sketch_feature.SketchHAN` for node classification
     and report macro-F1 on val / test masks.

This is the *minimum viable* end-to-end loop. It does not yet include:

  * Multi-seed averaging.
  * Per-meta-path encoder weight sharing for the typed identity universe.
  * Sparsifier-mode comparison (that lives in exp1/exp2/exp3).
  * Honest baselines (Simple-HGN, HAN-on-real-graph). The point of this
    script is to validate that sketch-as-feature trains and the loss
    decreases — quality comparisons come later.

Usage
-----
    python scripts/exp_sketch_feature_train.py HGB_DBLP \\
        --meta-paths "author_to_paper,paper_to_author" \\
                     "author_to_paper,paper_to_term,term_to_paper,paper_to_author" \\
        --k 32 --epochs 100

    python scripts/exp_sketch_feature_train.py HGB_DBLP --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.sketch_feature import (
    SketchBundle,
    SketchHAN,
    decode_sketches,
    extract_sketches,
)


# Default meta-paths per HGB dataset for the pilot. Both must be mirrored
# (start == end at the target type) for the extractor to accept them.
_DEFAULT_META_PATHS = {
    "HGB_DBLP": [
        "author_to_paper,paper_to_author",
        "author_to_paper,paper_to_term,term_to_paper,paper_to_author",
    ],
    "HGB_ACM": [
        "paper_to_author,author_to_paper",
        "paper_to_term,term_to_paper",
    ],
    "HGB_IMDB": [
        "movie_to_keyword,keyword_to_movie",
        "movie_to_actor,actor_to_movie",
        "movie_to_director,director_to_movie",
    ],
    "HNE_PubMed": [
        "disease_to_chemical,chemical_to_disease",
        "disease_to_gene,gene_to_disease",
    ],
}


def _macro_f1(y_pred: torch.Tensor, y_true: torch.Tensor, n_classes: int) -> float:
    """Macro-F1 averaged across classes; ignores classes absent in y_true."""
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


def _build_mp_edges_from_decoded(
    decoded: torch.Tensor,
    n_target: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Build the meta-path-induced adjacency from a decoded sketch.

    Each row v of ``decoded`` lists the bottom-k T_0 ids that v's meta-path
    sketch retained. Treat each (v, decoded[v, j]) for valid slots as an
    edge in the meta-path-induced graph, undirected (HANConv attends per
    relation in one direction only, but symmetrising helps the encoder
    see both ends).

    This is the sketch-as-sparsifier construction in service of the
    sketch-as-feature backbone: HAN gets meta-path-specific adjacencies
    derived from the same sketch its target features came from.
    """
    n, k = decoded.shape
    src = torch.arange(n).unsqueeze(1).expand(-1, k)              # [n, k]
    dst = decoded                                                  # [n, k]
    valid = dst >= 0
    src = src[valid]
    dst = dst[valid]

    # Undirected + optional self-loops, with PyG's coalesce for dedup.
    from torch_geometric.utils import coalesce, add_self_loops as _add_sl
    ei = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ]).long()
    if add_self_loops:
        ei, _ = _add_sl(ei, num_nodes=n_target)
    ei = coalesce(ei, num_nodes=n_target)
    return ei


def _macro_f1_multilabel(logits: torch.Tensor, y_true: torch.Tensor,
                         threshold: float = 0.5) -> float:
    """Per-class binary F1, averaged. logits: [N, C]; y_true: [N, C] in {0,1}."""
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--meta-paths", nargs="+",
                   help="Override default meta-paths (each comma-separated relations)")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--depth", type=int, default=2,
                   help="Number of HANConv layers")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--agg", choices=["mean", "sum", "attention"], default="mean")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--cache",
                   help="Path to cache the extracted SketchBundle (skip extraction "
                        "on rerun). Default: results/<dataset>/sketch_bundle_k<K>.pt")
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test mode: epochs=20, k=8")
    args = p.parse_args()

    if args.quick:
        args.epochs = 20
        args.k = 8

    torch.manual_seed(args.seed)

    cfg = config.get_dataset_config(args.dataset)
    target_type = cfg.target_node
    out_dir = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = args.meta_paths or _DEFAULT_META_PATHS.get(args.dataset)
    if not meta_paths:
        print(f"No default meta-paths registered for {args.dataset}; pass --meta-paths.",
              file=sys.stderr)
        return 2

    print(f"[load] dataset={args.dataset} target={target_type}")
    g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, target_type)
    n_target = g[target_type].num_nodes
    n_classes = int(info["num_classes"])
    labels = info["labels"]
    masks = info["masks"]
    train_mask = masks["train"].bool()
    val_mask = masks["val"].bool()
    test_mask = masks["test"].bool()

    # Detect labelling regime.
    multi_label = labels.dim() == 2 and labels.dtype in (torch.float32, torch.float64)
    # Single-label datasets may use -100 as ignore-index (HNE_PubMed). Mask
    # those out of every split so we never train / eval on them.
    if not multi_label:
        valid = labels != -100
        train_mask = train_mask & valid
        val_mask = val_mask & valid
        test_mask = test_mask & valid
    print(f"[labels] multi_label={multi_label}  "
          f"train={int(train_mask.sum())}  val={int(val_mask.sum())}  "
          f"test={int(test_mask.sum())}")

    # Extraction (cached).
    cache_path = Path(args.cache) if args.cache \
        else out_dir / f"sketch_bundle_k{args.k}_seed{args.seed}.pt"
    if cache_path.exists():
        print(f"[extract] reusing cached sketch bundle: {cache_path}")
        bundle = SketchBundle.load(cache_path)
        if bundle.k != args.k:
            print(f"  cache k={bundle.k} != --k {args.k}; rebuilding")
            cache_path.unlink()
            bundle = None
        elif sorted(bundle.meta_paths) != sorted(meta_paths):
            print(f"  cache meta-paths differ; rebuilding")
            cache_path.unlink()
            bundle = None
    else:
        bundle = None

    if bundle is None:
        t0 = time.perf_counter()
        bundle = extract_sketches(
            g, meta_paths, target_type=target_type,
            k=args.k, seed=args.seed, device="cpu",
        )
        t_extract = time.perf_counter() - t0
        print(f"[extract] propagated in {t_extract:.2f}s")
        bundle.save(cache_path)
        print(f"[extract] cached: {cache_path}")

    # Decode each meta-path's sketch into [N, k] T_0 ids.
    decoded_by_mp = {
        mp: decode_sketches(sk, bundle.sorted_hashes, bundle.sort_indices)
        for mp, sk in bundle.sketches_by_mp.items()
    }
    fill_rate = {
        mp: float((d >= 0).float().mean())
        for mp, d in decoded_by_mp.items()
    }
    print("[decode] fill rate per meta-path:")
    for mp, rate in fill_rate.items():
        print(f"   {mp}: {rate:.3f}")

    # Native target-type features (if any). Fusing them into the encoder is
    # important — the sketch alone gives at most |T_0|·k log slots of signal,
    # but the canonical HGB benchmarks already include text / topic features
    # per author, and ignoring them is a self-imposed handicap.
    base_x = None
    base_dim_by_type = None
    if "x" in g[target_type] and g[target_type].x is not None:
        base_x = g[target_type].x
        base_dim_by_type = {target_type: int(base_x.size(1))}
        print(f"[features] using native {target_type} features: {tuple(base_x.shape)}")

    node_types = [target_type]
    han = SketchHAN(
        n_t0=n_target,
        emb_dim=args.emb_dim,
        out_dim=n_classes,
        meta_paths=meta_paths,
        node_types=node_types,
        target_type=target_type,
        base_dim_by_type=base_dim_by_type,
        n_layers=args.depth,
        n_heads=args.n_heads,
        dropout=args.dropout,
        agg=args.agg,
    )

    # Per-meta-path edges decoded from the sketch — sketch-as-feature with
    # the meta-path adjacency also derived from the sketch (so HANConv's
    # per-relation attention has meaningful structure to attend over,
    # and we don't smuggle in the original heterogeneous graph).
    edge_index_dict = {}
    edge_counts = {}
    for mp in meta_paths:
        rel = "mp_" + SketchHAN._sanitize(mp)
        ei = _build_mp_edges_from_decoded(
            decoded_by_mp[mp], n_target=n_target, add_self_loops=True
        )
        edge_index_dict[(target_type, rel, target_type)] = ei
        edge_counts[mp] = int(ei.size(1))
    print(f"[edges] sketch-derived adjacency edge counts: {edge_counts}")

    sketch_inputs = {mp: (decoded_by_mp[mp], base_x) for mp in meta_paths}

    # Move to device.
    device = torch.device(args.device)
    print(f"[device] {device}")
    han = han.to(device)
    sketch_inputs = {
        mp: (d.to(device), bx.to(device) if bx is not None else None)
        for mp, (d, bx) in sketch_inputs.items()
    }
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    n_by_type = {target_type: n_target}
    labels_d = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    opt = torch.optim.Adam(han.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    wait = 0

    print(f"[train] epochs={args.epochs} lr={args.lr} dropout={args.dropout}")
    for epoch in range(1, args.epochs + 1):
        han.train()
        opt.zero_grad()
        logits = han(sketch_inputs, edge_index_dict, n_by_type)
        if multi_label:
            loss = F.binary_cross_entropy_with_logits(
                logits[train_mask], labels_d[train_mask].float()
            )
        else:
            loss = F.cross_entropy(logits[train_mask], labels_d[train_mask])
        loss.backward()
        opt.step()

        han.eval()
        with torch.no_grad():
            logits = han(sketch_inputs, edge_index_dict, n_by_type)
            if multi_label:
                train_f1 = _macro_f1_multilabel(
                    logits[train_mask], labels_d[train_mask]
                )
                val_f1 = _macro_f1_multilabel(
                    logits[val_mask], labels_d[val_mask]
                )
            else:
                pred = logits.argmax(dim=-1)
                train_f1 = _macro_f1(pred[train_mask], labels_d[train_mask], n_classes)
                val_f1 = _macro_f1(pred[val_mask], labels_d[val_mask], n_classes)

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in han.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}  loss={loss.item():.4f}  "
                  f"train_f1={train_f1:.4f}  val_f1={val_f1:.4f}  "
                  f"(best={best_val_f1:.4f}@{best_epoch})")

        if wait >= args.patience:
            print(f"  early stop at ep {epoch}")
            break

    if best_state is not None:
        han.load_state_dict(best_state)
    han.eval()
    with torch.no_grad():
        logits = han(sketch_inputs, edge_index_dict, n_by_type)
        if multi_label:
            test_f1 = _macro_f1_multilabel(logits[test_mask], labels_d[test_mask])
        else:
            pred = logits.argmax(dim=-1)
            test_f1 = _macro_f1(pred[test_mask], labels_d[test_mask], n_classes)
    print(f"[result] val_f1={best_val_f1:.4f} (ep {best_epoch})  test_f1={test_f1:.4f}")

    # Persist a small results JSON next to the bundle.
    out = {
        "dataset": args.dataset,
        "target_type": target_type,
        "meta_paths": meta_paths,
        "k": args.k,
        "depth": args.depth,
        "n_heads": args.n_heads,
        "emb_dim": args.emb_dim,
        "agg": args.agg,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "test_f1": test_f1,
        "fill_rate": fill_rate,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    res_path = out_dir / f"sketch_feature_pilot_k{args.k}_seed{args.seed}.json"
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[save] {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
