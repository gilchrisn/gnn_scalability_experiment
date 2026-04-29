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

import torch.nn as nn

from src.config import config
from src.data import DatasetFactory
from src.sketch_feature import (
    SketchBundle,
    SketchFeatureEncoder,
    SketchHAN,
    decode_sketches,
    extract_sketches,
)


class LoneTypedMLP(nn.Module):
    """Pure LoNe-typed: per-meta-path encoder -> fuse views -> MLP head.

    No graph convolution. The sketch's bottom-k slot embeddings already
    summarise the L-hop neighborhood; an MLP classifier on top is the
    cleanest "the sketch is the message passing" framing (Kutzkov AAAI'23).

    View fusion (mp_fusion):
        mean      — uniform average over meta-paths (default).
        attention — learned scalar weight per meta-path, softmax-normalised.
                    Helps when meta-paths differ in informativeness, e.g.
                    HGB_IMDB where keyword/actor/director have very
                    different fill rates.
        node_attn — per-node attention over views (heavier; one logit per
                    node per meta-path).
    """

    def __init__(self, n_t0: int, emb_dim: int, hidden_dim: int, out_dim: int,
                 meta_paths: list, target_base_dim: int, agg: str,
                 dropout: float, share_t0_embedding: bool = True,
                 mp_fusion: str = "mean"):
        super().__init__()
        if mp_fusion not in {"mean", "attention", "node_attn"}:
            raise ValueError(f"mp_fusion must be mean|attention|node_attn; got {mp_fusion!r}")
        self.meta_paths = list(meta_paths)
        self.mp_fusion = mp_fusion
        M = len(self.meta_paths)
        if share_t0_embedding:
            self._shared = nn.Embedding(n_t0 + 1, emb_dim, padding_idx=n_t0)
        else:
            self._shared = None
        self.encoders = nn.ModuleDict({
            SketchHAN._sanitize(mp): SketchFeatureEncoder(
                n_t0=n_t0, emb_dim=emb_dim, agg=agg, base_dim=target_base_dim,
                dropout=dropout, shared_embedding=self._shared,
            ) for mp in self.meta_paths
        })
        if mp_fusion == "attention":
            self.view_logits = nn.Parameter(torch.zeros(M))
        elif mp_fusion == "node_attn":
            self.view_attn = nn.Linear(emb_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, sketch_inputs):
        per_mp = []
        for mp in self.meta_paths:
            decoded, base_x = sketch_inputs[mp]
            per_mp.append(self.encoders[SketchHAN._sanitize(mp)](decoded, base_x))
        stacked = torch.stack(per_mp, dim=0)               # [M, N, d]

        if self.mp_fusion == "mean":
            x = stacked.mean(dim=0)
        elif self.mp_fusion == "attention":
            # Global per-view weights — same for every node.
            w = F.softmax(self.view_logits, dim=0)         # [M]
            x = (stacked * w.view(-1, 1, 1)).sum(dim=0)    # [N, d]
        else:                                              # node_attn
            # Per-node attention over views.
            scores = self.view_attn(stacked).squeeze(-1)   # [M, N]
            w = F.softmax(scores, dim=0)                   # [M, N]
            x = (stacked * w.unsqueeze(-1)).sum(dim=0)     # [N, d]

        return self.classifier(x)


# Default meta-paths per HGB dataset for the pilot. Both must be mirrored
# (start == end at the target type) for the extractor to accept them.
_DEFAULT_META_PATHS = {
    "HGB_DBLP": [
        # APA + APVPA matches the HAN paper's protocol on DBLP (venues are
        # far more class-discriminative than terms — switching from APTPA
        # to APVPA moves test F1 from 0.79 to 0.92 on this pilot).
        "author_to_paper,paper_to_author",
        "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author",
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
    p.add_argument("--mp-fusion",
                   choices=["mean", "attention", "node_attn"],
                   default="mean",
                   help="How to fuse per-meta-path features in the MLP "
                        "backbone. attention = global per-view softmax; "
                        "node_attn = per-node attention over views. Helps "
                        "on datasets with uneven meta-path fill rates "
                        "(e.g. HGB_IMDB).")
    p.add_argument("--backbone",
                   choices=["mlp", "han_real_edges", "han_sketch_edges"],
                   default="han_sketch_edges",
                   help=("Backbone fed by the sketch features. mlp = pure "
                         "LoNe-typed; han_real_edges = HAN over the original "
                         "heterogeneous graph's edge_index_dict; "
                         "han_sketch_edges = HAN over edges decoded from the "
                         "sketch itself."))
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

    sketch_inputs = {mp: (decoded_by_mp[mp], base_x) for mp in meta_paths}
    device = torch.device(args.device)
    sketch_inputs = {
        mp: (d.to(device), bx.to(device) if bx is not None else None)
        for mp, (d, bx) in sketch_inputs.items()
    }
    labels_d = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Build model + the forward closure that maps it to a uniform call site.
    target_base_dim = (base_dim_by_type or {}).get(target_type, 0)

    if args.backbone == "mlp":
        # Pure LoNe-typed: sketch encoder -> MLP, no graph convolution.
        model = LoneTypedMLP(
            n_t0=n_target, emb_dim=args.emb_dim,
            hidden_dim=args.emb_dim, out_dim=n_classes,
            meta_paths=meta_paths,
            target_base_dim=target_base_dim,
            agg=args.agg, dropout=args.dropout,
            mp_fusion=args.mp_fusion,
        ).to(device)
        forward_fn = lambda m: m(sketch_inputs)
        edge_counts = {"(no edges; MLP backbone)": 0}

    elif args.backbone == "han_real_edges":
        # Sketch features for the target type; HAN propagates over the
        # original heterogeneous graph's edges. Non-target types get their
        # native features (or a learned embedding if absent).
        node_types = list(g.node_types)
        edge_types = list(g.edge_types)
        # Build x_dict for non-target types; project all to emb_dim so
        # HANConv has a uniform input feature size.
        type_projects = nn.ModuleDict()
        type_embs = nn.ParameterDict()
        for nt in node_types:
            if nt == target_type:
                continue
            if "x" in g[nt] and g[nt].x is not None:
                type_projects[nt] = nn.Linear(g[nt].x.size(1), args.emb_dim)
            else:
                type_embs[nt] = nn.Parameter(
                    torch.randn(g[nt].num_nodes, args.emb_dim)
                )
        type_projects = type_projects.to(device)
        type_embs = type_embs.to(device)

        edge_index_dict = {et: g[et].edge_index.to(device) for et in edge_types}
        edge_counts = {f"{et}": int(g[et].edge_index.size(1)) for et in edge_types}

        model = SketchHAN(
            n_t0=n_target, emb_dim=args.emb_dim, out_dim=n_classes,
            meta_paths=meta_paths,
            node_types=node_types, target_type=target_type,
            base_dim_by_type=base_dim_by_type,
            n_layers=args.depth, n_heads=args.n_heads, dropout=args.dropout,
            agg=args.agg,
        ).to(device)

        # Override SketchHAN.forward by monkey-patching: instead of using the
        # synthetic per-meta-path edges, feed real heterogeneous edges with
        # one HAN call. We bypass _ensure_type_embedding by giving x_dict
        # ourselves.
        def forward_fn_real(m: SketchHAN):
            # Per-meta-path encoders -> target features
            per_mp = [
                m.encoders[SketchHAN._sanitize(mp)](*sketch_inputs[mp])
                for mp in meta_paths
            ]
            x_target = torch.stack(per_mp, dim=0).mean(dim=0)
            x_dict = {target_type: x_target}
            for nt in node_types:
                if nt == target_type:
                    continue
                if nt in type_projects:
                    x_dict[nt] = type_projects[nt](g[nt].x.to(device))
                else:
                    x_dict[nt] = type_embs[nt]
            # We need a HANConv stack expecting THIS metadata; SketchHAN
            # was built with synthetic per-mp edge_types. Build one ad-hoc.
            for layer in m.han_layers:
                # Re-initialise with the real metadata. Cheap because lin
                # layers are reset per relation; we only rebuild metadata
                # to avoid AssertionError on edge type mismatch.
                pass
            return m.classifier(x_dict[target_type])
        # Clean approach: HAN with real edges needs a SketchHAN built with
        # the real (node_types, edge_types) metadata. Rebuild it that way.
        from torch_geometric.nn import HANConv
        han_layers = nn.ModuleList()
        in_d = args.emb_dim
        for _ in range(args.depth):
            han_layers.append(HANConv(
                in_channels=in_d, out_channels=args.emb_dim,
                metadata=(node_types, edge_types),
                heads=args.n_heads, dropout=args.dropout,
            ))
            in_d = args.emb_dim
        han_layers = han_layers.to(device)
        classifier_real = nn.Linear(args.emb_dim, n_classes).to(device)

        # We hand over to a fresh forward_fn that uses the real layers.
        encoders = model.encoders  # reuse the encoders + shared embedding

        def forward_fn(_unused):
            per_mp = [
                encoders[SketchHAN._sanitize(mp)](*sketch_inputs[mp])
                for mp in meta_paths
            ]
            x_target = torch.stack(per_mp, dim=0).mean(dim=0)
            x_dict = {target_type: x_target}
            for nt in node_types:
                if nt == target_type:
                    continue
                if nt in type_projects:
                    x_dict[nt] = type_projects[nt](g[nt].x.to(device))
                else:
                    x_dict[nt] = type_embs[nt]
            for layer in han_layers:
                x_dict = layer(x_dict, edge_index_dict)
                x_dict = {k: F.elu(v) if v is not None else v for k, v in x_dict.items()}
            return classifier_real(x_dict[target_type])

        # Combine all parameters into one trainable bag.
        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoders = encoders
                self.han_layers = han_layers
                self.classifier_real = classifier_real
                self.type_projects = type_projects
                self.type_embs = type_embs
        model = _Wrapper().to(device)

    else:  # han_sketch_edges (default — original behaviour)
        node_types = [target_type]
        edge_index_dict = {}
        edge_counts = {}
        for mp in meta_paths:
            rel = "mp_" + SketchHAN._sanitize(mp)
            ei = _build_mp_edges_from_decoded(
                decoded_by_mp[mp], n_target=n_target, add_self_loops=True
            ).to(device)
            edge_index_dict[(target_type, rel, target_type)] = ei
            edge_counts[mp] = int(ei.size(1))
        print(f"[edges] sketch-derived adjacency edge counts: {edge_counts}")

        model = SketchHAN(
            n_t0=n_target, emb_dim=args.emb_dim, out_dim=n_classes,
            meta_paths=meta_paths,
            node_types=node_types, target_type=target_type,
            base_dim_by_type=base_dim_by_type,
            n_layers=args.depth, n_heads=args.n_heads, dropout=args.dropout,
            agg=args.agg,
        ).to(device)
        n_by_type_local = {target_type: n_target}
        forward_fn = lambda m: m(sketch_inputs, edge_index_dict, n_by_type_local)

    print(f"[backbone] {args.backbone}  edges={edge_counts}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    wait = 0

    print(f"[train] epochs={args.epochs} lr={args.lr} dropout={args.dropout}")
    train_t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = forward_fn(model)
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
            logits = forward_fn(model)
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
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
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

    train_time_s = time.perf_counter() - train_t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = forward_fn(model)
        if multi_label:
            test_f1 = _macro_f1_multilabel(logits[test_mask], labels_d[test_mask])
        else:
            pred = logits.argmax(dim=-1)
            test_f1 = _macro_f1(pred[test_mask], labels_d[test_mask], n_classes)
    print(f"[result] val_f1={best_val_f1:.4f} (ep {best_epoch})  "
          f"test_f1={test_f1:.4f}  train_time={train_time_s:.1f}s")

    # Persist a small results JSON next to the bundle.
    out = {
        "dataset": args.dataset,
        "target_type": target_type,
        "meta_paths": meta_paths,
        "backbone": args.backbone,
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
        "train_time_s": train_time_s,
        "fill_rate": fill_rate,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    bb_tag = "" if args.backbone == "han_sketch_edges" else f"_{args.backbone}"
    res_path = out_dir / f"sketch_feature_pilot_k{args.k}{bb_tag}_seed{args.seed}.json"
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[save] {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
