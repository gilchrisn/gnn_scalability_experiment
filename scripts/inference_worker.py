"""
inference_worker.py — SAGE inference subprocess.

Loads a pre-materialized graph (adjacency .adj or edge_index .pt) plus frozen
model weights, runs one forward pass, saves the embedding to disk, and reports
peak RSS + macro-F1 to stdout.

Because this runs as a subprocess with a fresh Python process, RSS starts at
zero — no contamination from materialization memory or other methods' tensors.

Stdout lines (parsed by exp3_inference.py)
------------------------------------------
    inf_peak_ram_mb: X.XX    — peak RSS during inference (MB)
    inf_time: X.XXXXXX       — wall-clock seconds for forward pass
    inf_f1: X.XXXXXX         — macro-F1 on test nodes
    inf_de: X.XXXXXX         — Dirichlet energy of embeddings

Usage (internal — called by exp3_inference.py)
-----------------------------------------------
    python scripts/inference_worker.py \\
        --graph-file  <adj_or_pt>   \\
        --graph-type  adj|pt        \\
        --feat-file   <x.pt>        \\
        --weights     <model.pt>    \\
        --z-out       <z.pt>        \\
        --labels-file <labels.pt>   \\
        --mask-file   <mask.pt>     \\
        --n-target    N             \\
        --node-offset O             \\
        --in-dim      D             \\
        --num-classes C             \\
        --num-layers  L
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.models import get_model
from src.config import config
from src.analysis.metrics import DirichletEnergyMetric


def _peak_ram_mb() -> float:
    """Current process RSS in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph-file",  required=True)
    p.add_argument("--graph-type",  choices=["adj", "pt"], required=True)
    p.add_argument("--feat-file",   required=True)
    p.add_argument("--weights",     required=True)
    p.add_argument("--z-out",       required=True)
    p.add_argument("--labels-file", required=True)
    p.add_argument("--mask-file",   required=True)
    p.add_argument("--n-target",    type=int, required=True)
    p.add_argument("--node-offset", type=int, required=True)
    p.add_argument("--in-dim",      type=int, required=True)
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--num-layers",  type=int, required=True)
    args = p.parse_args()

    device = torch.device("cpu")

    # Load materialized graph
    if args.graph_type == "adj":
        # Parse C++ adjacency list format: "u v1 v2 ..."
        srcs, dsts = [], []
        with open(args.graph_file) as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                if not parts:
                    continue
                u_local = parts[0] - args.node_offset
                if u_local < 0 or u_local >= args.n_target:
                    continue
                for v_global in parts[1:]:
                    v_local = v_global - args.node_offset
                    if 0 <= v_local < args.n_target:
                        srcs.append(u_local)
                        dsts.append(v_local)
        if srcs:
            from torch_geometric.utils import coalesce, add_self_loops
            ei = torch.tensor([srcs, dsts], dtype=torch.long)
            ei = coalesce(ei, num_nodes=args.n_target)
            ei, _ = add_self_loops(ei, num_nodes=args.n_target)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
    else:
        ei = torch.load(args.graph_file, weights_only=True)

    x       = torch.load(args.feat_file,  weights_only=True)
    labels  = torch.load(args.labels_file, weights_only=True)
    mask    = torch.load(args.mask_file,   weights_only=True)

    x = F.pad(x, (0, max(0, args.in_dim - x.size(1))))

    g = Data(edge_index=ei, num_nodes=args.n_target)
    g.x = x

    model = get_model("SAGE", args.in_dim, args.num_classes,
                      config.HIDDEN_DIM, num_layers=args.num_layers).to(device)
    model.load_state_dict(torch.load(args.weights, weights_only=True,
                                     map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Snapshot RSS AFTER loading everything — before forward pass.
    rss_before = _peak_ram_mb()

    t0 = time.perf_counter()
    with torch.no_grad():
        z = model(g.x.to(device), g.edge_index.to(device))
    inf_time = time.perf_counter() - t0

    rss_after = _peak_ram_mb()
    inf_peak  = max(0.0, rss_after - rss_before)

    # F1 on test mask
    from torchmetrics.functional import f1_score as _f1
    valid  = mask & (labels >= 0)
    if valid.sum() > 0:
        f1 = _f1(z[valid].argmax(1), labels[valid],
                 task="multiclass", num_classes=args.num_classes,
                 average="macro").item()
    else:
        f1 = 0.0

    # Dirichlet energy
    de_metric = DirichletEnergyMetric()
    try:
        de = de_metric.calculate(z, ei)
    except Exception:
        de = 0.0

    # Save embedding for CKA comparison in parent
    os.makedirs(os.path.dirname(os.path.abspath(args.z_out)), exist_ok=True)
    torch.save(z.cpu(), args.z_out)

    # Also save layerwise intermediates for layerwise CKA
    layers_out = args.z_out.replace(".pt", "_layers.pt")
    try:
        intermediates = []
        x_tmp = g.x.to(device)
        ei_tmp = g.edge_index.to(device)
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                x_tmp = layer(x_tmp, ei_tmp)
                if i < model.num_layers - 1:
                    x_tmp = F.relu(x_tmp)
                intermediates.append(x_tmp.cpu().clone())
        torch.save(intermediates, layers_out)
    except Exception:
        pass

    print(f"inf_peak_ram_mb: {inf_peak:.2f}")
    print(f"inf_time: {inf_time:.6f}")
    print(f"inf_f1: {f1:.6f}")
    print(f"inf_de: {de:.6f}")


if __name__ == "__main__":
    main()
