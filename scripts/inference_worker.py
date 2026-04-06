"""
inference_worker.py — SAGE inference subprocess.

Loads a pre-materialized graph (adjacency .adj or edge_index .pt) plus frozen
model weights, runs one forward pass, saves the embedding to disk, and reports
peak allocation + macro-F1 to stdout.

Because this runs as a subprocess with a fresh Python process, there is no
contamination from materialization memory or other methods' tensors.

Memory measurement
------------------
tracemalloc is started immediately before the forward pass and stopped after.
tracemalloc.clear_traces() resets the peak counter so the reported value is
the true peak Python-level allocation during the forward pass only, not
accumulated from earlier setup.  This captures all tensor constructions through
Python's PyMem_Raw* layer regardless of whether new OS pages are faulted in
(PyTorch's CPU caching allocator reuses mapped pages, which makes RSS-delta
approaches report 0 on small graphs).

Forward pass uses strict SpMM (torch.sparse_csr_tensor + torch.sparse.mm).
PyG's MessagePassing engine is bypassed entirely, so no dense [|E|, d] message
buffer is allocated at any point.  Each layer computes:

    agg_i  = (1 / deg_i) * sum_{j -> i} h_j        [SpMM, O(N*d) memory]
    h_i^+  = lin_l( agg_i ) + lin_r( h_i )          [SAGEConv mean aggregation]

Stdout lines (parsed by exp3_inference.py)
------------------------------------------
    inf_peak_ram_mb: X.XX    — peak tracemalloc allocation during forward (MB)
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
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.models import get_model
from src.config import config
from src.analysis.metrics import DirichletEnergyMetric


# ── Strict SpMM graph ops ────────────────────────────────────────────────────

def _build_normalized_adj_csr(
    ei: torch.Tensor,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Build D^{-1} A as a torch.sparse_csr_tensor.

    A[dst, src] = 1 / deg(dst)  for each edge (src -> dst).

    SpMM with this matrix computes mean neighbourhood aggregation:
        (D^{-1} A) @ x  =>  row i = mean_{j in N(i)} x_j
    Memory: O(N + E) for the sparse structure, O(N * d) for the matmul.
    No dense [|E|, d] buffer is ever allocated.
    """
    if ei.size(1) == 0:
        crow = torch.zeros(n + 1, dtype=torch.long, device=device)
        col  = torch.empty(0, dtype=torch.long, device=device)
        vals = torch.empty(0, dtype=torch.float32, device=device)
        return torch.sparse_csr_tensor(crow, col, vals, size=(n, n))

    src = ei[0].to(device)
    dst = ei[1].to(device)

    # Degree = number of incoming neighbours per dst node
    deg = torch.zeros(n, dtype=torch.float32, device=device)
    deg.scatter_add_(0, dst, torch.ones(src.size(0), dtype=torch.float32, device=device))
    deg_inv = deg.clamp(min=1.0).reciprocal()   # safe: no isolated-node NaN

    # Edge weights: 1 / deg(dst_node)
    vals = deg_inv[dst]                          # shape [|E|]

    # CSR requires rows (dst) in non-decreasing order.
    # Use lexicographic sort (dst, src) so duplicate (dst,src) pairs land
    # adjacently — coalesce was applied upstream so duplicates are absent.
    perm   = torch.argsort(dst * n + src)        # int64 safe for n < 3e9
    dst_s  = dst[perm]
    src_s  = src[perm]
    vals_s = vals[perm]

    # crow_indices: prefix-sum of per-row edge counts
    crow = torch.zeros(n + 1, dtype=torch.long, device=device)
    crow[1:].scatter_add_(0, dst_s, torch.ones_like(dst_s))
    crow = crow.cumsum(0)

    return torch.sparse_csr_tensor(crow, src_s, vals_s, size=(n, n))


def _sage_spmm_forward(
    model,
    x: torch.Tensor,
    adj_csr: torch.Tensor,
) -> torch.Tensor:
    """Strict SpMM forward pass for SAGE.  Bypasses PyG MessagePassing entirely.

    Each layer:
        agg   = adj_csr @ x                  # SpMM, O(N*d) — mean aggregation
        x_out = lin_l(agg) + lin_r(x)        # SAGEConv mean formula
        x_out = ReLU(x_out)                  # (all but last layer)

    adj_csr already encodes D^{-1} A, so the SpMM is the exact paper Eq. 1
    neighbourhood term without any intermediate message buffer.

    Raises RuntimeError if SAGEConv weight attributes cannot be located.
    """
    for i, layer in enumerate(model.layers):
        # Strict SpMM: O(N * in_dim) working memory, no [|E|, d] allocation
        agg = torch.sparse.mm(adj_csr, x)    # [N, in_dim]

        # SAGEConv weight attributes (PyG >= 2.0 uses lin_l / lin_r)
        if hasattr(layer, "lin_l") and hasattr(layer, "lin_r"):
            x_out = layer.lin_l(agg) + layer.lin_r(x)
        elif hasattr(layer, "lin") and hasattr(layer, "lin_r"):
            # Some older PyG builds use lin / lin_r
            x_out = layer.lin(agg) + layer.lin_r(x)
        else:
            available = [name for name, _ in layer.named_children()]
            raise RuntimeError(
                f"Cannot find SAGEConv weight attributes on layer {i}. "
                f"Detected children: {available}"
            )

        if i < model.num_layers - 1:
            x_out = F.relu(x_out)
        x = x_out
    return x


def _sage_spmm_layerwise(
    model,
    x: torch.Tensor,
    adj_csr: torch.Tensor,
) -> list[torch.Tensor]:
    """Same as _sage_spmm_forward but returns a list of per-layer outputs."""
    intermediates: list[torch.Tensor] = []
    for i, layer in enumerate(model.layers):
        agg = torch.sparse.mm(adj_csr, x)

        if hasattr(layer, "lin_l") and hasattr(layer, "lin_r"):
            x_out = layer.lin_l(agg) + layer.lin_r(x)
        elif hasattr(layer, "lin") and hasattr(layer, "lin_r"):
            x_out = layer.lin(agg) + layer.lin_r(x)
        else:
            available = [name for name, _ in layer.named_children()]
            raise RuntimeError(
                f"Cannot find SAGEConv weight attributes on layer {i}. "
                f"Detected children: {available}"
            )

        if i < model.num_layers - 1:
            x_out = F.relu(x_out)
        x = x_out
        intermediates.append(x.cpu().clone())
    return intermediates


# ── Main ─────────────────────────────────────────────────────────────────────

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

    # ── Load materialized graph ───────────────────────────────────────────
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

    x      = torch.load(args.feat_file,   weights_only=True)
    labels = torch.load(args.labels_file, weights_only=True)
    mask   = torch.load(args.mask_file,   weights_only=True)

    x = F.pad(x, (0, max(0, args.in_dim - x.size(1))))

    # ── Model ────────────────────────────────────────────────────────────
    model = get_model("SAGE", args.in_dim, args.num_classes,
                      config.HIDDEN_DIM, num_layers=args.num_layers).to(device)
    model.load_state_dict(torch.load(args.weights, weights_only=True,
                                     map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # ── Build D^{-1}A CSR — O(N+E), no dense buffer ──────────────────────
    # Self-loops were added above (adj format) or are expected in .pt files;
    # normalization includes them in the degree count.
    n = args.n_target
    adj_csr = _build_normalized_adj_csr(ei, n, device)

    # ── Inference — strict SpMM, measured with tracemalloc ───────────────
    # tracemalloc.clear_traces() resets the peak counter so we measure only
    # the forward pass, not allocations from graph loading or model init.
    x_dev = x.to(device)
    tracemalloc.start()
    tracemalloc.clear_traces()
    t0 = time.perf_counter()
    with torch.no_grad():
        z = _sage_spmm_forward(model, x_dev, adj_csr)
    inf_time = time.perf_counter() - t0
    _, inf_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    inf_peak = inf_peak_bytes / 1e6

    # ── F1 on test mask ───────────────────────────────────────────────────
    from torchmetrics.functional import f1_score as _f1
    valid = mask & (labels >= 0)
    if valid.sum() > 0:
        f1 = _f1(z[valid].argmax(1), labels[valid],
                 task="multiclass", num_classes=args.num_classes,
                 average="macro").item()
    else:
        f1 = 0.0

    # ── Dirichlet energy ──────────────────────────────────────────────────
    de_metric = DirichletEnergyMetric()
    try:
        de = de_metric.calculate(z, ei)
    except Exception:
        de = 0.0

    # ── Save embedding ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.z_out)), exist_ok=True)
    torch.save(z.cpu(), args.z_out)

    # ── Layerwise intermediates for CKA ──────────────────────────────────
    layers_out = args.z_out.replace(".pt", "_layers.pt")
    try:
        with torch.no_grad():
            intermediates = _sage_spmm_layerwise(model, x_dev, adj_csr)
        torch.save(intermediates, layers_out)
    except Exception:
        pass

    print(f"inf_peak_ram_mb: {inf_peak:.2f}")
    print(f"inf_time: {inf_time:.6f}")
    print(f"inf_f1: {f1:.6f}")
    print(f"inf_de: {de:.6f}")


if __name__ == "__main__":
    main()
