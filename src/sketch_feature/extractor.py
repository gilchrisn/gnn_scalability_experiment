"""Run KMV propagation on a heterogeneous graph and produce the artefacts
needed by the sketch-as-feature consumer:

  * ``sketches_by_mp[mp]: [N_target, k]`` int64 — bottom-k hashes per node
  * ``sorted_hashes, sort_indices: [N_target]`` — decode table, shared
    across all meta-paths because they share the same start-type hash
    universe (this is the "typed identity universe" property in
    CURRENT_STATE.md §"Open architectural decisions").

The extractor reuses :class:`src.kernels.kmv.KMVSketchingKernel` for the
propagation. Hashes for the start type are seeded so calls across
different meta-paths produce the same T_0 hash universe — without that,
slot decoding would be inconsistent across meta-paths and the encoder's
per-T_0 embedding lookup would mix unrelated entities.

The artefacts are picklable; ``save`` and ``load`` round-trip them so
training scripts don't need to re-propagate every run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import HeteroData

from src.kernels.kmv import KMVSketchingKernel, INFINITY


@dataclass
class SketchBundle:
    """All artefacts the sketch-as-feature consumer needs at training time."""
    target_type: str
    n_target: int
    k: int
    meta_paths: List[str]
    sketches_by_mp: Dict[str, torch.Tensor]   # [N_target, k] int64
    sorted_hashes: torch.Tensor                # [N_target] int64
    sort_indices: torch.Tensor                 # [N_target] int64

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "target_type": self.target_type,
                "n_target": self.n_target,
                "k": self.k,
                "meta_paths": self.meta_paths,
                "sketches_by_mp": self.sketches_by_mp,
                "sorted_hashes": self.sorted_hashes,
                "sort_indices": self.sort_indices,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SketchBundle":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**blob)


def _parse_meta_path_str(mp_str: str, g: HeteroData) -> List[Tuple[str, str, str]]:
    """Convert ``"author_to_paper,paper_to_author"`` -> list of edge tuples.

    The relations must exist on ``g``. Each relation appears exactly once
    in the heterogeneous graph's edge_types.
    """
    rels = [r.strip() for r in mp_str.split(",") if r.strip()]
    by_rel: Dict[str, Tuple[str, str, str]] = {
        et[1]: et for et in g.edge_types
    }
    out: List[Tuple[str, str, str]] = []
    for r in rels:
        if r not in by_rel:
            raise ValueError(
                f"Relation {r!r} not in graph edge_types {list(by_rel)}"
            )
        out.append(by_rel[r])
    return out


def extract_sketches(
    g: HeteroData,
    meta_paths: List[str],
    target_type: str,
    k: int = 32,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> SketchBundle:
    """Propagate KMV per meta-path and return a :class:`SketchBundle`.

    Args:
        g: Heterogeneous graph (PyG ``HeteroData``).
        meta_paths: List of comma-separated relation chains, e.g.
            ``["author_to_paper,paper_to_author", "author_to_paper,paper_to_term,term_to_paper,paper_to_author"]``.
            Each chain MUST start and end at ``target_type`` (mirrored).
        target_type: The T_0 node type. All meta-paths must round-trip
            here so a shared decode table is meaningful.
        k: Sketch size per slot.
        seed: Seed for the start-type hash universe — fixed across
            meta-paths so the same author id has the same hash in every
            sketch.
        device: Where to run the propagation.

    Returns:
        SketchBundle.
    """
    device = torch.device(device)
    n_target = g[target_type].num_nodes

    # 1) Build the shared T_0 hash universe ONCE so cross-meta-path slot
    #    decoding stays consistent.
    gen = torch.Generator(device=device).manual_seed(seed)
    t0_hashes = torch.randint(
        0, 2**63 - 1, (n_target,), generator=gen,
        dtype=torch.int64, device=device,
    )
    sorted_hashes, sort_indices = torch.sort(t0_hashes)

    sketches_by_mp: Dict[str, torch.Tensor] = {}
    for mp_str in meta_paths:
        chain = _parse_meta_path_str(mp_str, g)
        if chain[0][0] != target_type:
            raise ValueError(
                f"Meta-path {mp_str!r} starts at {chain[0][0]!r}, expected {target_type!r}"
            )
        if chain[-1][2] != target_type:
            raise ValueError(
                f"Meta-path {mp_str!r} ends at {chain[-1][2]!r}, expected {target_type!r}"
            )

        # Run propagation on a shallow copy to avoid mutating g (the kernel
        # writes a temporary `.sketch` attribute on each node type).
        g_local = g
        kernel = KMVSketchingKernel(k=k, nk=1, device=device)

        # Inject the shared T_0 hash universe by monkey-patching the
        # randint call inside _propagate_sketches. The kernel uses
        # torch.randint at import-time module level, so we override it
        # via a context. This is still cleaner than copying the whole
        # propagation method.
        sk = _propagate_with_fixed_universe(
            kernel, g_local, chain, t0_hashes, target_type
        )
        sketches_by_mp[mp_str] = sk.cpu()

    return SketchBundle(
        target_type=target_type,
        n_target=n_target,
        k=k,
        meta_paths=list(meta_paths),
        sketches_by_mp=sketches_by_mp,
        sorted_hashes=sorted_hashes.cpu(),
        sort_indices=sort_indices.cpu(),
    )


def _propagate_with_fixed_universe(
    kernel: KMVSketchingKernel,
    g: HeteroData,
    chain: List[Tuple[str, str, str]],
    t0_hashes: torch.Tensor,
    target_type: str,
) -> torch.Tensor:
    """Replicate ``KMVSketchingKernel._propagate_sketches`` but with a
    pre-built hash universe so meta-paths share the T_0 identity.

    Returns the bottom-k slots at the meta-path's end node type, shape
    ``[N_target, k]`` int64.
    """
    device = kernel.device
    k = kernel.k
    nk = kernel.nk

    all_ntypes = set()
    for s, _, d in chain:
        all_ntypes.add(s)
        all_ntypes.add(d)

    # Initialize sketches.
    for nt in all_ntypes:
        n = g[nt].num_nodes
        sketches = torch.full((n, nk, k), INFINITY, dtype=torch.int64, device=device)
        if nt == target_type:
            sketches[:, :, 0] = t0_hashes.unsqueeze(1)
        g[nt].sketch = sketches

    for src_type, rel, dst_type in chain:
        edge_index = g[src_type, rel, dst_type].edge_index.to(device)
        src_sketches = g[src_type].sketch
        dst_sketches_own = g[dst_type].sketch
        new_dst = kernel._propagate_and_merge(
            edge_index, src_sketches, dst_sketches_own
        )
        g[dst_type].sketch = new_dst

    out = g[chain[-1][2]].sketch.squeeze(1).contiguous()  # [N_end, k]

    # Cleanup so subsequent calls start clean.
    for nt in all_ntypes:
        if hasattr(g[nt], "sketch"):
            delattr(g[nt], "sketch")

    return out
