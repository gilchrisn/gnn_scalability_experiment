"""HGNN backbones consuming sketch features as input.

Two backbones are exposed as ablations per CURRENT_STATE.md §"Open
architectural decisions":

    SketchHAN       — HAN (Wang et al. WWW'19): meta-path-aware semantic
                      attention on top of per-node sketch features.
    SketchSimpleHGN — Simple-HGN (Lv et al. KDD'21): SOTA HGB baseline,
                      treats edge types via learnable type vectors.

Both backbones expect input features that have already been produced by
:class:`src.sketch_feature.encoder.SketchFeatureEncoder` — one per
meta-path. The HAN variant fuses across meta-paths via its built-in
semantic attention; the Simple-HGN variant concatenates them.

For Path 1's first pass, ``SketchHAN`` is the primary backbone (aligns with
the meta-path framing) and ``SketchSimpleHGN`` is the ablation. Both are
intentionally thin wrappers over ``torch_geometric.nn`` ops so the heavy
lifting is in well-tested upstream code.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HANConv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric.nn.HANConv unavailable; install PyG >= 2.4"
    ) from e

from .encoder import SketchFeatureEncoder


# ---------------------------------------------------------------------------
# HAN backbone
# ---------------------------------------------------------------------------


class SketchHAN(nn.Module):
    """HAN consuming sketch features per meta-path.

    The forward expects a dict ``sketch_inputs`` mapping each meta-path label
    to a ``(decoded, base_x)`` tuple, plus the heterogeneous ``edge_index_dict``
    of the original graph. Per meta-path:

        x_mp = SketchFeatureEncoder(decoded, base_x)   # [N, hidden]

    The per-meta-path feature tensors are then passed into ``HANConv`` as the
    target-type input, with stub features for non-target types (HAN's
    semantic attention produces one head per relation).

    Args:
        n_t0: Size of T_0 entity universe.
        emb_dim: Sketch embedding dim, == hidden dim into HANConv.
        out_dim: Output classes (NC) or embedding dim (LP / similarity).
        meta_paths: List of meta-path labels (used as keys in HANConv).
        node_types: List of all node types in the heterogeneous graph; HAN
            needs to know them to build its parameter dict.
        target_type: The ``T_0`` node type (where features and labels live).
        base_dim_by_type: Optional ``Dict[ntype, int]`` of native feature
            dims per node type. If a type is missing, a learnable type
            embedding is used instead.
        n_layers: Number of HANConv layers (depth L).
        n_heads: Number of attention heads.
        dropout: Dropout in encoder + HANConv.
        agg: Slot aggregation for the encoder ("mean" | "sum" | "attention").
    """

    def __init__(
        self,
        n_t0: int,
        emb_dim: int,
        out_dim: int,
        meta_paths: List[str],
        node_types: List[str],
        target_type: str,
        base_dim_by_type: Optional[Dict[str, int]] = None,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.5,
        agg: str = "mean",
        share_t0_embedding: bool = True,
    ) -> None:
        super().__init__()

        if target_type not in node_types:
            raise ValueError(f"target_type {target_type!r} not in node_types")
        self.target_type = target_type
        self.meta_paths = list(meta_paths)
        self.node_types = list(node_types)
        base_dim_by_type = base_dim_by_type or {}

        # Typed identity universe: by default share ONE embedding table E
        # for the T_0 entities across every per-meta-path encoder. Same
        # author id => same vector regardless of which sketch slot it
        # appears in. Set share_t0_embedding=False to ablate.
        if share_t0_embedding:
            self._shared_t0 = nn.Embedding(n_t0 + 1, emb_dim, padding_idx=n_t0)
        else:
            self._shared_t0 = None

        target_base_dim = base_dim_by_type.get(target_type, 0)
        self.encoders = nn.ModuleDict({
            self._sanitize(mp): SketchFeatureEncoder(
                n_t0=n_t0,
                emb_dim=emb_dim,
                agg=agg,
                base_dim=target_base_dim,
                dropout=dropout,
                shared_embedding=self._shared_t0,
            )
            for mp in self.meta_paths
        })

        # Stub linear projections for non-target node types so HANConv
        # receives a dense feature for every type. Either project from
        # native features or use a learned per-type embedding.
        self.type_projects = nn.ModuleDict()
        self.type_emb = nn.ParameterDict()
        for nt in node_types:
            if nt == target_type:
                continue
            d = base_dim_by_type.get(nt, 0)
            if d > 0:
                self.type_projects[nt] = nn.Linear(d, emb_dim)
            else:
                # Will be populated dynamically with the right N at forward
                # time; keep a placeholder name in the dict so we can do an
                # `nt in self.type_emb` check later.
                self.type_emb[nt] = nn.Parameter(torch.empty(0))

        # HAN stack. We aggregate per-meta-path encodings into one tensor for
        # the target type by stacking along a new "view" axis and feeding to
        # HANConv as multiple incoming relations.
        self.han_layers = nn.ModuleList()
        # Build metadata expected by HANConv: the metadata is (node_types,
        # edge_types). Edge types are synthesised below.
        edge_types = self._synth_edge_types(meta_paths, target_type, node_types)
        self.metadata = (list(node_types), edge_types)

        in_d = emb_dim
        for _ in range(n_layers):
            self.han_layers.append(
                HANConv(
                    in_channels=in_d,
                    out_channels=emb_dim,
                    metadata=self.metadata,
                    heads=n_heads,
                    dropout=dropout,
                )
            )
            in_d = emb_dim

        self.classifier = nn.Linear(emb_dim, out_dim)

    @staticmethod
    def _sanitize(mp: str) -> str:
        return mp.replace(",", "_").replace(" ", "_").replace("-", "_")

    @staticmethod
    def _synth_edge_types(
        meta_paths: List[str],
        target_type: str,
        node_types: List[str],
    ) -> List[Tuple[str, str, str]]:
        """One synthetic relation per meta-path so HANConv treats them as
        distinct semantic views over the target type.
        """
        edge_types: List[Tuple[str, str, str]] = []
        for mp in meta_paths:
            rel = f"mp_{SketchHAN._sanitize(mp)}"
            edge_types.append((target_type, rel, target_type))
        return edge_types

    def _ensure_type_embedding(self, nt: str, n: int, device: torch.device) -> torch.Tensor:
        """Lazily allocate the per-type embedding once we know N for type nt."""
        param = self.type_emb[nt]
        if param.numel() == 0:
            new = nn.Parameter(torch.randn(n, self.han_layers[0].out_channels, device=device))
            self.type_emb[nt] = new
            return new
        return param

    def forward(
        self,
        sketch_inputs: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        n_by_type: Dict[str, int],
    ) -> torch.Tensor:
        """Run HAN over sketch features.

        Args:
            sketch_inputs: ``{meta_path_label: (decoded[N, k], base_x or None)}``
                — one entry per meta-path. ``decoded`` is the output of
                :func:`src.sketch_feature.decoder.decode_sketches` for that
                meta-path's propagation pass.
            edge_index_dict: synthetic edge indices keyed by the relation
                tuples returned by :meth:`_synth_edge_types`. The caller is
                expected to provide one ``edge_index`` per meta-path that
                represents the consumer mode being ablated:

                  * sketch-as-feature only:  trivial self-loops on T_0
                    (HANConv reduces to per-meta-path MLP+softmax fusion).
                  * mixed:                   the sketch-decoded sparsifier
                    Ã built from each meta-path's bottom-k slots.

                Letting the caller decide keeps the backbone reusable.
            n_by_type: ``{ntype: count}`` for every type in ``self.node_types``.

        Returns:
            logits: [N_target, out_dim].
        """
        if set(sketch_inputs) != set(self.meta_paths):
            raise ValueError(
                f"sketch_inputs keys {sorted(sketch_inputs)} != meta_paths "
                f"{sorted(self.meta_paths)}"
            )

        device = next(iter(edge_index_dict.values())).device

        # 1) Encode the target type per meta-path; aggregate by mean across
        #    meta-paths to produce the input feature for HANConv. HAN's own
        #    semantic attention then re-weighs across views via the per-mp
        #    relations.
        per_mp = []
        for mp in self.meta_paths:
            decoded, base_x = sketch_inputs[mp]
            x_mp = self.encoders[self._sanitize(mp)](decoded, base_x)
            per_mp.append(x_mp)
        x_target = torch.stack(per_mp, dim=0).mean(dim=0)  # [N_target, emb]

        # 2) Build the typed feature dict for HANConv.
        x_dict = {self.target_type: x_target}
        for nt in self.node_types:
            if nt == self.target_type:
                continue
            n = n_by_type[nt]
            if nt in self.type_projects:
                # Caller didn't provide native features here; we expect them
                # to come in via base_x for non-target types only when the
                # backbone is extended to handle that. For now, projection
                # is unused — keep it as a hook.
                raise NotImplementedError(
                    "Non-target type native feature projection is not yet "
                    "wired through forward(); pass empty base_dim_by_type "
                    "or extend forward() to accept per-type x."
                )
            x_dict[nt] = self._ensure_type_embedding(nt, n, device)

        # 3) HAN stack.
        for layer in self.han_layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {k: F.elu(v) if v is not None else v for k, v in x_dict.items()}

        return self.classifier(x_dict[self.target_type])


# ---------------------------------------------------------------------------
# Simple-HGN backbone (ablation)
# ---------------------------------------------------------------------------


class SketchSimpleHGN(nn.Module):
    """Simple-HGN-style backbone consuming sketch features.

    Skeleton only — to be filled in once HAN is validated end to end on
    HGB_DBLP. The Simple-HGN paper provides the canonical reference
    implementation; this wrapper plans to:

      1. Encode per-meta-path sketch features as in :class:`SketchHAN`.
      2. Concatenate them as the initial target-type feature.
      3. Stack ``simple_hgn_conv.SimpleHGNConv`` layers (or PyG's
         ``HEATConv`` as a stand-in if a Simple-HGN PyG port lands).
      4. Apply edge-type embeddings on the original heterogeneous graph
         (NOT on synthetic per-meta-path edges).

    Path 1 ablation goal: show whether the framework's amortization claim
    holds independent of backbone choice.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SketchSimpleHGN is a stub for the Simple-HGN ablation. Filled "
            "in after the SketchHAN pilot lands. See "
            "final_report/research_notes/CURRENT_STATE.md "
            "§'Open architectural decisions' #1."
        )
