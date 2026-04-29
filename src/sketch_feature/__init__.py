"""Sketch-as-feature module (LoNe-typed) for Path 1 HGNN extension.

The sketch propagation pass produces, per target node v, a bottom-k array of
hash values K[v]. This module consumes that array as input *features* to a
heterogeneous GNN backbone, in contrast to the sketch-as-sparsifier path
that decodes each slot to a graph edge.

See `final_report/research_notes/CURRENT_STATE.md` for the locked Path 1
scope. See `final_report/research_notes/25_amortization_cost_analysis.md`
for the load-bearing cost theory; this module instantiates `C_consume_feature`.

Entry points:
    SketchFeatureEncoder — embeds + aggregates bottom-k slots per node.
    SketchHAN           — HAN backbone consuming sketch features.
    decode_sketches     — utility: hash array -> T_0 entity ids per node.
"""
from .decoder import decode_sketches, build_decode_table
from .encoder import SketchFeatureEncoder
from .backbone import SketchHAN, SketchSimpleHGN
from .extractor import SketchBundle, extract_sketches
from .utils import macro_f1, macro_f1_multilabel, build_mp_edges_from_decoded

__all__ = [
    "decode_sketches",
    "build_decode_table",
    "SketchFeatureEncoder",
    "SketchHAN",
    "SketchSimpleHGN",
    "SketchBundle",
    "extract_sketches",
    "macro_f1",
    "macro_f1_multilabel",
    "build_mp_edges_from_decoded",
]
