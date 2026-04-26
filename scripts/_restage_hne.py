"""Restage HNE_PubMed: full .dat + qnodes ALL diseases + DCD rule."""
import sys, os, types as _t, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ts = _t.ModuleType("torch_sparse"); _ts.spspmm = None
sys.modules.setdefault("torch_sparse", _ts)
warnings.filterwarnings("ignore")

from src.config import config
from src.data import DatasetFactory
from src.bridge.converter import PyGToCppAdapter
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes

cfg = config.get_dataset_config("HNE_PubMed")
g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
n_target = g[cfg.target_node].num_nodes
print(f"HNE_PubMed: target={cfg.target_node} n={n_target}")

PyGToCppAdapter("staging/HNE_PubMed").convert(g)
mp = "disease_to_chemical,chemical_to_disease"
compile_rule_for_cpp(mp, g, "staging/HNE_PubMed", "HNE_PubMed")
generate_qnodes("staging/HNE_PubMed", "HNE_PubMed", cfg.target_node, g,
                sample_size=n_target)
print("Done.")
