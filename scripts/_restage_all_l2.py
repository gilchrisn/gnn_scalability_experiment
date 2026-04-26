"""Restage DBLP/ACM/IMDB cleanly for L=2 KGRW bench:
  - rewrite meta.dat / node.dat / link.dat from current PyG graph
  - compile 4-hop rule (APAPA / PAPAP / MDMDM)
  - generate qnodes covering ALL target-type nodes
"""
import sys, os, types as _t, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ts = _t.ModuleType("torch_sparse"); _ts.spspmm = None
sys.modules.setdefault("torch_sparse", _ts)
warnings.filterwarnings("ignore")

from src.config import config
from src.data import DatasetFactory
from src.bridge.converter import PyGToCppAdapter
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes

JOBS = [
    ("HGB_DBLP", "HGBn-DBLP",
     "author_to_paper,paper_to_author,author_to_paper,paper_to_author"),
    ("HGB_ACM",  "HGBn-ACM",
     "paper_to_author,author_to_paper,paper_to_author,author_to_paper"),
    ("HGB_IMDB", "HGBn-IMDB",
     "movie_to_director,director_to_movie,movie_to_director,director_to_movie"),
]

for ds, folder, mp in JOBS:
    cfg = config.get_dataset_config(ds)
    g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    data_dir = f"staging/{folder}"
    print(f"\n=== {ds} ({cfg.target_node} count={g[cfg.target_node].num_nodes}) ===")
    PyGToCppAdapter(data_dir).convert(g)
    compile_rule_for_cpp(mp, g, data_dir, folder)
    generate_qnodes(data_dir, folder, cfg.target_node, g,
                    sample_size=g[cfg.target_node].num_nodes)
print("\nDone.")
