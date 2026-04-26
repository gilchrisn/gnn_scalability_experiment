"""Restage IMDB: full .dat files + qnodes covering ALL movies + 4-hop MDMDM rule.

Fix for: KMV undercount on IMDB (qnodes had only 99 entries, so graph_prep
sketch was propagating from a 99-node sample instead of all 4,932 movies).
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

cfg = config.get_dataset_config("HGB_IMDB")
g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)

print(f"Movies: {g[cfg.target_node].num_nodes}")

# 1. Re-write meta.dat / node.dat / link.dat from current PyG graph
PyGToCppAdapter("staging/HGBn-IMDB").convert(g)
print("Restaged .dat files")

# 2. Restore 4-hop MDMDM rule
mp = "movie_to_director,director_to_movie,movie_to_director,director_to_movie"
compile_rule_for_cpp(mp, g, "staging/HGBn-IMDB", "HGBn-IMDB")

# 3. Generate qnodes with ALL movies (for full Algorithm 2 KMV reconstruction)
n_movies = g[cfg.target_node].num_nodes
generate_qnodes("staging/HGBn-IMDB", "HGBn-IMDB", cfg.target_node, g,
                sample_size=n_movies)
print(f"qnodes: ALL {n_movies} movies")
