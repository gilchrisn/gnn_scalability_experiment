"""Restore the 4-hop rules used to generate kgrw_bench.csv (APAPA/PAPAP/MDMDM).

Run from project root:
    python scripts/_restore_rules_l2_bench.py
"""
import sys, os, types as _t, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ts = _t.ModuleType("torch_sparse"); _ts.spspmm = None
sys.modules.setdefault("torch_sparse", _ts)
warnings.filterwarnings("ignore")

from src.config import config
from src.data import DatasetFactory
from scripts.bench_utils import compile_rule_for_cpp

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
    print(f"[{ds}] compiling {mp} -> {data_dir}/cod-rules_{folder}.limit")
    compile_rule_for_cpp(mp, g, data_dir, folder)
    with open(f"{data_dir}/cod-rules_{folder}.limit") as fh:
        print(f"  rule bytecode: {fh.read().strip()}")
print("Done.")
