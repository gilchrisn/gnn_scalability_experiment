"""bench_metapath_sweep.py — sweep over metapath lengths and run KMV vs KGRW vs MPRW
at each one.

For each metapath:
    1. Compile the rule.
    2. Train SAGE (skipped if weights file exists).
    3. Materialize the exact graph (skipped if exact .adj exists).
    4. Generate z_exact for the chosen SAGE depth.
    5. Call bench_kgrw.py --metapath <mp> to run all three methods.

Usage
-----
    python scripts/bench_metapath_sweep.py \\
        --dataset HGB_DBLP \\
        --metapaths "author_to_paper,paper_to_author" \\
                    "author_to_paper,paper_to_author,author_to_paper,paper_to_author" \\
                    "author_to_paper,paper_to_author,author_to_paper,paper_to_author,author_to_paper,paper_to_author" \\
        --sage-L 2 \\
        --k-values 4 8 16 32 64 128 \\
        --w-values 2 4 8 16 32 64 128 256 512 \\
        --seeds 5

Each metapath's data goes into a separate CSV: results/<dataset>/metapath_<mp_safe>.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import types as _t
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_ts = _t.ModuleType("torch_sparse"); _ts.spspmm = None
sys.modules.setdefault("torch_sparse", _ts)
warnings.filterwarnings("ignore")

from src.config import config
from src.data import DatasetFactory
from src.bridge.converter import PyGToCppAdapter
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes


def _wsl(cmd_str: str) -> list[str]:
    if sys.platform == "win32":
        return ["wsl", "--exec", "bash", "-c", cmd_str]
    return ["bash", "-c", cmd_str]


def _materialize_exact(folder: str, exact_adj_win: str) -> bool:
    """Run graph_prep materialize for ExactD output. Returns True on success."""
    if Path(exact_adj_win).exists():
        print(f"  [exact] already exists: {exact_adj_win}")
        return True
    cmd = (f"cd /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/scalability_experiment && "
           f"bin/graph_prep materialize staging/{folder} "
           f"staging/{folder}/cod-rules_{folder}.limit {exact_adj_win}")
    r = subprocess.run(_wsl(cmd), capture_output=True, text=True, timeout=1800)
    ok = (r.returncode == 0 and Path(exact_adj_win).exists())
    if not ok:
        print(f"  [exact] FAILED: {r.stderr[-300:]}")
    return ok


def _train_sage(dataset: str, metapath: str, sage_L: int,
                partition_json: str, weights_path: Path) -> bool:
    """Train SAGE if weights file missing. Returns True on success/already-exists."""
    if weights_path.exists():
        print(f"  [train] weights exist: {weights_path.name}")
        return True
    print(f"  [train] training SAGE L={sage_L} on {metapath} ...")
    env = {**os.environ}
    stub = str(ROOT / "results" / "kgrw_eval" / "_stubs")
    if Path(stub).exists():
        env["PYTHONPATH"] = stub + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(ROOT / "scripts" / "exp2_train.py"),
           dataset, "--metapath", metapath, "--depth", str(sage_L),
           "--partition-json", partition_json, "--epochs", "200"]
    r = subprocess.run(cmd, capture_output=False, env=env, timeout=3600)
    return r.returncode == 0 and weights_path.exists()


def _gen_z_exact(dataset: str, exact_adj_win: str, sage_L: int) -> bool:
    """Run bench_kgrw.py --gen-exact to produce z_exact_<DS>_L<L>.pt."""
    z_exact = ROOT / "results" / "kgrw_eval" / f"z_exact_{dataset}_L{sage_L}.pt"
    cmd = [sys.executable, str(ROOT / "scripts" / "bench_kgrw.py"),
           "--dataset", dataset, "--L", str(sage_L),
           "--gen-exact", "--exact-adj", exact_adj_win]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return r.returncode == 0 and z_exact.exists()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--metapaths", required=True, nargs="+",
                   help="One or more comma-separated metapath strings")
    p.add_argument("--sage-L", type=int, default=2)
    p.add_argument("--k-values", type=int, nargs="+",
                   default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--w-values", type=int, nargs="+",
                   default=[2, 4, 8, 16, 32, 64, 128, 256, 512])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--skip-kgrw", action="store_true",
                   help="Skip KGRW; only KMV+MPRW")
    p.add_argument("--regenerate-exact", action="store_true",
                   help="Force re-materialization of exact .adj per metapath")
    args = p.parse_args()

    cfg    = config.get_dataset_config(args.dataset)
    folder = config.get_folder_name(args.dataset)
    g, _   = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    n_target = g[cfg.target_node].num_nodes

    # Restage once: PyG → .dat + qnodes ALL targets (avoids drift)
    print(f"=== Restaging {args.dataset} ===")
    PyGToCppAdapter(f"staging/{folder}").convert(g)
    generate_qnodes(f"staging/{folder}", folder, cfg.target_node, g,
                    sample_size=n_target)

    partition_json = ROOT / "results" / args.dataset / "partition.json"
    if not partition_json.exists():
        print(f"FATAL: partition file not found at {partition_json}")
        return

    for mp in args.metapaths:
        mp_safe = mp.replace(",", "_")
        n_hops  = len(mp.split(","))
        print(f"\n{'='*70}\nMetapath ({n_hops}-hop): {mp}\n{'='*70}")

        # 1. Rule
        compile_rule_for_cpp(mp, g, f"staging/{folder}", folder)

        # 2. Weights
        weights_path = ROOT / "results" / args.dataset / "weights" / f"{mp_safe}_L{args.sage_L}.pt"
        if not _train_sage(args.dataset, mp, args.sage_L,
                            str(partition_json), weights_path):
            print(f"  [SKIP] training failed for {mp}")
            continue

        # 3. Exact adjacency
        exact_adj_win = str(ROOT / "results" / args.dataset / f"exact_{mp_safe}.adj")
        exact_adj_wsl = exact_adj_win.replace("\\", "/").replace("C:/", "/mnt/c/")
        if args.regenerate_exact and Path(exact_adj_win).exists():
            Path(exact_adj_win).unlink()
        if not _materialize_exact(folder, exact_adj_wsl):
            print(f"  [SKIP] exact materialization failed for {mp}")
            continue

        # 4. z_exact reference
        if not _gen_z_exact(args.dataset, exact_adj_wsl, args.sage_L):
            print(f"  [SKIP] z_exact generation failed for {mp}")
            continue

        # 5. KMV + KGRW + MPRW sweep
        csv_out = ROOT / "results" / args.dataset / f"metapath_{mp_safe}.csv"
        cmd = [sys.executable, str(ROOT / "scripts" / "bench_kgrw.py"),
               "--dataset", args.dataset,
               "--metapath", mp,
               "--csv-out", str(csv_out),
               "--L", str(args.sage_L),
               "--k-values", *map(str, args.k_values),
               "--w-values", *map(str, args.w_values),
               "--seeds", str(args.seeds)]
        if args.skip_kgrw:
            cmd.append("--skip-kgrw")
        env = {**os.environ}
        stub = str(ROOT / "results" / "kgrw_eval" / "_stubs")
        if Path(stub).exists():
            env["PYTHONPATH"] = stub + os.pathsep + env.get("PYTHONPATH", "")
        print(f"  [bench] running KMV+KGRW+MPRW...")
        r = subprocess.run(cmd, env=env, timeout=14400)  # 4h cap
        print(f"  [bench] exit={r.returncode}  CSV: {csv_out}")


if __name__ == "__main__":
    main()
