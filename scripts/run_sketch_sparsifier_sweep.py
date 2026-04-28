"""Multi-seed × multi-dataset sweep for the sketch-as-sparsifier consumer
mode. Mirrors run_sketch_feature_sweep.py but invokes
exp_sketch_sparsifier_train.py.

Usage
-----
    python scripts/run_sketch_sparsifier_sweep.py
    python scripts/run_sketch_sparsifier_sweep.py --datasets HGB_DBLP --num-seeds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]


def _result_path(dataset: str, k: int, seed: int) -> Path:
    return _ROOT / "results" / dataset / f"sketch_sparsifier_pilot_k{k}_seed{seed}.json"


def _run_one(dataset: str, seed: int, k: int, args) -> bool:
    out = _result_path(dataset, k, seed)
    if out.exists() and not args.force:
        try:
            blob = json.loads(out.read_text())
            print(f"[{dataset} seed={seed}] cached test_f1={blob['test_f1']:.4f}")
            return True
        except Exception:
            pass
    cmd = [
        sys.executable, "scripts/exp_sketch_sparsifier_train.py", dataset,
        "--k", str(k),
        "--epochs", str(args.epochs),
        "--depth", str(args.depth),
        "--hidden-dim", str(args.hidden_dim),
        "--seed", str(seed),
    ]
    print(f"\n[{dataset} seed={seed}] $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


def _aggregate(dataset: str, k: int, seeds: List[int]) -> Dict:
    test_f1s, val_f1s, times = [], [], []
    for s in seeds:
        p = _result_path(dataset, k, s)
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        test_f1s.append(blob["test_f1"])
        val_f1s.append(blob["best_val_f1"])
        if "train_time_s" in blob:
            times.append(blob["train_time_s"])
    if not test_f1s:
        return {"n": 0}
    out = {
        "n": len(test_f1s),
        "test_mean": statistics.fmean(test_f1s),
        "test_std": statistics.pstdev(test_f1s) if len(test_f1s) > 1 else 0.0,
        "val_mean": statistics.fmean(val_f1s),
        "val_std": statistics.pstdev(val_f1s) if len(val_f1s) > 1 else 0.0,
    }
    if times:
        out["time_mean"] = statistics.fmean(times)
        out["time_std"] = statistics.pstdev(times) if len(times) > 1 else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=_DEFAULT_DATASETS)
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = [args.seed_base + i for i in range(args.num_seeds)]
    print(f"Sparsifier sweep: datasets={args.datasets}  seeds={seeds}  k={args.k}")

    n_fail = 0
    for ds in args.datasets:
        for s in seeds:
            if not _run_one(ds, s, args.k, args):
                n_fail += 1

    print("\n" + "=" * 78)
    print(f"Sketch-as-sparsifier multi-seed summary "
          f"(k={args.k}, depth={args.depth}, hidden={args.hidden_dim})")
    print("=" * 78)
    print(f"{'Dataset':<14}  {'n':>2}  {'test_f1':>16}  {'train_s':>10}")
    rows = []
    for ds in args.datasets:
        agg = _aggregate(ds, args.k, seeds)
        if agg["n"] == 0:
            print(f"{ds:<14}  -   (no results)")
            continue
        test = f"{agg['test_mean']:.4f} ± {agg['test_std']:.4f}"
        tm = (f"{agg['time_mean']:.1f}±{agg['time_std']:.1f}"
              if "time_mean" in agg else "-")
        print(f"{ds:<14}  {agg['n']:>2}  {test:>16}  {tm:>10}")
        rows.append({"dataset": ds, **agg})

    out_md = _ROOT / "results" / f"sketch_sparsifier_sweep_k{args.k}.md"
    with open(out_md, "w") as fh:
        fh.write(f"# Sketch-as-sparsifier multi-seed sweep (k={args.k})\n\n")
        fh.write(f"depth={args.depth}, hidden_dim={args.hidden_dim}, "
                 f"epochs={args.epochs}, seeds={seeds}\n\n")
        fh.write("| Dataset | n | test_f1 | val_f1 | train_time (s) |\n")
        fh.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            tm = (f"{r['time_mean']:.1f} ± {r['time_std']:.1f}"
                  if "time_mean" in r else "-")
            fh.write(f"| {r['dataset']} | {r['n']} | "
                     f"{r['test_mean']:.4f} ± {r['test_std']:.4f} | "
                     f"{r['val_mean']:.4f} ± {r['val_std']:.4f} | {tm} |\n")
    print(f"\n[saved] {out_md}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
