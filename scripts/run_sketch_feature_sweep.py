"""Multi-seed × multi-dataset sweep of the sketch-as-feature pilot.

Runs ``scripts/exp_sketch_feature_train.py`` once per (dataset, seed),
parses the per-run JSON outputs, and prints a markdown table of mean ± std
test-F1 per dataset for the paper.

Resume-safe: the underlying script writes
``results/<dataset>/sketch_feature_pilot_k{K}_seed{S}.json``; if it already
exists with matching k, this orchestrator skips that (dataset, seed).

Usage
-----
    python scripts/run_sketch_feature_sweep.py
    python scripts/run_sketch_feature_sweep.py --datasets HGB_DBLP HGB_ACM
    python scripts/run_sketch_feature_sweep.py --num-seeds 5
    python scripts/run_sketch_feature_sweep.py --quick  # 3 seeds, k=8
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


def _result_path(dataset: str, k: int, seed: int, backbone: str = "han_sketch_edges") -> Path:
    bb_tag = "" if backbone == "han_sketch_edges" else f"_{backbone}"
    return _ROOT / "results" / dataset / f"sketch_feature_pilot_k{k}{bb_tag}_seed{seed}.json"


def _run_one(dataset: str, seed: int, k: int, epochs: int, args) -> bool:
    """Run a single (dataset, seed). Returns True on success."""
    out = _result_path(dataset, k, seed, args.backbone)
    if out.exists() and not args.force:
        try:
            blob = json.loads(out.read_text())
            if blob.get("k") == k and blob.get("seed") == seed:
                print(f"[{dataset} seed={seed} {args.backbone}] cached test_f1={blob['test_f1']:.4f}")
                return True
        except Exception:
            pass

    cmd = [
        sys.executable, "scripts/exp_sketch_feature_train.py", dataset,
        "--k", str(k),
        "--epochs", str(epochs),
        "--emb-dim", str(args.emb_dim),
        "--n-heads", str(args.n_heads),
        "--agg", args.agg,
        "--seed", str(seed),
        "--backbone", args.backbone,
    ]
    print(f"\n[{dataset} seed={seed}] $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, text=True).returncode
    if rc != 0:
        print(f"[{dataset} seed={seed}] FAILED rc={rc}")
        return False
    return True


def _aggregate(dataset: str, k: int, seeds: List[int], backbone: str) -> Dict:
    test_f1s = []
    val_f1s = []
    train_times = []
    for s in seeds:
        p = _result_path(dataset, k, s, backbone)
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        test_f1s.append(blob["test_f1"])
        val_f1s.append(blob["best_val_f1"])
        if "train_time_s" in blob:
            train_times.append(blob["train_time_s"])
    if not test_f1s:
        return {"n": 0}
    out = {
        "n": len(test_f1s),
        "test_mean": statistics.fmean(test_f1s),
        "test_std": statistics.pstdev(test_f1s) if len(test_f1s) > 1 else 0.0,
        "val_mean": statistics.fmean(val_f1s),
        "val_std": statistics.pstdev(val_f1s) if len(val_f1s) > 1 else 0.0,
    }
    if train_times:
        out["time_mean"] = statistics.fmean(train_times)
        out["time_std"] = statistics.pstdev(train_times) if len(train_times) > 1 else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--datasets", nargs="+", default=_DEFAULT_DATASETS)
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--agg", choices=["mean", "sum", "attention"], default="attention")
    ap.add_argument("--backbone",
                    choices=["mlp", "han_real_edges", "han_sketch_edges"],
                    default="mlp",
                    help="Sketch-as-feature backbone (default: mlp = pure LoNe-typed)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if a result JSON already exists")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 2 seeds, k=8, 30 epochs")
    args = ap.parse_args()

    if args.quick:
        args.num_seeds = 2
        args.k = 8
        args.epochs = 30

    seeds = [args.seed_base + i for i in range(args.num_seeds)]

    print(f"Sweep: datasets={args.datasets}  seeds={seeds}  k={args.k}  backbone={args.backbone}")

    n_fail = 0
    for ds in args.datasets:
        for s in seeds:
            if not _run_one(ds, s, args.k, args.epochs, args):
                n_fail += 1

    # Aggregate.
    print("\n" + "=" * 78)
    print(f"Sketch-as-feature multi-seed summary "
          f"(backbone={args.backbone}, k={args.k}, emb={args.emb_dim}, "
          f"heads={args.n_heads}, agg={args.agg})")
    print("=" * 78)
    print(f"{'Dataset':<14}  {'n':>2}  {'test_f1':>16}  {'val_f1':>16}  {'train_s':>10}")
    print("-" * 70)
    rows = []
    for ds in args.datasets:
        agg = _aggregate(ds, args.k, seeds, args.backbone)
        if agg["n"] == 0:
            print(f"{ds:<14}  {'-':>2}  {'(no results)':>16}")
            continue
        test = f"{agg['test_mean']:.4f} ± {agg['test_std']:.4f}"
        val = f"{agg['val_mean']:.4f} ± {agg['val_std']:.4f}"
        if "time_mean" in agg:
            tm = f"{agg['time_mean']:.1f}±{agg['time_std']:.1f}"
        else:
            tm = "-"
        print(f"{ds:<14}  {agg['n']:>2}  {test:>16}  {val:>16}  {tm:>10}")
        rows.append({"dataset": ds, **agg})

    bb_tag = f"_{args.backbone}" if args.backbone != "han_sketch_edges" else ""
    out_md = _ROOT / "results" / f"sketch_feature_sweep_k{args.k}{bb_tag}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write(f"# Sketch-as-feature multi-seed sweep "
                 f"(backbone={args.backbone}, k={args.k})\n\n")
        fh.write(f"Configuration: emb={args.emb_dim}, heads={args.n_heads}, "
                 f"agg={args.agg}, epochs={args.epochs}, seeds={seeds}\n\n")
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
