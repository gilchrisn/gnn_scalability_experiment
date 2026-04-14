"""
run_full_pipeline.py -- Full HGNN inference pipeline orchestrator.

Runs the four-stage pipeline for a single (dataset, metapath, depth):

  Stage 1  exp1_partition.py  -- temporal/stratified split (--train-frac, default 0.4)
  Stage 2  exp2_train.py      -- train SAGE on V_train (exact subgraph)
  Stage 3  exp3_inference.py  -- inference sweep:
               Exact   x  1  run  (deterministic, run_id=0)
               KMV     x --kmv-reps   independent hash seeds  (default 50)
               MPRW    x --mprw-reps  independent walk seeds  (default 10)
                 Each rep runs the full w-sweep (w=1,2,4,...,512) independently.
  Stage 4  exp4_visualize.py  -- generate paper figures from master_results.csv

MPRW methodology
----------------
MPRW runs as a w-sweep: each w in [1, 2, 4, 8, ..., 512] is one independent
`mprw_exec materialize` call.  No calibration, no density matching.
Comparison against KMV uses edge count (density) as the common axis.

Resume safety
-------------
Each exp3 invocation is resume-safe: (MetaPath, L, Method, k_value, w_value, Seed)
tuples already present in master_results.csv are skipped automatically.
Re-running the orchestrator after a partial failure is safe.

Usage
-----
    # Full run (default 40/60 split, k=2..64, w=1..512, 50 KMV reps, 10 MPRW reps)
    python scripts/run_full_pipeline.py \\
        --dataset HGB_ACM \\
        --metapath "paper_to_term,term_to_paper" \\
        --target-type paper

    # Custom settings
    python scripts/run_full_pipeline.py \\
        --dataset HGB_DBLP \\
        --metapath "author_to_paper,paper_to_author" \\
        --target-type author \\
        --train-frac 0.4 --depth 2 \\
        --k-values 2 4 8 16 32 64 \\
        --w-values 1 2 4 8 16 32 64 128 256 512 \\
        --kmv-reps 50 --mprw-reps 10

    # Skip stages already done
    python scripts/run_full_pipeline.py \\
        --dataset HGB_ACM \\
        --metapath "paper_to_term,term_to_paper" \\
        --target-type paper \\
        --skip-partition --skip-train
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str, timeout: Optional[int] = None) -> None:
    """Run a subprocess, stream its output, raise on failure."""
    log.info("\n%s\n$ %s", "=" * 70, " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        timeout=timeout,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"[{label}] subprocess exited with code {result.returncode} "
            f"after {elapsed:.1f}s"
        )
    log.info("[%s] done in %.1fs", label, elapsed)


def _python(*args) -> list[str]:
    return [sys.executable, *args]


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage1_partition(
    dataset: str,
    target_type: str,
    train_frac: float,
    seed: int,
    out_dir: Path,
    timeout: int,
) -> Path:
    part_path = out_dir / "partition.json"
    cmd = _python(
        "scripts/exp1_partition.py",
        "--dataset", dataset,
        "--target-type", target_type,
        "--train-frac", str(train_frac),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    )
    _run(cmd, "exp1_partition", timeout=timeout)
    assert part_path.exists(), f"partition.json not created at {part_path}"
    return part_path


def stage2_train(
    dataset: str,
    metapath: str,
    depth: list[int],
    epochs: int,
    partition_json: Path,
    out_dir: Path,
    seed: int,
    timeout: int,
) -> Path:
    weights_dir = out_dir / "weights"
    cmd = _python(
        "scripts/exp2_train.py",
        dataset,
        "--metapath", metapath,
        "--depth", *[str(d) for d in depth],
        "--epochs", str(epochs),
        "--partition-json", str(partition_json),
        "--seed", str(seed),
    )
    _run(cmd, "exp2_train", timeout=timeout)
    return weights_dir


def stage3_exp3(
    dataset: str,
    metapath: str,
    depth: list[int],
    k_values: list[int],
    w_values: list[int],
    partition_json: Path,
    weights_dir: Path,
    csv_path: Path,
    run_id: int,
    hash_seed: int,
    timeout: int,
    inf_timeout: int,
    *,
    skip_exact: bool = False,
    skip_kmv: bool = False,
    skip_mprw: bool = False,
    max_rss_gb: Optional[float] = None,
) -> None:
    cmd = _python(
        "scripts/exp3_inference.py",
        dataset,
        "--metapath", metapath,
        "--depth", *[str(d) for d in depth],
        "--k-values", *[str(k) for k in k_values],
        "--w-values", *[str(w) for w in w_values],
        "--partition-json", str(partition_json),
        "--weights-dir", str(weights_dir),
        "--output", str(csv_path),
        "--run-id", str(run_id),
        "--hash-seed", str(hash_seed),
        "--timeout", str(timeout),
        "--inf-timeout", str(inf_timeout),
    )
    if skip_exact:
        cmd.append("--skip-exact")
    if skip_kmv:
        cmd.append("--skip-kmv")
    if skip_mprw:
        cmd.append("--skip-mprw")
    if max_rss_gb is not None:
        cmd += ["--max-rss-gb", str(max_rss_gb)]

    label = f"exp3 run_id={run_id} hash_seed={hash_seed}"
    if skip_exact and skip_mprw:
        label += " [KMV-only]"
    elif skip_exact and skip_kmv:
        label += " [MPRW-only]"
    elif not skip_exact and not skip_kmv and not skip_mprw:
        label += " [full: Exact+KMV+MPRW]"
    _run(cmd, label, timeout=None)  # no outer timeout; exp3 has its own


def stage4_visualize(out_dir: Path, datasets: list[str], timeout: int) -> None:
    cmd = _python(
        "scripts/exp4_visualize.py",
        "--datasets", *datasets,
        "--transfer-dir", str(out_dir.parent),
        "--out-dir", str(out_dir / "figures"),
    )
    _run(cmd, "exp4_visualize", timeout=timeout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",      required=True,
                        help="Dataset name, e.g. HGB_ACM")
    parser.add_argument("--metapath",     required=True,
                        help="Comma-separated edge names, e.g. paper_to_term,term_to_paper")
    parser.add_argument("--target-type",  required=True,
                        help="Target node type for partition, e.g. paper / author")
    parser.add_argument("--train-frac",   type=float, default=0.4,
                        help="Training fraction for exp1 (default 0.4 = 40/60 split)")
    parser.add_argument("--depth",        type=int,   nargs="+", default=[2],
                        help="SAGE depth(s) (default: 2)")
    parser.add_argument("--k-values",     type=int,   nargs="+",
                        default=[2, 4, 8, 16, 32, 64],
                        help="KMV sketch sizes (default: 2 4 8 16 32 64)")
    parser.add_argument("--w-values",     type=int,   nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        help="MPRW walk budgets (default: 1 2 4 ... 512)")
    parser.add_argument("--kmv-reps",     type=int,   default=50,
                        help="Number of independent KMV seed replicates (default: 50)")
    parser.add_argument("--mprw-reps",    type=int,   default=10,
                        help="Number of independent MPRW seed replicates (default: 10)")
    parser.add_argument("--epochs",       type=int,   default=200,
                        help="Training epochs for exp2 (default: 200)")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Master seed for partition + training (default: 42)")
    parser.add_argument("--timeout",      type=int,   default=600,
                        help="Materialization timeout in seconds per exp3 run (default: 600)")
    parser.add_argument("--inf-timeout",  type=int,   default=600,
                        help="Inference subprocess timeout per exp3 run (default: 600)")
    parser.add_argument("--max-rss-gb",   type=float, default=None,
                        help="RSS guard in GB (passed to exp3 --max-rss-gb)")
    parser.add_argument("--out-dir",      type=str,   default=None,
                        help="Results directory (default: results/<dataset>)")
    parser.add_argument("--skip-partition", action="store_true",
                        help="Skip exp1 (partition.json must already exist)")
    parser.add_argument("--skip-train",    action="store_true",
                        help="Skip exp2 (weights must already exist)")
    parser.add_argument("--skip-exact",    action="store_true",
                        help="Skip the Exact run in exp3 (exact rows must be in CSV)")
    parser.add_argument("--skip-visualize", action="store_true",
                        help="Skip exp4_visualize at the end")
    args = parser.parse_args()

    # -- Logging --------------------------------------------------------------
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "pipeline.log", encoding="utf-8"),
        ],
    )

    log.info("=" * 70)
    log.info("Pipeline start")
    log.info("  dataset=%s  metapath=%s  target_type=%s",
             args.dataset, args.metapath, args.target_type)
    log.info("  train_frac=%.0f%%  k_values=%s  w_values=%s  depth=%s",
             args.train_frac * 100, args.k_values, args.w_values, args.depth)
    log.info("  kmv_reps=%d  mprw_reps=%d",
             args.kmv_reps, args.mprw_reps)
    log.info("=" * 70)

    partition_json = out_dir / "partition.json"
    weights_dir    = out_dir / "weights"
    csv_path       = out_dir / "master_results.csv"

    t_pipeline = time.perf_counter()

    # -- Stage 1: Partition ----------------------------------------------------
    if args.skip_partition:
        log.info("[Stage 1] Skipped (--skip-partition).  partition.json=%s", partition_json)
        assert partition_json.exists(), \
            f"--skip-partition given but {partition_json} does not exist"
    else:
        log.info("[Stage 1] Partitioning dataset (train_frac=%.0f%%)...", args.train_frac * 100)
        stage1_partition(
            args.dataset, args.target_type, args.train_frac,
            args.seed, out_dir, timeout=300,
        )

    # -- Stage 2: Train -------------------------------------------------------
    if args.skip_train:
        log.info("[Stage 2] Skipped (--skip-train).  weights_dir=%s", weights_dir)
    else:
        log.info("[Stage 2] Training SAGE%s on V_train...", args.depth)
        stage2_train(
            args.dataset, args.metapath, args.depth, args.epochs,
            partition_json, out_dir, args.seed, timeout=7200,
        )

    # -- Stage 3: Seed 0 — Exact + KMV + MPRW w-sweep -------------------------
    if args.skip_exact:
        log.info("[Stage 3 / seed 0] Skipping Exact (--skip-exact). "
                 "Running KMV seed 0 + MPRW w-sweep seed 0.")
    else:
        log.info("[Stage 3 / seed 0] Exact + KMV seed 0 + MPRW w-sweep seed 0...")

    stage3_exp3(
        args.dataset, args.metapath, args.depth,
        k_values=args.k_values,
        w_values=args.w_values,
        partition_json=partition_json,
        weights_dir=weights_dir,
        csv_path=csv_path,
        run_id=0, hash_seed=0,
        timeout=args.timeout, inf_timeout=args.inf_timeout,
        skip_exact=args.skip_exact, skip_kmv=False, skip_mprw=False,
        max_rss_gb=args.max_rss_gb,
    )

    # -- Stage 3: KMV seed replicates 1..kmv_reps-1 --------------------------
    log.info("[Stage 3] KMV sweep: seeds 1..%d  (k=%s)", args.kmv_reps - 1, args.k_values)
    for i in range(1, args.kmv_reps):
        log.info("  KMV rep %d / %d", i + 1, args.kmv_reps)
        stage3_exp3(
            args.dataset, args.metapath, args.depth,
            k_values=args.k_values,
            w_values=args.w_values,
            partition_json=partition_json,
            weights_dir=weights_dir,
            csv_path=csv_path,
            run_id=i, hash_seed=i,
            timeout=args.timeout, inf_timeout=args.inf_timeout,
            skip_exact=True, skip_kmv=False, skip_mprw=True,
            max_rss_gb=args.max_rss_gb,
        )

    # -- Stage 3: MPRW seed replicates 1..mprw_reps-1 -----------------------
    # Each rep runs the full w-sweep independently with a different walk seed.
    log.info("[Stage 3] MPRW w-sweep: seeds 1..%d  (w=%s)",
             args.mprw_reps - 1, args.w_values)
    for i in range(1, args.mprw_reps):
        log.info("  MPRW rep %d / %d", i + 1, args.mprw_reps)
        stage3_exp3(
            args.dataset, args.metapath, args.depth,
            k_values=args.k_values,
            w_values=args.w_values,
            partition_json=partition_json,
            weights_dir=weights_dir,
            csv_path=csv_path,
            run_id=i, hash_seed=i,
            timeout=args.timeout, inf_timeout=args.inf_timeout,
            skip_exact=True, skip_kmv=True, skip_mprw=False,
            max_rss_gb=args.max_rss_gb,
        )

    log.info("[Stage 3] All reps done.  CSV -> %s", csv_path)

    # -- Stage 4: Visualize ---------------------------------------------------
    if args.skip_visualize:
        log.info("[Stage 4] Skipped (--skip-visualize).")
    else:
        log.info("[Stage 4] Generating figures...")
        try:
            stage4_visualize(out_dir, [args.dataset], timeout=300)
        except Exception as e:
            log.warning("[Stage 4] exp4_visualize failed (non-fatal): %s", e)

    t_total = time.perf_counter() - t_pipeline
    log.info("=" * 70)
    log.info("Pipeline complete.  Total wall time: %.1fs (%.1f min)",
             t_total, t_total / 60)
    log.info("Results -> %s", csv_path)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
