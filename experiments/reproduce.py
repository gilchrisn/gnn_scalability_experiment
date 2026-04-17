"""
reproduce.py — Single-entry reproduction script for ALL paper experiments.

Exact config used for the April 2026 overnight run that produced the results
in results/<DATASET>/master_results.csv and figures/*.pdf.

Usage
-----
# Regenerate figures only (fast, ~20s, works locally):
python experiments/reproduce.py --stage visualize

# Re-run inference for one dataset (needs server):
python experiments/reproduce.py --stage inference --datasets HGB_ACM

# Full pipeline for all datasets (run on server, takes hours):
python experiments/reproduce.py --stage all

# Individual datasets:
python experiments/reproduce.py --stage all --datasets HGB_ACM HGB_DBLP

# Skip stages already done:
python experiments/reproduce.py --stage all --skip-partition --skip-train

Stages
------
  partition   exp1_partition.py  — train/test split (once per dataset)
  train       exp2_train.py      — SAGE training on exact V_train subgraph
  inference   run_full_pipeline  — KMV + MPRW sweep (50 KMV reps, 10 MPRW reps)
  visualize   exp4_visualize.py  — generate all paper figures from CSV

Environment
-----------
Stages 1-3 MUST run in WSL/Linux (need /usr/bin/time -v for memory measurement).
Stage 4 (visualize) runs fine on Windows.

On server:
    source .venv/bin/activate && python experiments/reproduce.py --stage all

Locally (figures only):
    source .venv/Scripts/activate && python experiments/reproduce.py --stage visualize
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — exact parameters used for the April 2026 overnight run
# ══════════════════════════════════════════════════════════════════════════════

DATASETS = {
    "HGB_ACM": {
        "metapath":    "paper_to_term,term_to_paper",
        "target_type": "paper",
    },
    "HGB_DBLP": {
        "metapath":    "author_to_paper,paper_to_term,term_to_paper,paper_to_author",
        "target_type": "author",
    },
    "HGB_IMDB": {
        "metapath":    "movie_to_keyword,keyword_to_movie",
        "target_type": "movie",
    },
    "HNE_PubMed": {
        "metapath":    "disease_to_chemical,chemical_to_disease",
        "target_type": "disease",
    },
}

# Sweep parameters
TRAIN_FRAC  = 0.4
DEPTH       = [1, 2, 3, 4]
K_VALUES    = [2, 4, 8, 16, 32, 64]
W_VALUES    = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
KMV_REPS    = 50      # independent hash seeds for KMV
MPRW_REPS   = 10      # independent walk seeds for MPRW
MAX_RSS_GB  = 80.0    # server memory cap

# Output directories
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Visualization fixed params (for table2, plot_k_sweep, etc.)
K_FIXED  = 32
W_FIXED  = 32
L_FIXED  = 2

# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PYTHON = sys.executable


def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess, stream output, return True if successful."""
    log.info("Running: %s", label)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    if result.returncode == 0:
        log.info("[OK] %s in %.0fs", label, elapsed)
        return True
    else:
        log.error("[FAIL] %s (rc=%d) in %.0fs", label, result.returncode, elapsed)
        return False


# ── Stage 1: Partition ────────────────────────────────────────────────────────

def stage_partition(datasets: list[str]) -> None:
    for ds in datasets:
        cfg = DATASETS[ds]
        out = RESULTS_DIR / ds / "partition.json"
        if out.exists():
            log.info("[SKIP] partition for %s (already exists)", ds)
            continue
        run([
            PYTHON, "scripts/exp1_partition.py",
            "--dataset",     ds,
            "--target-type", cfg["target_type"],
            "--train-frac",  str(TRAIN_FRAC),
            "--seed",        "42",
        ], f"partition:{ds}")


# ── Stage 2: Train ────────────────────────────────────────────────────────────

def stage_train(datasets: list[str]) -> None:
    for ds in datasets:
        cfg = DATASETS[ds]
        run([
            PYTHON, "scripts/exp2_train.py",
            "--dataset",     ds,
            "--metapath",    cfg["metapath"],
            "--target-type", cfg["target_type"],
            "--L-values",    *[str(l) for l in DEPTH],
            "--hidden",      "64",
            "--epochs",      "200",
        ], f"train:{ds}")


# ── Stage 3: Inference sweep ──────────────────────────────────────────────────

def stage_inference(datasets: list[str]) -> None:
    """
    Runs run_full_pipeline.py which internally calls exp3_inference.py
    for all seeds and methods. Resume-safe — already-done rows are skipped.
    """
    for ds in datasets:
        cfg = DATASETS[ds]
        run([
            PYTHON, "scripts/run_full_pipeline.py",
            "--dataset",     ds,
            "--metapath",    cfg["metapath"],
            "--target-type", cfg["target_type"],
            "--train-frac",  str(TRAIN_FRAC),
            "--depth",       *[str(l) for l in DEPTH],
            "--k-values",    *[str(k) for k in K_VALUES],
            "--w-values",    *[str(w) for w in W_VALUES],
            "--kmv-reps",    str(KMV_REPS),
            "--mprw-reps",   str(MPRW_REPS),
            "--max-rss-gb",  str(MAX_RSS_GB),
            "--skip-visualize",          # visualize separately
        ], f"inference:{ds}")


# ── Stage 4: Visualize ────────────────────────────────────────────────────────

def stage_visualize(datasets: list[str]) -> None:
    """
    Reads results/<DS>/master_results.csv for all datasets,
    produces all paper figures and tables into figures/.

    This is the FAST stage (~20s). Run this locally to regenerate plots.
    """
    run([
        PYTHON, "scripts/exp4_visualize.py",
        "--datasets",    *datasets,
        "--transfer-dir", str(RESULTS_DIR),
        "--out-dir",      str(FIGURES_DIR),
        "--k-fixed",      str(K_FIXED),
        "--w-fixed",      str(W_FIXED),
        "--l-fixed",      str(L_FIXED),
    ], "visualize:all")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["partition", "train", "inference", "visualize", "all"],
                   default="visualize",
                   help="Which stage to run (default: visualize)")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()),
                   help="Which datasets to process (default: all 4)")
    p.add_argument("--skip-partition", action="store_true")
    p.add_argument("--skip-train",     action="store_true")
    p.add_argument("--skip-inference", action="store_true")
    args = p.parse_args()

    ds = args.datasets
    log.info("Datasets: %s", ds)
    log.info("Stage:    %s", args.stage)

    if args.stage in ("partition", "all") and not args.skip_partition:
        stage_partition(ds)

    if args.stage in ("train", "all") and not args.skip_train:
        stage_train(ds)

    if args.stage in ("inference", "all") and not args.skip_inference:
        stage_inference(ds)

    if args.stage in ("visualize", "all"):
        stage_visualize(ds)

    log.info("Done.")


if __name__ == "__main__":
    main()
