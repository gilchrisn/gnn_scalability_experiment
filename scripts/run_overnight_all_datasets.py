"""run_overnight_all_datasets.py -- Run the full KMV-vs-MPRW pipeline end-to-end
across ACM / DBLP / IMDB / PubMed, then regenerate the multi-dataset figures
that the presentation references.

Flow
----
For each dataset:
    1. scripts/run_full_pipeline.py  (exp1 partition + exp2 train +
       exp3 Exact+KMV+MPRW w-sweep + exp4 per-dataset plots).
Then once:
    2. scripts/exp4_visualize.py  across all 4 datasets, writing to figures/
       (the directory referenced by presentation/slides.tex).
Optional:
    3. scripts/exp9_plot_dblp_aptpa.py  regenerate the starved-budget plot
       from the fresh DBLP CSV.
    4. scripts/exp10_is_mechanism_dblp.py  regenerate the IS-mechanism plot.

Each dataset is resume-safe -- re-running after a crash picks up where it left off
(exp3 skips (MetaPath, L, Method, k/w, Seed) rows already in master_results.csv).

The default configuration is the post-fix methodology described in
markdown/exp_redesign_mprw_fairness.md: KMV k in {2..64}, MPRW w in {1..512},
no calibration, natural-mode sweep on a common density axis.

Usage
-----
    python scripts/run_overnight_all_datasets.py
    python scripts/run_overnight_all_datasets.py --only HGB_DBLP
    python scripts/run_overnight_all_datasets.py --skip HGB_ACM HNE_PubMed
    python scripts/run_overnight_all_datasets.py --quick  # smaller budgets for a dry run
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger("overnight")


# ---------------------------------------------------------------------------
# Dataset registry -- densest meta-path per dataset, matching the paper
# ---------------------------------------------------------------------------

@dataclass
class DatasetJob:
    """Definition of a full-pipeline run for one dataset."""
    name: str
    metapath: str
    target_type: str
    k_values: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    w_values: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    kmv_reps: int = 50
    mprw_reps: int = 10
    depth: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


JOBS: list[DatasetJob] = [
    # Short meta-paths (L=2 densest) where KMV and MPRW are expected to be
    # interchangeable -- this is the bulk of the paper evidence.
    DatasetJob(
        name="HGB_ACM",
        metapath="paper_to_term,term_to_paper",
        target_type="paper",
    ),
    DatasetJob(
        name="HGB_DBLP",
        # Long APTPA (L=4) is the starved-budget failure regime for MPRW.
        # Using the 4-hop path makes exp9/exp10 plots regenerate correctly.
        metapath="author_to_paper,paper_to_term,term_to_paper,paper_to_author",
        target_type="author",
    ),
    DatasetJob(
        name="HGB_IMDB",
        metapath="movie_to_keyword,keyword_to_movie",
        target_type="movie",
    ),
    DatasetJob(
        name="HNE_PubMed",
        metapath="disease_to_chemical,chemical_to_disease",
        target_type="disease",
    ),
]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> int:
    log.info("\n%s\n$ %s", "=" * 78, " ".join(cmd))
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, text=True).returncode
    dt = time.perf_counter() - t0
    if rc != 0:
        log.error("[%s] FAILED rc=%d after %.1fs", label, rc, dt)
    else:
        log.info("[%s] OK in %.1fs", label, dt)
    return rc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--only",         nargs="+", metavar="DATASET",
                    help="Run only these datasets (default: all)")
    ap.add_argument("--skip",         nargs="+", default=[], metavar="DATASET",
                    help="Skip these datasets")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="Jump straight to exp4/exp9/exp10 using existing CSVs")
    ap.add_argument("--skip-visualize", action="store_true",
                    help="Do not run the final exp4 multi-dataset visualize step")
    ap.add_argument("--skip-exp9-exp10", action="store_true",
                    help="Do not re-run the DBLP-APTPA failure-case scripts")
    ap.add_argument("--quick", action="store_true",
                    help="Reduced budgets for a smoke test (k<=16, w<=64, 3+3 reps)")
    ap.add_argument("--train-frac", type=float, default=0.4,
                    help="Partition fraction for exp1 (default 0.4)")
    ap.add_argument("--max-rss-gb", type=float, default=None,
                    help="Pass --max-rss-gb to exp3 (OOM guard)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going to the next dataset if a pipeline run fails")
    ap.add_argument("--out-figures",  default="figures",
                    help="Directory for multi-dataset figures (default: figures/)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(_ROOT / "results" / "overnight.log",
                                encoding="utf-8"),
        ],
    )

    # Filter job list
    selected = list(JOBS)
    if args.only:
        selected = [j for j in selected if j.name in args.only]
    if args.skip:
        selected = [j for j in selected if j.name not in args.skip]
    if not selected:
        log.error("No datasets selected after --only/--skip filtering.")
        return 2

    # Quick-mode override
    if args.quick:
        for j in selected:
            j.k_values = [2, 4, 8, 16]
            j.w_values = [1, 2, 4, 8, 16, 32, 64]
            j.kmv_reps = 3
            j.mprw_reps = 3

    log.info("=" * 78)
    log.info("Overnight run starting.  %d datasets: %s",
             len(selected), [j.name for j in selected])
    log.info("Train frac: %.2f | quick=%s | max_rss=%s | skip_pipeline=%s",
             args.train_frac, args.quick, args.max_rss_gb, args.skip_pipeline)
    log.info("=" * 78)

    t_all = time.perf_counter()
    failures: list[str] = []

    # ---- Stage A: per-dataset pipelines ------------------------------------
    if not args.skip_pipeline:
        for job in selected:
            cmd = [
                sys.executable, "scripts/run_full_pipeline.py",
                "--dataset",     job.name,
                "--metapath",    job.metapath,
                "--target-type", job.target_type,
                "--train-frac",  str(args.train_frac),
                "--depth",       *[str(d) for d in job.depth],
                "--k-values",    *[str(k) for k in job.k_values],
                "--w-values",    *[str(w) for w in job.w_values],
                "--kmv-reps",    str(job.kmv_reps),
                "--mprw-reps",   str(job.mprw_reps),
                # Skip the per-dataset exp4 -- we run one multi-dataset exp4
                # at the end instead, to produce the 4-panel suites.
                "--skip-visualize",
            ]
            if args.max_rss_gb is not None:
                cmd += ["--max-rss-gb", str(args.max_rss_gb)]

            rc = _run(cmd, f"pipeline:{job.name}")
            if rc != 0:
                failures.append(job.name)
                if not args.continue_on_error:
                    log.error("Stopping (no --continue-on-error).  Use "
                              "--skip-pipeline to jump to visualize later.")
                    return 1

    # ---- Stage B: multi-dataset exp4 visualize -----------------------------
    if not args.skip_visualize:
        cmd = [
            sys.executable, "scripts/exp4_visualize.py",
            "--datasets",    *[j.name for j in selected],
            "--transfer-dir", "results",
            "--out-dir",      args.out_figures,
        ]
        rc = _run(cmd, "exp4_visualize:multi")
        if rc != 0:
            failures.append("exp4_visualize")

    # ---- Stage C: DBLP-APTPA failure-case plots ----------------------------
    if not args.skip_exp9_exp10:
        dblp_csv = _ROOT / "results" / "HGB_DBLP" / "master_results.csv"
        if dblp_csv.exists():
            rc = _run(
                [sys.executable, "scripts/exp9_plot_dblp_aptpa.py",
                 "--csv", str(dblp_csv),
                 "--out", "figures/exp9/dblp_aptpa_starved.pdf"],
                "exp9:dblp_aptpa",
            )
            if rc != 0:
                failures.append("exp9")

            rc = _run(
                [sys.executable, "scripts/exp10_is_mechanism_dblp.py",
                 "--out", "figures/exp10/is_mechanism_dblp_aptpa.pdf"],
                "exp10:is_mechanism",
            )
            if rc != 0:
                failures.append("exp10")
        else:
            log.warning("DBLP CSV missing (%s) -- skipping exp9/exp10", dblp_csv)

    dt = time.perf_counter() - t_all
    log.info("=" * 78)
    log.info("Overnight run done in %.1f min", dt / 60)
    if failures:
        log.error("Failures: %s", failures)
        return 1
    log.info("All stages OK.  Figures -> %s", args.out_figures)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
