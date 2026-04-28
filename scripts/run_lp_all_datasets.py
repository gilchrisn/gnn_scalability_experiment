"""Run the LP pipeline (partition + train + inference + analyze) across the
HGB / HNE datasets covered by Path 1.

Per ``final_report/research_notes/CURRENT_STATE.md``, the LP pipeline is in
scope as a *demonstration that the same KMV propagation supports LP queries*
— not as a quality competition with MPRW. This orchestrator reuses
``exp1_partition.py``, ``exp_lp_train.py`` and ``exp_lp_inference.py``,
running each dataset to completion before moving on.

Per-dataset configuration (densest meta-path, target type, feature
projection) is encoded in :class:`LPJob`. IMDB (in_dim=3489) requires
``feat_proj_dim`` to avoid first-layer message-passing OOM — this was the
blocker in `19_pilot_results_dblp.md` and is now handled via
``exp_lp_train.py --feat-proj-dim``.

Usage
-----
    python scripts/run_lp_all_datasets.py
    python scripts/run_lp_all_datasets.py --only HGB_IMDB
    python scripts/run_lp_all_datasets.py --skip HGB_DBLP HGB_ACM --dry-run
    python scripts/run_lp_all_datasets.py --quick  # smaller sweep for smoke test

Resume safety
-------------
- ``exp1_partition.py`` is a no-op when the partition file already exists.
- ``exp_lp_train.py`` skips already-trained ``(metapath, L)`` rows.
- ``exp_lp_inference.py`` skips already-completed
  ``(MetaPath, L, Method, k, w, Seed)`` rows in ``master_lp_results.csv``.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger("lp_overnight")


# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class LPJob:
    """One end-to-end LP run on a single dataset / meta-path.

    ``feat_proj_dim`` is the input-feature projection dim used by the
    SAGE encoder. 0 means no projection (default; see
    :class:`scripts.exp_lp_train.LPEncoder`).
    """
    name: str
    metapath: str
    target_type: str
    train_frac: float = 0.4
    feat_proj_dim: int = 0
    embedding_dim: int = 64
    epochs: int = 100
    depth: List[int] = field(default_factory=lambda: [2])
    k_values: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    w_values: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    kmv_reps: int = 5
    mprw_reps: int = 5
    max_test_edges: int = 2000
    n_negs_per_pos: int = 50


# Pilot results in `final_report/research_notes/19_pilot_results_dblp.md`
# established that:
#   - DBLP APA: KMV / MPRW both within ~5% of Exact MRR.
#   - ACM PTP: H is saturated (near-complete), AUC ~0.5 — switch to PAP.
#   - IMDB MKM: in_dim=3489 OOMs without feature projection — fixed via
#     feat_proj_dim=128.
#   - PubMed DCD: pilot complete; expand to LP sweep.
#   - Freebase: not yet started in the LP pipeline.
JOBS: List[LPJob] = [
    LPJob(
        name="HGB_DBLP",
        metapath="author_to_paper,paper_to_author",
        target_type="author",
    ),
    LPJob(
        name="HGB_ACM",
        # PTP saturates; PAP is sparser and a meaningful LP benchmark.
        metapath="paper_to_author,author_to_paper",
        target_type="paper",
    ),
    LPJob(
        name="HGB_IMDB",
        metapath="movie_to_keyword,keyword_to_movie",
        target_type="movie",
        feat_proj_dim=128,
    ),
    LPJob(
        name="HNE_PubMed",
        metapath="disease_to_chemical,chemical_to_disease",
        target_type="disease",
    ),
    # Freebase is staged for future runs once a sensible meta-path is chosen.
    # Keep it commented to avoid running an underspecified job unintentionally.
    # LPJob(
    #     name="HGB_Freebase",
    #     metapath="<TBD>",
    #     target_type="<TBD>",
    # ),
]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

_DRY_RUN = False


def _run(cmd: List[str], label: str, allow_fail: bool = False) -> int:
    log.info("\n%s\n$ %s", "=" * 78, " ".join(cmd))
    if _DRY_RUN:
        log.info("[%s] DRY-RUN (not executed)", label)
        return 0
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, text=True).returncode
    dt = time.perf_counter() - t0
    if rc != 0:
        if allow_fail:
            log.warning("[%s] FAILED rc=%d after %.1fs (continuing)", label, rc, dt)
        else:
            log.error("[%s] FAILED rc=%d after %.1fs", label, rc, dt)
    else:
        log.info("[%s] OK in %.1fs", label, dt)
    return rc


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def _partition_path(job: LPJob) -> Path:
    return _ROOT / "results" / job.name / "partition.json"


def _run_partition(job: LPJob) -> int:
    pj = _partition_path(job)
    if pj.exists():
        log.info("[%s] partition.json already at %s — reusing", job.name, pj)
        return 0
    cmd = [
        sys.executable, "scripts/exp1_partition.py",
        "--dataset", job.name,
        "--target-type", job.target_type,
        "--train-frac", str(job.train_frac),
        "--seed", "42",
    ]
    return _run(cmd, f"{job.name}/partition")


def _run_train(job: LPJob) -> int:
    cmd = [
        sys.executable, "scripts/exp_lp_train.py", job.name,
        "--metapath", job.metapath,
        "--depth", *[str(L) for L in job.depth],
        "--epochs", str(job.epochs),
        "--partition-json", str(_partition_path(job)),
        "--embedding-dim", str(job.embedding_dim),
    ]
    if job.feat_proj_dim > 0:
        cmd += ["--feat-proj-dim", str(job.feat_proj_dim)]
    return _run(cmd, f"{job.name}/lp_train")


def _run_inference(job: LPJob, run_id: int, hash_seed: int,
                   skip: List[str]) -> int:
    """One inference rep; ``skip`` lists methods to skip (Exact/KMV/MPRW)."""
    cmd = [
        sys.executable, "scripts/exp_lp_inference.py", job.name,
        "--metapath", job.metapath,
        "--depth", str(job.depth[0]),  # primary depth
        "--partition-json", str(_partition_path(job)),
        "--embedding-dim", str(job.embedding_dim),
        "--k-values", *[str(k) for k in job.k_values],
        "--w-values", *[str(w) for w in job.w_values],
        "--run-id", str(run_id),
        "--hash-seed", str(hash_seed),
        "--max-test-edges", str(job.max_test_edges),
        "--n-negs-per-pos", str(job.n_negs_per_pos),
    ]
    for m in skip:
        cmd.append(f"--skip-{m.lower()}")
    label = f"{job.name}/lp_inf seed={run_id}" + (f" skip={skip}" if skip else "")
    return _run(cmd, label, allow_fail=True)


def _run_one(job: LPJob, args: argparse.Namespace) -> int:
    log.info("\n%s\n# Dataset: %s\n%s", "#" * 78, job.name, "#" * 78)

    if not args.skip_partition:
        rc = _run_partition(job)
        if rc != 0 and not args.continue_on_error:
            return rc

    if not args.skip_train:
        rc = _run_train(job)
        if rc != 0 and not args.continue_on_error:
            return rc

    # Seed 0 runs everything; subsequent seeds split KMV / MPRW reps.
    if not args.skip_inference:
        rc = _run_inference(job, run_id=0, hash_seed=0, skip=[])
        if rc != 0 and not args.continue_on_error:
            return rc
        for i in range(1, max(job.kmv_reps, job.mprw_reps)):
            if i < job.kmv_reps:
                _run_inference(job, run_id=i, hash_seed=i,
                               skip=["exact", "mprw"])
            if i < job.mprw_reps:
                _run_inference(job, run_id=i + 1000, hash_seed=i,
                               skip=["exact", "kmv"])
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--only",  nargs="+", metavar="DATASET",
                    help="Only run these datasets (default: all in JOBS)")
    ap.add_argument("--skip",  nargs="+", default=[], metavar="DATASET",
                    help="Skip these datasets")
    ap.add_argument("--skip-partition",  action="store_true")
    ap.add_argument("--skip-train",      action="store_true")
    ap.add_argument("--skip-inference",  action="store_true")
    ap.add_argument("--skip-analyze",    action="store_true")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going to the next dataset if a step fails")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: reduced k/w grid + 1+1 seeds")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global _DRY_RUN
    _DRY_RUN = args.dry_run

    log_path = _ROOT / "results" / "lp_overnight.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )

    selected = list(JOBS)
    if args.only:
        selected = [j for j in selected if j.name in args.only]
    if args.skip:
        selected = [j for j in selected if j.name not in args.skip]
    if not selected:
        log.error("No datasets selected after filtering.")
        return 2

    if args.quick:
        for j in selected:
            j.k_values = [4, 16]
            j.w_values = [4, 16, 64]
            j.kmv_reps = 1
            j.mprw_reps = 1
            j.epochs = 20
            j.max_test_edges = 200

    log.info("LP overnight: %d datasets — %s",
             len(selected), [j.name for j in selected])

    n_fail = 0
    for j in selected:
        rc = _run_one(j, args)
        if rc != 0:
            n_fail += 1
            if not args.continue_on_error:
                log.error("Aborting (use --continue-on-error to skip past failures).")
                return rc

    if not args.skip_analyze and not _DRY_RUN:
        # exp_lp_analyze.py expects the resulting CSVs to be in place.
        ds = [j.name for j in selected]
        _run([sys.executable, "scripts/exp_lp_analyze.py",
              "--datasets", *ds],
             "lp_analyze", allow_fail=True)

    log.info("Done. %d datasets ran, %d failures.", len(selected), n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
