"""Phase 4 — multi-query amortization measurement.

The headline claim of Path 1: ONE KMV propagation pass supports multiple
downstream queries (NC, similarity, LP). For Q queries, the total cost is:

    KMV path        :  extract_once  +  Σ_q  consume_q
    Sampling-HGNN   :                  Σ_q  train_from_scratch_q

This script measures both sides on a single dataset and writes a
markdown table + JSON record so the cost equation in
final_report/research_notes/25_amortization_cost_analysis.md becomes
real numbers, not symbols.

Query classes evaluated
-----------------------

1. **NC**  : node classification on the target type. Both methods.
2. **Sim**: pairwise Jaccard similarity on a S-node sample.
   - KMV side: read from sketch (basically free).
   - Baseline: compute from exact meta-path adjacency.

LP is intentionally out of scope for this script — it requires a held-out
edge split and a separate decoder; adding it doubles the script length
without changing the qualitative finding. The cost-equation crossover is
already crossed at Q=2 (NC + similarity), which is the message.

Usage
-----
    python scripts/exp_multi_query_amortization.py HGB_DBLP --k 32
    python scripts/exp_multi_query_amortization.py HGB_ACM --quick
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def _run(cmd: list, label: str) -> int:
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, text=True).returncode
    dt = time.perf_counter() - t0
    print(f"[{label}] rc={rc}  wall={dt:.1f}s")
    return rc


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--quick", action="store_true",
                   help="Smaller epochs / sample for smoke test")
    args = p.parse_args()

    if args.quick:
        args.epochs = 30
        args.sample = 100

    out_dir = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    KMV_NC_RES = out_dir / f"sketch_feature_pilot_k{args.k}_mlp_seed{args.seed}.json"
    KMV_SIM_RES = out_dir / f"sketch_similarity_pilot_k{args.k}_seed{args.seed}.json"
    SHGN_NC_RES = out_dir / f"simple_hgn_baseline_seed{args.seed}.json"

    # ------------------------------------------------------------------
    # KMV path: extract once (implicit in each tool that calls extract;
    # they all share the cache file, so the second+ tools incur ~0s
    # extract cost). We measure end-to-end wall clock per tool here, then
    # subtract the extract time recorded in the JSON to attribute the
    # one-time precompute correctly.
    # ------------------------------------------------------------------
    print("=" * 78)
    print(f"KMV-path queries (one sketch, served {args.dataset})")
    print("=" * 78)

    # Make sure we re-extract by deleting cache.
    cache_path = out_dir / f"sketch_bundle_k{args.k}_seed{args.seed}.pt"
    if cache_path.exists():
        cache_path.unlink()
        print(f"[reset] removed stale sketch cache so we measure cold path")

    # NC via sketch-as-feature MLP. The script writes train_time_s + (we
    # added) extract_time_s won't be there yet — we reconstruct precompute
    # below from KMV_SIM_RES which DOES record extract_time_s.
    rc = _run(
        [py, "scripts/exp_sketch_feature_train.py", args.dataset,
         "--k", str(args.k), "--seed", str(args.seed),
         "--epochs", str(args.epochs),
         "--emb-dim", "128", "--n-heads", "8", "--agg", "attention",
         "--backbone", "mlp"],
        "KMV/NC",
    )
    if rc != 0:
        print("KMV NC failed; aborting"); return 1
    nc_blob = _read_json(KMV_NC_RES)
    kmv_nc_train_s = float(nc_blob["train_time_s"])

    # Similarity from the SAME cached sketch. extract_time_s ≈ 0 because
    # the bundle is cached. The script does record extract_time_s = 0 in
    # that path; the actual extraction cost is captured by the NC run above.
    rc = _run(
        [py, "scripts/exp_sketch_similarity.py", args.dataset,
         "--k", str(args.k), "--seed", str(args.seed),
         "--sample", str(args.sample)],
        "KMV/Sim",
    )
    if rc != 0:
        print("KMV Sim failed; aborting"); return 1
    sim_blob = _read_json(KMV_SIM_RES)
    sim_sketch_total_s = sum(
        v["sketch_time_s"] for v in sim_blob["per_meta_path"].values()
    )
    sim_exact_total_s = sum(
        v["exact_time_s"] for v in sim_blob["per_meta_path"].values()
    )

    # The sketch precompute is whatever the NC run paid for it; that's
    # baked into kmv_nc_train_s + a one-time term. The NC script's
    # train_time_s does NOT include extract; extract is logged separately
    # in stdout. We re-extract here cleanly and time it standalone for
    # an honest precompute number.
    cache_path.unlink(missing_ok=True)
    print("\n[precompute] timing standalone sketch extraction ...")
    t0 = time.perf_counter()
    rc = _run(
        [py, "-c",
         "import sys; sys.path.insert(0,'.'); "
         "from src.config import config; from src.data import DatasetFactory; "
         "from src.sketch_feature import extract_sketches; "
         "from scripts.exp_sketch_feature_train import _DEFAULT_META_PATHS; "
         f"cfg=config.get_dataset_config('{args.dataset}'); "
         "g,info=DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node); "
         f"b=extract_sketches(g, _DEFAULT_META_PATHS['{args.dataset}'], cfg.target_node, "
         f"k={args.k}, seed={args.seed}, device='cpu'); "
         f"b.save('{cache_path.as_posix()}'); print('extract OK')"],
        "KMV/precompute",
    )
    extract_time_s = time.perf_counter() - t0
    if rc != 0:
        print("Precompute timing failed"); return 1

    # ------------------------------------------------------------------
    # Baseline path: Simple-HGN trains end-to-end for NC; for similarity,
    # we time the EXACT computation that is currently the only Simple-HGN-
    # compatible reference (since Simple-HGN doesn't natively expose
    # node-similarity on a meta-path). exp_sketch_similarity.py already
    # measured both sketch and exact per meta-path; we reuse that number.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"Baseline path (Simple-HGN) — {args.dataset}")
    print("=" * 78)

    if not SHGN_NC_RES.exists():
        rc = _run(
            [py, "scripts/exp_simple_hgn_baseline.py", args.dataset,
             "--num-seeds", "1", "--seed-base", str(args.seed),
             "--epochs", str(args.epochs)],
            "SHGN/NC",
        )
        if rc != 0:
            print("Simple-HGN NC failed; continuing with sketch-only numbers")
    shgn_blob = _read_json(SHGN_NC_RES) if SHGN_NC_RES.exists() else None
    shgn_nc_train_s = float(shgn_blob["train_time_s"]) if shgn_blob else float("nan")

    # ------------------------------------------------------------------
    # Build the table.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"Multi-query amortization — {args.dataset}, k={args.k}, seed={args.seed}")
    print("=" * 78)
    print(f"{'Query':<10}  {'KMV consume (s)':>18}  {'baseline (s)':>14}  {'speedup':>8}")
    print("-" * 60)
    print(f"{'NC':<10}  {kmv_nc_train_s:>18.1f}  {shgn_nc_train_s:>14.1f}  "
          f"{shgn_nc_train_s / max(kmv_nc_train_s, 1e-9):>7.2f}x")
    print(f"{'Sim':<10}  {sim_sketch_total_s:>18.3f}  "
          f"{sim_exact_total_s:>14.3f}  "
          f"{sim_exact_total_s / max(sim_sketch_total_s, 1e-9):>7.2f}x")
    print()
    print(f"KMV one-time precompute (sketch propagate): {extract_time_s:.2f}s")
    print()
    print("Total wall-clock to serve {NC, Sim}:")
    kmv_total = extract_time_s + kmv_nc_train_s + sim_sketch_total_s
    base_total = shgn_nc_train_s + sim_exact_total_s
    print(f"  KMV path  : {kmv_total:.1f}s "
          f"(precompute {extract_time_s:.1f} + NC {kmv_nc_train_s:.1f} + Sim {sim_sketch_total_s:.3f})")
    print(f"  Baseline  : {base_total:.1f}s "
          f"(SHGN-NC {shgn_nc_train_s:.1f} + exact-Sim {sim_exact_total_s:.1f})")
    speedup = base_total / max(kmv_total, 1e-9)
    print(f"  speedup   : {speedup:.2f}x")

    # JSON record for downstream plotting.
    out = {
        "dataset": args.dataset,
        "k": args.k,
        "seed": args.seed,
        "kmv": {
            "precompute_s": extract_time_s,
            "nc_consume_s": kmv_nc_train_s,
            "sim_consume_s": sim_sketch_total_s,
            "nc_test_f1": float(nc_blob["test_f1"]),
        },
        "baseline": {
            "shgn_nc_train_s": shgn_nc_train_s,
            "exact_sim_s": sim_exact_total_s,
            "shgn_nc_test_f1": float(shgn_blob["test_f1"]) if shgn_blob else None,
        },
        "totals_q2_nc_sim": {
            "kmv": kmv_total,
            "baseline": base_total,
            "speedup_x": speedup,
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    res_path = out_dir / f"multi_query_amortization_k{args.k}_seed{args.seed}.json"
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[save] {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
