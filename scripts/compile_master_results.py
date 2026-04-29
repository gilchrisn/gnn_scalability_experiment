"""Compile every per-(dataset, method, seed) result JSON under results/
into one master CSV + a paper-ready markdown table.

Sources walked:
  - results/<DS>/sketch_feature_pilot_k*_seed*.json    (LoNe MLP / HAN variants)
  - results/<DS>/sketch_sparsifier_pilot_k*_seed*.json
  - results/<DS>/simple_hgn_baseline_seed*.json
  - results/<DS>/sketch_similarity_pilot_k*_seed*.json
  - results/<DS>/multi_query_amortization_k*_seed*.json

Outputs:
  - results/master_results.csv             flat per-row table
  - results/master_table_quality.md        per-(dataset, method) F1 + train_time
  - results/master_table_amortization.md   per-dataset KMV vs baseline cost
  - results/master_table_similarity.md     Jaccard fidelity per meta-path

Read-only: this script never overwrites the per-run JSONs, only summarises.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]


# ---------------------------------------------------------------------------
# Loaders — one function per result-file family
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[warn] could not parse {path}: {e}")
        return {}


def load_sketch_feature_results(ds: str) -> List[dict]:
    rows = []
    for p in sorted((RES / ds).glob("sketch_feature_pilot_k*_seed*.json")):
        b = _load_json(p)
        if not b:
            continue
        # Some old files don't have backbone field; default to han_sketch_edges.
        backbone = b.get("backbone", "han_sketch_edges")
        rows.append({
            "dataset": ds,
            "method": f"sketch_feature_{backbone}",
            "k": b.get("k"),
            "seed": b.get("seed"),
            "test_f1": b.get("test_f1"),
            "val_f1": b.get("best_val_f1"),
            "train_time_s": b.get("train_time_s"),
            "extract_time_s": "",  # not recorded in feature pilot
            "epoch": b.get("best_epoch"),
            "src_file": p.name,
        })
    return rows


def load_sketch_sparsifier_results(ds: str) -> List[dict]:
    rows = []
    for p in sorted((RES / ds).glob("sketch_sparsifier_pilot_k*_seed*.json")):
        b = _load_json(p)
        if not b:
            continue
        rows.append({
            "dataset": ds,
            "method": "sketch_sparsifier_SAGE",
            "k": b.get("k"),
            "seed": b.get("seed"),
            "test_f1": b.get("test_f1"),
            "val_f1": b.get("best_val_f1"),
            "train_time_s": b.get("train_time_s"),
            "extract_time_s": b.get("extract_time_s"),
            "epoch": b.get("best_epoch"),
            "src_file": p.name,
        })
    return rows


def load_simple_hgn_results(ds: str) -> List[dict]:
    rows = []
    for p in sorted((RES / ds).glob("simple_hgn_baseline_seed*.json")):
        b = _load_json(p)
        if not b:
            continue
        rows.append({
            "dataset": ds,
            "method": "simple_hgn_pyg_port",
            "k": "",
            "seed": b.get("seed"),
            "test_f1": b.get("test_f1"),
            "val_f1": b.get("best_val_f1"),
            "train_time_s": b.get("train_time_s"),
            "extract_time_s": "",
            "epoch": b.get("best_epoch"),
            "src_file": p.name,
        })
    return rows


def load_similarity_results(ds: str) -> List[dict]:
    """Returns one row per (dataset, k, seed, meta_path)."""
    rows = []
    for p in sorted((RES / ds).glob("sketch_similarity_pilot_k*_seed*.json")):
        b = _load_json(p)
        if not b:
            continue
        for mp, m in b.get("per_meta_path", {}).items():
            rows.append({
                "dataset": ds,
                "method": "sketch_similarity_jaccard",
                "k": b.get("k"),
                "seed": b.get("seed"),
                "meta_path": mp,
                "n_pairs": m.get("n_pairs"),
                "sketch_time_s": m.get("sketch_time_s"),
                "exact_time_s": m.get("exact_time_s"),
                "speedup_x": m.get("speedup_x"),
                "mae": m.get("mae"),
                "pearson": m.get("pearson"),
                "src_file": p.name,
            })
    return rows


def load_amortization_results(ds: str) -> List[dict]:
    rows = []
    for p in sorted((RES / ds).glob("multi_query_amortization_k*_seed*.json")):
        b = _load_json(p)
        if not b:
            continue
        rows.append({
            "dataset": ds,
            "k": b.get("k"),
            "seed": b.get("seed"),
            "kmv_precompute_s": b.get("kmv", {}).get("precompute_s"),
            "kmv_nc_s": b.get("kmv", {}).get("nc_consume_s"),
            "kmv_sim_s": b.get("kmv", {}).get("sim_consume_s"),
            "kmv_nc_test_f1": b.get("kmv", {}).get("nc_test_f1"),
            "shgn_nc_train_s": b.get("baseline", {}).get("shgn_nc_train_s"),
            "exact_sim_s": b.get("baseline", {}).get("exact_sim_s"),
            "shgn_nc_test_f1": b.get("baseline", {}).get("shgn_nc_test_f1"),
            "kmv_total_q2": b.get("totals_q2_nc_sim", {}).get("kmv"),
            "baseline_total_q2": b.get("totals_q2_nc_sim", {}).get("baseline"),
            "speedup_q2_x": b.get("totals_q2_nc_sim", {}).get("speedup_x"),
            "src_file": p.name,
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _agg_seeds(rows: List[dict], key: str) -> Dict:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def quality_table_md(quality_rows: List[dict]) -> str:
    """Build a markdown table comparing methods per dataset on F1 + train_time."""
    methods_order = [
        "sketch_feature_mlp",
        "sketch_feature_han_sketch_edges",
        "sketch_feature_han_real_edges",
        "sketch_sparsifier_SAGE",
        "simple_hgn_pyg_port",
    ]
    method_labels = {
        "sketch_feature_mlp":               "Sketch-feature (LoNe MLP)",
        "sketch_feature_han_sketch_edges":  "Sketch-feature (HAN, sketch-edges)",
        "sketch_feature_han_real_edges":    "Sketch-feature (HAN, real edges)",
        "sketch_sparsifier_SAGE":           "Sketch-sparsifier (SAGE)",
        "simple_hgn_pyg_port":              "Simple-HGN (PyG port)",
    }

    lines = []
    lines.append("# Quality table (NC test macro-F1)\n")
    lines.append(
        "| Dataset | Method | n seeds | test_f1 | val_f1 | train_time (s) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")

    by_ds_method: Dict[str, Dict[str, List[dict]]] = {}
    for r in quality_rows:
        by_ds_method.setdefault(r["dataset"], {}).setdefault(r["method"], []).append(r)

    for ds in DATASETS:
        if ds not in by_ds_method:
            continue
        for m in methods_order:
            rows = by_ds_method[ds].get(m, [])
            if not rows:
                continue
            f1 = _agg_seeds(rows, "test_f1")
            vf = _agg_seeds(rows, "val_f1")
            tt = _agg_seeds(rows, "train_time_s")
            f1_s = (f"{f1['mean']:.4f} ± {f1['std']:.4f}" if f1["n"]
                    else "—")
            vf_s = (f"{vf['mean']:.4f} ± {vf['std']:.4f}" if vf["n"]
                    else "—")
            tt_s = (f"{tt['mean']:.1f} ± {tt['std']:.1f}" if tt["n"]
                    else "—")
            lines.append(
                f"| {ds} | {method_labels[m]} | {f1['n']} "
                f"| {f1_s} | {vf_s} | {tt_s} |"
            )
    lines.append("")
    return "\n".join(lines)


def amortization_table_md(amort_rows: List[dict]) -> str:
    lines = []
    lines.append("# Multi-query amortization (NC + Similarity)\n")
    lines.append(
        "Per-task wall-clock breakdown. KMV cost = "
        "precompute (one-time) + per-task consume. Baseline cost = "
        "Simple-HGN train + exact Jaccard.\n"
    )
    lines.append(
        "| Dataset | seed | precompute | KMV NC | KMV Sim | "
        "SHGN NC | exact Sim | KMV total Q=2 | Baseline total Q=2 | "
        "speedup |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in amort_rows:
        lines.append(
            f"| {r['dataset']} | {r['seed']} | "
            f"{r['kmv_precompute_s']:.2f}s | "
            f"{r['kmv_nc_s']:.2f}s | "
            f"{r['kmv_sim_s']:.2f}s | "
            f"{r['shgn_nc_train_s']:.2f}s | "
            f"{r['exact_sim_s']:.2f}s | "
            f"{r['kmv_total_q2']:.2f}s | "
            f"{r['baseline_total_q2']:.2f}s | "
            f"**{r['speedup_q2_x']:.2f}×** |"
        )
    lines.append("")
    lines.append(
        "Caveat: this comparison contrasts KMV's task-agnostic "
        "representation against per-task SHGN trainings. A fairer "
        "multi-query baseline would be a multi-task SHGN with shared "
        "encoder + per-task heads (not measured this session).\n"
    )
    return "\n".join(lines)


def similarity_table_md(sim_rows: List[dict]) -> str:
    lines = []
    lines.append("# Similarity fidelity (Jaccard from sketch vs exact)\n")
    lines.append(
        "| Dataset | meta-path | seed | n_pairs | sketch_t (s) | "
        "exact_t (s) | speedup | MAE | Pearson |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sim_rows:
        pearson = r["pearson"]
        pearson_s = (f"{pearson:.4f}" if pearson is not None and not (pearson != pearson)
                     else "nan†")
        lines.append(
            f"| {r['dataset']} | {r['meta_path']} | {r['seed']} | "
            f"{r['n_pairs']} | {r['sketch_time_s']:.3f} | "
            f"{r['exact_time_s']:.3f} | {r['speedup_x']:.2f}× | "
            f"{r['mae']:.4f} | {pearson_s} |"
        )
    lines.append("")
    lines.append(
        "†Pearson = nan when the exact ground truth has zero variance "
        "(e.g. ACM PTP graph is saturated; every paper-pair shares ~all "
        "terms). MAE = 0 in that case is the meaningful fidelity statistic.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    quality_rows: List[dict] = []
    sim_rows: List[dict] = []
    amort_rows: List[dict] = []

    for ds in DATASETS:
        quality_rows += load_sketch_feature_results(ds)
        quality_rows += load_sketch_sparsifier_results(ds)
        quality_rows += load_simple_hgn_results(ds)
        sim_rows += load_similarity_results(ds)
        amort_rows += load_amortization_results(ds)

    # CSV — one row per (dataset, method, seed) for quality + extras for sim/amort.
    csv_path = RES / "master_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        keys = ["dataset", "method", "k", "seed", "test_f1", "val_f1",
                "train_time_s", "extract_time_s", "epoch", "src_file"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in quality_rows:
            w.writerow(r)
    print(f"[csv]  {len(quality_rows)} quality rows -> {csv_path}")

    # Quality markdown.
    qpath = RES / "master_table_quality.md"
    qpath.write_text(quality_table_md(quality_rows), encoding="utf-8")
    print(f"[md]   quality table -> {qpath}")

    # Amortization markdown.
    if amort_rows:
        apath = RES / "master_table_amortization.md"
        apath.write_text(amortization_table_md(amort_rows), encoding="utf-8")
        print(f"[md]   amortization table -> {apath}  ({len(amort_rows)} rows)")

    # Similarity markdown + CSV.
    if sim_rows:
        spath = RES / "master_table_similarity.md"
        spath.write_text(similarity_table_md(sim_rows), encoding="utf-8")
        sim_csv = RES / "master_results_similarity.csv"
        with open(sim_csv, "w", newline="", encoding="utf-8") as fh:
            keys = ["dataset", "method", "k", "seed", "meta_path", "n_pairs",
                    "sketch_time_s", "exact_time_s", "speedup_x", "mae",
                    "pearson", "src_file"]
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in sim_rows:
                w.writerow(r)
        print(f"[md]   similarity table -> {spath}")
        print(f"[csv]  {len(sim_rows)} sim rows -> {sim_csv}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
