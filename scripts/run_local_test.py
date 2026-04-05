"""
run_local_test.py — Quick end-to-end pipeline test for HGB_DBLP.

Runs exp1 → exp2 → exp3 for one metapath, then prints a formatted
comparison table of Exact vs KMV vs MPRW across k values.

Usage
-----
    python scripts/run_local_test.py
    python scripts/run_local_test.py --dataset HGB_ACM --epochs 30
    python scripts/run_local_test.py --no-train   # skip exp2 if weights exist
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly from scripts/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",    default="HGB_DBLP")
    p.add_argument("--metapath",   default="author_to_paper,paper_to_author",
                   help="Metapath to test (default: author_to_paper,paper_to_author)")
    p.add_argument("--depth",      type=int, nargs="+", default=[2, 3])
    p.add_argument("--k-values",   type=int, nargs="+", default=[8, 32, 128])
    p.add_argument("--epochs",     type=int, default=50,
                   help="Training epochs for exp2 (default: 50)")
    p.add_argument("--train-frac", type=float, default=0.1)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--no-train",   action="store_true",
                   help="Skip exp2 if weights already exist")
    p.add_argument("--max-rss-gb", type=float, default=None,
                   help="RSS memory guard for exp3 (GB)")
    return p.parse_args()


# ── Subprocess helpers ────────────────────────────────────────────────────

def run(cmd: list[str], step: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {step}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {step} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


# ── Table printer ─────────────────────────────────────────────────────────

_COL_ORDER = ["Method", "k_value", "Density_Matched_w", "L", "Edge_Count",
              "Materialization_Time", "Inference_Time", "Peak_RAM_MB",
              "Macro_F1", "Pred_Similarity", "Dirichlet_Energy",
              "CKA_L1", "CKA_L2", "CKA_L3", "CKA_L4"]

_HEADERS = {
    "Method":               "Method",
    "k_value":              "k",
    "Density_Matched_w":    "w",
    "L":                    "L",
    "Edge_Count":           "Edges",
    "Materialization_Time": "Mat(s)",
    "Inference_Time":       "Inf(s)",
    "Peak_RAM_MB":          "RAM(MB)",
    "Macro_F1":             "MacroF1",
    "Pred_Similarity":      "PredSim",
    "Dirichlet_Energy":     "DirE",
    "CKA_L1":               "CKA_L1",
    "CKA_L2":               "CKA_L2",
    "CKA_L3":               "CKA_L3",
    "CKA_L4":               "CKA_L4",
}

_WIDTHS = {
    "Method": 7, "k_value": 5, "Density_Matched_w": 6, "L": 3, "Edge_Count": 12,
    "Materialization_Time": 7, "Inference_Time": 7, "Peak_RAM_MB": 8,
    "Macro_F1": 8, "Pred_Similarity": 8, "Dirichlet_Energy": 8,
    "CKA_L1": 8, "CKA_L2": 8, "CKA_L3": 8, "CKA_L4": 8,
}


def _fmt(val: str, col: str) -> str:
    if val == "":
        return "—"
    try:
        f = float(val)
        if col in ("k_value", "L", "Edge_Count"):
            return str(int(f)) if val.strip() != "" else "—"
        if col in ("Materialization_Time", "Inference_Time"):
            return f"{f:.2f}"
        if col == "Peak_RAM_MB":
            return f"{f:.0f}"
        return f"{f:.4f}"
    except ValueError:
        return val


def _print_table(rows: list[dict], metapath: str, dataset: str) -> None:
    print(f"\n{'─'*80}")
    print(f"  Dataset: {dataset}   Metapath: {metapath}")
    print(f"{'─'*80}")

    # Header
    hdr = "  ".join(_HEADERS[c].rjust(_WIDTHS[c]) for c in _COL_ORDER)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    prev_method = None
    for row in rows:
        if row.get("Method") == "MPRW" and row.get("exact_status") == "MPRW_PENDING":
            continue  # old placeholder — skip
        method = row.get("Method", "")
        if method != prev_method and prev_method is not None:
            print()   # blank line between method groups
        prev_method = method

        cells = []
        for c in _COL_ORDER:
            val = row.get(c, "")
            cells.append(_fmt(val, c).rjust(_WIDTHS[c]))
        print("  ".join(cells))

    print(f"{'─'*80}\n")


def _load_csv(path: Path, metapath: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("MetaPath", "") == metapath:
                rows.append(row)

    # Sort: Exact first, then KMV by k, then MPRW by k; within each by L
    order = {"Exact": 0, "KMV": 1, "MPRW": 2}
    def _key(r):
        k_val = int(r.get("k_value") or 0)
        l_val = int(r.get("L") or 0)
        return (order.get(r.get("Method", ""), 9), k_val, l_val)

    rows.sort(key=_key)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    py   = sys.executable

    part_json   = Path(f"results/{args.dataset}/partition.json")
    weights_dir = Path(f"results/{args.dataset}/weights")
    csv_path    = Path(f"results/{args.dataset}/master_results.csv")
    depths_str  = list(map(str, args.depth))
    k_str       = list(map(str, args.k_values))

    # ── Exp 1: partition ──────────────────────────────────────────────────
    from src.config import config
    cfg         = config.get_dataset_config(args.dataset)
    target_type = cfg.target_node

    run([py, "scripts/exp1_partition.py",
         "--dataset",    args.dataset,
         "--target-type", target_type,
         "--train-frac", str(args.train_frac),
         "--seed",       str(args.seed)],
        "EXP1: Temporal partition")

    # ── Exp 2: train ──────────────────────────────────────────────────────
    # Check if we can skip
    mp_safe = args.metapath.replace(",", "_").replace("/", "_")
    weights_exist = all(
        (weights_dir / f"{mp_safe}_L{L}.pt").exists()
        for L in args.depth
    )

    if args.no_train and weights_exist:
        print(f"\n[skip] exp2 — weights found for all depths in {weights_dir}")
    else:
        run([py, "scripts/exp2_train.py", args.dataset,
             "--metapath",      args.metapath,
             "--depth",         *depths_str,
             "--epochs",        str(args.epochs),
             "--partition-json", str(part_json),
             "--seed",          str(args.seed)],
            "EXP2: Train SAGE")

    # ── Exp 3: inference (Exact + KMV + MPRW) ────────────────────────────
    # Delete the whole CSV so the fresh run writes a header with current schema.
    if csv_path.exists():
        csv_path.unlink()
        print(f"[reset] Deleted {csv_path} (schema may have changed)")

    exp3_cmd = [py, "scripts/exp3_inference.py", args.dataset,
                "--metapath",      args.metapath,
                "--depth",         *depths_str,
                "--k-values",      *k_str,
                "--partition-json", str(part_json),
                "--weights-dir",    str(weights_dir)]
    if args.max_rss_gb is not None:
        exp3_cmd += ["--max-rss-gb", str(args.max_rss_gb)]

    run(exp3_cmd, "EXP3: Inference — Exact + KMV + MPRW")

    # ── Print results ─────────────────────────────────────────────────────
    if not csv_path.exists():
        print(f"[ERROR] Results CSV not found: {csv_path}")
        sys.exit(1)

    rows = _load_csv(csv_path, args.metapath)
    if not rows:
        print(f"[WARN] No rows found for metapath '{args.metapath}' in {csv_path}")
    else:
        _print_table(rows, args.metapath, args.dataset)


def _purge_metapath_rows(csv_path: Path, metapath: str) -> None:
    """Remove rows matching this metapath from the CSV so exp3 reruns them."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        fieldnames = reader.fieldnames

    kept = [r for r in all_rows if r.get("MetaPath", "") != metapath]
    removed = len(all_rows) - len(kept)

    if removed > 0:
        print(f"[purge] Removed {removed} existing rows for metapath '{metapath}'")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(kept)


if __name__ == "__main__":
    main()
