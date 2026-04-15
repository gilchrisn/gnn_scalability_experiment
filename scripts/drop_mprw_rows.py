"""drop_mprw_rows.py -- Remove MPRW rows from each dataset's master_results.csv.

Use after a failed MPRW run (e.g. mprw_exec missing) so the overnight
orchestrator can re-fill them on the next pass.  Exact and KMV rows are kept.

Usage
-----
    python scripts/drop_mprw_rows.py
    python scripts/drop_mprw_rows.py --datasets HGB_ACM HGB_DBLP
    python scripts/drop_mprw_rows.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASETS = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "HNE_PubMed"]


def drop_mprw(csv_path: Path, dry_run: bool) -> tuple[int, int]:
    if not csv_path.exists():
        print(f"[skip] {csv_path} does not exist")
        return 0, 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        print(f"[skip] {csv_path} is empty")
        return 0, 0
    header, body = rows[0], rows[1:]
    try:
        method_idx = header.index("Method")
    except ValueError:
        print(f"[skip] {csv_path}: no 'Method' column")
        return 0, 0
    keep = [r for r in body if len(r) > method_idx and r[method_idx] != "MPRW"]
    dropped = len(body) - len(keep)
    print(f"[{csv_path.relative_to(_ROOT)}] keep={len(keep)}  drop={dropped}")
    if dry_run or dropped == 0:
        return len(keep), dropped
    backup = csv_path.with_suffix(csv_path.suffix + ".bak")
    shutil.copy2(csv_path, backup)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(keep)
    print(f"           backup -> {backup.relative_to(_ROOT)}")
    return len(keep), dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=_DEFAULT_DATASETS)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print row counts without modifying any file")
    args = ap.parse_args()

    total_kept = total_dropped = 0
    for ds in args.datasets:
        kept, dropped = drop_mprw(_ROOT / "results" / ds / "master_results.csv",
                                  args.dry_run)
        total_kept += kept
        total_dropped += dropped

    print("-" * 60)
    print(f"total kept = {total_kept}   total dropped (MPRW) = {total_dropped}")
    if args.dry_run:
        print("DRY-RUN: nothing was modified.")
    else:
        print("Done.  Re-run:  python scripts/run_overnight_all_datasets.py "
              "--continue-on-error --max-rss-gb 80")


if __name__ == "__main__":
    main()
