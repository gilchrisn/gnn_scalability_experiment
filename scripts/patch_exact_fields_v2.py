"""Backfill null Exact fields in v2 JSONs from each cell's seed=42 k=8 producer run.

Bug: 240/300 v2 JSONs at results/approach_a_2026_05_07/<DS>/<arch>/<mp>_seed<S>_k<K>.json
have null macro_f1_exact, n_edges_exact, *_exact_s, *_exact_mb. Only the first (cell, seed)
run wrote Exact metrics; later k values reused cached Z without re-emitting scalars.

Fix: for each (dataset, arch, mp, seed), find the JSON that has non-null macro_f1_exact
(typically k=8) and copy the Exact fields into the other 4 (k=16,32,64,128) JSONs of
that same (cell, seed). Exact is deterministic and cell-determined, so this is safe.

Caveat: inference_time_exact_s, mat_time_exact_s, *_exact_mb fields carry the producer
run's measurements rather than the actual cached-reuse latency. Paper text must report
Exact wallclock from the producer (k=8) run only.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "approach_a_2026_05_07"

EXACT_FIELDS = [
    "macro_f1_exact", "micro_f1_exact",
    "macro_f1_gap", "micro_f1_gap", "f1_gap",
    "n_edges_exact",
    "inference_time_exact_s", "mat_time_exact_s",
    "mat_peak_rss_mb_exact", "inf_peak_rss_mb_exact",
]


def main():
    paths = list(RES.glob("*/*/*.json"))
    paths = [p for p in paths if "/sanity/" not in str(p).replace("\\", "/")]
    print(f"[load] {len(paths)} JSONs")

    by_cell_seed = defaultdict(list)
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"[skip] cannot read {p}: {e}")
            continue
        key = (d.get("dataset"), d.get("arch"), d.get("meta_path"), d.get("seed"))
        by_cell_seed[key].append((p, d))

    n_cells = len(by_cell_seed)
    n_patched = 0
    n_no_donor = 0
    n_already_ok = 0

    for key, entries in by_cell_seed.items():
        donor = next((d for _, d in entries
                      if d.get("macro_f1_exact") is not None), None)
        if donor is None:
            n_no_donor += 1
            print(f"[no donor] {key}: no JSON with macro_f1_exact set")
            continue

        donor_micro = donor.get("micro_f1_exact")
        donor_macro = donor.get("macro_f1_exact")
        donor_n_edges = donor.get("n_edges_exact")
        donor_inf_time = donor.get("inference_time_exact_s")
        donor_mat_time = donor.get("mat_time_exact_s")
        donor_mat_ram = donor.get("mat_peak_rss_mb_exact")
        donor_inf_ram = donor.get("inf_peak_rss_mb_exact")

        for p, d in entries:
            if d.get("macro_f1_exact") is not None:
                n_already_ok += 1
                continue
            d["macro_f1_exact"] = donor_macro
            d["micro_f1_exact"] = donor_micro
            d["n_edges_exact"] = donor_n_edges
            d["inference_time_exact_s"] = donor_inf_time
            d["mat_time_exact_s"] = donor_mat_time
            d["mat_peak_rss_mb_exact"] = donor_mat_ram
            d["inf_peak_rss_mb_exact"] = donor_inf_ram
            kmv_macro = d.get("macro_f1_kmv")
            kmv_micro = d.get("micro_f1_kmv")
            d["macro_f1_gap"] = (None if (kmv_macro is None or donor_macro is None)
                                 else float(kmv_macro) - float(donor_macro))
            d["micro_f1_gap"] = (None if (kmv_micro is None or donor_micro is None)
                                 else float(kmv_micro) - float(donor_micro))
            d["f1_gap"] = d["macro_f1_gap"]
            d["_patch_2026_05_07"] = (
                "Exact fields backfilled from donor run "
                f"(seed={donor.get('seed')}, k={donor.get('kmv_k')}). "
                "*_exact_s and *_exact_mb fields are producer-run measurements."
            )
            p.write_text(json.dumps(d, indent=2))
            n_patched += 1

    print(f"[done] cells: {n_cells}, patched: {n_patched}, "
          f"already-ok: {n_already_ok}, no-donor: {n_no_donor}")


if __name__ == "__main__":
    main()
