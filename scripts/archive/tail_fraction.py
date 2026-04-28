"""tail_fraction.py — measure |T|/|V_L| from exact meta-path adjacency.

For each dataset's exact_*.adj, compute r_L(w) = number of distinct sources
reaching endpoint w under the meta-path, then report the tail fraction
|{w : r_L(w) <= k}| / |V_L| for k in {4, 8, 16, 32, 64}.

Usage:
    python scripts/tail_fraction.py

Output:
    Prints table to stdout.
    Writes results/tail_fraction.csv.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K_VALUES = [4, 8, 16, 32, 64]

TARGETS = [
    ("HGB_DBLP", "apapa",          "exact_apapa.adj"),
    ("HGB_ACM",  "papap",          "exact_papap.adj"),
    ("HGB_IMDB", "mdmdm",          "exact_mdmdm.adj"),
    ("HGB_IMDB", "mam",            "exact_movie_to_actor_actor_to_movie.adj"),
    ("HGB_IMDB", "mdm",            "exact_movie_to_director_director_to_movie.adj"),
]


def endpoint_rdeg(adj_path: Path) -> Counter:
    """Return Counter mapping endpoint w -> r(w) = number of distinct sources."""
    rdeg: Counter[int] = Counter()
    with open(adj_path) as f:
        for line in f:
            toks = line.split()
            if len(toks) < 2:
                continue
            # toks[0] = source u, toks[1:] = endpoints w
            dsts = set(int(t) for t in toks[1:])
            for w in dsts:
                rdeg[w] += 1
    return rdeg


def main() -> None:
    rows = []
    print(f"\n{'Dataset':<12} {'Metapath':<10} {'|V_L|':>8} {'max r':>8} {'mean r':>8} "
          + "  ".join(f"|T|/|V_L|@k={k}" for k in K_VALUES))
    print("-" * 110)

    for dataset, mp, fname in TARGETS:
        adj_path = ROOT / "results" / dataset / fname
        if not adj_path.exists():
            print(f"{dataset:<12} {mp:<10} MISSING ({fname})")
            continue

        rdeg = endpoint_rdeg(adj_path)
        V_L = len(rdeg)
        if V_L == 0:
            continue
        max_r = max(rdeg.values())
        mean_r = sum(rdeg.values()) / V_L

        fracs = []
        for k in K_VALUES:
            tail = sum(1 for r in rdeg.values() if r <= k)
            fracs.append(tail / V_L)

        frac_str = "  ".join(f"{f:>13.3f}" for f in fracs)
        print(f"{dataset:<12} {mp:<10} {V_L:>8} {max_r:>8} {mean_r:>8.1f}  {frac_str}")

        row = {"dataset": dataset, "metapath": mp, "V_L": V_L,
               "max_r": max_r, "mean_r": round(mean_r, 2)}
        for k, f in zip(K_VALUES, fracs):
            row[f"tail_frac_k{k}"] = round(f, 4)
        rows.append(row)

    # Write CSV
    out_csv = ROOT / "results" / "tail_fraction.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out_csv}")

    # Degree distribution summary (quantiles)
    print("\n" + "=" * 110)
    print("Reverse-degree quantiles (r_L(w) distribution over endpoints)")
    print(f"{'Dataset':<12} {'Metapath':<10} {'min':>6} {'p10':>6} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6} {'p99':>6} {'max':>8}")
    print("-" * 90)
    for dataset, mp, fname in TARGETS:
        adj_path = ROOT / "results" / dataset / fname
        if not adj_path.exists():
            continue
        rdeg = endpoint_rdeg(adj_path)
        if not rdeg:
            continue
        vals = sorted(rdeg.values())
        n = len(vals)
        def q(p): return vals[min(n - 1, int(p * n))]
        print(f"{dataset:<12} {mp:<10} {vals[0]:>6} {q(.10):>6} {q(.25):>6} "
              f"{q(.50):>6} {q(.75):>6} {q(.90):>6} {q(.99):>6} {vals[-1]:>8}")


if __name__ == "__main__":
    main()
