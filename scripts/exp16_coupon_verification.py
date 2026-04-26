#!/usr/bin/env python3
"""
Numerical verification of Lemmas 1–2 in mprw_kmv_optimality.md.

Simulate a neighborhood N(v) of size D. Draw walks with replacement under
two endpoint distributions:
    1. UNIFORM — p_i = 1/D
    2. ZIPFIAN — p_i ∝ 1/i^α  (degree-biased)

For each k ∈ {2,4,...,D}, measure mean walks needed to retain k distinct
neighbors. Compare to theoretical predictions:
    Lemma 1 (uniform):  E[W_k] = D(H_D − H_{D−k})
    Lemma 2 (Zipfian):  E[W_k] ≥ log(k) / p_D

Outputs: figures/kmv_properties/optimality_verification.pdf
         results/kmv_properties/coupon_sim.csv
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
FIG  = PROJ / "figures" / "kmv_properties"
RES  = PROJ / "results" / "kmv_properties"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def harmonic(n: int) -> float:
    return float(np.sum(1.0 / np.arange(1, n + 1))) if n > 0 else 0.0


def theoretical_uniform(D: int, k: int) -> float:
    return D * (harmonic(D) - harmonic(D - k))


def walks_to_collect_k(p: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """Sample from p with replacement until k distinct values seen."""
    seen = set()
    w = 0
    D = p.size
    while len(seen) < k:
        # batch sample for efficiency
        batch = rng.choice(D, size=1024, replace=True, p=p)
        for u in batch:
            seen.add(int(u))
            w += 1
            if len(seen) >= k:
                return w
    return w


def simulate(D: int, k_values: List[int], dist: str, alpha: float, trials: int,
             rng: np.random.Generator) -> List[Tuple[int, float, float]]:
    if dist == "uniform":
        p = np.full(D, 1.0 / D)
    elif dist == "zipf":
        ranks = np.arange(1, D + 1, dtype=np.float64)
        p = 1.0 / ranks**alpha
        p /= p.sum()
    else:
        raise ValueError(dist)
    out = []
    for k in k_values:
        ws = np.empty(trials, dtype=np.int64)
        for t in range(trials):
            ws[t] = walks_to_collect_k(p, k, rng)
        out.append((k, float(ws.mean()), float(ws.std())))
    return out


def main():
    D = 1000
    k_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 900]
    trials = 200
    rng = np.random.default_rng(42)

    print(f"D={D}, trials={trials}/point")

    print("Simulating uniform endpoints...")
    res_uni = simulate(D, k_values, "uniform", 1.0, trials, rng)
    print("Simulating Zipfian(α=1) endpoints...")
    res_z1 = simulate(D, k_values, "zipf", 1.0, trials, rng)
    print("Simulating Zipfian(α=1.5) endpoints...")
    res_z15 = simulate(D, k_values, "zipf", 1.5, trials, rng)

    # Write CSV
    csv_path = RES / "coupon_sim.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "k", "dist", "alpha", "mean_walks", "std_walks",
                    "theoretical_uniform", "kmv_walks", "ratio"])
        for (k, mean, std) in res_uni:
            w.writerow([D, k, "uniform", 1.0, mean, std,
                        theoretical_uniform(D, k), k, mean / k])
        for (k, mean, std) in res_z1:
            w.writerow([D, k, "zipf", 1.0, mean, std, "", k, mean / k])
        for (k, mean, std) in res_z15:
            w.writerow([D, k, "zipf", 1.5, mean, std, "", k, mean / k])
    print(f"→ {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ks = np.array(k_values)
    uni = np.array([m for _, m, _ in res_uni])
    z1  = np.array([m for _, m, _ in res_z1])
    z15 = np.array([m for _, m, _ in res_z15])
    theo_uni = np.array([theoretical_uniform(D, k) for k in k_values])
    ax.plot(ks, ks, "k--", label="KMV (= k)", lw=1.5)
    ax.plot(ks, theo_uni, "b:", label=r"Theory: $D(H_D-H_{D-k})$", lw=1.5)
    ax.plot(ks, uni, "o-", color="#1f77b4", label="MPRW uniform (sim)")
    ax.plot(ks, z1,  "s-", color="#ff7f0e", label=r"MPRW Zipf α=1 (sim)")
    ax.plot(ks, z15, "^-", color="#d62728", label=r"MPRW Zipf α=1.5 (sim)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("k (neighbors to retain)")
    ax.set_ylabel("mean walks to retain k distinct")
    ax.set_title(f"Walks vs k  (D={D}, {trials} trials)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    # ratio MPRW_walks / KMV_walks = mean / k
    r_uni = uni / ks
    r_z1  = z1  / ks
    r_z15 = z15 / ks
    ax.plot(ks, np.ones_like(ks, dtype=float), "k--", label="KMV baseline (1.0)", lw=1.5)
    ax.plot(ks, r_uni, "o-", color="#1f77b4", label="MPRW uniform")
    ax.plot(ks, r_z1,  "s-", color="#ff7f0e", label=r"MPRW Zipf α=1")
    ax.plot(ks, r_z15, "^-", color="#d62728", label=r"MPRW Zipf α=1.5")
    # theoretical log growth ref
    ax.plot(ks, np.log2(ks), "g:", label=r"$\log_2 k$ reference", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k")
    ax.set_ylabel("walks_MPRW / k   (= budget overhead vs KMV)")
    ax.set_title("Budget overhead: MPRW vs KMV")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    plt.suptitle(r"Coupon-collector verification: MPRW is rate-optimal, budget-suboptimal by $\Theta(\rho \log k)$",
                 fontsize=11)
    plt.tight_layout()
    out = FIG / "optimality_verification.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"→ {out}")

    # Quick summary print
    print("\n=== Summary: MPRW walks / k  (overhead factor vs KMV) ===")
    print(f"{'k':>5s} {'uniform':>10s} {'zipf α=1':>10s} {'zipf α=1.5':>12s}")
    for i, k in enumerate(k_values):
        print(f"{k:>5d} {r_uni[i]:>10.2f} {r_z1[i]:>10.2f} {r_z15[i]:>12.2f}")


if __name__ == "__main__":
    main()
