#!/usr/bin/env python3
"""
Rigor check on KGRW vs MPRW gap:
    * How many seeds per cell?
    * Mean and std of CKA/F1 per cell?
    * Is the KGRW-MPRW gap > 2σ (clean win), 1-2σ (suggestive), <1σ (noise)?

Inputs: results/<DS>/kgrw_bench.csv
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent


def run_dataset(ds: str):
    p = PROJ / "results" / ds / "kgrw_bench.csv"
    if not p.exists():
        print(f"[{ds}] NO DATA")
        return
    df = pd.read_csv(p, low_memory=False)
    for c in ["Edge_Count", "Mat_Time_s", "Macro_F1", "CKA", "Pred_Agreement"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["L"] == 2].copy()

    # k is NaN for MPRW (only w_prime matters); replace with -1 to keep in groupby.
    df_grp = df.copy()
    df_grp["k"] = df_grp["k"].fillna(-1)

    # Seed counts per cell
    seed_counts = (df_grp.groupby(["Method", "k", "w_prime"])["Seed"].count()
                   .reset_index().rename(columns={"Seed": "n_seeds"}))
    print(f"\n=== {ds} ===")
    print("Seed coverage per cell:")
    print(seed_counts.to_string(index=False))

    # Compute mean + std per cell
    agg = (df_grp.groupby(["Method", "k", "w_prime"])
             .agg(edges_mean=("Edge_Count", "mean"),
                  edges_std=("Edge_Count", "std"),
                  cka_mean=("CKA", "mean"),
                  cka_std=("CKA", "std"),
                  f1_mean=("Macro_F1", "mean"),
                  f1_std=("Macro_F1", "std"),
                  time_mean=("Mat_Time_s", "mean"),
                  time_std=("Mat_Time_s", "std"),
                  n=("Seed", "count"))
             .reset_index())

    mprw = agg[agg["Method"] == "MPRW"].copy().reset_index(drop=True)
    kgrw = agg[agg["Method"] == "KGRW"].copy().reset_index(drop=True)

    if mprw.empty or kgrw.empty:
        print(f"[{ds}] MPRW rows: {len(mprw)}, KGRW rows: {len(kgrw)} — nothing to compare")
        return

    # For each KGRW cell, find closest MPRW by edge count and compute gap + pooled std
    rows = []
    for _, kr in kgrw.iterrows():
        diffs = (mprw["edges_mean"] - kr["edges_mean"]).abs()
        mr = mprw.iloc[int(diffs.values.argmin())]
        # pooled std (2-sample)
        n1, n2 = kr["n"], mr["n"]
        if n1 < 2 or n2 < 2:
            continue
        pooled_cka_std = np.sqrt((kr["cka_std"]**2 + mr["cka_std"]**2) / 2)
        pooled_f1_std  = np.sqrt((kr["f1_std"]**2  + mr["f1_std"]**2)  / 2)
        se_cka = pooled_cka_std * np.sqrt(1/n1 + 1/n2)
        se_f1  = pooled_f1_std  * np.sqrt(1/n1 + 1/n2)
        gap_cka = kr["cka_mean"] - mr["cka_mean"]
        gap_f1  = kr["f1_mean"]  - mr["f1_mean"]
        # Welch-like t-stat
        t_cka = gap_cka / se_cka if se_cka > 0 else float("nan")
        t_f1  = gap_f1  / se_f1  if se_f1  > 0 else float("nan")
        rows.append({
            "k": int(kr["k"]),
            "w'": int(kr["w_prime"]),
            "n_k": int(kr["n"]),
            "n_m": int(mr["n"]),
            "K_edges": kr["edges_mean"],
            "M_edges": mr["edges_mean"],
            "K_cka":   kr["cka_mean"],
            "M_cka":   mr["cka_mean"],
            "cka_gap": gap_cka,
            "cka_SE":  se_cka,
            "t_cka":   t_cka,
            "f1_gap":  gap_f1,
            "f1_SE":   se_f1,
            "t_f1":    t_f1,
        })
    rdf = pd.DataFrame(rows).sort_values("K_edges").reset_index(drop=True)
    print("\nKGRW vs closest-density MPRW (L=2), t-stat = gap / SE:")
    print(rdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Flag significant wins / losses
    sig_wins  = rdf[(rdf["t_cka"] >  2.0) & (rdf["cka_gap"] > 0)]
    sig_loss  = rdf[(rdf["t_cka"] < -2.0) & (rdf["cka_gap"] < 0)]
    print(f"\n  Significant KGRW wins  (|t_CKA|>2): {len(sig_wins)}/{len(rdf)} cells")
    print(f"  Significant KGRW losses (|t_CKA|>2): {len(sig_loss)}/{len(rdf)} cells")
    if not sig_wins.empty:
        print("  Winning cells (CKA):")
        print(sig_wins[["k", "w'", "K_edges", "M_edges", "cka_gap", "t_cka",
                       "f1_gap", "t_f1"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    for ds in ["HGB_DBLP", "HGB_ACM", "HGB_IMDB"]:
        run_dataset(ds)


if __name__ == "__main__":
    main()
