#!/usr/bin/env python3
"""
Cross-dataset H1 test using existing master_results.csv files.

Each dataset was run with a single meta-path (its "current pick"). The
meta-paths span the homo_ratio spectrum (DBLP APA=2.63, IMDB MKM=1.11,
ACM PTP=1.00, PubMed DCD=1.05). H1 predicts:
    MPRW > KMV gap on CKA/F1/PA shrinks as homo_ratio → 1.

For each dataset we pair each KMV k-row with the MPRW row at closest edge-count
(density-matched), then report:
    gap_CKA   = MPRW_CKA  - KMV_CKA
    gap_F1    = MPRW_F1   - KMV_F1
    gap_PA    = MPRW_PA   - KMV_PA

Aggregation: mean across seeds.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent

# (ds_key, homo_ratio from exp14) — sorted heterophilic → homophilic
DATASETS = [
    ("HGB_ACM",    1.00),   # PTP, saturated
    ("HNE_PubMed", 1.05),   # DCD, near-heterophilic
    ("HGB_IMDB",   1.11),   # MKM, mildly homophilic
    ("HGB_DBLP",   2.63),   # APA, strongly homophilic
]


def load_csv(ds: str) -> pd.DataFrame | None:
    f = PROJ / "results" / ds / "master_results.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, low_memory=False)
    # normalize w column name across schemas
    if "w_value" not in df.columns and "Density_Matched_w" in df.columns:
        df = df.rename(columns={"Density_Matched_w": "w_value"})
    return df


def summarize(ds: str, homo_ratio: float, df: pd.DataFrame) -> pd.DataFrame:
    # aggregate across seeds: mean Edge_Count, CKA_L2, Pred_Similarity, Macro_F1
    metrics = ["Edge_Count", "CKA_L2", "Pred_Similarity", "Macro_F1"]
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    # Use L=2 rows only for the head-to-head at fixed depth
    df2 = df[df["L"] == 2].copy()

    # KMV: group by k_value, mean across seeds
    kmv = (df2[df2["Method"] == "KMV"]
           .groupby("k_value")[metrics].mean()
           .sort_index())
    kmv.index = kmv.index.astype(int)

    # MPRW: group by w_value, mean across seeds
    mprw = df2[df2["Method"] == "MPRW"].copy()
    # keep rows with a sensible w_value
    mprw["w_value"] = pd.to_numeric(mprw["w_value"], errors="coerce")
    mprw = mprw.dropna(subset=["w_value"])
    mprw = mprw.groupby("w_value")[metrics].mean().sort_index()

    if kmv.empty or mprw.empty:
        return pd.DataFrame()

    # For each KMV row, find MPRW with closest Edge_Count
    rows = []
    for k, kr in kmv.iterrows():
        diffs = (mprw["Edge_Count"] - kr["Edge_Count"]).abs()
        w_best = diffs.idxmin()
        mr = mprw.loc[w_best]
        rows.append({
            "dataset":   ds,
            "homo_ratio": homo_ratio,
            "k":          int(k),
            "w_matched":  int(w_best),
            "kmv_edges":  kr["Edge_Count"],
            "mprw_edges": mr["Edge_Count"],
            "density_ratio":  mr["Edge_Count"] / max(1.0, kr["Edge_Count"]),
            "kmv_CKA":    kr["CKA_L2"],
            "mprw_CKA":   mr["CKA_L2"],
            "gap_CKA":    mr["CKA_L2"] - kr["CKA_L2"],
            "kmv_PA":     kr["Pred_Similarity"],
            "mprw_PA":    mr["Pred_Similarity"],
            "gap_PA":     mr["Pred_Similarity"] - kr["Pred_Similarity"],
            "kmv_F1":     kr["Macro_F1"],
            "mprw_F1":    mr["Macro_F1"],
            "gap_F1":     mr["Macro_F1"] - kr["Macro_F1"],
        })
    return pd.DataFrame(rows)


def main():
    out_rows = []
    for ds, ratio in DATASETS:
        df = load_csv(ds)
        if df is None:
            print(f"[skip] {ds}: no master_results.csv")
            continue
        summary = summarize(ds, ratio, df)
        if summary.empty:
            print(f"[skip] {ds}: no L=2 KMV+MPRW rows")
            continue
        out_rows.append(summary)
        print(f"\n=== {ds} (homo_ratio = {ratio:.2f}) ===")
        print(summary[["k", "w_matched", "density_ratio",
                       "kmv_CKA", "mprw_CKA", "gap_CKA",
                       "kmv_PA", "mprw_PA", "gap_PA",
                       "kmv_F1", "mprw_F1", "gap_F1"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if not out_rows:
        print("No data to aggregate.")
        return

    all_rows = pd.concat(out_rows, ignore_index=True)

    # Mean gap per dataset
    mean_gap = (all_rows.groupby(["dataset", "homo_ratio"])
                [["gap_CKA", "gap_PA", "gap_F1"]].mean()
                .reset_index()
                .sort_values("homo_ratio"))
    print("\n\n=== MEAN MPRW-KMV GAP BY DATASET (sorted heterophilic → homophilic) ===")
    print(mean_gap.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # Correlation between homo_ratio and each gap
    print("\nPearson r(homo_ratio vs gap):")
    for g in ["gap_CKA", "gap_PA", "gap_F1"]:
        x = mean_gap["homo_ratio"].values.astype(float)
        y = mean_gap[g].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            print(f"  {g}: n<3, skip")
            continue
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        print(f"  {g}: r = {r:+.3f}  ({'MPRW advantage scales WITH homophily' if r>0 else 'MPRW advantage scales AGAINST homophily'})")

    out = PROJ / "results" / "kmv_properties" / "h1_crossdataset.csv"
    all_rows.to_csv(out, index=False)
    mean_gap.to_csv(PROJ / "results" / "kmv_properties" / "h1_meangap.csv", index=False)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
