"""
Plot all extension experiment results.
Usage: python scripts/plot_extension.py
Output: results/plots/*.png
"""
import os
import sys
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

PLOT_DIR = os.path.join(project_root, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DATASETS = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB"]
K_VALS = [4, 8, 16, 32, 64]


def load_extension(dataset, name="extension"):
    path = os.path.join(project_root, "results", dataset, f"{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["snapshot"] != "FAILED"].copy()
    for col in ["fraction", "k", "n_edges_exact", "n_edges_kmv", "adj_mb_exact",
                 "adj_mb_kmv", "t_exact_mat", "t_kmv_mat", "f1_exact", "f1_kmv",
                 "cka_kmv", "pred_agreement", "dirichlet_exact", "dirichlet_kmv",
                 "speedup_kmv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_table4(dataset):
    path = os.path.join(project_root, "results", dataset, "table4.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def shorten_mp(mp):
    parts = mp.split(",")
    if len(parts) <= 2:
        return mp.replace("_to_", "->").replace("rev_", "r")
    return f"{parts[0].replace('_to_','->')}...({len(parts)}hop)"


def plot_cka_vs_fraction():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        df = load_extension(ds)
        if df is None:
            ax.set_title(f"{ds}\n(no data)")
            continue
        for mp in df["metapath"].unique():
            sub = df[df["metapath"] == mp].sort_values("fraction")
            sub = sub.dropna(subset=["cka_kmv"])
            if len(sub) > 0:
                ax.plot(sub["fraction"], sub["cka_kmv"], "o-", label=shorten_mp(mp), markersize=5)
        ax.axhline(0.85, color="red", linestyle="--", alpha=0.5, label="0.85 threshold")
        ax.set_xlabel("Graph Fraction")
        ax.set_title(ds.replace("HGB_", ""))
        ax.set_ylim(0.7, 1.02)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("CKA")
    fig.suptitle("CKA vs Graph Fraction (Scale Curve)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "cka_vs_fraction.png"), bbox_inches="tight")
    print("  Saved cka_vs_fraction.png")


def plot_f1_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, ds in zip(axes, DATASETS):
        df = load_extension(ds)
        if df is None:
            ax.set_title(f"{ds}\n(no data)")
            continue
        valid = df.dropna(subset=["f1_exact", "f1_kmv"])
        for mp in valid["metapath"].unique():
            sub = valid[valid["metapath"] == mp]
            ax.scatter(sub["f1_exact"], sub["f1_kmv"], label=shorten_mp(mp), s=30, alpha=0.7)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("F1 (Exact)")
        ax.set_title(ds.replace("HGB_", ""))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("F1 (KMV)")
    fig.suptitle("F1: Exact vs KMV", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "f1_comparison.png"), bbox_inches="tight")
    print("  Saved f1_comparison.png")


def plot_pred_agreement():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        df = load_extension(ds)
        if df is None:
            ax.set_title(f"{ds}\n(no data)")
            continue
        valid = df.dropna(subset=["pred_agreement"])
        for mp in valid["metapath"].unique():
            sub = valid[valid["metapath"] == mp].sort_values("fraction")
            ax.plot(sub["fraction"], sub["pred_agreement"], "o-",
                    label=shorten_mp(mp), markersize=5)
        ax.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95%")
        ax.set_xlabel("Graph Fraction")
        ax.set_title(ds.replace("HGB_", ""))
        ax.set_ylim(0.75, 1.02)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Prediction Agreement")
    fig.suptitle("Prediction Agreement vs Graph Fraction", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "pred_agreement.png"), bbox_inches="tight")
    print("  Saved pred_agreement.png")


def plot_dirichlet():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, ds in zip(axes, DATASETS):
        df = load_extension(ds)
        if df is None:
            ax.set_title(f"{ds}\n(no data)")
            continue
        valid = df.dropna(subset=["dirichlet_exact", "dirichlet_kmv"])
        for mp in valid["metapath"].unique():
            sub = valid[valid["metapath"] == mp].sort_values("fraction")
            ax.plot(sub["fraction"], sub["dirichlet_exact"], "o-",
                    label=f"{shorten_mp(mp)} exact", markersize=4)
            ax.plot(sub["fraction"], sub["dirichlet_kmv"], "s--",
                    label=f"{shorten_mp(mp)} KMV", markersize=4)
        ax.set_xlabel("Graph Fraction")
        ax.set_title(ds.replace("HGB_", ""))
        ax.set_yscale("log")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Dirichlet Energy (log)")
    fig.suptitle("Dirichlet Energy: Exact vs KMV (both >> 0 = no over-smoothing)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "dirichlet_energy.png"), bbox_inches="tight")
    print("  Saved dirichlet_energy.png")


def plot_depthwise_cka():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        df = load_extension(ds)
        if df is None:
            ax.set_title(f"{ds}\n(no data)")
            continue
        full = df[df["fraction"] == 1.0].copy()
        valid = full.dropna(subset=["depthwise_cka"])
        valid = valid[valid["depthwise_cka"] != ""]
        labels, layer1_vals, layer2_vals = [], [], []
        for mp in valid["metapath"].unique():
            row = valid[valid["metapath"] == mp].iloc[0]
            try:
                layers = ast.literal_eval(row["depthwise_cka"])
                layer1_vals.append(layers[0])
                layer2_vals.append(layers[1] if len(layers) > 1 else layers[0])
                labels.append(shorten_mp(mp))
            except:
                continue
        if labels:
            width = 0.35
            x = range(len(labels))
            ax.bar([i - width/2 for i in x], layer1_vals, width, label="Layer 1", alpha=0.8)
            ax.bar([i + width/2 for i in x], layer2_vals, width, label="Layer 2", alpha=0.8)
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.axhline(0.85, color="red", linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)
        ax.set_title(ds.replace("HGB_", ""))
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("CKA")
    fig.suptitle("Depth-wise CKA at 100% (Layer 1 vs Layer 2)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "depthwise_cka.png"), bbox_inches="tight")
    print("  Saved depthwise_cka.png")


def plot_memory():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, ds in zip(axes, DATASETS):
        df = load_extension(ds)
        if df is None:
            ax.set_title(f"{ds}\n(no data)")
            continue
        valid = df.dropna(subset=["adj_mb_exact", "adj_mb_kmv"])
        for mp in valid["metapath"].unique():
            sub = valid[valid["metapath"] == mp].sort_values("fraction")
            ax.plot(sub["fraction"], sub["adj_mb_exact"], "o-",
                    label=f"{shorten_mp(mp)} exact", markersize=4)
            ax.plot(sub["fraction"], sub["adj_mb_kmv"], "s--",
                    label=f"{shorten_mp(mp)} KMV", markersize=4)
        ax.set_xlabel("Graph Fraction")
        ax.set_title(ds.replace("HGB_", ""))
        ax.set_yscale("log")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Adjacency Size (MB, log)")
    fig.suptitle("Memory: Exact vs KMV Adjacency Size", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "memory_comparison.png"), bbox_inches="tight")
    print("  Saved memory_comparison.png")


def plot_speedup():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        df = load_extension(ds)
        if df is None:
            ax.set_title(f"{ds}\n(no data)")
            continue
        valid = df.dropna(subset=["speedup_kmv"])
        for mp in valid["metapath"].unique():
            sub = valid[valid["metapath"] == mp].sort_values("fraction")
            ax.plot(sub["fraction"], sub["speedup_kmv"], "o-",
                    label=shorten_mp(mp), markersize=5)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Graph Fraction")
        ax.set_title(ds.replace("HGB_", ""))
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Speedup (Exact / KMV)")
    fig.suptitle("Materialization Speedup vs Graph Fraction", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "speedup_vs_fraction.png"), bbox_inches="tight")
    print("  Saved speedup_vs_fraction.png")


def plot_k_sensitivity():
    dfs = {}
    for k in K_VALS:
        name = "extension" if k == 32 else f"extension_k{k}"
        df = load_extension("HGB_ACM", name)
        if df is not None:
            dfs[k] = df[df["fraction"] == 1.0]
    if not dfs:
        print("  No k-sweep data, skipping.")
        return
    all_mps = set()
    for df in dfs.values():
        v = df.dropna(subset=["cka_kmv"])
        all_mps.update(v["metapath"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # CKA vs k
    ax = axes[0]
    for mp in sorted(all_mps):
        ks, ckas = [], []
        for k, df in sorted(dfs.items()):
            row = df[df["metapath"] == mp]
            if len(row) > 0 and pd.notna(row.iloc[0]["cka_kmv"]):
                ks.append(k); ckas.append(row.iloc[0]["cka_kmv"])
        if ks:
            ax.plot(ks, ckas, "o-", label=shorten_mp(mp), markersize=6)
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("k (sketch size)"); ax.set_ylabel("CKA")
    ax.set_title("CKA vs k (ACM, 100%)")
    ax.set_xscale("log", base=2); ax.set_xticks(K_VALS); ax.set_xticklabels(K_VALS)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # F1 vs k
    ax = axes[1]
    for mp in sorted(all_mps):
        ks, f1s = [], []
        for k, df in sorted(dfs.items()):
            row = df[df["metapath"] == mp]
            if len(row) > 0 and pd.notna(row.iloc[0]["f1_kmv"]):
                ks.append(k); f1s.append(row.iloc[0]["f1_kmv"])
        if ks:
            ax.plot(ks, f1s, "o-", label=shorten_mp(mp), markersize=6)
    ax.set_xlabel("k (sketch size)"); ax.set_ylabel("F1 (KMV)")
    ax.set_title("F1 vs k (ACM, 100%)")
    ax.set_xscale("log", base=2); ax.set_xticks(K_VALS); ax.set_xticklabels(K_VALS)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("k-Sensitivity Sweep (HGB_ACM at 100%)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "k_sensitivity.png"), bbox_inches="tight")
    print("  Saved k_sensitivity.png")


def plot_table4_speedup():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ds in zip(axes, DATASETS):
        df = load_table4(ds)
        if df is None:
            ax.set_title(f"{ds}\n(no data)"); continue
        mps = df["metapath"].unique()
        labels, speedups_d, speedups_h = [], [], []
        for mp in mps:
            sub = df[df["metapath"] == mp]
            wcd = sub[sub["method"] == "WcD"]["avg_time_s"].values
            wch = sub[sub["method"] == "WcH"]["avg_time_s"].values
            glod = sub[sub["method"] == "GloD"]["avg_time_s"].values
            gloh = sub[sub["method"] == "GloH"]["avg_time_s"].values
            speedups_d.append(wcd[0]/glod[0] if len(wcd)>0 and len(glod)>0 and glod[0]>0 else 0)
            speedups_h.append(wch[0]/gloh[0] if len(wch)>0 and len(gloh)>0 and gloh[0]>0 else 0)
            labels.append(shorten_mp(mp))
        x = range(len(labels)); width = 0.35
        ax.bar([i-width/2 for i in x], speedups_d, width, label="GloD/WcD", alpha=0.8)
        ax.bar([i+width/2 for i in x], speedups_h, width, label="GloH/WcH", alpha=0.8)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(ds.replace("HGB_", "")); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Speedup")
    fig.suptitle("Hub Query Speedup: GloD/WcD and GloH/WcH", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "table4_speedup.png"), bbox_inches="tight")
    print("  Saved table4_speedup.png")


def plot_table4_f1():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    methods = ["GloD", "PerD", "PerD+", "GloH", "PerH", "PerH+"]
    for ax, ds in zip(axes, DATASETS):
        df = load_table4(ds)
        if df is None:
            ax.set_title(f"{ds}\n(no data)"); continue
        avg = df[df["method"].isin(methods)].groupby("method")["f1_or_acc"].mean().reindex(methods)
        colors = ["#1f77b4"]*3 + ["#ff7f0e"]*3
        avg.plot(kind="bar", ax=ax, color=colors, alpha=0.8)
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.5)
        ax.set_title(ds.replace("HGB_", "")); ax.set_ylim(0, 1.1)
        ax.set_xticklabels(methods, rotation=30, ha="right"); ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Avg F1 / Accuracy")
    fig.suptitle("Hub Query Effectiveness", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "table4_f1.png"), bbox_inches="tight")
    print("  Saved table4_f1.png")


def plot_summary():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Top-left: CKA all datasets
    ax = axes[0, 0]
    for ds in DATASETS:
        df = load_extension(ds)
        if df is None: continue
        valid = df.dropna(subset=["cka_kmv"])
        for mp in valid["metapath"].unique():
            sub = valid[valid["metapath"] == mp].sort_values("fraction")
            ax.plot(sub["fraction"], sub["cka_kmv"], "o-", markersize=3,
                    label=f"{ds.replace('HGB_','')}: {shorten_mp(mp)}", alpha=0.7)
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Graph Fraction"); ax.set_ylabel("CKA")
    ax.set_title("CKA vs Graph Fraction"); ax.set_ylim(0.7, 1.02)
    ax.legend(fontsize=5, ncol=2, loc="lower left"); ax.grid(True, alpha=0.3)

    # Top-right: k-sensitivity
    ax = axes[0, 1]
    dfs_k = {}
    for k in K_VALS:
        name = "extension" if k == 32 else f"extension_k{k}"
        d = load_extension("HGB_ACM", name)
        if d is not None: dfs_k[k] = d[d["fraction"] == 1.0]
    all_mps = set()
    for d in dfs_k.values():
        v = d.dropna(subset=["cka_kmv"]); all_mps.update(v["metapath"].unique())
    for mp in sorted(all_mps):
        ks, ckas = [], []
        for k, d in sorted(dfs_k.items()):
            row = d[d["metapath"] == mp]
            if len(row)>0 and pd.notna(row.iloc[0]["cka_kmv"]):
                ks.append(k); ckas.append(row.iloc[0]["cka_kmv"])
        if ks: ax.plot(ks, ckas, "o-", label=shorten_mp(mp), markersize=5)
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("k"); ax.set_ylabel("CKA"); ax.set_title("CKA vs k (ACM)")
    ax.set_xscale("log", base=2); ax.set_xticks(K_VALS); ax.set_xticklabels(K_VALS)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bottom-left: Memory at 100%
    ax = axes[1, 0]
    mem_data = []
    for ds in DATASETS:
        df = load_extension(ds)
        if df is None: continue
        full = df[df["fraction"] == 1.0].dropna(subset=["adj_mb_exact", "adj_mb_kmv"])
        for _, row in full.iterrows():
            mem_data.append({"label": f"{ds.replace('HGB_','')}\n{shorten_mp(row['metapath'])}",
                             "exact": row["adj_mb_exact"], "kmv": row["adj_mb_kmv"]})
    if mem_data:
        mdf = pd.DataFrame(mem_data); x = range(len(mdf)); width = 0.35
        ax.bar([i-width/2 for i in x], mdf["exact"], width, label="Exact", alpha=0.8)
        ax.bar([i+width/2 for i in x], mdf["kmv"], width, label="KMV", alpha=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels(mdf["label"], fontsize=6, rotation=30, ha="right")
        ax.set_yscale("log"); ax.legend()
    ax.set_ylabel("Adj Size (MB, log)"); ax.set_title("Memory at 100%"); ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Speedup
    ax = axes[1, 1]
    for ds in DATASETS:
        df = load_extension(ds)
        if df is None: continue
        valid = df.dropna(subset=["speedup_kmv"])
        for mp in valid["metapath"].unique():
            sub = valid[valid["metapath"] == mp].sort_values("fraction")
            ax.plot(sub["fraction"], sub["speedup_kmv"], "o-", markersize=3,
                    label=f"{ds.replace('HGB_','')}: {shorten_mp(mp)}", alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Graph Fraction"); ax.set_ylabel("Speedup")
    ax.set_title("Materialization Speedup")
    ax.legend(fontsize=5, ncol=2, loc="best"); ax.grid(True, alpha=0.3)

    fig.suptitle("KMV Extension Experiments - Summary", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "summary.png"), bbox_inches="tight")
    print("  Saved summary.png")


if __name__ == "__main__":
    print("Generating plots...\n")
    plot_cka_vs_fraction()
    plot_f1_comparison()
    plot_pred_agreement()
    plot_dirichlet()
    plot_depthwise_cka()
    plot_memory()
    plot_speedup()
    plot_k_sensitivity()
    plot_table4_speedup()
    plot_table4_f1()
    plot_summary()
    print(f"\nAll plots saved to {PLOT_DIR}/")
