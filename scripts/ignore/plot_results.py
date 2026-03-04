"""
Plot E2E Sweep Results
======================
Generates 4 publication-ready figures from the exhaustive sweep CSV.

Figures produced:
  1. accuracy_vs_k.png       — KMV accuracy vs K per dataset (main result)
  2. mattime_vs_k.png        — Materialisation time vs K per dataset
  3. pareto_frontier.png     — Accuracy vs mat_time trade-off (the paper figure)
  4. edge_retention_vs_k.png — What fraction of exact edges does each K retain?

Usage:
    python scripts/plot_results.py --input output/results/exhaustive_sweep.csv
    python scripts/plot_results.py --input output/results/exhaustive_sweep.csv --output output/plots
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

STYLE = {
    'figure.facecolor':    '#0f1117',
    'axes.facecolor':      '#1a1d27',
    'axes.edgecolor':      '#3a3d4d',
    'axes.labelcolor':     '#e0e0e0',
    'axes.titlecolor':     '#ffffff',
    'axes.grid':           True,
    'grid.color':          '#2a2d3d',
    'grid.linewidth':      0.6,
    'xtick.color':         '#a0a0b0',
    'ytick.color':         '#a0a0b0',
    'text.color':          '#e0e0e0',
    'legend.facecolor':    '#1a1d27',
    'legend.edgecolor':    '#3a3d4d',
    'legend.labelcolor':   '#e0e0e0',
    'font.family':         'monospace',
    'font.size':           10,
    'axes.titlesize':      12,
    'axes.labelsize':      10,
}

# One colour per dataset — colourblind-safe palette
DATASET_COLORS = {
    'HGB_ACM':      '#4fc3f7',   # sky blue
    'HGB_DBLP':     '#81c784',   # green
    'HGB_IMDB':     '#ffb74d',   # amber
    'HGB_Freebase': '#f06292',   # pink
}

# One marker per metapath within a dataset
METAPATH_MARKERS = ['o', 's', '^', 'D', 'v', 'P']

DPI    = 180
LWIDTH = 1.8


# ---------------------------------------------------------------------------
# Data loading + preprocessing
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Coerce numerics (some cells may be 'N/A' strings)
    for col in ['k', 'mat_time', 'train_time', 'total_time',
                'test_acc', 'num_edges', 'num_nodes',
                'graph_density', 'isolated_nodes',
                'final_val_loss', 'final_val_acc', 'epochs_run']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Short metapath label for legends
    df['mp_short'] = df['metapath'].apply(_shorten_metapath)

    # Unique label per (dataset, metapath) combination
    df['series'] = df['dataset'] + '/' + df['mp_short']

    # Attach exact-pipeline num_edges to every KMV row for retention calculation
    exact_edges = (
        df[df['method'] == 'Exact']
        .groupby('series')['num_edges']
        .first()
        .rename('exact_edges')
    )
    df = df.merge(exact_edges, on='series', how='left')
    df['edge_retention'] = df['num_edges'] / df['exact_edges'].replace(0, np.nan)

    # Exact accuracy for gap computation
    exact_acc = (
        df[df['method'] == 'Exact']
        .groupby('series')['test_acc']
        .first()
        .rename('exact_acc')
    )
    df = df.merge(exact_acc, on='series', how='left')
    df['acc_gap'] = df['exact_acc'] - df['test_acc']   # positive = exact is better

    return df


def _shorten_metapath(mp: str) -> str:
    """'paper_to_author,author_to_paper' → 'PAP'"""
    hops    = [h.strip() for h in mp.split(',')]
    letters = []
    for h in hops:
        parts = h.split('_to_')
        if parts:
            letters.append(parts[0][0].upper())
    last = hops[-1].split('_to_')
    if len(last) >= 2:
        letters.append(last[-1][0].upper())
    return ''.join(letters)


def _series_style(series_name: str, all_series: list):
    """Returns (color, marker) for a (dataset/metapath) series."""
    dataset = series_name.split('/')[0]
    color   = DATASET_COLORS.get(dataset, '#aaaaaa')
    idx     = all_series.index(series_name)
    marker  = METAPATH_MARKERS[idx % len(METAPATH_MARKERS)]
    return color, marker


# ---------------------------------------------------------------------------
# Figure 1 — Accuracy vs K
# ---------------------------------------------------------------------------

def plot_accuracy_vs_k(df: pd.DataFrame, out_dir: str) -> None:
    """
    One subplot per dataset. KMV accuracy curves + exact baseline dashed line.
    This is the main result figure.
    """
    kmv    = df[df['method'] == 'KMV'].copy().sort_values('k')
    exact  = df[df['method'] == 'Exact'].copy()
    datasets = sorted(df['dataset'].unique())
    n      = len(datasets)
    cols   = min(n, 2)
    rows   = (n + cols - 1) // cols

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(7 * cols, 4.5 * rows),
                                 squeeze=False)
        fig.suptitle('KMV Accuracy vs Sketch Size K\n(dashed = Exact baseline)',
                     fontsize=14, fontweight='bold', y=1.01)

        for i, ds in enumerate(datasets):
            ax   = axes[i // cols][i % cols]
            ds_kmv   = kmv[kmv['dataset'] == ds]
            ds_exact = exact[exact['dataset'] == ds]
            all_series = sorted(ds_kmv['series'].unique().tolist())

            for j, series in enumerate(all_series):
                color, marker = _series_style(series, all_series)
                mp_short = series.split('/')[1]
                sdata    = ds_kmv[ds_kmv['series'] == series].dropna(subset=['k', 'test_acc'])

                ax.plot(sdata['k'], sdata['test_acc'],
                        color=color, marker=marker, linewidth=LWIDTH,
                        markersize=6, label=f'KMV – {mp_short}', zorder=4)

                # Exact baseline for this series
                ex = ds_exact[ds_exact['series'] == series]
                if not ex.empty:
                    ax.axhline(float(ex['test_acc'].iloc[0]),
                               color=color, linestyle='--',
                               linewidth=1.2, alpha=0.7,
                               label=f'Exact – {mp_short}')

            ax.set_xscale('log', base=2)
            ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_xlabel('Sketch size K')
            ax.set_ylabel('Test accuracy')
            ax.set_title(ds, fontweight='bold')
            ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.02))
            ax.legend(fontsize=8, framealpha=0.8, loc='lower right')

        # Hide unused subplots
        for i in range(n, rows * cols):
            axes[i // cols][i % cols].set_visible(False)

        plt.tight_layout()
        path = os.path.join(out_dir, 'accuracy_vs_k.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
    print(f"[Fig 1] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Materialisation time vs K
# ---------------------------------------------------------------------------

def plot_mattime_vs_k(df: pd.DataFrame, out_dir: str) -> None:
    """
    Materialisation time per method. Reveals where KMV stops being cheaper.
    """
    kmv    = df[df['method'] == 'KMV'].copy().sort_values('k')
    exact  = df[df['method'] == 'Exact'].copy()
    datasets = sorted(df['dataset'].unique())
    n    = len(datasets)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(7 * cols, 4.5 * rows),
                                 squeeze=False)
        fig.suptitle('Materialisation Time vs K\n(dashed = Exact baseline)',
                     fontsize=14, fontweight='bold', y=1.01)

        for i, ds in enumerate(datasets):
            ax       = axes[i // cols][i % cols]
            ds_kmv   = kmv[kmv['dataset'] == ds]
            ds_exact = exact[exact['dataset'] == ds]
            all_series = sorted(ds_kmv['series'].unique().tolist())

            for series in all_series:
                color, marker = _series_style(series, all_series)
                mp_short = series.split('/')[1]
                sdata    = ds_kmv[ds_kmv['series'] == series].dropna(
                    subset=['k', 'mat_time'])

                ax.plot(sdata['k'], sdata['mat_time'],
                        color=color, marker=marker, linewidth=LWIDTH,
                        markersize=6, label=f'KMV – {mp_short}', zorder=4)

                ex = ds_exact[ds_exact['series'] == series]
                if not ex.empty:
                    ax.axhline(float(ex['mat_time'].iloc[0]),
                               color=color, linestyle='--',
                               linewidth=1.2, alpha=0.7,
                               label=f'Exact – {mp_short}')

            ax.set_xscale('log', base=2)
            ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_xlabel('Sketch size K')
            ax.set_ylabel('Materialisation time (s)')
            ax.set_title(ds, fontweight='bold')
            ax.legend(fontsize=8, framealpha=0.8)

        for i in range(n, rows * cols):
            axes[i // cols][i % cols].set_visible(False)

        plt.tight_layout()
        path = os.path.join(out_dir, 'mattime_vs_k.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
    print(f"[Fig 2] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Pareto Frontier (THE paper figure)
# ---------------------------------------------------------------------------

def plot_pareto_frontier(df: pd.DataFrame, out_dir: str) -> None:
    """
    Accuracy vs materialisation time. Every point is one (dataset, metapath, K).
    Oracle = star marker. KMV points annotated with K value.

    The ideal result: KMV points cluster near the Oracle accuracy at lower cost
    (left of the Oracle stars on the x-axis).
    """
    kmv   = df[df['method'] == 'KMV'].copy()
    exact = df[df['method'] == 'Exact'].copy()
    all_series = sorted(df['series'].unique().tolist())

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Accuracy vs Materialisation Time — Pareto Frontier\n'
                     '(★ = Exact oracle   •  = KMV sketch)',
                     fontsize=13, fontweight='bold')

        for series in all_series:
            color, marker = _series_style(series, all_series)
            ds       = series.split('/')[0]
            mp_short = series.split('/')[1]

            # KMV points
            sk = kmv[kmv['series'] == series].dropna(
                subset=['mat_time', 'test_acc', 'k']).sort_values('k')
            if not sk.empty:
                sc = ax.scatter(sk['mat_time'], sk['test_acc'],
                                c=color, marker=marker, s=55,
                                label=f'{ds} / {mp_short}',
                                zorder=4, alpha=0.9)
                # Annotate K values on the curve
                for _, row in sk.iterrows():
                    ax.annotate(f"K={int(row['k'])}",
                                xy=(row['mat_time'], row['test_acc']),
                                xytext=(4, 3), textcoords='offset points',
                                fontsize=6.5, color=color, alpha=0.85)

            # Oracle point — larger star
            ex = exact[exact['series'] == series].dropna(
                subset=['mat_time', 'test_acc'])
            if not ex.empty:
                ax.scatter(float(ex['mat_time'].iloc[0]),
                           float(ex['test_acc'].iloc[0]),
                           c=color, marker='*', s=250,
                           edgecolors='white', linewidths=0.5,
                           zorder=5)

        ax.set_xlabel('Materialisation time (s)', fontsize=11)
        ax.set_ylabel('Test accuracy', fontsize=11)

        # Annotate regions
        ax.text(0.02, 0.97,
                '← Faster materialisation',
                transform=ax.transAxes, fontsize=8,
                color='#888888', va='top', ha='left')
        ax.text(0.98, 0.02,
                'Higher accuracy →',
                transform=ax.transAxes, fontsize=8,
                color='#888888', va='bottom', ha='right', rotation=90)

        # Custom legend: datasets by colour, Oracle by star
        legend_elements = []
        seen_datasets = []
        for series in all_series:
            ds = series.split('/')[0]
            if ds not in seen_datasets:
                color, _ = _series_style(series, all_series)
                legend_elements.append(
                    Line2D([0], [0], color=color, linewidth=0,
                           marker='o', markersize=7, label=ds))
                seen_datasets.append(ds)
        legend_elements.append(
            Line2D([0], [0], color='white', linewidth=0,
                   marker='*', markersize=12, label='Exact oracle'))

        ax.legend(handles=legend_elements, fontsize=9,
                  framealpha=0.85, loc='lower right')

        plt.tight_layout()
        path = os.path.join(out_dir, 'pareto_frontier.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
    print(f"[Fig 3] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Edge retention vs K
# ---------------------------------------------------------------------------

def plot_edge_retention(df: pd.DataFrame, out_dir: str) -> None:
    """
    What fraction of exact edges does each K value retain?
    Explains accuracy drop at low K — the sketch is too sparse.
    """
    kmv      = df[(df['method'] == 'KMV') & df['edge_retention'].notna()].copy()
    datasets = sorted(kmv['dataset'].unique())
    n    = len(datasets)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(7 * cols, 4.5 * rows),
                                 squeeze=False)
        fig.suptitle('Edge Retention vs K\n'
                     '(fraction of Exact edges recovered by KMV sketch)',
                     fontsize=14, fontweight='bold', y=1.01)

        for i, ds in enumerate(datasets):
            ax     = axes[i // cols][i % cols]
            ds_kmv = kmv[kmv['dataset'] == ds]
            all_series = sorted(ds_kmv['series'].unique().tolist())

            for series in all_series:
                color, marker = _series_style(series, all_series)
                mp_short = series.split('/')[1]
                sdata    = ds_kmv[ds_kmv['series'] == series].sort_values('k')

                ax.plot(sdata['k'], sdata['edge_retention'] * 100,
                        color=color, marker=marker, linewidth=LWIDTH,
                        markersize=6, label=mp_short, zorder=4)

            ax.axhline(100, color='#ffffff', linestyle='--',
                       linewidth=1.0, alpha=0.4, label='Exact (100%)')
            ax.set_xscale('log', base=2)
            ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_xlabel('Sketch size K')
            ax.set_ylabel('Edge retention (%)')
            ax.set_title(ds, fontweight='bold')
            ax.set_ylim(0, 110)
            ax.legend(fontsize=9, framealpha=0.8)

        for i in range(n, rows * cols):
            axes[i // cols][i % cols].set_visible(False)

        plt.tight_layout()
        path = os.path.join(out_dir, 'edge_retention_vs_k.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
    print(f"[Fig 4] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 5 — Summary table heatmap (bonus: quick visual of all numbers)
# ---------------------------------------------------------------------------

def plot_summary_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """
    Heatmap: rows = dataset/metapath, cols = K, cells = test accuracy.
    Exact column appended on the right. Good for a one-glance overview.
    """
    kmv   = df[df['method'] == 'KMV'].copy()
    exact = df[df['method'] == 'Exact'].copy()

    k_vals  = sorted(kmv['k'].dropna().unique().astype(int).tolist())
    series  = sorted(df['series'].unique().tolist())

    # Build matrix: rows = series, cols = K values + Exact
    matrix = np.full((len(series), len(k_vals) + 1), np.nan)
    for i, s in enumerate(series):
        for j, k in enumerate(k_vals):
            row = kmv[(kmv['series'] == s) & (kmv['k'] == k)]
            if not row.empty:
                matrix[i, j] = float(row['test_acc'].iloc[0])
        ex = exact[exact['series'] == s]
        if not ex.empty:
            matrix[i, -1] = float(ex['test_acc'].iloc[0])

    col_labels = [str(k) for k in k_vals] + ['Exact']
    row_labels  = [s.replace('HGB_', '') for s in series]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(len(col_labels) * 0.9 + 2,
                                        len(row_labels) * 0.65 + 1.5))
        fig.suptitle('Test Accuracy Heatmap  (rows = dataset/metapath, cols = K)',
                     fontsize=12, fontweight='bold')

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                       vmin=max(0, np.nanmin(matrix) - 0.05),
                       vmax=min(1, np.nanmax(matrix) + 0.01))

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel('K  (Exact = rightmost column)', fontsize=10)

        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = matrix[i, j]
                if not np.isnan(val):
                    txt_color = 'black' if val > 0.65 else 'white'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=7.5, color=txt_color, fontweight='bold')

        # Vertical separator before Exact column
        ax.axvline(len(k_vals) - 0.5, color='#ffffff', linewidth=1.5, alpha=0.6)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Test accuracy', fontsize=9)

        plt.tight_layout()
        path = os.path.join(out_dir, 'accuracy_heatmap.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
    print(f"[Fig 5] Saved → {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*65}")
    print("SWEEP RESULT SUMMARY")
    print(f"{'='*65}")

    exact = df[df['method'] == 'Exact']
    kmv   = df[df['method'] == 'KMV']

    print(f"\nDatasets run  : {sorted(df['dataset'].unique().tolist())}")
    print(f"K values      : {sorted(kmv['k'].dropna().unique().astype(int).tolist())}")
    print(f"Total rows    : {len(df)}  ({len(exact)} Exact, {len(kmv)} KMV)")

    print(f"\n{'─'*65}")
    print(f"{'Series':<28} {'Exact acc':>9} {'Best KMV':>9} {'Best K':>7} {'Gap':>7}")
    print(f"{'─'*65}")

    for series in sorted(df['series'].unique()):
        ex = exact[exact['series'] == series]
        km = kmv[kmv['series'] == series]
        if ex.empty or km.empty:
            continue
        exact_acc = float(ex['test_acc'].iloc[0])
        best_row  = km.loc[km['test_acc'].idxmax()]
        best_acc  = float(best_row['test_acc'])
        best_k    = int(best_row['k'])
        gap       = exact_acc - best_acc
        gap_str   = f"{gap:+.4f}"
        gap_color = '' if gap <= 0.02 else '  ⚠'
        print(f"  {series:<26} {exact_acc:>9.4f} {best_acc:>9.4f} {best_k:>7} "
              f"{gap_str:>7}{gap_color}")

    # Mat time comparison
    print(f"\n{'─'*65}")
    print("Materialisation time — Exact vs KMV at K=32 (if available)")
    print(f"{'─'*65}")
    for series in sorted(df['series'].unique()):
        ex = exact[exact['series'] == series]
        k32 = kmv[(kmv['series'] == series) & (kmv['k'] == 32)]
        if ex.empty or k32.empty:
            continue
        t_ex  = float(ex['mat_time'].iloc[0])
        t_kmv = float(k32['mat_time'].iloc[0])
        speedup = t_ex / t_kmv if t_kmv > 0 else float('inf')
        print(f"  {series:<26}  Exact={t_ex:.2f}s  KMV={t_kmv:.2f}s  "
              f"speedup={speedup:.1f}×")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot E2E sweep results")
    parser.add_argument('--input',  type=str, required=True,
                        help='Path to exhaustive_sweep.csv')
    parser.add_argument('--output', type=str,
                        default='output/plots',
                        help='Output directory for plots')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[Error] CSV not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"[Plot] Loading: {args.input}")
    df = load_and_prepare(args.input)

    if df.empty:
        print("[Plot] CSV is empty — nothing to plot.")
        sys.exit(1)

    print_summary(df)

    print(f"\n[Plot] Generating figures → {args.output}/")
    plot_accuracy_vs_k(df,      args.output)
    plot_mattime_vs_k(df,       args.output)
    plot_pareto_frontier(df,    args.output)
    plot_edge_retention(df,     args.output)
    plot_summary_heatmap(df,    args.output)

    print(f"\n[Done] All figures saved to {args.output}/")
    print("  accuracy_vs_k.png       — main accuracy result")
    print("  mattime_vs_k.png        — cost comparison")
    print("  pareto_frontier.png     — the paper figure")
    print("  edge_retention_vs_k.png — explains low-K accuracy drop")
    print("  accuracy_heatmap.png    — one-glance full summary")


if __name__ == '__main__':
    main()