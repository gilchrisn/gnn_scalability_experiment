"""exp9_plot_dblp_aptpa.py — Visualize KMV vs MPRW on DBLP APTPA (L=4 meta-path,
starved budgets) from master_results.csv.

This is the experiment that reveals the IS-vs-US distinction: longer meta-path
+ small budget means walks have room to converge toward the stationary (hub)
distribution, so MPRW's sampling bias actually bites, while KMV's uniform
per-vertex sampling holds ground.

Outputs a 4-panel figure:
  (a) Macro-F1 vs edge density  (KMV vs MPRW, Exact as horizontal reference)
  (b) Prediction Agreement vs edge density
  (c) Dirichlet Energy vs edge density  (over-smoothing proxy)
  (d) CKA vs edge density  (shows the paradox: MPRW higher CKA but lower F1)

Usage:
  python scripts/exp9_plot_dblp_aptpa.py \
      --csv results/HGB_DBLP/master_results.csv \
      --out figures/exp9/dblp_aptpa_starved.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def aggregate(df: pd.DataFrame, keys: list[str],
              metrics: list[str]) -> pd.DataFrame:
    """mean + std across seeds for each (k_value | w_value) combination."""
    g = df.groupby(keys, dropna=False)
    agg_dict = {'edges': ('Edge_Count', 'mean')}
    for m in metrics:
        agg_dict[f'{m}_mean'] = (m, 'mean')
        agg_dict[f'{m}_std']  = (m, 'std')
    return g.agg(**agg_dict).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/HGB_DBLP/master_results.csv')
    ap.add_argument('--L',   type=int, default=2, help='SAGE depth to plot')
    ap.add_argument('--out', default='figures/exp9/dblp_aptpa_starved.pdf')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df['L'] == args.L].copy()
    # Exact is a single row (Seed=0, deterministic)
    exact_row = df[df['Method'] == 'Exact']
    kmv = df[df['Method'] == 'KMV']
    mprw = df[df['Method'] == 'MPRW']

    metrics_list = ['Macro_F1', 'Pred_Similarity', 'Dirichlet_Energy', 'CKA_L2']
    kmv_agg  = aggregate(kmv,  ['k_value'], metrics_list).sort_values('k_value').reset_index(drop=True)
    mprw_agg = aggregate(mprw, ['w_value'], metrics_list).sort_values('w_value').reset_index(drop=True)

    print('KMV:')
    print(kmv_agg[['k_value', 'edges', 'Macro_F1_mean', 'Macro_F1_std',
                   'Pred_Similarity_mean', 'Dirichlet_Energy_mean',
                   'CKA_L2_mean']].to_string(index=False))
    print('\nMPRW:')
    print(mprw_agg[['w_value', 'edges', 'Macro_F1_mean', 'Macro_F1_std',
                    'Pred_Similarity_mean', 'Dirichlet_Energy_mean',
                    'CKA_L2_mean']].to_string(index=False))

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    COL_EXACT = '#1f77b4'
    COL_KMV   = '#2ca02c'
    COL_MPRW  = '#d62728'

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    metrics = [
        ('Macro_F1',         'Macro-F1',              'higher better'),
        ('Pred_Similarity',  'Prediction Agreement',  'higher better'),
        ('Dirichlet_Energy', 'Dirichlet Energy',      'over-smoothing proxy'),
        ('CKA_L2',           'CKA vs Exact (output)', 'representation similarity'),
    ]

    exact_means = {m[0]: exact_row[m[0]].iloc[0] if not exact_row.empty
                       and m[0] in exact_row.columns
                       and not exact_row[m[0]].isna().all()
                       else None for m in metrics}

    for ax, (metric, title, sub) in zip(axes, metrics):
        # KMV
        x = kmv_agg['edges'].values
        y = kmv_agg[f'{metric}_mean'].values
        yerr = kmv_agg[f'{metric}_std'].fillna(0).values
        ax.errorbar(x, y, yerr=yerr, marker='o', color=COL_KMV,
                    label='KMV (uniform)', capsize=3, linewidth=1.8, markersize=8)
        for xi, yi, kv in zip(x, y, kmv_agg['k_value'].values):
            ax.annotate(f'k={int(kv)}', (xi, yi), fontsize=7,
                        xytext=(4, 4), textcoords='offset points')
        # MPRW
        x = mprw_agg['edges'].values
        y = mprw_agg[f'{metric}_mean'].values
        yerr = mprw_agg[f'{metric}_std'].fillna(0).values
        ax.errorbar(x, y, yerr=yerr, marker='^', color=COL_MPRW,
                    label='MPRW (importance)', capsize=3, linewidth=1.8, markersize=8)
        for xi, yi, wv in zip(x, y, mprw_agg['w_value'].values):
            ax.annotate(f'w={int(wv)}', (xi, yi), fontsize=7,
                        xytext=(4, -12), textcoords='offset points')

        # Exact reference
        if exact_means[metric] is not None and not np.isnan(exact_means[metric]):
            ax.axhline(exact_means[metric], linestyle='--', color=COL_EXACT,
                       linewidth=1.5, alpha=0.8,
                       label=f'Exact = {exact_means[metric]:.3f}')

        ax.set_xscale('log')
        ax.set_xlabel('Edge count (density)')
        ax.set_ylabel(title)
        ax.set_title(f'{title}\n({sub})', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3, which='both')

    fig.suptitle('HGB_DBLP — APTPA (L=4 meta-path), starved budgets\n'
                 'KMV vs MPRW: the IS-vs-US distinction emerges on long meta-paths',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches='tight')
    print(f'\n[pdf] wrote {out}')


if __name__ == '__main__':
    main()
