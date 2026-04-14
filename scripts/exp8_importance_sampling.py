"""exp8_importance_sampling.py — Direct empirical test: MPRW = importance sampling,
KMV = uniform sampling.

For each method, measure the distribution of endpoint degrees (in the Exact graph)
over sampled edges. Predictions:

  - KMV samples each of v's neighbors with probability ~ min(1, k/d(v)).
    → distribution of sampled-edge endpoint degrees should match Exact's distribution
    → no degree bias.

  - MPRW samples edge (u,v) with probability ~ walk-visits(u,v) ~ stationary(u)·transition
    → over-weights hub-to-hub edges
    → distribution of endpoint degrees shifts right (hub-biased).

Also computes:
  - hub-hub edge fraction (both endpoints in top-10% degree): MPRW > KMV expected.
  - homophily of sampled edges (among labeled-labeled node pairs): MPRW should
    sample MORE homophilous edges because hubs of the same "type" cluster
    (importance sampling concentrates on the stationary).

Interpretation for presentation:
  - "MPRW is importance sampling weighted by degree."
  - "On heterophilous meta-paths (where hubs have different labels than their
    neighbors) this bias is destructive — the sampler ignores the edges carrying
    class-discriminative signal."
  - "KMV's uniform sampling is dataset-agnostic: neither amplifies nor attenuates
    any particular edge class."

Outputs:
  - results/HNE_PubMed/exp8_importance_sampling.csv  (summary stats per method)
  - figures/exp8/importance_sampling.pdf              (4-panel: degree CDF, hub-hub
                                                       fraction, homophily bar, mean deg)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch

TARGET_OFFSET = 26522
N_TARGET      = 20163


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def load_edges(path: Path, n: int = N_TARGET, offset: int = TARGET_OFFSET
               ) -> tuple[np.ndarray, np.ndarray]:
    """Read adjacency list file → (src, dst) arrays, de-duplicated & undirected."""
    rows, cols = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            src = int(parts[0]) - offset
            if src < 0 or src >= n:
                continue
            for tok in parts[1:]:
                dst = int(tok) - offset
                if dst < 0 or dst >= n or dst == src:
                    continue
                # keep as undirected edge: only one direction
                u, v = (src, dst) if src < dst else (dst, src)
                rows.append(u)
                cols.append(v)
    arr = np.unique(np.stack([rows, cols], axis=1), axis=0)
    return arr[:, 0], arr[:, 1]


def load_labels(labels_pt: Path) -> np.ndarray:
    """Load per-node label tensor → numpy array with -100 for unlabeled."""
    y = torch.load(labels_pt, map_location='cpu', weights_only=False)
    if hasattr(y, 'numpy'):
        y = y.numpy()
    return np.asarray(y)


# -----------------------------------------------------------------------------
# Summary stats per method
# -----------------------------------------------------------------------------
def summarize(label: str, src: np.ndarray, dst: np.ndarray,
              d_exact: np.ndarray, y: np.ndarray | None,
              hub_thresh: float) -> dict:
    """Compute endpoint-degree stats + hub-hub fraction + homophily."""
    du, dv = d_exact[src], d_exact[dst]
    d_sum  = du + dv
    d_prod = du.astype(np.float64) * dv.astype(np.float64)
    hub_u  = d_exact[src] >= hub_thresh
    hub_v  = d_exact[dst] >= hub_thresh
    hub_hub = (hub_u & hub_v).mean()

    homophily = np.nan
    hom_n    = 0
    if y is not None:
        labeled = (y[src] >= 0) & (y[dst] >= 0)
        if labeled.any():
            homophily = float((y[src][labeled] == y[dst][labeled]).mean())
            hom_n    = int(labeled.sum())

    return {
        'label': label,
        'n_edges': len(src),
        'mean_d_sum':  float(d_sum.mean()),
        'median_d_sum': float(np.median(d_sum)),
        'p90_d_sum':   float(np.percentile(d_sum, 90)),
        'mean_log_d_prod': float(np.log1p(d_prod).mean()),
        'hub_hub_frac':    float(hub_hub),
        'homophily':       homophily,
        'n_homophily_edges': hom_n,
        '_d_sum': d_sum,  # for plotting only
        '_d_prod': d_prod,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exact', default='HNE_PubMed/mat_exact.adj')
    ap.add_argument('--kmv',   default='HNE_PubMed/mat_sketch_0')
    ap.add_argument('--kmv-k', type=int, default=128)
    ap.add_argument('--mprw-dir', default='HNE_PubMed/mprw_work')
    ap.add_argument('--mprw-w', nargs='+', type=int, default=[128, 256])
    ap.add_argument('--labels-pt',
        default='results/HNE_PubMed/inf_scratch/disease_to_chemical_chemical_to_disease/labels.pt')
    ap.add_argument('--hub-percentile', type=float, default=90.0,
                    help='percentile degree threshold for "hub"')
    ap.add_argument('--out-csv', default='results/HNE_PubMed/exp8_importance_sampling.csv')
    ap.add_argument('--out-pdf', default='figures/exp8/importance_sampling.pdf')
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Load exact and derive d_exact, hub threshold
    # -------------------------------------------------------------------------
    print('[load] Exact')
    src_e, dst_e = load_edges(Path(args.exact))
    print(f'       {len(src_e):,} undirected edges')
    # Build degree from symmetrized edges
    deg = np.zeros(N_TARGET, dtype=np.int64)
    np.add.at(deg, src_e, 1)
    np.add.at(deg, dst_e, 1)
    # Only non-isolated nodes count toward hub percentile
    hub_thresh = float(np.percentile(deg[deg > 0], args.hub_percentile))
    print(f'       hub threshold (p{int(args.hub_percentile)}): deg >= {hub_thresh:.0f}')

    # -------------------------------------------------------------------------
    # Labels (optional — used for homophily)
    # -------------------------------------------------------------------------
    y = None
    labels_pt = Path(args.labels_pt)
    if labels_pt.exists():
        print(f'[load] labels from {labels_pt}')
        y = load_labels(labels_pt)
        # Label tensor may be target-local indexing already — assume so (shape [N_TARGET])
        if y.shape[0] != N_TARGET:
            print(f'[warn] label shape {y.shape} != N_TARGET={N_TARGET}; skipping homophily')
            y = None
        else:
            print(f'       {int((y >= 0).sum())} labeled / {len(y)}')
    else:
        print(f'[warn] labels not found at {labels_pt}; skipping homophily')

    # -------------------------------------------------------------------------
    # Summarize each method
    # -------------------------------------------------------------------------
    stats = [summarize('Exact', src_e, dst_e, deg, y, hub_thresh)]

    print('[load] KMV k=128')
    s, d = load_edges(Path(args.kmv))
    stats.append(summarize(f'KMV k={args.kmv_k}', s, d, deg, y, hub_thresh))

    for w in args.mprw_w:
        p = Path(f'{args.mprw_dir}/mat_mprw_{w}.adj')
        if not p.exists():
            print(f'[skip] MPRW w={w}: missing {p}')
            continue
        print(f'[load] MPRW w={w}')
        s, d = load_edges(p)
        stats.append(summarize(f'MPRW w={w}', s, d, deg, y, hub_thresh))

    # -------------------------------------------------------------------------
    # Print + CSV
    # -------------------------------------------------------------------------
    print('\n' + '-' * 90)
    print(f"{'method':<16}{'n_edges':>12}{'mean(du+dv)':>15}"
          f"{'p90(du+dv)':>12}{'hub-hub%':>12}{'homophily':>12}")
    print('-' * 90)
    for s in stats:
        hom_str = f"{s['homophily']:.4f}" if not np.isnan(s['homophily']) else '    --'
        print(f"{s['label']:<16}{s['n_edges']:>12,}{s['mean_d_sum']:>15.1f}"
              f"{s['p90_d_sum']:>12.0f}{s['hub_hub_frac']*100:>11.2f}%{hom_str:>12}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w_csv = csv.writer(f)
        w_csv.writerow(['method', 'n_edges', 'mean_d_sum', 'median_d_sum',
                        'p90_d_sum', 'mean_log_d_prod', 'hub_hub_frac',
                        'homophily', 'n_homophily_edges'])
        for s in stats:
            w_csv.writerow([s['label'], s['n_edges'], s['mean_d_sum'],
                            s['median_d_sum'], s['p90_d_sum'],
                            s['mean_log_d_prod'], s['hub_hub_frac'],
                            s['homophily'], s['n_homophily_edges']])
    print(f'\n[csv] wrote {out_csv}')

    # -------------------------------------------------------------------------
    # Plots — 4 panels
    # -------------------------------------------------------------------------
    colors = {'Exact': '#1f77b4'}
    for s in stats:
        if s['label'].startswith('KMV'):
            colors[s['label']] = '#2ca02c'
        elif s['label'].startswith('MPRW'):
            # shade MPRW by w value
            colors[s['label']] = '#d62728'

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # Panel A — CDF of log(d(u)+d(v)) per method
    ax = axes[0]
    for s in stats:
        x = np.sort(np.log10(s['_d_sum']))
        y_cdf = np.arange(1, len(x) + 1) / len(x)
        ls = '-' if s['label'] == 'Exact' else ('--' if s['label'].startswith('KMV') else ':')
        lw = 2.4 if s['label'] == 'Exact' else 1.8
        ax.plot(x, y_cdf, label=s['label'], color=colors[s['label']],
                linestyle=ls, linewidth=lw)
    ax.set_xlabel(r'$\log_{10}(d(u) + d(v))$  (endpoints in Exact graph)')
    ax.set_ylabel('CDF over sampled edges')
    ax.set_title('(a) Endpoint-degree distribution\nRight-shift = importance sampling')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    # Panel B — hub-hub edge fraction
    ax = axes[1]
    labels_bar = [s['label'] for s in stats]
    vals = [s['hub_hub_frac'] * 100 for s in stats]
    bar_colors = [colors[s['label']] for s in stats]
    bars = ax.bar(range(len(stats)), vals, color=bar_colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(labels_bar, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(f'% edges with both endpoints in top-{100-int(args.hub_percentile)}% degree')
    ax.set_title('(b) Hub–hub edge fraction\nMPRW > KMV = IS signature')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Panel C — homophily of sampled edges (only if labels available)
    ax = axes[2]
    has_hom = not all(np.isnan(s['homophily']) for s in stats)
    if has_hom:
        hom_vals = [s['homophily'] if not np.isnan(s['homophily']) else 0 for s in stats]
        bars = ax.bar(range(len(stats)), hom_vals, color=bar_colors,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(labels_bar, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('P(y_u == y_v | edge kept, both labeled)')
        ax.set_title('(c) Homophily of sampled edges\nHigher = sampler concentrates on same-label pairs')
        for bar, v, s in zip(bars, hom_vals, stats):
            if not np.isnan(s['homophily']):
                ax.text(bar.get_x() + bar.get_width()/2, v,
                        f'{v:.3f}\n(n={s["n_homophily_edges"]})',
                        ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, max(hom_vals) * 1.2 if max(hom_vals) > 0 else 1)
        ax.grid(alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No labels available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('(c) Homophily of sampled edges')

    # Panel D — mean sampled-edge endpoint degree, annotated
    ax = axes[3]
    mean_vals = [s['mean_d_sum'] for s in stats]
    bars = ax.bar(range(len(stats)), mean_vals, color=bar_colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(labels_bar, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(r'Mean of $d(u) + d(v)$ over sampled edges')
    ax.set_title('(d) Average endpoint degree\nIS bias: MPRW > KMV > Exact?')
    for bar, v in zip(bars, mean_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.0f}',
                ha='center', va='bottom', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f'[pdf] wrote {out_pdf}')


if __name__ == '__main__':
    main()
