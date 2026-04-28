"""exp10_is_mechanism_dblp.py — Direct measurement of IS bias on HGB_DBLP APTPA.

On HGB_DBLP, meta-path author→paper→term→paper→author (L=4, APTPA), at starved
budgets, KMV beats MPRW by ~1.5-2pp F1 (see figures/exp9/dblp_aptpa_starved.pdf).
This script shows *why*: endpoint-degree distribution of sampled edges.

If MPRW is importance sampling, then its sampled edges should preferentially
have high-degree endpoints in the exact APTPA graph (hub bias). KMV, sampling
near-uniformly per node, should match Exact's distribution.

Panels:
  (a) CDF of log(d(u)+d(v)) per method. Right-shift = IS signature.
  (b) Hub-hub edge fraction (both endpoints in top-10% degree of exact).
  (c) Mean (d(u)+d(v)) bar chart.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# DBLP author-type constants
N_TARGET = 4057   # author nodes
OFFSET   = 0      # author is first alphabetically


def load_edges(path: Path, n: int = N_TARGET, offset: int = OFFSET
               ) -> tuple[np.ndarray, np.ndarray]:
    """Read adjacency list → (src, dst) arrays, de-duplicated & undirected."""
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
                u, v = (src, dst) if src < dst else (dst, src)
                rows.append(u)
                cols.append(v)
    arr = np.unique(np.stack([rows, cols], axis=1), axis=0)
    return arr[:, 0], arr[:, 1]


def degree_from_edges(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    return deg


def summarize(label: str, src: np.ndarray, dst: np.ndarray,
              d_exact: np.ndarray, hub_thresh: float) -> dict:
    du, dv = d_exact[src], d_exact[dst]
    d_sum = (du + dv).astype(np.float64)
    hub_u = d_exact[src] >= hub_thresh
    hub_v = d_exact[dst] >= hub_thresh
    hub_hub = (hub_u & hub_v).mean()
    return {
        'label': label,
        'n_edges': int(len(src)),
        'mean_d_sum': float(d_sum.mean()),
        'median_d_sum': float(np.median(d_sum)),
        'p90_d_sum': float(np.percentile(d_sum, 90)),
        'hub_hub_frac': float(hub_hub),
        '_d_sum': d_sum,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exact', default='results/HGB_DBLP/exact_author_to_paper_paper_to_term_term_to_paper_paper_to_author.adj')
    ap.add_argument('--kmv', default='HGBn-DBLP/mat_sketch_0',
                    help='KMV .adj (last k from pipeline was k=16)')
    ap.add_argument('--kmv-k', type=int, default=16)
    ap.add_argument('--mprw-dir', default='HGBn-DBLP/mprw_work')
    ap.add_argument('--mprw-w', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument('--hub-percentile', type=float, default=90.0)
    ap.add_argument('--out', default='figures/exp10/is_mechanism_dblp_aptpa.pdf')
    args = ap.parse_args()

    # Exact: compute author-level degree in APTPA graph
    print(f'[load] exact APTPA from {args.exact}')
    src_e, dst_e = load_edges(Path(args.exact))
    print(f'       {len(src_e):,} undirected edges')
    deg = degree_from_edges(src_e, dst_e, N_TARGET)
    nz_mask = deg > 0
    hub_thresh = float(np.percentile(deg[nz_mask], args.hub_percentile))
    print(f'       mean degree (nonzero): {deg[nz_mask].mean():.1f}  '
          f'hub threshold (p{int(args.hub_percentile)}): deg >= {hub_thresh:.0f}')

    stats = [summarize('Exact', src_e, dst_e, deg, hub_thresh)]

    # KMV (single k)
    if Path(args.kmv).exists():
        print(f'[load] KMV k={args.kmv_k} from {args.kmv}')
        s, d = load_edges(Path(args.kmv))
        stats.append(summarize(f'KMV k={args.kmv_k}', s, d, deg, hub_thresh))
    else:
        print(f'[skip] KMV missing: {args.kmv}')

    # MPRW sweep
    for w in args.mprw_w:
        p = Path(f'{args.mprw_dir}/mat_mprw_{w}.adj')
        if not p.exists():
            print(f'[skip] MPRW w={w}: {p} missing')
            continue
        print(f'[load] MPRW w={w} from {p}')
        s, d = load_edges(p)
        stats.append(summarize(f'MPRW w={w}', s, d, deg, hub_thresh))

    # Console summary
    print('\n' + '-' * 82)
    print(f"{'method':<16}{'n_edges':>12}{'mean(du+dv)':>15}"
          f"{'p90(du+dv)':>12}{'hub-hub%':>12}")
    print('-' * 82)
    for s in stats:
        print(f"{s['label']:<16}{s['n_edges']:>12,}{s['mean_d_sum']:>15.1f}"
              f"{s['p90_d_sum']:>12.0f}{s['hub_hub_frac']*100:>11.2f}%")

    # Plot
    colors = {'Exact': '#1f77b4'}
    for s in stats:
        if s['label'].startswith('KMV'):
            colors[s['label']] = '#2ca02c'
        elif s['label'].startswith('MPRW'):
            colors[s['label']] = '#d62728'

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel A — degree distribution of Exact (context)
    ax = axes[0]
    deg_nz = deg[deg > 0]
    ax.hist(deg_nz, bins=60, color='#1f77b4', edgecolor='black', linewidth=0.5,
            alpha=0.8)
    ax.axvline(deg_nz.mean(), color='red', linewidth=1.2, linestyle='--',
               label=f'mean = {deg_nz.mean():.0f}')
    ax.axvline(np.percentile(deg_nz, 10), color='orange', linewidth=1.2,
               linestyle=':', label=f'p10 = {np.percentile(deg_nz, 10):.0f}')
    ax.set_xlabel('author degree in Exact APTPA')
    ax.set_ylabel('count of authors')
    ax.set_title(f'(a) Exact APTPA is near-complete\n'
                 f'N={N_TARGET}, mean deg={deg_nz.mean():.0f}/{N_TARGET-1} '
                 f'= {100*deg_nz.mean()/(N_TARGET-1):.1f}% saturation')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    # Panel B — hub-hub fraction (the visible IS signature in this regime)
    ax = axes[1]
    labels_bar = [s['label'] for s in stats]
    vals = [s['hub_hub_frac'] * 100 for s in stats]
    bar_colors = [colors[s['label']] for s in stats]
    bars = ax.bar(range(len(stats)), vals, color=bar_colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(labels_bar, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(f'% edges with both endpoints in top-{100-int(args.hub_percentile)}% author degree')
    ax.set_title('(b) Hub-hub edge fraction of sampled edges\n'
                 'MPRW > KMV > Exact: IS over-represents the dense core')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Panel C — edge count per method (shows MPRW's coupon-collector waste)
    ax = axes[2]
    e_counts = [s['n_edges'] for s in stats if s['label'] != 'Exact']
    e_labels = [s['label'] for s in stats if s['label'] != 'Exact']
    e_colors = [colors[lab] for lab in e_labels]
    bars = ax.bar(range(len(e_counts)), e_counts, color=e_colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(e_counts)))
    ax.set_xticklabels(e_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('unique sampled edges')
    ax.set_title('(c) Edges sampled per method\n'
                 'KMV k=16 > MPRW w=16 edges at matched k/w = KMV is denser-for-same-budget')
    for bar, v in zip(bars, e_counts):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:,}',
                ha='center', va='bottom', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle('HGB_DBLP — APTPA (L=4) is nearly complete: endpoint-degree IS signature '
                 'is suppressed; bias expresses as hub-hub pair over-sampling\n'
                 f'Exact: {stats[0]["hub_hub_frac"]*100:.0f}% hub-hub → '
                 f'MPRW: {[s["hub_hub_frac"] for s in stats if s["label"].startswith("MPRW")][-1]*100:.0f}% hub-hub',
                 fontsize=11, y=1.04)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches='tight')
    print(f'\n[pdf] wrote {out}')


if __name__ == '__main__':
    main()
