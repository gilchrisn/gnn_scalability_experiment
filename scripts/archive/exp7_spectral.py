"""exp7_spectral.py — Spectral comparison of Exact vs KMV vs MPRW sparsifiers.

Frames the KMV vs MPRW question as a spectral sparsification question.
Since meta-path materialization is matrix multiplication and SAGE iterates
D^{-1/2} A D^{-1/2}, the *spectrum* of the sparsified normalized adjacency
determines GNN behavior.

Predictions (to test empirically):
  - KMV samples edges with keep-prob ~ 1/d(u)+1/d(v) → close to Spielman-
    Srivastava effective-resistance sampling → near-uniform eigenvalue scaling.
  - MPRW samples edges with keep-prob ~ stationary × access ~ d(u)·d(v)
    → over-weights hub-to-hub edges → amplifies top eigenvector, compresses
    spectral gap → faster over-smoothing, higher CKA (top-component alignment)
    but lower F1 (principal-component dominance is class-agnostic).

Outputs:
  - results/HNE_PubMed/exp7_spectrum.csv  (long-format: method, i, lambda_i, n_edges)
  - figures/exp7/spectrum.pdf              (3-panel: raw, ratio vs exact, gap-vs-density)

No new C++ runs — consumes existing .adj files.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Constants for HNE_PubMed + disease→chemical→chemical→disease meta-path
# -----------------------------------------------------------------------------
TARGET_OFFSET = 26522   # disease type offset (per HNE_PubMed/offsets.json)
N_TARGET      = 20163   # gene offset (46685) - disease offset (26522)
TOP_K_EIG     = 100     # top eigenvalues to compute


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def load_adj(path: Path, n: int = N_TARGET, offset: int = TARGET_OFFSET) -> sp.csr_matrix:
    """Load C++ adjacency list file → symmetric sparse CSR.

    Format: each line is "src_id dst1 dst2 ... dstK" with global IDs.
    We shift IDs into [0, n) using offset, build sparse matrix, symmetrize,
    clip to binary. Self-loops are removed.
    """
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
                rows.append(src)
                cols.append(dst)
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrize and binarize (some files already symmetric, some directed).
    A = A + A.T
    A.data = np.ones_like(A.data)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def count_unique_edges(A: sp.csr_matrix) -> int:
    """Upper-triangle nnz = undirected edge count."""
    return int(sp.triu(A, k=1).nnz)


def giant_component_mask(A: sp.csr_matrix) -> np.ndarray:
    """Boolean mask over nodes: True iff node in largest connected component.

    Sparsifiers tend to fragment the graph — each small component contributes
    a trivial eigenvalue 1.0 that swamps the top of the spectrum. We need a
    common node set (the exact graph's GCC) to make spectra comparable.
    """
    n_comp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    gcc_label = int(np.argmax(sizes))
    mask = labels == gcc_label
    return mask


def restrict_and_normalize(A: sp.csr_matrix, mask: np.ndarray) -> sp.csr_matrix:
    """Restrict A to nodes in `mask`, then compute D^{-1/2} A D^{-1/2}.

    If some rows become disconnected (d=0) after restriction, their
    1/sqrt(d) := 0 → that row and column get zeroed out.
    """
    idx = np.flatnonzero(mask)
    A_sub = A[idx][:, idx]
    return normalized_adj(A_sub)


# -----------------------------------------------------------------------------
# Spectrum
# -----------------------------------------------------------------------------
def normalized_adj(A: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric normalized adjacency D^{-1/2} A D^{-1/2}.

    Isolated nodes (d=0) get 1/sqrt(d) := 0 → zero rows/cols (benign).
    """
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(deg, dtype=np.float64)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D = sp.diags(d_inv_sqrt)
    return (D @ A @ D).astype(np.float64)


def top_eigenvalues(A_sym: sp.csr_matrix, k: int = TOP_K_EIG) -> np.ndarray:
    """Top-k eigenvalues of symmetric matrix, sorted descending."""
    # eigsh expects float64 for stability
    vals = eigsh(A_sym, k=k, which='LA', return_eigenvectors=False, tol=1e-5)
    return np.sort(vals)[::-1]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exact',   default='HNE_PubMed/mat_exact.adj')
    ap.add_argument('--kmv',     default='HNE_PubMed/mat_sketch_0')
    ap.add_argument('--kmv-k',   type=int, default=128)
    ap.add_argument('--mprw-dir', default='HNE_PubMed/mprw_work')
    ap.add_argument('--mprw-w',   nargs='+', type=int, default=[8, 32, 128, 512])
    ap.add_argument('--n-eig',    type=int, default=TOP_K_EIG)
    ap.add_argument('--out-csv',  default='results/HNE_PubMed/exp7_spectrum.csv')
    ap.add_argument('--out-pdf',  default='figures/exp7/spectrum.pdf')
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Assemble the method list: (label, path, kind)
    # -------------------------------------------------------------------------
    plans = [('Exact', args.exact, 'exact'),
             (f'KMV k={args.kmv_k}', args.kmv, 'kmv')]
    for w in args.mprw_w:
        plans.append((f'MPRW w={w}',
                      f'{args.mprw_dir}/mat_mprw_{w}.adj', 'mprw'))

    # -------------------------------------------------------------------------
    # Load exact first, find its giant connected component → common node set.
    # All other methods are restricted to this same nodeset so spectra are
    # comparable (otherwise each sparsifier's fragmentation pattern dominates).
    # -------------------------------------------------------------------------
    exact_plan = next(p for p in plans if p[2] == 'exact')
    print(f'[load] {exact_plan[0]} (for GCC extraction)')
    A_exact_full = load_adj(Path(exact_plan[1]))
    gcc_mask = giant_component_mask(A_exact_full)
    n_gcc = int(gcc_mask.sum())
    n_edges_exact = count_unique_edges(A_exact_full)
    print(f'       {n_edges_exact:,} undirected edges, '
          f'GCC = {n_gcc}/{len(gcc_mask)} nodes')

    # -------------------------------------------------------------------------
    # Load, restrict-to-GCC, normalize, eigendecompose each
    # -------------------------------------------------------------------------
    results = []
    for label, path, kind in plans:
        p = Path(path)
        if not p.exists():
            print(f'[skip] {label}: missing {p}')
            continue
        if kind == 'exact':
            A = A_exact_full
        else:
            print(f'[load] {label} from {p}')
            A = load_adj(p)
        n_edges_full = count_unique_edges(A)

        # Restrict to exact's GCC (common node set across methods)
        A_sub_sym = restrict_and_normalize(A, gcc_mask)
        A_sub_unrm = A[np.flatnonzero(gcc_mask)][:, np.flatnonzero(gcc_mask)]
        n_edges_gcc = count_unique_edges(A_sub_unrm)
        # How many nodes become isolated within GCC?
        deg_sub = np.asarray(A_sub_unrm.sum(axis=1)).ravel()
        n_isolated = int((deg_sub == 0).sum())

        print(f'       {n_edges_full:,} edges total → '
              f'{n_edges_gcc:,} edges within GCC, '
              f'{n_isolated} isolated-in-GCC')
        print(f'[eig]  top-{args.n_eig}')
        eig = top_eigenvalues(A_sub_sym, k=args.n_eig)
        # Trim trivial 1.0 eigenvalues at the top (each ~= one component).
        n_trivial = int((eig > 1 - 1e-4).sum())
        print(f'       lambda_1={eig[0]:.4f}, lambda_2={eig[1]:.4f}, '
              f'n_trivial(=1)={n_trivial}, '
              f'first_nontrivial={eig[n_trivial] if n_trivial < len(eig) else float("nan"):.4f}')
        results.append({'label': label, 'kind': kind, 'path': str(p),
                        'n_edges': n_edges_full, 'n_edges_gcc': n_edges_gcc,
                        'n_isolated_in_gcc': n_isolated,
                        'n_trivial_eig': n_trivial, 'eig': eig})

    # -------------------------------------------------------------------------
    # Save long-format CSV (incl. fragmentation diagnostics)
    # -------------------------------------------------------------------------
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method_kind', 'label', 'n_edges', 'n_edges_gcc',
                    'n_isolated_in_gcc', 'n_trivial_eig',
                    'i', 'lambda_i'])
        for r in results:
            for i, lam in enumerate(r['eig']):
                w.writerow([r['kind'], r['label'], r['n_edges'],
                            r['n_edges_gcc'], r['n_isolated_in_gcc'],
                            r['n_trivial_eig'], i, f'{lam:.6f}'])
    print(f'[csv] wrote {out_csv}')

    # -------------------------------------------------------------------------
    # Plot 4 panels
    # -------------------------------------------------------------------------
    exact = next(r for r in results if r['kind'] == 'exact')
    colors = {'exact': '#1f77b4', 'kmv': '#2ca02c', 'mprw': '#d62728'}

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    # Panel A — 1 - λ_i (distance from 1), log-y.
    # Trivial (=1) components show up as a drop-off; shape below is the
    # "real" spectrum of the connected piece. Good sparsifier tracks Exact.
    ax = axes[0]
    for r in results:
        ls = '-' if r['kind'] == 'exact' else ('--' if r['kind'] == 'kmv' else ':')
        alpha = 1.0 if r['kind'] != 'mprw' else 0.6
        lw = 2.2 if r['kind'] == 'exact' else 1.5
        y = np.clip(1.0 - r['eig'], 1e-6, None)
        ax.plot(np.arange(1, len(y) + 1), y,
                label=r['label'], color=colors[r['kind']],
                linestyle=ls, linewidth=lw, alpha=alpha)
    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue rank $i$ (1=top)')
    ax.set_ylabel(r'$1 - \lambda_i$')
    ax.set_title('(a) Distance from $\\lambda=1$\n(flat top = extra components)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3, which='both')

    # Panel B — fragmentation of exact's GCC
    ax = axes[1]
    labels = [r['label'] for r in results]
    n_trivial = [r['n_trivial_eig'] for r in results]
    bar_colors = [colors[r['kind']] for r in results]
    bars = ax.bar(range(len(results)), n_trivial, color=bar_colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(r'# eigenvalues within $10^{-4}$ of 1')
    ax.set_title('(b) Fragmentation of Exact GCC\n(extra connected components)')
    for bar, v in zip(bars, n_trivial):
        ax.text(bar.get_x() + bar.get_width()/2, v,
                str(v), ha='center', va='bottom', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # Panel C — ratio of non-trivial eigenvalues vs Exact
    # Use a common starting index: max n_trivial across methods, so every
    # series is past its trivial-1 plateau.
    ax = axes[2]
    i_start = max(r['n_trivial_eig'] for r in results)
    eig_exact = exact['eig']
    for r in results:
        if r['kind'] == 'exact':
            continue
        ratio = (1 - r['eig']) / np.maximum(1 - eig_exact, 1e-9)
        ls = '--' if r['kind'] == 'kmv' else ':'
        alpha = 1.0 if r['kind'] == 'kmv' else 0.6
        ax.plot(np.arange(i_start + 1, len(ratio) + 1), ratio[i_start:],
                label=r['label'], color=colors[r['kind']],
                linestyle=ls, alpha=alpha, linewidth=1.5)
    ax.axhline(1.0, color=colors['exact'], linewidth=1.2, alpha=0.7,
               label='Exact')
    ax.set_xlabel('Eigenvalue rank $i$')
    ax.set_ylabel(r'$(1-\lambda_i^{\tilde A})\,/\,(1-\lambda_i^{\rm Exact})$')
    ax.set_title(f'(c) Ratio vs Exact (i ≥ {i_start+1})\n'
                 '(flat @1 = uniform approx)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

    # Panel D — Fiedler-like gap: 1 - λ[n_trivial] vs density
    ax = axes[3]
    for r in results:
        nt = r['n_trivial_eig']
        if nt >= len(r['eig']):
            continue
        fiedler = 1.0 - r['eig'][nt]   # first "real" gap below the 1-plateau
        marker = '*' if r['kind'] == 'exact' else ('o' if r['kind'] == 'kmv' else '^')
        size = 220 if r['kind'] == 'exact' else 110
        ax.scatter(r['n_edges'], fiedler, s=size, color=colors[r['kind']],
                   marker=marker, label=r['label'],
                   edgecolor='black', linewidth=0.6, zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel('Undirected edges (density)')
    ax.set_ylabel(r'$1 - \lambda_{\rm 1st nontrivial}$')
    ax.set_title('(d) First-nontrivial gap vs density\n(higher = better mixing)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f'[pdf] wrote {out_pdf}')


if __name__ == '__main__':
    main()
