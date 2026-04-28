#!/usr/bin/env python3
"""
Meta-path inductive-bias profiler.

For every meta-path listed in src/config.DATASETS[<ds>].suggested_paths,
load the cached exact materialized adjacency (produced by prior graph_prep
runs), and report:
    * |E| retained (undirected edges between target-type nodes)
    * density = |E| / (n_target * (n_target-1) / 2)
    * homophily = P(label(u) == label(v) | (u,v) retained, both labelled)
    * base rate = 1/|C|  (uniform-label chance)
    * homo_ratio = homophily / base_rate   (> 1 homophilic, ≈ 1 neutral, < 1 heterophilic)
    * spectral radius λ_max of retained adjacency
    * mean degree

Intent: decide which meta-path to use per dataset for the MPRW-vs-KMV
comparison. The homo_ratio is the key: H1 (MPRW homophily bias) predicts
MPRW wins on homophilic meta-paths and loses on heterophilic ones.

Reads only cached .adj files under results/<ds>/. Missing paths are reported
as TODO (run graph_prep materialize for them).

Outputs:
    results/kmv_properties/metapath_profile.csv
    stdout table sorted by homo_ratio (ascending → heterophilic first).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.config import config
from src.data.factory import DatasetFactory


DATASETS = {
    "HGB_ACM":    {"source": "HGB", "name": "ACM",    "target": "paper"},
    "HGB_DBLP":   {"source": "HGB", "name": "DBLP",   "target": "author"},
    "HGB_IMDB":   {"source": "HGB", "name": "IMDB",   "target": "movie"},
    "HNE_PubMed": {"source": "HNE", "name": "PubMed", "target": "disease"},
}


def slugify(mp: str) -> str:
    return mp.replace(",", "_")


def node_offset_for_target(g_hetero, target: str) -> int:
    off = 0
    for nt in sorted(g_hetero.node_types):
        if nt == target:
            break
        off += g_hetero[nt].num_nodes
    return off


def load_adj(path: Path, node_offset: int, n_target: int) -> Tuple[np.ndarray, np.ndarray]:
    srcs: List[int] = []
    dsts: List[int] = []
    with path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u_global = int(parts[0])
            u_local = u_global - node_offset
            if not (0 <= u_local < n_target):
                continue
            for tok in parts[1:]:
                v_global = int(tok)
                v_local = v_global - node_offset
                if 0 <= v_local < n_target and v_local != u_local:
                    srcs.append(u_local)
                    dsts.append(v_local)
    return np.asarray(srcs, dtype=np.int64), np.asarray(dsts, dtype=np.int64)


def homophily(src: np.ndarray, dst: np.ndarray, labels: np.ndarray) -> Tuple[float, int]:
    if src.size == 0:
        return float("nan"), 0
    lu = labels[src]
    lv = labels[dst]
    mask = (lu >= 0) & (lv >= 0)
    if mask.sum() == 0:
        return float("nan"), 0
    return float((lu[mask] == lv[mask]).mean()), int(mask.sum())


def spectral_radius(src: np.ndarray, dst: np.ndarray, n: int) -> float:
    if src.size == 0:
        return float("nan")
    A = sp.coo_matrix((np.ones(src.size), (src, dst)), shape=(n, n))
    A = A.maximum(A.T).tocsr()
    A.data = np.minimum(A.data, 1.0)
    try:
        from scipy.sparse.linalg import eigsh
        vals = eigsh(A.astype(np.float64), k=1, which="LM",
                     return_eigenvectors=False, maxiter=500, tol=1e-3)
        return float(vals[0])
    except Exception:
        return float("nan")


def find_cached_adj(ds_key: str, metapath: str) -> Optional[Path]:
    """Locate exact_<slug>.adj under results/<ds_key>/. Try multiple slug variants."""
    results_dir = PROJ / "results" / ds_key
    if not results_dir.exists():
        return None
    slug = slugify(metapath)
    candidates = [
        results_dir / f"exact_{slug}.adj",
        results_dir / f"mat_exact_{slug}.adj",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    # short-name abbreviations (e.g., PAP, APA, MKM, DCD)
    short = shortname(metapath)
    if short:
        for c in [results_dir / f"exact_{short.lower()}.adj"]:
            if c.exists() and c.stat().st_size > 0:
                return c
    # densest shortcut: only if this is the first path listed (densest in the paper)
    return None


def shortname(metapath: str) -> str:
    """MKM, APA, PAP etc. by initial of each node type in the path."""
    tokens = metapath.split(",")
    letters: List[str] = []
    for i, tok in enumerate(tokens):
        src_type, dst_type = parse_edge(tok)
        if i == 0 and src_type:
            letters.append(src_type[0].upper())
        if dst_type:
            letters.append(dst_type[0].upper())
    return "".join(letters)


def parse_edge(tok: str) -> Tuple[str, str]:
    tok = tok.strip()
    if "_to_" in tok:
        a, b = tok.split("_to_", 1)
        return a, b
    return "", ""


def profile_metapath(ds_key: str, metapath: str, g_hetero, labels: np.ndarray,
                     n_target: int, offset: int, n_classes: int) -> Dict:
    adj = find_cached_adj(ds_key, metapath)
    row: Dict = {
        "dataset": ds_key,
        "metapath": metapath,
        "short": shortname(metapath),
        "L_hops": len(metapath.split(",")),
        "n_target": n_target,
        "n_classes": n_classes,
        "base_rate": (1.0 / n_classes) if n_classes > 0 else float("nan"),
    }
    if adj is None:
        row.update({
            "cached": False, "edges": 0, "density": float("nan"),
            "homophily": float("nan"), "homo_ratio": float("nan"),
            "n_labeled_edges": 0, "lambda_max": float("nan"),
            "mean_degree": float("nan"),
        })
        return row

    src, dst = load_adj(adj, offset, n_target)
    # dedup undirected
    lo = np.minimum(src, dst); hi = np.maximum(src, dst)
    pair_key = lo.astype(np.int64) * n_target + hi.astype(np.int64)
    uniq = np.unique(pair_key)
    n_edges = int(uniq.size)
    density = n_edges / (n_target * (n_target - 1) / 2.0) if n_target > 1 else float("nan")
    homo, n_le = homophily(src, dst, labels)
    homo_ratio = homo / row["base_rate"] if (n_classes > 0 and np.isfinite(homo)) else float("nan")
    lam = spectral_radius(src, dst, n_target)
    mean_deg = src.size / n_target if n_target > 0 else float("nan")

    row.update({
        "cached": True,
        "edges": n_edges,
        "density": density,
        "homophily": homo,
        "homo_ratio": homo_ratio,
        "n_labeled_edges": n_le,
        "lambda_max": lam,
        "mean_degree": mean_deg,
        "adj_path": str(adj.relative_to(PROJ)),
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--out", default=str(PROJ / "results" / "kmv_properties" / "metapath_profile.csv"))
    args = ap.parse_args()

    rows: List[Dict] = []
    for ds_key in args.datasets:
        if ds_key not in DATASETS:
            print(f"[skip] {ds_key} not in known list")
            continue
        spec = DATASETS[ds_key]
        ds_cfg = config.get_dataset_config(ds_key)
        metapaths = ds_cfg.suggested_paths
        if not metapaths:
            print(f"[{ds_key}] no suggested_paths in config.DATASETS; skip")
            continue
        print(f"\n=== {ds_key} (target={spec['target']}, {len(metapaths)} paths) ===")
        g, _ = DatasetFactory.get_data(spec["source"], spec["name"], spec["target"])
        target = spec["target"]
        n_target = g[target].num_nodes
        offset = node_offset_for_target(g, target)
        labels = g[target].y.cpu().numpy() if hasattr(g[target], "y") else np.full(n_target, -1, dtype=np.int64)
        labels = labels.astype(np.int64, copy=False)
        n_classes = int(labels.max() + 1) if (labels >= 0).any() else 0
        label_cov = float((labels >= 0).mean())
        print(f"  n_target={n_target}  classes={n_classes}  label_cov={label_cov:.1%}")

        for mp in metapaths:
            row = profile_metapath(ds_key, mp, g, labels, n_target, offset, n_classes)
            row["label_coverage"] = label_cov
            rows.append(row)
            if row["cached"]:
                print(f"  {row['short']:<8s} L={row['L_hops']} |E|={row['edges']:>10,}  "
                      f"dens={row['density']:.4f}  homo={row['homophily']:.3f}  "
                      f"ratio={row['homo_ratio']:.2f}  λ={row['lambda_max']:.1f}  "
                      f"(n_le={row['n_labeled_edges']})")
            else:
                print(f"  {row['short']:<8s} L={row['L_hops']}  [NO CACHED .adj — need to run graph_prep]")

    # Ranked table
    print("\n\n=== RANKED BY homo_ratio (heterophilic → homophilic) ===")
    rows_sorted = sorted([r for r in rows if r["cached"] and np.isfinite(r["homo_ratio"])],
                         key=lambda r: r["homo_ratio"])
    print(f"{'dataset':<12s} {'short':<10s} {'L':<3s} {'|E|':>12s} {'density':>9s} "
          f"{'homo':>7s} {'base':>6s} {'ratio':>6s} {'λ_max':>8s}")
    for r in rows_sorted:
        print(f"{r['dataset']:<12s} {r['short']:<10s} {r['L_hops']:<3d} "
              f"{r['edges']:>12,} {r['density']:>9.4f} "
              f"{r['homophily']:>7.3f} {r['base_rate']:>6.3f} "
              f"{r['homo_ratio']:>6.2f} {r['lambda_max']:>8.1f}")

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "metapath", "short", "L_hops", "n_target", "n_classes",
              "base_rate", "label_coverage", "cached", "edges", "density",
              "homophily", "homo_ratio", "n_labeled_edges", "lambda_max", "mean_degree",
              "adj_path"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
