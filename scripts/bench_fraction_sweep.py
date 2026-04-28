"""bench_fraction_sweep.py — Fraction-sweep version of the 3-way bench.

Subsamples a dataset's target type by hash to {6.25%, 12.5%, 25%, 50%, 100%},
stages each fraction as its own C++ input directory, then runs KMV / KGRW /
MPRW with the same measurement protocol as `bench_kgrw.py`. Writes one row
per (Fraction, Method, k, w', Seed) cell into
`results/<DATASET>/kgrw_bench_fractions.csv`.

Two operating modes
-------------------
* PubMed (`--mode trained`): trains SAGE(L=2) once per fraction on V_train of
  that fraction, generates Z_exact reference, computes F1 + CKA + PA against
  Exact. 5 seeds.
* OGB    (`--mode untrained`): random-init SAGE weights (no training, no
  exact), uses KMV at the largest swept k as the CKA/PA reference. F1 stays
  blank (untrained predictions are noise). 1 seed.

The driver is resume-safe — already-present (Dataset, Fraction, L, Method,
k, w', Seed) rows are skipped.

Usage
-----
    python scripts/bench_fraction_sweep.py --dataset HNE_PubMed  --mode trained
    python scripts/bench_fraction_sweep.py --dataset OGB_MAG     --mode untrained

Outputs
-------
    staging/<DATASET>_f<NN>/                       — per-fraction C++ inputs
    results/<DATASET>/weights/<mp_safe>_L2_f<NN>.pt — frozen θ* per fraction
    results/<DATASET>/kgrw_bench_fractions.csv     — long-form per-cell rows
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

# Stub torch_sparse before importing src/* (some modules transitively import it)
import types as _t
_ts = _t.ModuleType("torch_sparse"); _ts.spspmm = None
sys.modules.setdefault("torch_sparse", _ts)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.models import get_model
from scripts.bench_utils import compile_rule_for_cpp

warnings.filterwarnings("ignore")


# ─── constants ─────────────────────────────────────────────────────────────

FRACTIONS         = [0.0625, 0.125, 0.25, 0.5, 1.0]
KMV_KS            = [4, 8, 16, 32, 64, 128, 256]
MPRW_WS           = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# KGRW = two slices: fix k sweep w', fix w' sweep k. Common cell deduped.
KGRW_FIX_K        = 16
KGRW_FIX_K_WPS    = [1, 2, 4, 8, 16, 32]      # k=16  × these w'
KGRW_FIX_WP       = 4
KGRW_FIX_WP_KS    = [4, 8, 16, 32, 64]        # these k × w'=4
DEFAULT_L         = 2

GRAPH_PREP_BIN = "bin/graph_prep"
MPRW_BIN       = "bin/mprw_exec"


def _wsl(cmd: str) -> list[str]:
    """Wrap a shell command for WSL on Windows; bash on linux."""
    if sys.platform == "win32":
        return ["wsl", "--exec", "bash", "-c", cmd]
    return ["bash", "-c", cmd]


# All C++ wrappers run with cwd=ROOT so `bin/graph_prep` resolves correctly
# and side-files land where the existing pipeline expects them. On Windows,
# wsl.exe automatically maps a Windows cwd (C:\...) to its /mnt/c/... view
# inside the WSL session — no manual path translation needed.
ROOT_STR = str(ROOT)


def _stage_tag(frac: float) -> str:
    """`0.0625` → `f0625` (filesystem-safe, sortable)."""
    return f"f{int(round(frac * 10000)):04d}"


# ─── subsampling ───────────────────────────────────────────────────────────

def subsample_hetero(g: HeteroData, target_type: str, frac: float, seed: int
                     ) -> tuple[HeteroData, torch.Tensor]:
    """Sample frac% of EDGES per edge type, induce subgraph on incident nodes.

    Returns (subsampled HeteroData, kept_target_indices_in_original).

    Policy: independently sample fraction of edges in each edge type. All
    target-type nodes are retained (so |qnodes| is constant across fractions
    and only edge density varies). Non-target nodes are kept iff they appear
    in at least one surviving edge.
    """
    rng = np.random.default_rng(seed)

    # Step 1: subsample edges per edge type into a fresh HeteroData
    g_edges = HeteroData()
    for nt in g.node_types:
        for key, val in g[nt].items():
            g_edges[nt][key] = val
        g_edges[nt].num_nodes = g[nt].num_nodes
    for et, store in g.edge_items():
        ei = store.edge_index
        n_e = ei.size(1)
        if frac >= 1.0 or n_e == 0:
            keep_idx = torch.arange(n_e, dtype=torch.long)
        else:
            n_keep = max(1, int(round(n_e * frac)))
            arr = rng.choice(n_e, size=n_keep, replace=False)
            keep_idx = torch.tensor(np.sort(arr), dtype=torch.long)
        g_edges[et].edge_index = ei[:, keep_idx]
        for k, v in store.items():
            if k == "edge_index": continue
            if isinstance(v, torch.Tensor) and v.size(0) == n_e:
                g_edges[et][k] = v[keep_idx]
            else:
                g_edges[et][k] = v

    # Step 2: induce node sets — keep all target-type nodes, plus any
    # non-target node that participates in a surviving edge.
    n_target = g[target_type].num_nodes
    target_keep = torch.arange(n_target, dtype=torch.long)
    keep_per_type: dict[str, set] = {nt: set() for nt in g.node_types}
    keep_per_type[target_type] = set(range(n_target))
    for (s_t, _, d_t), store in g_edges.edge_items():
        ei = store.edge_index
        if ei.size(1) == 0: continue
        keep_per_type[s_t].update(ei[0].tolist())
        keep_per_type[d_t].update(ei[1].tolist())

    node_dict: dict[str, torch.Tensor] = {}
    for nt in g.node_types:
        ids = sorted(keep_per_type[nt])
        node_dict[nt] = torch.tensor(ids, dtype=torch.long) if ids else torch.zeros(0, dtype=torch.long)

    g_sub = g_edges.subgraph(node_dict)
    return g_sub, target_keep


# ─── staging ────────────────────────────────────────────────────────────────

def stage_fraction(g_sub: HeteroData, dataset: str, target_type: str,
                   metapath: str, frac: float, seed: int) -> dict:
    """Write staging/<dataset>_<tag>/{node,link,meta,qnodes,cod-rules}.dat
    plus partition.json. Returns dict with paths + ids needed downstream."""
    folder = f"{dataset}_{_stage_tag(frac)}"
    stage_dir = ROOT / "staging" / folder
    stage_dir.mkdir(parents=True, exist_ok=True)

    # node.dat / link.dat / meta.dat / offsets.json
    PyGToCppAdapter(str(stage_dir)).convert(g_sub)
    # cod-rules_<folder>.limit  +  <folder>-cod-global-rules.dat
    compile_rule_for_cpp(metapath, g_sub, str(stage_dir), folder)

    # qnodes_<folder>.dat — ALL target-type nodes (HGNN inference setting),
    # not the small 100-sample variant from bench_utils.generate_qnodes.
    sorted_ntypes = sorted(g_sub.node_types)
    offset = sum(g_sub[nt].num_nodes for nt in sorted_ntypes if nt < target_type)
    n_target = g_sub[target_type].num_nodes
    qnode_path = stage_dir / f"qnodes_{folder}.dat"
    qnode_path.write_text("\n".join(str(offset + i) for i in range(n_target)) + "\n",
                           encoding="utf-8")

    # partition.json — temporal split if year present, else hash
    rng = np.random.default_rng(seed)
    train_frac = 0.4  # matches existing protocol
    if "year" in g_sub[target_type] and g_sub[target_type].year.numel() == n_target:
        years = g_sub[target_type].year.cpu().numpy()
        cutoff = int(np.quantile(years, train_frac))
        train_mask_arr = years <= cutoff
        is_temporal = True
    else:
        n_train = int(round(n_target * train_frac))
        train_idx = rng.choice(n_target, size=n_train, replace=False)
        train_mask_arr = np.zeros(n_target, dtype=bool)
        train_mask_arr[train_idx] = True
        is_temporal = False
        cutoff = None

    train_ids = np.where(train_mask_arr)[0].tolist()
    test_ids  = np.where(~train_mask_arr)[0].tolist()

    part = {
        "dataset": dataset, "target_type": target_type, "fraction": frac,
        "train_frac": train_frac, "seed": seed, "is_temporal": is_temporal,
        "cutoff_year": cutoff,
        "train_node_ids": train_ids, "test_node_ids": test_ids,
    }
    (stage_dir / "partition.json").write_text(json.dumps(part), encoding="utf-8")

    return {
        "folder":     folder,
        "stage_dir":  stage_dir,
        "rule_file":  f"staging/{folder}/cod-rules_{folder}.limit",
        "data_dir":   f"staging/{folder}",
        "qnodes":     qnode_path,
        "n_target":   n_target,
        "node_offset": offset,
        "partition":  part,
    }


# ─── training (PubMed mode) ────────────────────────────────────────────────

def _train_subgraph(g_sub: HeteroData, target_type: str, train_ids: list[int],
                    in_dim: int, num_classes: int, L: int, epochs: int,
                    seed: int) -> torch.nn.Module:
    """Train SAGE(L) on the V_train subgraph (target-type endpoints both in
    train_ids). Mirrors exp2_train.py but inlined to avoid coupling."""
    torch.manual_seed(seed); np.random.seed(seed)

    n_target = g_sub[target_type].num_nodes
    keep = torch.tensor(sorted(train_ids), dtype=torch.long)
    id_map = torch.full((n_target,), -1, dtype=torch.long)
    id_map[keep] = torch.arange(keep.size(0))

    # Build a homogeneous edge_index with global IDs offset by sorted-type offsets.
    sorted_ntypes = sorted(g_sub.node_types)
    type_off = {nt: sum(g_sub[t2].num_nodes for t2 in sorted_ntypes if t2 < nt)
                for nt in sorted_ntypes}
    n_total = sum(g_sub[nt].num_nodes for nt in sorted_ntypes)

    src_list, dst_list = [], []
    for (s_t, _, d_t), store in g_sub.edge_items():
        ei = store.edge_index
        valid = torch.ones(ei.size(1), dtype=torch.bool)
        if s_t == target_type:
            valid &= id_map[ei[0]] >= 0
        if d_t == target_type:
            valid &= id_map[ei[1]] >= 0
        if valid.sum() == 0:
            continue
        s = ei[0][valid] + type_off[s_t]
        d = ei[1][valid] + type_off[d_t]
        src_list.append(s); dst_list.append(d)
    if src_list:
        edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)])
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Build x for ALL types (zero-pad non-target to in_dim)
    x_full = torch.zeros((n_total, in_dim), dtype=torch.float32)
    feat_target = g_sub[target_type].x.float()
    x_full[type_off[target_type]:type_off[target_type] + n_target] = feat_target

    labels = g_sub[target_type].y if hasattr(g_sub[target_type], "y") else None
    if labels is None:
        raise RuntimeError(f"No labels on {target_type}")

    train_mask_global = torch.zeros(n_total, dtype=torch.bool)
    # Validation: random 10% of train_ids
    keep_perm = keep[torch.randperm(keep.size(0))]
    n_val = max(1, int(0.1 * keep.size(0)))
    val_local = keep_perm[:n_val]
    train_local = keep_perm[n_val:]
    val_mask_global = torch.zeros(n_total, dtype=torch.bool)
    train_mask_global[type_off[target_type] + train_local] = True
    val_mask_global[  type_off[target_type] + val_local]   = True

    labels_full = torch.full((n_total,) + labels.shape[1:], -1,
                             dtype=labels.dtype)
    labels_full[type_off[target_type]:type_off[target_type] + n_target] = labels

    model = get_model("SAGE", in_dim, num_classes, config.HIDDEN_DIM, num_layers=L)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    is_multi = labels.dim() == 2
    valid_train = train_mask_global & (
        (labels_full.sum(dim=1) > 0) if is_multi else (labels_full >= 0))
    valid_val = val_mask_global & (
        (labels_full.sum(dim=1) > 0) if is_multi else (labels_full >= 0))

    best_loss = float("inf"); best_state = None; wait = 0; patience = 30
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        out = model(x_full, edge_index)
        if is_multi:
            loss = F.binary_cross_entropy_with_logits(
                out[valid_train], labels_full[valid_train].float())
        else:
            loss = F.cross_entropy(out[valid_train], labels_full[valid_train])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            out_e = model(x_full, edge_index)
            if valid_val.sum() > 0:
                if is_multi:
                    vl = F.binary_cross_entropy_with_logits(
                        out_e[valid_val], labels_full[valid_val].float()).item()
                else:
                    vl = F.cross_entropy(
                        out_e[valid_val], labels_full[valid_val]).item()
            else:
                vl = loss.item()

        if vl < best_loss:
            best_loss = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def init_random_sage(in_dim: int, num_classes: int, L: int, seed: int
                     ) -> torch.nn.Module:
    """Random-init SAGE(L) for OGB untrained mode."""
    torch.manual_seed(seed)
    return get_model("SAGE", in_dim, num_classes, config.HIDDEN_DIM, num_layers=L).eval()


# ─── C++ subprocess wrappers ───────────────────────────────────────────────

def _density(edges: int, n_target: int) -> float:
    """Directed density of the materialized target-to-target adjacency."""
    if n_target <= 1:
        return 0.0
    return edges / (n_target * (n_target - 1))


def _count_edges(adj_wsl: str) -> int:
    r = subprocess.run(_wsl(f"awk '{{s += NF-1}} END {{print s+0}}' {adj_wsl}"),
                       capture_output=True, text=True)
    return int(r.stdout.strip()) if r.stdout.strip().isdigit() else 0


def _parse_time(stdout: str) -> float:
    for line in stdout.split("\n"):
        if line.startswith("time:"):
            try: return float(line.split(":", 1)[1].strip())
            except ValueError: return 0.0
    return 0.0


def _parse_peak_kb(stderr: str) -> float:
    for line in stderr.split("\n"):
        if "Maximum resident set size" in line:
            try: return float(line.split(":", 1)[1].strip())
            except ValueError: return 0.0
    return 0.0


def run_kmv(data_dir: str, rule_file: str, out_base_wsl: str, k: int, seed: int
           ) -> tuple[float, int, str, float]:
    """`graph_prep sketch` wrapped with /usr/bin/time -v.
    Returns (algo_time_s, edge_count, actual_adj_wsl_path, peak_rss_mb)."""
    cmd = (f"/usr/bin/time -v "
           f"{GRAPH_PREP_BIN} sketch {data_dir} {rule_file} {out_base_wsl} {k} 1 {seed}")
    r = subprocess.run(_wsl(cmd), capture_output=True, text=True, timeout=3600, cwd=ROOT_STR)
    if r.returncode != 0:
        raise RuntimeError(f"graph_prep sketch failed (k={k}): {r.stderr[-300:]}")
    adj = f"{out_base_wsl}_0"
    return _parse_time(r.stdout), _count_edges(adj), adj, _parse_peak_kb(r.stderr) / 1024


def run_mprw(data_dir: str, rule_file: str, out_adj_wsl: str, w: int, seed: int
            ) -> tuple[float, int, float]:
    cmd = (f"/usr/bin/time -v "
           f"{MPRW_BIN} materialize {data_dir} {rule_file} {out_adj_wsl} {w} {seed}")
    r = subprocess.run(_wsl(cmd), capture_output=True, text=True, timeout=7200, cwd=ROOT_STR)
    if r.returncode != 0:
        raise RuntimeError(f"mprw_exec materialize failed (w={w}): {r.stderr[-300:]}")
    return _parse_time(r.stdout), _count_edges(out_adj_wsl), _parse_peak_kb(r.stderr) / 1024


def run_kgrw(data_dir: str, rule_file: str, out_adj_wsl: str, k: int, w: int,
             seed: int) -> tuple[float, int, float]:
    cmd = (f"/usr/bin/time -v "
           f"{MPRW_BIN} kgrw {data_dir} {rule_file} {out_adj_wsl} {k} {w} {seed}")
    r = subprocess.run(_wsl(cmd), capture_output=True, text=True, timeout=3600, cwd=ROOT_STR)
    if r.returncode != 0:
        raise RuntimeError(f"mprw_exec kgrw failed (k={k}, w={w}): {r.stderr[-300:]}")
    return _parse_time(r.stdout), _count_edges(out_adj_wsl), _parse_peak_kb(r.stderr) / 1024


def run_exact(data_dir: str, rule_file: str, out_adj_wsl: str
             ) -> tuple[float, int, float]:
    cmd = (f"/usr/bin/time -v "
           f"{GRAPH_PREP_BIN} materialize {data_dir} {rule_file} {out_adj_wsl}")
    r = subprocess.run(_wsl(cmd), capture_output=True, text=True, timeout=14400, cwd=ROOT_STR)
    if r.returncode != 0:
        raise RuntimeError(f"graph_prep materialize failed: {r.stderr[-400:]}")
    return _parse_time(r.stdout), _count_edges(out_adj_wsl), _parse_peak_kb(r.stderr) / 1024


# ─── inference (call inference_worker.py as subprocess) ───────────────────

def run_inference(adj_wsl: str, feat_file: str, weights_path: str, z_out: str,
                  labels_file: str, mask_file: str, n_target: int,
                  node_offset: int, in_dim: int, num_classes: int, L: int,
                  scratch: Path) -> dict:
    win_adj = str(scratch / f"tmp_inf_{int(time.time()*1000)%10**9}.adj")
    win_adj_wsl = win_adj.replace("\\", "/").replace("C:/", "/mnt/c/")
    subprocess.run(_wsl(f"cp {adj_wsl} {win_adj_wsl}"),
                   capture_output=True, text=True)

    cmd = [sys.executable, str(ROOT / "scripts" / "inference_worker.py"),
           "--graph-file",  win_adj,
           "--graph-type",  "adj",
           "--feat-file",   feat_file,
           "--weights",     weights_path,
           "--z-out",       z_out,
           "--labels-file", labels_file,
           "--mask-file",   mask_file,
           "--n-target",    str(n_target),
           "--node-offset", str(node_offset),
           "--in-dim",      str(in_dim),
           "--num-classes", str(num_classes),
           "--num-layers",  str(L)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out: dict = {}
    if r.returncode != 0:
        return out
    for line in r.stdout.split("\n"):
        for key in ("inf_f1", "inf_peak_ram_mb", "inf_time", "inf_de"):
            if line.strip().lower().startswith(f"{key}:"):
                try: out[key] = float(line.split(":", 1)[1].strip())
                except ValueError: pass
    try: os.remove(win_adj)
    except OSError: pass
    return out


# ─── main loop ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", choices=["trained", "untrained"], required=True)
    p.add_argument("--metapath", default=None,
                   help="Override metapath. Default: cfg.metapaths[0]")
    p.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    p.add_argument("--kmv-k",  type=int, nargs="+", default=KMV_KS)
    p.add_argument("--mprw-w", type=int, nargs="+", default=MPRW_WS)
    p.add_argument("--kgrw-fix-k",      type=int, default=KGRW_FIX_K,
                   help="Fixed k value for the w'-sweep slice.")
    p.add_argument("--kgrw-fix-k-wps",  type=int, nargs="+", default=KGRW_FIX_K_WPS,
                   help="w' values to sweep at the fixed k.")
    p.add_argument("--kgrw-fix-wp",     type=int, default=KGRW_FIX_WP,
                   help="Fixed w' value for the k-sweep slice.")
    p.add_argument("--kgrw-fix-wp-ks",  type=int, nargs="+", default=KGRW_FIX_WP_KS,
                   help="k values to sweep at the fixed w'.")
    p.add_argument("--seeds", type=int, default=None,
                   help="Default: 5 (trained) / 1 (untrained)")
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--L", type=int, default=DEFAULT_L)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--skip-exact", action="store_true",
                   help="In trained mode, skip Exact materialization (already done).")
    p.add_argument("--skip-restage", action="store_true",
                   help="Reuse existing staging dirs without re-subsampling.")
    p.add_argument("--methods", nargs="+",
                   choices=["KMV", "MPRW", "KGRW"],
                   default=["KMV", "MPRW", "KGRW"],
                   help="Only run these methods. Default: all three. "
                        "Useful when KMV/MPRW are already done and you want "
                        "to top up KGRW only: --methods KGRW")
    args = p.parse_args()
    if args.seeds is None:
        args.seeds = 5 if args.mode == "trained" else 1
    seed_list = list(range(args.seed_base, args.seed_base + args.seeds))

    cfg = config.get_dataset_config(args.dataset)
    metapath = args.metapath or cfg.suggested_paths[0]
    mp_safe = metapath.replace(",", "_")
    target_type = cfg.target_node

    print(f"\n=== {args.dataset} | mode={args.mode} | metapath={metapath} ===")
    print(f"Fractions: {args.fractions} | seeds: {seed_list}")
    print(f"KMV k:  {args.kmv_k}")
    print(f"MPRW w: {args.mprw_w}")
    # Build the KGRW (k, w') cell list — two slices, dedup'd
    kgrw_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for wp in args.kgrw_fix_k_wps:
        cell = (args.kgrw_fix_k, wp)
        if cell not in seen: kgrw_cells.append(cell); seen.add(cell)
    for k in args.kgrw_fix_wp_ks:
        cell = (k, args.kgrw_fix_wp)
        if cell not in seen: kgrw_cells.append(cell); seen.add(cell)
    print(f"KGRW:   slice-A k={args.kgrw_fix_k}, w'∈{args.kgrw_fix_k_wps}")
    print(f"        slice-B k∈{args.kgrw_fix_wp_ks}, w'={args.kgrw_fix_wp}")
    print(f"        → {len(kgrw_cells)} unique cells: {kgrw_cells}\n")

    # Load full graph once
    print(f"[load] {cfg.source}/{cfg.dataset_name} target={target_type}")
    g_full, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, target_type)

    # CSV setup
    out_csv = ROOT / "results" / args.dataset / "kgrw_bench_fractions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Dataset", "Fraction", "MetaPath", "L", "Method", "k", "w_prime",
              "Seed", "Edge_Count", "Density", "Mat_Time_s", "Mat_Peak_RAM_MB",
              "Macro_F1", "CKA", "Pred_Agreement", "n_target"]
    existing: set[tuple] = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add((row.get("Dataset", ""),
                              row.get("Fraction", ""),
                              row.get("L", ""),
                              row.get("Method", ""),
                              row.get("k", ""),
                              row.get("w_prime", ""),
                              row.get("Seed", "")))

    fout = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=fields)
    if out_csv.stat().st_size == 0:
        writer.writeheader()

    # ── Resume summary: how many rows already in CSV per (Fraction, Method)
    if existing:
        from collections import Counter
        tally = Counter((row[1], row[3]) for row in existing)  # (frac, method)
        print(f"\n[resume] {len(existing)} rows already in CSV — breakdown:")
        for frac in sorted({r[1] for r in existing}):
            line = f"  frac={frac}: " + ", ".join(
                f"{m}={tally.get((frac, m), 0)}"
                for m in ("Exact", "KMV", "MPRW", "KGRW") if (frac, m) in tally)
            print(line)
        print(f"[resume] running methods: {args.methods}\n")

    weights_dir = ROOT / "results" / args.dataset / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "results" / args.dataset / "fraction_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    # Reuse a single random-init weights file across all OGB fractions
    # (input dim and num classes are constant — fraction only changes graph).
    untrained_weights_cache: torch.nn.Module | None = None

    for frac in args.fractions:
        tag = _stage_tag(frac)
        print(f"\n────── FRACTION {frac:.4f}  ({tag}) ──────")
        stage = ROOT / "staging" / f"{args.dataset}_{tag}"
        if args.skip_restage and (stage / "node.dat").exists() and (stage / "partition.json").exists():
            paths = {"folder":     stage.name,
                     "stage_dir":  stage,
                     "rule_file":  f"staging/{stage.name}/cod-rules_{stage.name}.limit",
                     "data_dir":   f"staging/{stage.name}",
                     "qnodes":     stage / f"qnodes_{stage.name}.dat",
                     "partition":  json.loads((stage / "partition.json").read_text())}
            # Need n_target + node_offset — derive from staged node.dat
            n_target = sum(1 for _ in (stage / f"qnodes_{stage.name}.dat").open())
            paths["n_target"] = n_target
            print(f"[stage] reusing {stage}")
            # Need g_sub for training/inference features; load from a saved snapshot
            snap = stage / "data.pt"
            if snap.exists():
                g_sub = torch.load(snap, weights_only=False)
            else:
                # fall back to subsampling
                g_sub, _ = subsample_hetero(g_full, target_type, frac, args.seed_base)
                torch.save(g_sub, snap)
        else:
            print(f"[stage] subsampling → {stage}")
            g_sub, _ = subsample_hetero(g_full, target_type, frac, args.seed_base)
            paths = stage_fraction(g_sub, args.dataset, target_type, metapath,
                                   frac, args.seed_base)
            torch.save(g_sub, paths["stage_dir"] / "data.pt")

        n_target = g_sub[target_type].num_nodes
        in_dim = g_sub[target_type].x.size(1)
        num_classes = info["num_classes"]
        sorted_ntypes = sorted(g_sub.node_types)
        node_offset = sum(g_sub[nt].num_nodes for nt in sorted_ntypes if nt < target_type)
        print(f"[stage] |target|={n_target}  in_dim={in_dim}  classes={num_classes}")

        # Save tensors that inference_worker needs
        feat_file = str(scratch / f"x_{tag}.pt")
        labels_file = str(scratch / f"labels_{tag}.pt")
        mask_file = str(scratch / f"mask_{tag}.pt")
        torch.save(g_sub[target_type].x.cpu(), feat_file)
        labels_t = g_sub[target_type].y if hasattr(g_sub[target_type], "y") \
                   else torch.full((n_target,), -1, dtype=torch.long)
        torch.save(labels_t.cpu(), labels_file)
        test_ids = torch.tensor(paths["partition"]["test_node_ids"], dtype=torch.long)
        test_mask = torch.zeros(n_target, dtype=torch.bool); test_mask[test_ids] = True
        torch.save(test_mask, mask_file)

        # Weights
        weights_path = weights_dir / f"{mp_safe}_L{args.L}_{tag}.pt"
        if args.mode == "trained":
            if not weights_path.exists():
                t0 = time.perf_counter()
                model = _train_subgraph(
                    g_sub, target_type, paths["partition"]["train_node_ids"],
                    in_dim, num_classes, args.L, args.epochs, args.seed_base)
                torch.save(model.state_dict(), weights_path)
                print(f"[train] L={args.L} → {weights_path.name} ({time.perf_counter()-t0:.1f}s)")
        else:
            if untrained_weights_cache is None:
                untrained_weights_cache = init_random_sage(
                    in_dim, num_classes, args.L, args.seed_base)
                torch.save(untrained_weights_cache.state_dict(), weights_path)
            elif not weights_path.exists():
                torch.save(untrained_weights_cache.state_dict(), weights_path)

        # ── Reference Z (for CKA / PA) ────────────────────────────────────
        ref_z: torch.Tensor | None = None
        if args.mode == "trained" and not args.skip_exact:
            run_key = (args.dataset, f"{frac:.4f}", str(args.L), "Exact", "", "", "0")
            ex_adj = f"/tmp/exact_{args.dataset}_{tag}.adj"
            ex_z   = scratch / f"z_exact_{tag}.pt"
            if run_key not in existing or not ex_z.exists():
                try:
                    t_mat, edges, ram = run_exact(paths["data_dir"], paths["rule_file"], ex_adj)
                    inf = run_inference(ex_adj, feat_file, str(weights_path),
                                        str(ex_z), labels_file, mask_file,
                                        n_target, node_offset, in_dim, num_classes,
                                        args.L, scratch)
                    f1 = inf.get("inf_f1", float("nan"))
                    if run_key not in existing:
                        writer.writerow({
                            "Dataset": args.dataset, "Fraction": f"{frac:.4f}",
                            "MetaPath": metapath, "L": args.L,
                            "Method": "Exact", "k": "", "w_prime": "", "Seed": 0,
                            "Edge_Count": edges, "Density": round(_density(edges, n_target), 8),
                    "Mat_Time_s": round(t_mat, 4),
                            "Mat_Peak_RAM_MB": round(ram, 1),
                            "Macro_F1": round(f1, 6) if f1 == f1 else "",
                            "CKA": 1.0, "Pred_Agreement": 1.0,
                            "n_target": n_target,
                        })
                        fout.flush(); existing.add(run_key)
                    print(f"[Exact] edges={edges:,}  t={t_mat:.2f}s  ram={ram:.0f}MB  F1={f1:.4f}")
                except Exception as e:
                    print(f"[Exact] FAILED: {e}")
            if ex_z.exists():
                ref_z = torch.load(str(ex_z), weights_only=True)

        # If untrained mode, KMV at largest k will be the reference — generated below
        # in the KMV loop and then used for CKA/PA on subsequent runs.
        ref_k_max = max(args.kmv_k) if args.mode == "untrained" else None

        from src.analysis.cka import LinearCKA
        cka_calc = LinearCKA(device=torch.device("cpu"))

        def _cka_pa(z_path: Path) -> tuple[float, float]:
            if ref_z is None or not z_path.exists():
                return float("nan"), float("nan")
            try:
                z = torch.load(str(z_path), weights_only=True)
                cka = float(cka_calc.calculate(ref_z[test_mask], z[test_mask]))
                pa  = (ref_z[test_mask].argmax(1) == z[test_mask].argmax(1)
                      ).float().mean().item()
                return cka, pa
            except Exception as ex:
                print(f"  CKA error: {ex}"); return float("nan"), float("nan")

        # ── KMV sweep ────────────────────────────────────────────────────
        for k in (sorted(args.kmv_k) if "KMV" in args.methods else []):
            for seed in seed_list:
                run_key = (args.dataset, f"{frac:.4f}", str(args.L), "KMV", str(k), "", str(seed))
                if run_key in existing:
                    print(f"[KMV k={k:>3} s={seed}] skip (exists)"); continue
                out_base = f"/tmp/kmv_{args.dataset}_{tag}_k{k}_s{seed}"
                try:
                    t_mat, edges, adj, ram = run_kmv(
                        paths["data_dir"], paths["rule_file"], out_base, k, seed)
                except Exception as e:
                    print(f"[KMV k={k} s={seed}] FAILED: {e}"); continue
                z_out = scratch / f"z_kmv_{tag}_k{k}_s{seed}.pt"
                inf = run_inference(adj, feat_file, str(weights_path), str(z_out),
                                    labels_file, mask_file, n_target, node_offset,
                                    in_dim, num_classes, args.L, scratch)
                f1 = inf.get("inf_f1", float("nan"))
                # First reference for untrained mode = KMV at largest k, seed_base
                if (args.mode == "untrained" and ref_z is None and k == ref_k_max
                        and seed == seed_list[0] and z_out.exists()):
                    ref_z = torch.load(str(z_out), weights_only=True)
                cka, pa = _cka_pa(z_out)
                writer.writerow({
                    "Dataset": args.dataset, "Fraction": f"{frac:.4f}",
                    "MetaPath": metapath, "L": args.L,
                    "Method": "KMV", "k": k, "w_prime": "", "Seed": seed,
                    "Edge_Count": edges, "Density": round(_density(edges, n_target), 8),
                    "Mat_Time_s": round(t_mat, 4),
                    "Mat_Peak_RAM_MB": round(ram, 1),
                    "Macro_F1": round(f1, 6) if f1 == f1 else "",
                    "CKA": round(cka, 6) if cka == cka else "",
                    "Pred_Agreement": round(pa, 6) if pa == pa else "",
                    "n_target": n_target,
                })
                fout.flush(); existing.add(run_key)
                print(f"[KMV k={k:>3} s={seed}] edges={edges:>10,} t={t_mat:6.2f}s "
                      f"ram={ram:>6.0f}MB F1={f1:.4f} CKA={cka:.4f} PA={pa:.4f}")

        # If we still have no ref in untrained mode (KMV failed at ref_k_max),
        # try smaller k references in descending order.
        if args.mode == "untrained" and ref_z is None:
            for k in sorted(args.kmv_k, reverse=True):
                cand = scratch / f"z_kmv_{tag}_k{k}_s{seed_list[0]}.pt"
                if cand.exists():
                    ref_z = torch.load(str(cand), weights_only=True)
                    print(f"[ref] using KMV k={k} as CKA/PA reference"); break

        # ── MPRW sweep ───────────────────────────────────────────────────
        for w in (sorted(args.mprw_w) if "MPRW" in args.methods else []):
            for seed in seed_list:
                run_key = (args.dataset, f"{frac:.4f}", str(args.L), "MPRW", "", str(w), str(seed))
                if run_key in existing:
                    print(f"[MPRW w={w:>3} s={seed}] skip (exists)"); continue
                out_adj = f"/tmp/mprw_{args.dataset}_{tag}_w{w}_s{seed}.adj"
                try:
                    t_mat, edges, ram = run_mprw(
                        paths["data_dir"], paths["rule_file"], out_adj, w, seed)
                except Exception as e:
                    print(f"[MPRW w={w} s={seed}] FAILED: {e}"); continue
                z_out = scratch / f"z_mprw_{tag}_w{w}_s{seed}.pt"
                inf = run_inference(out_adj, feat_file, str(weights_path), str(z_out),
                                    labels_file, mask_file, n_target, node_offset,
                                    in_dim, num_classes, args.L, scratch)
                f1 = inf.get("inf_f1", float("nan"))
                cka, pa = _cka_pa(z_out)
                writer.writerow({
                    "Dataset": args.dataset, "Fraction": f"{frac:.4f}",
                    "MetaPath": metapath, "L": args.L,
                    "Method": "MPRW", "k": "", "w_prime": w, "Seed": seed,
                    "Edge_Count": edges, "Density": round(_density(edges, n_target), 8),
                    "Mat_Time_s": round(t_mat, 4),
                    "Mat_Peak_RAM_MB": round(ram, 1),
                    "Macro_F1": round(f1, 6) if f1 == f1 else "",
                    "CKA": round(cka, 6) if cka == cka else "",
                    "Pred_Agreement": round(pa, 6) if pa == pa else "",
                    "n_target": n_target,
                })
                fout.flush(); existing.add(run_key)
                print(f"[MPRW w={w:>3} s={seed}] edges={edges:>10,} t={t_mat:6.2f}s "
                      f"ram={ram:>6.0f}MB F1={f1:.4f} CKA={cka:.4f} PA={pa:.4f}")

        # ── KGRW sweep (two-slice fix-and-sweep, dedup'd) ────────────────
        print(f"[KGRW] starting sweep over {len(kgrw_cells)} cells × {len(seed_list)} seeds "
              f"(frac={frac:.4f})", flush=True)
        if not kgrw_cells:
            print("[KGRW] ⚠ kgrw_cells is empty — check --kgrw-fix-k-wps / --kgrw-fix-wp-ks",
                  flush=True)
        for k, w in (kgrw_cells if "KGRW" in args.methods else []):
            for seed in seed_list:
                run_key = (args.dataset, f"{frac:.4f}", str(args.L), "KGRW", str(k), str(w), str(seed))
                if run_key in existing:
                    print(f"[KGRW k={k} w={w} s={seed}] skip (exists)"); continue
                out_adj = f"/tmp/kgrw_{args.dataset}_{tag}_k{k}_w{w}_s{seed}.adj"
                try:
                    t_mat, edges, ram = run_kgrw(
                        paths["data_dir"], paths["rule_file"], out_adj, k, w, seed)
                except Exception as e:
                    print(f"[KGRW k={k} w={w} s={seed}] FAILED: {e}"); continue
                z_out = scratch / f"z_kgrw_{tag}_k{k}_w{w}_s{seed}.pt"
                inf = run_inference(out_adj, feat_file, str(weights_path), str(z_out),
                                    labels_file, mask_file, n_target, node_offset,
                                    in_dim, num_classes, args.L, scratch)
                f1 = inf.get("inf_f1", float("nan"))
                cka, pa = _cka_pa(z_out)
                writer.writerow({
                    "Dataset": args.dataset, "Fraction": f"{frac:.4f}",
                    "MetaPath": metapath, "L": args.L,
                    "Method": "KGRW", "k": k, "w_prime": w, "Seed": seed,
                    "Edge_Count": edges, "Density": round(_density(edges, n_target), 8),
                    "Mat_Time_s": round(t_mat, 4),
                    "Mat_Peak_RAM_MB": round(ram, 1),
                    "Macro_F1": round(f1, 6) if f1 == f1 else "",
                    "CKA": round(cka, 6) if cka == cka else "",
                    "Pred_Agreement": round(pa, 6) if pa == pa else "",
                    "n_target": n_target,
                })
                fout.flush(); existing.add(run_key)
                print(f"[KGRW k={k} w={w} s={seed}] edges={edges:>10,} t={t_mat:6.2f}s "
                      f"ram={ram:>6.0f}MB F1={f1:.4f} CKA={cka:.4f} PA={pa:.4f}")

    fout.close()
    print(f"\nWrote → {out_csv}")


if __name__ == "__main__":
    main()
