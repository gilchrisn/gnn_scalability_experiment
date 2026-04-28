"""Quick one-shot evaluation of KGRW adjacency files.

Runs frozen SAGE inference on KGRW (and Exact/KMV/MPRW for reference),
computes F1, CKA, PredAgreement on test nodes.

Usage:
    python scripts/eval_kgrw.py --dataset HGB_DBLP --L 2 \
        --kgrw-adj /tmp/kgrw_dblp.adj \
        --exact-adj HGBn-DBLP/mat_exact.adj
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub torch_sparse so src/__init__.py doesn't crash (not needed here)
import types as _types
_ts = _types.ModuleType('torch_sparse')
_ts.spspmm = None
sys.modules.setdefault('torch_sparse', _ts)

import torch, torch.nn.functional as F
from src.config import config
from src.data import DatasetFactory
from src.analysis.cka import LinearCKA


def _ensure_stubs():
    """Create a torch_sparse stub module so subprocesses don't crash on Windows."""
    stub_dir = ROOT / "results" / "kgrw_eval" / "_stubs" / "torch_sparse"
    stub_dir.mkdir(parents=True, exist_ok=True)
    init_f = stub_dir / "__init__.py"
    if not init_f.exists():
        init_f.write_text("spspmm = None\n")
    return str(stub_dir.parent)

_STUB_PATH = None

def _run_inf(graph_file, graph_type, feat_file, weights_path, z_out,
             labels_file, mask_file, n_target, node_offset, in_dim,
             num_classes, num_layers, label=""):
    global _STUB_PATH
    if _STUB_PATH is None:
        _STUB_PATH = _ensure_stubs()

    env = {**os.environ}
    env["PYTHONPATH"] = _STUB_PATH + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, str(ROOT / "scripts" / "inference_worker.py"),
        "--graph-file",  graph_file,
        "--graph-type",  graph_type,
        "--feat-file",   feat_file,
        "--weights",     weights_path,
        "--z-out",       z_out,
        "--labels-file", labels_file,
        "--mask-file",   mask_file,
        "--n-target",    str(n_target),
        "--node-offset", str(node_offset),
        "--in-dim",      str(in_dim),
        "--num-classes",  str(num_classes),
        "--num-layers",  str(num_layers),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"  [{label}] FAILED (rc={res.returncode})")
        print(f"  stderr: {res.stderr[-500:]}")
        return {}
    out = {}
    for line in res.stdout.split("\n"):
        for key in ("inf_peak_ram_mb", "inf_time", "inf_f1", "inf_de"):
            if line.strip().lower().startswith(f"{key}:"):
                try: out[key] = float(line.split(":", 1)[1].strip())
                except ValueError: pass
    print(f"  [{label}] F1={out.get('inf_f1',0):.4f}  DE={out.get('inf_de',0):.4f}  "
          f"inf_time={out.get('inf_time',0):.3f}s  wall={dt:.1f}s")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--kgrw-adj", required=True, help="Path to KGRW adjacency file")
    p.add_argument("--exact-adj", default=None, help="Path to Exact adjacency (optional)")
    p.add_argument("--metapath", default=None)
    p.add_argument("--partition-json", default=None)
    p.add_argument("--weights-dir", default=None)
    args = p.parse_args()

    cfg = config.get_dataset_config(args.dataset)
    folder = config.get_folder_name(args.dataset)
    data_dir = config.get_staging_dir(args.dataset)
    target_ntype = cfg.target_node

    # Partition
    part_path = args.partition_json or str(ROOT / "results_new" / args.dataset / "partition.json")
    if not os.path.exists(part_path):
        part_path = str(ROOT / "results" / args.dataset / "partition.json")
    with open(part_path) as f:
        part = json.load(f)

    # Dataset
    g_full, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    num_classes = info["num_classes"]
    labels_full = info["labels"]

    # Test mask
    test_ids = torch.tensor(part["test_node_ids"], dtype=torch.long)
    n_target = g_full[target_ntype].num_nodes
    test_mask = torch.zeros(n_target, dtype=torch.bool)
    test_mask[test_ids] = True

    # Node offset
    s_ntypes = sorted(g_full.node_types)
    node_offset = sum(g_full[nt].num_nodes for nt in s_ntypes if nt < target_ntype)

    x_full = g_full[target_ntype].x
    in_dim = x_full.size(1)

    # Metapath for weights path
    mp_lookup = {
        "HGB_ACM": "paper_to_term_term_to_paper",
        "HGB_DBLP": "author_to_paper_paper_to_term_term_to_paper_paper_to_author",
        "HGB_IMDB": "movie_to_keyword_keyword_to_movie",
        "HNE_PubMed": "disease_to_chemical_chemical_to_disease",
    }
    mp_safe = args.metapath.replace(",", "_") if args.metapath else mp_lookup.get(args.dataset, "")

    weights_dir = Path(args.weights_dir) if args.weights_dir else ROOT / "results" / args.dataset / "weights"
    weights_path = str(weights_dir / f"{mp_safe}_L{args.L}.pt")
    if not os.path.exists(weights_path):
        # Try results_new
        weights_path = str(ROOT / "results_new" / args.dataset / "weights" / f"{mp_safe}_L{args.L}.pt")
    print(f"Weights: {weights_path}")
    print(f"  exists: {os.path.exists(weights_path)}")

    # Save temp files
    tmp_dir = ROOT / "results" / "kgrw_eval"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    feat_file = str(tmp_dir / "x.pt")
    labels_file = str(tmp_dir / "labels.pt")
    mask_file = str(tmp_dir / "mask.pt")
    torch.save(x_full.cpu(), feat_file)
    torch.save(labels_full.cpu(), labels_file)
    torch.save(test_mask.cpu(), mask_file)

    print(f"\nDataset: {args.dataset}  target={target_ntype}  n_target={n_target}"
          f"  node_offset={node_offset}  in_dim={in_dim}  classes={num_classes}"
          f"  L={args.L}  test={test_mask.sum().item()}")

    z_exact_path = str(tmp_dir / f"z_exact_L{args.L}.pt")
    z_kgrw_path  = str(tmp_dir / f"z_kgrw_L{args.L}.pt")

    # Run Exact inference
    if args.exact_adj and os.path.exists(args.exact_adj):
        print(f"\n--- Exact inference (adj: {args.exact_adj}) ---")
        _run_inf(args.exact_adj, "adj", feat_file, weights_path, z_exact_path,
                 labels_file, mask_file, n_target, node_offset, in_dim,
                 num_classes, args.L, "Exact")
    else:
        print("  [Exact] skipped (no --exact-adj or file missing)")

    # Run KGRW inference
    print(f"\n--- KGRW inference (adj: {args.kgrw_adj}) ---")
    _run_inf(args.kgrw_adj, "adj", feat_file, weights_path, z_kgrw_path,
             labels_file, mask_file, n_target, node_offset, in_dim,
             num_classes, args.L, "KGRW")

    # CKA comparison
    if os.path.exists(z_exact_path) and os.path.exists(z_kgrw_path):
        print("\n--- CKA / PredAgreement (KGRW vs Exact) ---")
        z_exact = torch.load(z_exact_path, weights_only=True)
        z_kgrw  = torch.load(z_kgrw_path, weights_only=True)
        mask_dev = test_mask

        pred_agree = (z_exact[mask_dev].argmax(1) == z_kgrw[mask_dev].argmax(1)).float().mean().item()
        cka = LinearCKA(device=torch.device("cpu"))
        cka_val = cka.calculate(z_exact[mask_dev], z_kgrw[mask_dev])

        print(f"  CKA:              {cka_val:.4f}")
        print(f"  PredAgreement:    {pred_agree:.4f}")


if __name__ == "__main__":
    main()
