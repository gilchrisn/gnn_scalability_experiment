"""
bench_kgrw.py — KGRW quality benchmark against MPRW across seeds.

For each (k, w', seed) combination:
  1. Runs `mprw_exec kgrw`       → adjacency file
  2. Runs `mprw_exec materialize` → MPRW reference at matched edges
  3. Runs inference_worker.py for both → F1, CKA, PredAgreement
  Aggregates mean ± std across seeds.

Usage
-----
# Single seed quick check:
python scripts/bench_kgrw.py --dataset HGB_DBLP --L 2

# Multi-seed rigorous run (recommended):
python scripts/bench_kgrw.py --dataset HGB_DBLP --L 2 --seeds 10 \
    --k-values 4 8 16 32 --w-values 1 2 4 8 16

# Resume-safe: already-done (L, method, k, w', seed) rows are skipped.

Output
------
  results/<dataset>/kgrw_bench.csv  — raw per-seed rows
  Printed mean±std summary table
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub torch_sparse (not needed here but src imports it)
import types as _t
_ts = _t.ModuleType("torch_sparse"); _ts.spspmm = None
sys.modules.setdefault("torch_sparse", _ts)

import torch
import torch.nn.functional as F
from src.config import config
from src.data import DatasetFactory

import warnings; warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_K_VALUES  = [4, 8, 16, 32]
DEFAULT_W_PRIMES  = [1, 2, 4, 8]
SEED = 42

MPRW_BIN = str(ROOT / "bin" / "mprw_exec")
WSL_PREFIX = ["wsl", "--exec", "bash", "-c"]   # used on Windows to call ELF binary


def _wsl(cmd_str: str) -> list[str]:
    """Wrap a shell command string for WSL execution on Windows."""
    if sys.platform == "win32":
        return ["wsl", "--exec", "bash", "-c", cmd_str]
    return ["bash", "-c", cmd_str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metapath_safe(mp: str) -> str:
    return mp.replace(",", "_")


def _find_weights(dataset: str, mp_safe: str, L: int) -> Path | None:
    for base in [ROOT / "results" / dataset / "weights",
                 ROOT / "results_new" / dataset / "weights"]:
        p = base / f"{mp_safe}_L{L}.pt"
        if p.exists():
            return p
    return None


def _run_kgrw(dataset_dir: str, rule_file: str, out_adj: str,
              k: int, w_prime: int, seed: int) -> tuple[float, int]:
    """Run mprw_exec kgrw, return (algo_time_s, edge_count)."""
    cmd = f"cd /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/scalability_experiment && " \
          f"bin/mprw_exec kgrw {dataset_dir} {rule_file} {out_adj} {k} {w_prime} {seed}"
    result = subprocess.run(_wsl(cmd), capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"mprw_exec kgrw failed: {result.stderr[-300:]}")

    time_s = 0.0
    for line in result.stdout.split("\n"):
        if line.startswith("time:"):
            time_s = float(line.split(":", 1)[1].strip())

    # Count edges from adj file (each line: "u v1 v2 ...")
    # File is in WSL /tmp — read directly
    adj_path = out_adj.replace("/tmp/", "/tmp/")
    try:
        # Convert WSL path to Windows path for reading
        win_adj = out_adj  # /tmp/ is accessible via wsl
        count_cmd = f"awk '{{s += NF-1}} END {{print s+0}}' {out_adj}"
        cr = subprocess.run(_wsl(count_cmd), capture_output=True, text=True)
        edges = int(cr.stdout.strip()) if cr.stdout.strip().isdigit() else 0
    except Exception:
        edges = 0

    return time_s, edges


def _run_inference(adj_wsl_path: str, feat_file: str, weights_path: str,
                   z_out: str, labels_file: str, mask_file: str,
                   n_target: int, node_offset: int, in_dim: int,
                   num_classes: int, num_layers: int, label: str = "") -> dict:
    """Run inference_worker.py as subprocess, return metrics dict."""
    env = {**os.environ}
    stub = str(ROOT / "results" / "kgrw_eval" / "_stubs")
    env["PYTHONPATH"] = stub + os.pathsep + env.get("PYTHONPATH", "")

    # Convert WSL adj path to Windows path for inference_worker
    # /tmp/file.adj → \\wsl$\Ubuntu\tmp\file.adj  (not reliable cross-distro)
    # Instead, copy to a Windows-accessible temp location first
    win_adj = str(ROOT / "results" / "kgrw_eval" / f"tmp_{label}.adj")
    copy_cmd = f"cp {adj_wsl_path} /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/scalability_experiment/results/kgrw_eval/tmp_{label}.adj"
    subprocess.run(_wsl(copy_cmd), capture_output=True)

    cmd = [
        sys.executable, str(ROOT / "scripts" / "inference_worker.py"),
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
        "--num-layers",  str(num_layers),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    wall = time.perf_counter() - t0

    if res.returncode != 0:
        print(f"  [{label}] inference FAILED: {res.stderr[-200:]}")
        return {}

    out = {}
    for line in res.stdout.split("\n"):
        for key in ("inf_f1", "inf_peak_ram_mb", "inf_time", "inf_de"):
            if line.strip().lower().startswith(f"{key}:"):
                try:
                    out[key] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return out


def _load_reference(dataset: str, metapath: str, L: int) -> dict:
    """Load KMV and MPRW rows from master_results.csv for reference."""
    csv_path = ROOT / "results" / dataset / "master_results.csv"
    if not csv_path.exists():
        return {}

    ref = {"exact": None, "kmv": {}, "mprw": {}}
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("L", "")) != str(L):
                continue
            # Loose metapath match (first segment)
            mp_col = row.get("MetaPath", "")
            if mp_col and metapath.split(",")[0] not in mp_col:
                continue
            method = row.get("Method", "")
            try:
                edges = int(float(row.get("Edge_Count", 0) or 0))
                f1    = float(row.get("Macro_F1", 0) or 0)
                cka_col = f"CKA_L{L}"
                cka   = float(row.get(cka_col, 0) or 0)
                pa    = float(row.get("Pred_Similarity", 0) or 0)
                t_mat = float(row.get("Materialization_Time", 0) or 0)
            except (ValueError, TypeError):
                continue

            if method == "Exact" and ref["exact"] is None:
                ref["exact"] = {"edges": edges, "f1": f1, "cka": cka, "pa": pa, "time": t_mat}
            elif method == "KMV":
                k = row.get("k_value", "")
                if k:
                    key = str(int(float(k)))
                    if key not in ref["kmv"] or ref["kmv"][key]["f1"] < f1:
                        ref["kmv"][key] = {"edges": edges, "f1": f1, "cka": cka, "pa": pa, "time": t_mat}
            elif method == "MPRW":
                w_col = "w_value" if "w_value" in row else "Density_Matched_w"
                w = row.get(w_col, "")
                if w:
                    key = str(int(float(w)))
                    if key not in ref["mprw"] or ref["mprw"][key]["f1"] < f1:
                        ref["mprw"][key] = {"edges": edges, "f1": f1, "cka": cka, "pa": pa, "time": t_mat}
    return ref


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--L", type=int, nargs="+", default=[2])
    p.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    p.add_argument("--w-values", type=int, nargs="+", default=DEFAULT_W_PRIMES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of independent seeds to run (default 1). "
                        "Seeds are 42, 43, 44, ...")
    p.add_argument("--gen-exact", action="store_true",
                   help="Generate z_exact_<dataset>_L<L>.pt files from exact adj, then exit.")
    p.add_argument("--exact-adj", default=None,
                   help="Path to exact adjacency file (Windows path). "
                        "If omitted, looks in results/<dataset>/exact_papap.adj etc.")
    args = p.parse_args()
    args.seed_list = list(range(args.seed, args.seed + args.seeds))

    cfg         = config.get_dataset_config(args.dataset)
    folder      = config.get_folder_name(args.dataset)
    dataset_dir = folder  # relative, e.g. HGBn-DBLP
    rule_file   = f"{folder}/cod-rules_{folder}.limit"
    metapath    = cfg.metapaths[0] if hasattr(cfg, "metapaths") else ""

    # Infer metapath from rule file / config
    # (use whatever compile_rule_for_cpp last wrote)
    g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    num_classes = info["num_classes"]
    labels_full = info["labels"]
    n_target    = g[cfg.target_node].num_nodes
    in_dim      = g[cfg.target_node].x.size(1)
    sorted_ntypes = sorted(g.node_types)
    node_offset = sum(g[nt].num_nodes for nt in sorted_ntypes if nt < cfg.target_node)

    # Load partition
    part_path = ROOT / "results" / args.dataset / "partition.json"
    if not part_path.exists():
        part_path = ROOT / "results_new" / args.dataset / "partition.json"
    with open(part_path) as f:
        part = json.load(f)

    test_ids  = torch.tensor(part["test_node_ids"], dtype=torch.long)
    test_mask = torch.zeros(n_target, dtype=torch.bool)
    test_mask[test_ids] = True

    # Infer metapath name from weights directory
    weights_dir = ROOT / "results" / args.dataset / "weights"
    pt_files = list(weights_dir.glob("*.pt")) if weights_dir.exists() else []
    mp_safe = pt_files[0].stem.rsplit("_L", 1)[0] if pt_files else "unknown"
    metapath = mp_safe.replace("_", ",")

    print(f"\nDataset:  {args.dataset}")
    print(f"Target:   {cfg.target_node}  ({n_target} nodes, {test_mask.sum()} test)")
    print(f"Metapath: {metapath}")
    print(f"Depths:   {args.L}")
    print(f"k sweep:  {args.k_values}")
    print(f"w' sweep: {args.w_values}")

    # Save temp tensors
    tmp_dir = ROOT / "results" / "kgrw_eval"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    feat_file   = str(tmp_dir / f"x_{args.dataset}.pt")
    labels_file = str(tmp_dir / f"labels_{args.dataset}.pt")
    mask_file   = str(tmp_dir / f"mask_{args.dataset}.pt")
    torch.save(g[cfg.target_node].x.cpu(), feat_file)
    torch.save(labels_full.cpu(), labels_file)
    torch.save(test_mask.cpu(), mask_file)

    if args.gen_exact:
        # Find exact adj file
        exact_adj = args.exact_adj
        if exact_adj is None:
            candidates = sorted((ROOT / "results" / args.dataset).glob("exact_*.adj"))
            if not candidates:
                print("ERROR: no exact adj found. Pass --exact-adj <path>."); return
            exact_adj = str(candidates[0])
        # Convert to WSL path for inference
        exact_adj_wsl = exact_adj.replace("\\", "/").replace("C:/", "/mnt/c/")
        print(f"Generating z_exact files from {exact_adj}")
        for L in args.L:
            weights_path = _find_weights(args.dataset, mp_safe, L)
            if weights_path is None:
                print(f"  [SKIP] no weights for L={L}"); continue
            z_out = str(tmp_dir / f"z_exact_{args.dataset}_L{L}.pt")
            _run_inference(exact_adj_wsl, feat_file, str(weights_path), z_out,
                           labels_file, mask_file,
                           n_target, node_offset, in_dim, num_classes, L,
                           f"exact_{args.dataset}_L{L}")
            if os.path.exists(z_out):
                z = torch.load(z_out, weights_only=True)
                print(f"  L={L}: saved z_exact shape={z.shape}")
            else:
                print(f"  L={L}: FAILED to generate z_exact")
        print("Done. Now re-run bench_kgrw.py without --gen-exact.")
        return

    # Output CSV — seed is now a column for multi-seed aggregation
    out_csv = ROOT / "results" / args.dataset / "kgrw_bench.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = ["Dataset", "L", "Method", "k", "w_prime", "Seed",
                  "Edge_Count", "Mat_Time_s",
                  "Macro_F1", "CKA", "Pred_Agreement"]
    existing_runs: set[tuple] = set()

    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_runs.add((row["L"], row["Method"],
                                   row["k"], row["w_prime"],
                                   row.get("Seed", "0")))

    fout   = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=csv_fields)
    if out_csv.stat().st_size == 0:
        writer.writeheader()

    for L in args.L:
        weights_path = _find_weights(args.dataset, mp_safe, L)
        if weights_path is None:
            print(f"\n[SKIP] no weights found for L={L}"); continue

        z_exact_path = tmp_dir / f"z_exact_{args.dataset}_L{L}.pt"
        # Fallback to legacy name (no dataset prefix) for backwards compat
        if not z_exact_path.exists():
            z_exact_path = tmp_dir / f"z_exact_L{L}.pt"
        z_exact = None
        if z_exact_path.exists():
            loaded = torch.load(str(z_exact_path), weights_only=True)
            if loaded.shape[0] == n_target:
                z_exact = loaded
            else:
                print(f"  [WARN] z_exact shape {loaded.shape} != n_target={n_target}, skipping")

        print(f"\n{'='*65}  L={L}")
        print(f"  Weights: {weights_path.name}")
        print(f"  CKA reference: {'OK' if z_exact is not None else 'MISSING (run exact inference first)'}")
        print(f"\n  {'Method':<22} {'Edges':>8} {'F1':>7} {'CKA':>7} {'PA':>7}  (mean across {len(args.seed_list)} seed(s))")
        print(f"  {'-'*65}")

        # ── MPRW reference: run fresh on this meta-path across seeds ──────
        for w in args.w_values:
            mprw_metrics: list[dict] = []
            for seed in args.seed_list:
                run_key = (str(L), "MPRW", "", str(w), str(seed))
                if run_key in existing_runs:
                    continue
                adj_wsl = f"/tmp/mprw_bench_{w}_s{seed}_{args.dataset}.adj"
                cmd = (f"cd /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/"
                       f"scalability_experiment && "
                       f"bin/mprw_exec materialize {dataset_dir} {rule_file} "
                       f"{adj_wsl} {w} {seed}")
                r = subprocess.run(_wsl(cmd), capture_output=True, text=True, timeout=300)
                mat_time = 0.0
                for line in r.stdout.split("\n"):
                    if line.startswith("time:"):
                        mat_time = float(line.split(":", 1)[1].strip())
                cr = subprocess.run(_wsl(f"awk '{{s += NF-1}} END {{print s+0}}' {adj_wsl}"),
                                    capture_output=True, text=True)
                edges = int(cr.stdout.strip()) if cr.stdout.strip().isdigit() else 0

                label  = f"mprw_w{w}_s{seed}_L{L}"
                z_out  = str(tmp_dir / f"z_{label}.pt")
                inf    = _run_inference(adj_wsl, feat_file, str(weights_path), z_out,
                                        labels_file, mask_file,
                                        n_target, node_offset, in_dim,
                                        num_classes, L, label)
                f1 = inf.get("inf_f1", float("nan"))
                cka_val = pa_val = float("nan")
                if z_exact is not None and os.path.exists(z_out):
                    try:
                        from src.analysis.cka import LinearCKA
                        z_m  = torch.load(z_out, weights_only=True)
                        cka_val = float(LinearCKA(device=torch.device("cpu")).calculate(
                            z_exact[test_mask], z_m[test_mask]))
                        pa_val = (z_exact[test_mask].argmax(1) ==
                                  z_m[test_mask].argmax(1)).float().mean().item()
                    except Exception as e:
                        print(f"    CKA error (MPRW): {e}")

                writer.writerow({"Dataset": args.dataset, "L": L,
                                 "Method": "MPRW", "k": "", "w_prime": w, "Seed": seed,
                                 "Edge_Count": edges, "Mat_Time_s": round(mat_time, 4),
                                 "Macro_F1": round(f1, 6) if f1 == f1 else "",
                                 "CKA": round(cka_val, 6) if cka_val == cka_val else "",
                                 "Pred_Agreement": round(pa_val, 6) if pa_val == pa_val else ""})
                fout.flush()
                existing_runs.add(run_key)
                mprw_metrics.append({"f1": f1, "cka": cka_val, "pa": pa_val,
                                     "edges": edges, "time": mat_time})

        # ── KGRW sweep ────────────────────────────────────────────────────
        for k in args.k_values:
            for wp in args.w_values:
                kgrw_metrics: list[dict] = []
                for seed in args.seed_list:
                    run_key = (str(L), "KGRW", str(k), str(wp), str(seed))
                    if run_key in existing_runs:
                        continue
                    label   = f"kgrw_k{k}_wp{wp}_s{seed}_L{L}"
                    adj_wsl = f"/tmp/{label}_{args.dataset}.adj"
                    z_out   = str(tmp_dir / f"z_{label}.pt")
                    try:
                        mat_time, edges = _run_kgrw(dataset_dir, rule_file,
                                                    adj_wsl, k, wp, seed)
                    except Exception as e:
                        print(f"  KGRW k={k} w'={wp} s={seed}: FAILED ({e})"); continue

                    inf = _run_inference(adj_wsl, feat_file, str(weights_path), z_out,
                                        labels_file, mask_file,
                                        n_target, node_offset, in_dim, num_classes, L, label)
                    f1 = inf.get("inf_f1", float("nan"))
                    cka_val = pa_val = float("nan")
                    if z_exact is not None and os.path.exists(z_out):
                        try:
                            from src.analysis.cka import LinearCKA
                            z_k = torch.load(z_out, weights_only=True)
                            cka_val = float(LinearCKA(device=torch.device("cpu")).calculate(
                                z_exact[test_mask], z_k[test_mask]))
                            pa_val = (z_exact[test_mask].argmax(1) ==
                                      z_k[test_mask].argmax(1)).float().mean().item()
                        except Exception as e:
                            print(f"    CKA error (KGRW): {e}")

                    writer.writerow({"Dataset": args.dataset, "L": L,
                                     "Method": "KGRW", "k": k, "w_prime": wp, "Seed": seed,
                                     "Edge_Count": edges, "Mat_Time_s": round(mat_time, 4),
                                     "Macro_F1": round(f1, 6) if f1 == f1 else "",
                                     "CKA": round(cka_val, 6) if cka_val == cka_val else "",
                                     "Pred_Agreement": round(pa_val, 6) if pa_val == pa_val else ""})
                    fout.flush()
                    existing_runs.add(run_key)
                    kgrw_metrics.append({"f1": f1, "cka": cka_val, "pa": pa_val,
                                         "edges": edges, "time": mat_time})

        # ── Print aggregated summary from CSV ─────────────────────────────
        print(f"\n  --- Aggregated (mean ± std, {len(args.seed_list)} seeds) ---")
        import statistics
        all_rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
        def _agg(method, k_val, w_val):
            rows_ = [r for r in all_rows
                     if r["L"] == str(L) and r["Method"] == method
                     and r["k"] == str(k_val) and r["w_prime"] == str(w_val)]
            if not rows_: return None
            def ms(col):
                vals = [float(r[col]) for r in rows_ if r[col]]
                if not vals: return float("nan"), float("nan")
                return (statistics.mean(vals),
                        statistics.stdev(vals) if len(vals) > 1 else 0.0)
            edges = int(float(rows_[0]["Edge_Count"])) if rows_[0]["Edge_Count"] else 0
            f1m, f1s   = ms("Macro_F1")
            ckam, ckas = ms("CKA")
            pam, pas   = ms("Pred_Agreement")
            return edges, f1m, f1s, ckam, ckas, pam, pas

        for w in sorted(args.w_values):
            a = _agg("MPRW", "", w)
            if a:
                e,f,fs,c,cs,p,ps = a
                print(f"  {'MPRW w='+str(w):<22} {e:>8,} {f:.4f}±{fs:.3f} {c:.4f}±{cs:.3f} {p:.4f}±{ps:.3f}")
        for k in sorted(args.k_values):
            for w in sorted(args.w_values):
                a = _agg("KGRW", k, w)
                if a:
                    e,f,fs,c,cs,p,ps = a
                    print(f"  {'KGRW k='+str(k)+' w='+str(w):<22} {e:>8,} {f:.4f}±{fs:.3f} {c:.4f}±{cs:.3f} {p:.4f}±{ps:.3f}")

    fout.close()
    print(f"\nRaw rows: {out_csv}")


if __name__ == "__main__":
    main()
