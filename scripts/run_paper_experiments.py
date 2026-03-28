"""
Master experiment runner for paper reproduction.

Mines (or loads cached) metapaths for a dataset, validates them through the
full pipeline (parse → normalize → mirror → validate → dedup), then runs every
paper experiment across all valid metapaths in a single command.

Output  →  results/<DATASET>/
    table4.csv   — per-method F1 + time for each metapath  (Table IV)
    figure4.csv  — per-method (|E*|, time) scatter          (Figure 4)
    figure5.csv  — per-method F1/acc vs lambda sweep        (Figure 5)
    figure6.csv  — per-method F1/acc vs k sweep             (Figure 6)
    run_<timestamp>.log — full debug log for verification

Resume-safe: metapaths already in table4.csv are skipped on re-run.
This means you can Ctrl+C and restart without losing progress.

Usage:
    python scripts/run_paper_experiments.py HGB_ACM
    python scripts/run_paper_experiments.py HGB_DBLP --max-metapaths 50
    python scripts/run_paper_experiments.py HGB_ACM  --skip-sweeps
    python scripts/run_paper_experiments.py HGB_ACM  --force-remine
    python scripts/run_paper_experiments.py HGB_ACM  --min-conf 0.05
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import (
    PyGToCppAdapter,
    GraphPrepRunner,
    AnyBURLRunner,
    load_validated_metapaths,
    BoolAPConverter,
    BoolAPRunner,
)
from scripts.bench_utils import (
    compile_rule_for_cpp,
    generate_qnodes,
    setup_global_res_dirs,
    run_cpp,
)


DEFAULT_TOPR  = "0.05"
DEFAULT_MIN_CONF = 0.1          # paper uses 0.1 (not 0.001)
LAMBDAS       = ["0.02", "0.03", "0.04", "0.05"]
K_VALUES      = [2, 4, 8, 16, 32]
DEFAULT_K_DEG = 32
DEFAULT_K_HID = 4
BETA          = 0.1

# (kind, task, default_k, beta_or_None)
_METHOD_SPECS: List[Tuple[str, str, int, Optional[float]]] = [
    ("glo", "GloD",  DEFAULT_K_DEG, None),
    ("per", "PerD",  DEFAULT_K_DEG, None),
    ("per", "PerD+", DEFAULT_K_DEG, BETA),
    ("glo", "GloH",  DEFAULT_K_HID, None),
    ("per", "PerH",  DEFAULT_K_HID, None),
    ("per", "PerH+", DEFAULT_K_HID, BETA),
]

_SCATTER_RE = re.compile(r"SCATTER_DATA:\s*[0-9.]+,([0-9.]+),([0-9.]+)")

_T3_FIELDS = ["dataset", "metapath", "avg_edges", "peer_size", "density"]
_T4_FIELDS = ["dataset", "metapath", "method", "f1_or_acc", "avg_time_s", "rule_count"]
_F4_FIELDS = ["dataset", "metapath", "method", "edges", "time_s"]
_F5_FIELDS = ["dataset", "metapath", "method", "lambda", "f1_or_acc"]
_F6_FIELDS = ["dataset", "metapath", "method", "k", "f1_or_acc"]


def _setup_logging(out_dir: Path, dataset: str) -> logging.Logger:
    """
    Set up a logger with two handlers:
      - Console (INFO):  clean one-line progress messages
      - File   (DEBUG):  full timestamped detail for verification
    """
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"run_{ts}.log"

    logger = logging.getLogger(f"paper_experiments.{dataset}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_file}")
    return logger


def _open_csv(path: Path, fields: List[str]) -> Tuple:
    """Open CSV in append mode; write header only if the file is new/empty."""
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if is_new:
        writer.writeheader()
    return fh, writer


def _done_metapaths(path: Path) -> Set[str]:
    """Return the set of metapath strings already written to a CSV file."""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["metapath"] for row in csv.DictReader(fh)}


def _done_metapaths_for_method(path: Path, method: str) -> Set[str]:
    """Return metapaths that already have a row for a specific method."""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["metapath"] for row in csv.DictReader(fh)
                if row.get("method") == method}


def _failed_metapaths(path: Path) -> Set[str]:
    """Return metapaths permanently marked FAILED (timed out / OOM on a previous run)."""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["metapath"] for row in csv.DictReader(fh)
                if row.get("method", "").startswith("FAILED")}


_KMV_METHODS = {"GloD", "PerD", "PerD+", "GloH", "PerH", "PerH+"}


def _done_kmv_metapaths(path: Path) -> Set[str]:
    """Return metapaths where all 6 KMV methods are already present."""
    if not path.exists():
        return set()
    from collections import defaultdict
    counts: dict = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = row.get("method", "")
            if m in _KMV_METHODS:
                counts[row["metapath"]].add(m)
    return {mp for mp, methods in counts.items() if methods >= _KMV_METHODS}


def _call(runner: GraphPrepRunner, kind: str, task: str, folder: str,
          topr: str, k: int, beta: Optional[float]):
    """Dispatch to run_glo or run_per with the correct arguments."""
    if kind == "glo":
        return runner.run_glo(task, folder, topr=topr, k=k)
    kwargs = {"k": k}
    if beta is not None:
        kwargs["beta"] = beta
    return runner.run_per(task, folder, topr=topr, **kwargs)


def _score(result) -> float:
    return result.avg_f1 if hasattr(result, "avg_f1") else result.avg_accuracy


def _run_hg_stats(
    dataset:  str,
    folder:   str,
    metapath: str,
    t3_w:     csv.DictWriter,
    log:      logging.Logger,
) -> None:
    """Run hg_stats for one metapath and write a row to table3 CSV."""
    stdout = run_cpp(config.CPP_EXECUTABLE, ["hg_stats", folder],
                     print_output=False, timeout=600)
    edges_m = re.findall(r"RAW_EDGES_E\*:\s*([0-9.eE+\-]+)", stdout)
    peer_m  = re.search(r"~\|peer\|:\s*([0-9.eE+\-]+)",       stdout)
    dens_m  = re.search(r"~dens:\s*([0-9.eE+\-]+)",            stdout)

    if not edges_m or not dens_m:
        log.warning("  table3 | hg_stats: no parseable output for %s", metapath[:60])
        return

    avg_edges = sum(float(m) for m in edges_m) / len(edges_m)
    peer_size = float(peer_m.group(1)) if peer_m else float("nan")
    density   = float(dens_m.group(1))

    t3_w.writerow({
        "dataset":   dataset,
        "metapath":  metapath,
        "avg_edges": round(avg_edges, 2),
        "peer_size": round(peer_size, 2),
        "density":   round(density, 8),
    })
    log.debug("  table3 | avg_edges=%.1f  peer_size=%.1f  density=%.6f",
              avg_edges, peer_size, density)


def _run_main(
    runner:   GraphPrepRunner,
    dataset:  str,
    folder:   str,
    metapath: str,
    topr:     str,
    t4_w:     csv.DictWriter,
    f4_w:     csv.DictWriter,
    log:      logging.Logger,
) -> None:
    """
    Ground truth + all 6 KMV methods at default k.
    Writes one row per method to both table4 and figure4 CSVs.
    """
    gt = runner.run_ground_truth(folder, topr=topr)

    # --- Record WcD / WcH exact baseline times ---
    for exact_task in ["ExactD", "ExactH"]:
        # Skip h-index exact baseline if ground truth is unavailable
        if "H" in exact_task and gt.hf1_inclusive is None:
            log.debug("  table4 | WcH    SKIPPED (no h-index ground truth)")
            continue
        wc_name = "WcD" if "D" in exact_task else "WcH"
        exact_r = runner.run_exact(exact_task, folder, topr=topr)
        wc_time = None
        for line in exact_r.stdout.split('\n'):
            line = line.strip().lower()
            if line.startswith("time:"):
                try:
                    wc_time = float(line.split(":")[1].strip().replace(" s", ""))
                except ValueError:
                    pass
        if wc_time is not None:
            t4_w.writerow({
                "dataset":    dataset,
                "metapath":   metapath,
                "method":     wc_name,
                "f1_or_acc":  1.0,
                "avg_time_s": round(wc_time, 6),
                "rule_count": 1,
            })
            log.debug("  table4 | %-6s  f1/acc=1.0000  time=%.4fs  (exact baseline)", wc_name, wc_time)

    for kind, task, k, beta in _METHOD_SPECS:
        # Skip h-index methods if ground truth is unavailable
        if "H" in task and gt.hf1_inclusive is None:
            log.debug("  table4 | %-6s  SKIPPED (no h-index ground truth)", task)
            continue
        r = _call(runner, kind, task, folder, topr, k, beta)
        score = _score(r)

        t4_w.writerow({
            "dataset":    dataset,
            "metapath":   metapath,
            "method":     task,
            "f1_or_acc":  round(score, 6),
            "avg_time_s": round(r.avg_time_s, 6),
            "rule_count": r.rule_count,
        })

        log.debug(
            "  table4 | %-6s  f1/acc=%.4f  time=%.4fs  rules=%d",
            task, score, r.avg_time_s, r.rule_count,
        )

        m = _SCATTER_RE.search(r.stdout)
        if m:
            f4_w.writerow({
                "dataset":  dataset,
                "metapath": metapath,
                "method":   task,
                "edges":    m.group(1),
                "time_s":   m.group(2),
            })
            log.debug("  figure4 | %-6s  edges=%s  time_s=%s", task, m.group(1), m.group(2))
        else:
            log.debug("  figure4 | %-6s  NO SCATTER_DATA in stdout", task)


def _run_sweeps(
    runner:   GraphPrepRunner,
    dataset:  str,
    folder:   str,
    metapath: str,
    f5_w:     csv.DictWriter,
    f6_w:     csv.DictWriter,
    log:      logging.Logger,
) -> None:
    """
    Lambda sweep (Figure 5) and k sweep (Figure 6).
    Uses default k values per method for lambda sweep;
    uses the swept k for all methods in the k sweep.
    """
    log.debug("  sweeps: lambda in %s", LAMBDAS)

    # Figure 5 — vary lambda, k fixed at method defaults
    for lam in LAMBDAS:
        runner.run_ground_truth(folder, topr=lam)
        for kind, task, k, beta in _METHOD_SPECS:
            r = _call(runner, kind, task, folder, lam, k, beta)
            f5_w.writerow({
                "dataset":   dataset,
                "metapath":  metapath,
                "method":    task,
                "lambda":    lam,
                "f1_or_acc": round(_score(r), 6),
            })

    log.debug("  sweeps: k in %s", K_VALUES)

    # Figure 6 — vary k, lambda fixed at default
    runner.run_ground_truth(folder, topr=DEFAULT_TOPR)
    for k in K_VALUES:
        for kind, task, _, beta in _METHOD_SPECS:
            r = _call(runner, kind, task, folder, DEFAULT_TOPR, k, beta)
            f6_w.writerow({
                "dataset":   dataset,
                "metapath":  metapath,
                "method":    task,
                "k":         k,
                "f1_or_acc": round(_score(r), 6),
            })


def _run_boolap_table4(
    converter:    BoolAPConverter,
    boolap_run:   BoolAPRunner,
    g_hetero,
    dataset:      str,
    folder:       str,
    metapath:     str,
    method_label: str,
    t4_w:         csv.DictWriter,
    log:          logging.Logger,
) -> None:
    """Run one BoolAP binary for Table IV; silently skips on any error."""
    try:
        prefix = f"{folder}_{method_label}"
        files  = converter.convert(g_hetero, metapath, prefix)
        result = boolap_run.run(files, timeout=600)
        t4_w.writerow({
            "dataset":    dataset,
            "metapath":   metapath,
            "method":     method_label,
            "f1_or_acc":  1.0,          # exact baseline — always correct
            "avg_time_s": round(result.total_time_s, 6),
            "rule_count": 1,
        })
        log.debug("  table4 | %-8s  time=%.4fs  (BoolAP exact)", method_label, result.total_time_s)
    except Exception as exc:
        log.warning("  table4 | %s skipped — %s", method_label, exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all paper experiments for a dataset across all metapaths."
    )
    parser.add_argument("dataset",
                        help="Dataset key (e.g. HGB_ACM, HGB_DBLP).")
    parser.add_argument("--max-metapaths", type=int, default=500,
                        help="Cap on number of metapaths to process (default 500).")
    parser.add_argument("--topr",          type=str, default=DEFAULT_TOPR,
                        help="Top-R threshold for Table IV and Figure 4 (default 0.05).")
    parser.add_argument("--min-conf",      type=float, default=DEFAULT_MIN_CONF,
                        help="Min AnyBURL confidence for metapath selection "
                             "(default 0.1 — matches original paper).")
    parser.add_argument("--skip-sweeps",   action="store_true",
                        help="Skip Figure 5 & 6 sweeps. Much faster; produces Table IV + Fig 4 only.")
    parser.add_argument("--force-remine",  action="store_true",
                        help="Re-run AnyBURL even if cached rules exist.")
    parser.add_argument("--force-mine",    action="store_true",
                        help="Use AnyBURL mining even if config has suggested_paths.")
    parser.add_argument("--mining-timeout", type=int, default=10,
                        help="AnyBURL snapshot timeout in seconds (default 10 — matches paper).")
    parser.add_argument("--timeout",           type=int,   default=600,
                        help="Per-C++-call subprocess timeout in seconds (default 600). "
                             "Metapaths that exceed this are marked FAILED and skipped on resume.")
    parser.add_argument("--boolap-binary",     type=str, default=None,
                        help="Path to BoolAPCoreD binary (adds BoolAP row to Table IV).")
    parser.add_argument("--boolap-plus-binary", type=str, default=None,
                        help="Path to BoolAPCoreG binary (adds BoolAP+ row to Table IV).")
    args = parser.parse_args()

    dataset  = args.dataset
    folder   = config.get_folder_name(dataset)
    data_dir = os.path.join(project_root, folder)
    out_dir  = Path(project_root) / "results" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logging(out_dir, dataset)

    log.info("")
    log.info("=" * 60)
    log.info("  Paper Experiments: %s", dataset)
    log.info("  topr=%-6s  min_conf=%-5s  max_metapaths=%d  mining_timeout=%ds  timeout=%ds",
             args.topr, args.min_conf, args.max_metapaths, args.mining_timeout, args.timeout)
    log.info("  sweeps: %s", "disabled (--skip-sweeps)" if args.skip_sweeps else "enabled")
    log.info("  Output → %s/", out_dir)
    log.info("=" * 60)

    # ---- Load graph --------------------------------------------------------
    log.info("")
    log.info("[1/4] Loading graph...")
    cfg           = config.get_dataset_config(dataset)
    g_hetero, _   = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    log.info("      node types  : %s", [nt for nt in g_hetero.node_types])
    log.info("      edge types  : %s", [et[1] for et in g_hetero.edge_types])
    log.info("      target node : %s", cfg.target_node)

    # ---- Load metapaths (config first, then mine if needed) -----------------
    log.info("")
    log.info("[2/4] Loading + validating metapaths...")

    # Prefer pre-configured metapaths from config (curated, known to work)
    # --force-mine overrides this for HGB datasets that need mining
    if cfg.suggested_paths and not args.force_mine:
        metapaths = list(cfg.suggested_paths)
        log.info("      Using %d pre-configured metapaths from config.", len(metapaths))
        for mp in metapaths:
            log.info("        %s", mp)
    else:
        # Fallback: mine via AnyBURL
        work_dir   = os.path.join(config.DATA_DIR, f"mining_{dataset}")
        anyburl    = AnyBURLRunner(work_dir, config.ANYBURL_JAR)
        anyburl.export_for_mining(g_hetero)

        rules_file = os.path.join(work_dir, "anyburl_rules.txt")
        if args.force_remine or not os.path.exists(rules_file):
            log.info("      Mining metapaths via AnyBURL (snapshot at %ds)...", args.mining_timeout)
            anyburl.run_mining(timeout=args.mining_timeout, max_length=6, num_threads=4)
        else:
            log.info("      Using cached rules (pass --force-remine to re-run).")

        log.info("      Running validation pipeline (parse → normalize → mirror → validate)...")
        metapaths, stats = load_validated_metapaths(
            rules_file=rules_file,
            g_hetero=g_hetero,
            target_node=cfg.target_node,
            min_conf=args.min_conf,
            max_n=args.max_metapaths,
        )
        # Filter out long metapaths (>4 hops)
        max_hops = 4
        before = len(metapaths)
        metapaths = [mp for mp in metapaths if len(mp.split(",")) <= max_hops]
        if len(metapaths) < before:
            log.info("      Filtered %d metapaths with >%d hops (%d remaining)",
                     before - len(metapaths), max_hops, len(metapaths))

        log.info("")
        log.info("  Metapath validation summary:")
        log.info("    total parsed lines   : %d", stats["total_parsed"])
        log.info("    after conf >= %-5s  : %d", args.min_conf, stats["after_conf_filter"])
        log.info("    unique raw paths     : %d", stats["unique_raw_paths"])
        log.info("    valid mirrored paths : %d", stats["valid_mirrored"])
        log.info("    failed (schema miss) : %d", stats["fail_schema"])
        log.info("    failed (no target)   : %d", stats["fail_trim"])
        log.info("    failed (asymmetric)  : %d", stats["fail_symmetry"])
        log.info("    returned (capped)    : %d  (cap=%d)", stats["returned"], args.max_metapaths)
    log.info("")

    if not metapaths:
        log.error("[ERROR] No valid metapaths found.")
        log.error("  Try: --force-remine, lower --min-conf, or inspect with:")
        log.error("  python scripts/inspect_mined_paths.py %s --min-conf %s", dataset, args.min_conf)
        sys.exit(1)

    # ---- Stage C++ data (once) ---------------------------------------------
    log.info("[3/4] Staging C++ data → %s/", data_dir)
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder, target_node_type=cfg.target_node, g_hetero=g_hetero)
    setup_global_res_dirs(folder, project_root)

    runner = GraphPrepRunner(
        binary=config.CPP_EXECUTABLE,
        working_dir=project_root,
        timeout=args.timeout,
        verbose=False,
    )

    # Optional BoolAP baselines
    boolap_conv = BoolAPConverter(data_dir) if (args.boolap_binary or args.boolap_plus_binary) else None
    boolap_run  = BoolAPRunner(args.boolap_binary)      if args.boolap_binary      else None
    boolap_plus = BoolAPRunner(args.boolap_plus_binary) if args.boolap_plus_binary else None
    if boolap_run or boolap_plus:
        log.info("      BoolAP baselines: %s",
                 ", ".join(filter(None, [
                     "BoolAP"  if boolap_run  else None,
                     "BoolAP+" if boolap_plus else None,
                 ])))

    # ---- Open CSV files (resume-safe) --------------------------------------
    log.info("")
    log.info("[4/4] Running experiments...")
    t3_path = out_dir / "table3.csv"
    t4_path = out_dir / "table4.csv"
    f4_path = out_dir / "figure4.csv"
    f5_path = out_dir / "figure5.csv"
    f6_path = out_dir / "figure6.csv"

    done_kmv       = _done_kmv_metapaths(t4_path)
    done_failed    = _failed_metapaths(t4_path)
    done_boolap    = _done_metapaths_for_method(t4_path, "BoolAP")  if boolap_run  else set()
    done_boolap_p  = _done_metapaths_for_method(t4_path, "BoolAP+") if boolap_plus else set()
    done_t3        = _done_metapaths(t3_path)
    total          = len(metapaths)
    n_done         = sum(1 for mp in metapaths
                         if mp in done_kmv
                         and (not boolap_run  or mp in done_boolap)
                         and (not boolap_plus or mp in done_boolap_p))
    n_todo  = total - n_done
    log.info("      %d/%d fully done, %d to run (or partially complete).", n_done, total, n_todo)
    log.info("")

    t3_fh, t3_w = _open_csv(t3_path, _T3_FIELDS)
    t4_fh, t4_w = _open_csv(t4_path, _T4_FIELDS)
    f4_fh, f4_w = _open_csv(f4_path, _F4_FIELDS)
    f5_fh, f5_w = _open_csv(f5_path, _F5_FIELDS)
    f6_fh, f6_w = _open_csv(f6_path, _F6_FIELDS)

    n_failed = 0

    try:
        for idx, metapath in enumerate(metapaths, start=1):
            need_kmv      = metapath not in done_kmv and metapath not in done_failed
            need_boolap   = boolap_run  and metapath not in done_boolap
            need_boolap_p = boolap_plus and metapath not in done_boolap_p

            if not need_kmv and not need_boolap and not need_boolap_p:
                log.info("  [%3d/%d] skip  %s", idx, total, metapath[:70])
                continue

            parts = (["KMV"] if need_kmv else []) + \
                    (["BoolAP"] if need_boolap else []) + \
                    (["BoolAP+"] if need_boolap_p else [])
            log.info("  [%3d/%d] run %s  %s", idx, total, "+".join(parts), metapath[:70])
            log.debug("         full metapath: %s", metapath)

            try:
                compile_rule_for_cpp(metapath, g_hetero, data_dir, folder)

                if need_kmv:
                    if metapath not in done_t3:
                        try:
                            _run_hg_stats(dataset, folder, metapath, t3_w, log)
                            t3_fh.flush()
                        except Exception as e:
                            log.warning("  hg_stats failed (%s) — continuing with experiments", e)

                    _run_main(runner, dataset, folder, metapath, args.topr, t4_w, f4_w, log)
                    t4_fh.flush()
                    f4_fh.flush()

                    if not args.skip_sweeps:
                        _run_sweeps(runner, dataset, folder, metapath, f5_w, f6_w, log)
                        f5_fh.flush()
                        f6_fh.flush()

                if need_boolap:
                    _run_boolap_table4(boolap_conv, boolap_run,  g_hetero, dataset,
                                       folder, metapath, "BoolAP",  t4_w, log)
                    t4_fh.flush()
                if need_boolap_p:
                    _run_boolap_table4(boolap_conv, boolap_plus, g_hetero, dataset,
                                       folder, metapath, "BoolAP+", t4_w, log)
                    t4_fh.flush()

            except SystemExit as exc:
                n_failed += 1
                reason = f"FAILED:CRASH(exit={exc.code})"
                log.warning("  [%3d/%d] %s: %s", idx, total, reason, metapath[:70])
                log.debug("Full traceback:", exc_info=True)
                if need_kmv:
                    t4_w.writerow({"dataset": dataset, "metapath": metapath,
                                   "method": reason, "f1_or_acc": "", "avg_time_s": "", "rule_count": ""})
                    t4_fh.flush()
                    done_failed.add(metapath)
            except Exception as exc:
                n_failed += 1
                exc_str = str(exc)
                if "timed out" in exc_str:
                    reason = f"FAILED:TIMEOUT({args.timeout}s)"
                elif "bad_alloc" in exc_str or "SIGKILL" in exc_str or "Cannot allocate" in exc_str:
                    reason = "FAILED:OOM"
                else:
                    reason = f"FAILED:{exc_str[:80]}"
                log.warning("  [%3d/%d] %s: %s", idx, total, reason, metapath[:70])
                log.debug("Full traceback:", exc_info=True)
                if need_kmv:
                    t4_w.writerow({"dataset": dataset, "metapath": metapath,
                                   "method": reason, "f1_or_acc": "", "avg_time_s": "", "rule_count": ""})
                    t4_fh.flush()
                    done_failed.add(metapath)

    finally:
        for fh in [t3_fh, t4_fh, f4_fh, f5_fh, f6_fh]:
            fh.close()

    # ---- Summary -----------------------------------------------------------
    n_ok = total - n_done - n_failed
    log.info("")
    log.info("=" * 60)
    log.info("  Done.  Results in %s/", out_dir)
    log.info("    metapaths : %d total | %d run | %d skipped (resume) | %d failed",
             total, n_ok, n_done, n_failed)
    log.info("    table3.csv  (%d metapaths — |E*| and density)", total)
    log.info("    table4.csv  (%d metapaths x 6 methods%s)", total,
             " + BoolAP baselines" if (boolap_run or boolap_plus) else "")
    log.info("    figure4.csv (scatter: |E*| vs time)")
    if not args.skip_sweeps:
        sweeps = f"{len(LAMBDAS)} lambdas + {len(K_VALUES)} k values"
        log.info("    figure5.csv (F1 vs lambda — %s)", sweeps)
        log.info("    figure6.csv (F1 vs k)")
    if n_failed:
        log.warning("  %d metapath(s) failed — check log for details.", n_failed)
    log.info("=" * 60)
    log.info("")


if __name__ == "__main__":
    main()
