"""
Figure 4: Sensitivity to |E*| — Scatter Plot Data Extraction.

Runs KMV approximate centrality methods (and optionally the BoolAP baseline)
across a list of metapaths.  Each metapath produces one scatter-plot data
point per method: (active_edges, time_s).

Usage — single metapath:
    python scripts/test_figure4.py HGB_ACM --metapath "author_to_paper,paper_to_author"

Usage — multiple metapaths from a file (one per line):
    python scripts/test_figure4.py HGB_ACM --metapaths-file paths/dblp_metapaths.txt

Add the BoolAP baseline (requires compiled BoolAPCoreD binary):
    python scripts/test_figure4.py HGB_ACM \\
        --metapaths-file paths/dblp_metapaths.txt \\
        --baseline \\
        --boolap-binary parallel-k-P-core-decomposition-code/BoolAPCoreD
"""
import argparse
import os
import re
import sys
from typing import List, Optional

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import (
    PyGToCppAdapter,
    GraphPrepRunner,
    BoolAPConverter,
    BoolAPRunner,
    BoolAPFiles,
    BoolAPResult,
)
from scripts.bench_utils import (
    compile_rule_for_cpp,
    generate_qnodes,
    setup_global_res_dirs,
)


_KMV_METHODS = [
    ("glo",  "GloD",  dict(k=32)),
    ("per",  "PerD",  dict(k=32)),
    ("per",  "PerD+", dict(k=32, beta=0.1)),
    ("glo",  "GloH",  dict(k=4)),
    ("per",  "PerH",  dict(k=4)),
    ("per",  "PerH+", dict(k=4, beta=0.1)),
]

_BOOLAP_METHOD_LABEL = "BoolAPCoreD"

_SCATTER_RE = re.compile(r"SCATTER_DATA:\s*([0-9.]+),([0-9.]+),([0-9.]+)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Raw scatter-plot data extraction for Figure 4."
    )
    p.add_argument("dataset",  type=str,
                   help="Dataset key (e.g. HGB_ACM, HGB_DBLP).")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--metapath",       type=str,
                       help="Single comma-separated metapath string.")
    group.add_argument("--metapaths-file", type=str,
                       help="Path to a text file with one metapath per line.")

    p.add_argument("--topr",          type=str, default="0.05")
    p.add_argument("--baseline",      action="store_true",
                   help="Also run BoolAPCoreD as a baseline.")
    p.add_argument("--boolap-binary", type=str, default=None,
                   help="Path to the compiled BoolAPCoreD binary "
                        "(required when --baseline is set).")
    return p


def _load_metapaths(args: argparse.Namespace) -> List[str]:
    """Return the list of metapaths from CLI args (single string or file)."""
    if args.metapath:
        return [args.metapath]
    with open(args.metapaths_file, "r") as fh:
        return [line.strip() for line in fh if line.strip()]


def _parse_scatter(stdout: str, method: str) -> Optional[tuple]:
    """
    Extract the first SCATTER_DATA line from stdout.

    Returns:
        (rule_id, edges, time_s) strings, or None if the token is absent.
    """
    m = _SCATTER_RE.search(stdout)
    if m is None:
        print(f"  [WARNING] No SCATTER_DATA for {method}. Check binary output.")
        return None
    return m.group(1), m.group(2), m.group(3)


def _run_kmv_methods(
    runner: GraphPrepRunner,
    folder_name: str,
    topr: str,
) -> List[tuple]:
    """
    Run all six KMV methods and return a list of (method, rule_id, edges, time_s).

    Args:
        runner:      Initialised GraphPrepRunner.
        folder_name: Dataset folder name (e.g. 'HGBn-ACM').
        topr:        Top-R threshold string.

    Returns:
        List of (method_label, rule_id, edges, time_s) for each method that
        produced a valid SCATTER_DATA line.
    """
    rows: List[tuple] = []
    for kind, task, kwargs in _KMV_METHODS:
        if kind == "glo":
            result = runner.run_glo(task, folder_name, topr=topr, **kwargs)
        else:
            result = runner.run_per(task, folder_name, topr=topr, **kwargs)

        point = _parse_scatter(result.stdout, task)
        if point is not None:
            rows.append((task, *point))
    return rows


def _run_boolap_baseline(
    boolap_runner: BoolAPRunner,
    boolap_converter: BoolAPConverter,
    g_hetero,
    metapath: str,
    folder_name: str,
    data_dir: str,
    reference_edges: Optional[str],
) -> Optional[tuple]:
    """
    Convert the graph, run BoolAPCoreD, and return a result row.

    The x-axis value (edges) is taken from GloD's SCATTER_DATA so that BoolAP
    and KMV data points share a common x-axis (same metapath → same |E*|).

    Args:
        boolap_runner:    Initialised BoolAPRunner.
        boolap_converter: Initialised BoolAPConverter.
        g_hetero:         Loaded PyG HeteroData.
        metapath:         Comma-separated metapath string.
        folder_name:      Dataset folder name used as file prefix.
        data_dir:         Directory for BoolAP input files.
        reference_edges:  Edge count string from KMV SCATTER_DATA (x-axis).

    Returns:
        (method_label, rule_id, edges, time_s) or None on failure.
    """
    try:
        files = boolap_converter.convert(g_hetero, metapath, prefix=folder_name)
        result: BoolAPResult = boolap_runner.run(files)
        edges = reference_edges if reference_edges is not None else "0"
        return (_BOOLAP_METHOD_LABEL, "0", edges, f"{result.total_time_s:.6f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARNING] BoolAP run failed: {exc}")
        return None


def main() -> None:
    """Entry point: stage dataset, loop over metapaths, print scatter table."""
    args   = _build_parser().parse_args()
    metapaths = _load_metapaths(args)

    if args.baseline and not args.boolap_binary:
        print("[ERROR] --baseline requires --boolap-binary to be set.")
        sys.exit(1)

    # ---- One-time dataset staging ----------------------------------------
    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir    = os.path.join(project_root, folder_name)

    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)

    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name,
                    target_node_type=cfg.target_node, g_hetero=g_hetero)
    setup_global_res_dirs(folder_name, project_root)

    runner = GraphPrepRunner(
        binary=config.CPP_EXECUTABLE,
        working_dir=project_root,
        verbose=True,
    )

    boolap_converter: Optional[BoolAPConverter] = None
    boolap_runner:    Optional[BoolAPRunner]    = None
    if args.baseline:
        boolap_converter = BoolAPConverter(out_dir=data_dir)
        boolap_runner    = BoolAPRunner(binary_path=args.boolap_binary)

    # ---- Print header -------------------------------------------------------
    print(f"\n[FIGURE 4]  {args.dataset}  topr={args.topr}")
    print(f"  {'Method':<14}  {'mp_idx':>6}  {'edges':>10}  {'time_s':>10}")
    print(f"  {'-' * 46}")

    # ---- Per-metapath loop --------------------------------------------------
    all_rows: List[tuple] = []

    for mp_idx, metapath in enumerate(metapaths):
        print(f"\n  -- metapath {mp_idx}: {metapath}")

        compile_rule_for_cpp(metapath, g_hetero, data_dir, folder_name)
        runner.run_ground_truth(folder_name, topr=args.topr)

        kmv_rows = _run_kmv_methods(runner, folder_name, args.topr)
        for row in kmv_rows:
            all_rows.append((row[0], mp_idx, row[2], row[3]))

        # Use GloD edges as shared x-axis for BoolAP
        glod_edges = next(
            (row[2] for row in kmv_rows if row[0] == "GloD"), None
        )

        if args.baseline:
            boolap_row = _run_boolap_baseline(
                boolap_runner, boolap_converter,
                g_hetero, metapath, folder_name, data_dir,
                reference_edges=glod_edges,
            )
            if boolap_row is not None:
                all_rows.append((boolap_row[0], mp_idx, boolap_row[2], boolap_row[3]))

    # ---- Print results ------------------------------------------------------
    for method, mp_idx, edges, time_s in all_rows:
        print(f"  {method:<14}  {mp_idx:>6}  {edges:>10}  {time_s:>10}")


if __name__ == "__main__":
    main()
