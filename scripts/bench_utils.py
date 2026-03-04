"""
Shared utilities for C++ benchmarking scripts.

Eliminates copy-paste across test_table3.py, test_table4.py,
test_figure4.py, test_figure5_6.py, and run_prereqs.py.
"""
import os
import sys
import random
import subprocess
from typing import Optional

from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# Data Staging
# ---------------------------------------------------------------------------

def generate_qnodes(data_dir: str, folder_name: str, target_node_type: str,
                    g_hetero, max_scan: int = 50000, sample_size: int = 100) -> None:
    """
    Selects `sample_size` nodes of `target_node_type` as query nodes.
    Filtering by type is critical: the C++ engine traverses from qnodes as
    starting points, so qnodes of the wrong type silently yield 0 peers.
    """
    from src.utils import SchemaMatcher

    # Get the global ID range for target_node_type from the PyG graph.
    # PyGToCppAdapter assigns IDs by concatenating node types in sorted order.
    sorted_node_types = sorted(g_hetero.node_types)
    offset = sum(
        g_hetero[nt].num_nodes
        for nt in sorted_node_types
        if nt < target_node_type
    )
    n_target = g_hetero[target_node_type].num_nodes
    all_ids = [str(offset + i) for i in range(n_target)]

    selected = random.sample(all_ids, min(sample_size, len(all_ids)))

    qnode_path = os.path.join(data_dir, f"qnodes_{folder_name}.dat")
    with open(qnode_path, 'w') as f:
        f.write("\n".join(selected))

    print(f"   [Staging] Generated {len(selected)} query nodes "
          f"(type='{target_node_type}', global IDs {offset}–{offset+n_target-1}) "
          f"→ {qnode_path}")


def compile_rule_for_cpp(
    metapath_str: str,
    g_hetero: HeteroData,
    data_dir: str,
    folder_name: str,
    instance_id: int = -1,
) -> None:
    """
    Compiles a human-readable metapath string into C++ stack-machine bytecode
    and writes it to both expected file locations.

    The C++ engine is schizophrenic about filenames:
      - `hg_stats` reads:          cod-rules_<folder>.limit
      - Centrality tasks read:     <folder>-cod-global-rules.dat
    Both are written here so any command works without extra setup.

    Args:
        metapath_str: Comma-separated relation names (e.g. 'author_to_paper,paper_to_author').
        g_hetero:     The loaded PyG graph (needed for edge-type → integer mapping).
        data_dir:     Directory where rule files will be written.
        folder_name:  Dataset folder name (e.g. 'HGBn-DBLP').
        instance_id:  -1 for variable mode (default), else a specific node ID.
    """
    # Lazy import to keep this module importable without src on path
    from src.utils import SchemaMatcher

    sorted_edges = sorted(list(g_hetero.edge_types))
    edge_map = {et: i for i, et in enumerate(sorted_edges)}

    path_list = [s.strip() for s in metapath_str.split(',')]
    parts = []

    for i, rel_str in enumerate(path_list):
        direction = "-3" if rel_str.startswith("rev_") else "-2"
        matched_edge = SchemaMatcher.match(rel_str, g_hetero)
        try:
            eid = edge_map[matched_edge]
        except KeyError as e:
            raise RuntimeError(f"Mined rule contains edge {e} not found in schema.")

        parts.append(direction)

        # The termination opcode MUST come immediately before the final edge integer.
        # Opcode -1 sets state=VARIABLE; the very next integer both pushes the edge
        # AND fires the rule. Placing -1 after all edges (the old behaviour) meant
        # the C++ parser never saw a valid rule trigger → rule_count:0 everywhere.
        if i == len(path_list) - 1:
            if instance_id == -1:
                parts.append("-1")                      # Variable mode
            else:
                parts.extend(["-5", str(instance_id)]) # Instance mode

        parts.append(str(eid))

    for _ in path_list:
        parts.append("-4")                              # Pop stack

    rule_content = " ".join(parts) + "\n"

    file_limit = os.path.join(data_dir, f"cod-rules_{folder_name}.limit")
    file_dat   = os.path.join(data_dir, f"{folder_name}-cod-global-rules.dat")

    with open(file_limit, "w") as f:
        f.write(rule_content)
    with open(file_dat, "w") as f:
        f.write(rule_content)

    print(f"   [Staging] Rule bytecode written → {os.path.basename(file_limit)}, "
          f"{os.path.basename(file_dat)}")
    print(f"   rule content: {rule_content.strip()}  (instance_id={instance_id})")


def setup_global_res_dirs(folder_name: str, project_root: str):
    """
    Creates the rigid directory structure required by C++ exact-baseline tracking.

    Expected layout (hardcoded inside the C++ binary):
        global_res/<folder_name>/df1/   ← Degree centrality baselines
        global_res/<folder_name>/hf1/   ← H-Index centrality baselines

    Args:
        folder_name:  Dataset folder name (e.g. 'HGBn-ACM').
        project_root: Absolute path to the repository root.

    Returns:
        Tuple of (df1_dir, hf1_dir) absolute paths.
    """
    df1_dir = os.path.join(project_root, "global_res", folder_name, "df1")
    hf1_dir = os.path.join(project_root, "global_res", folder_name, "hf1")
    os.makedirs(df1_dir, exist_ok=True)
    os.makedirs(hf1_dir, exist_ok=True)
    return df1_dir, hf1_dir


# ---------------------------------------------------------------------------
# C++ Execution
# ---------------------------------------------------------------------------

def run_cpp(
    binary: str,
    args: list,
    redirect_path: Optional[str] = None,
    timeout: int = 1200,
    print_output: bool = True,
) -> str:
    """
    Runs the C++ binary, optionally redirects stdout to a file, and returns
    the raw stdout string.

    Raises SystemExit on non-zero return codes (fatal C++ crashes).

    Args:
        binary:        Path to the graph_prep executable.
        args:          List of CLI arguments (task name first).
        redirect_path: If set, stdout is written to this file.
        timeout:       Subprocess timeout in seconds.
        print_output:  Whether to echo the raw C++ stdout to the terminal.

    Returns:
        Raw stdout string from the C++ process.
    """
    cmd = [binary] + args
    print(f"  [Engine] Executing: {' '.join(cmd)}")
    if redirect_path:
        print(f"           Redirecting stdout → {redirect_path}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] C++ Crashed! Exit Code: {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        print(f"--- STDERR ---\n{e.stderr.strip()}")
        print(f"--- STDOUT ---\n{e.stdout.strip()}")
        sys.exit(1)

    if redirect_path:
        with open(redirect_path, 'w') as f:
            f.write(result.stdout)

    if print_output:
        print("\n" + "-" * 40)
        print(f"--- RAW C++ STDOUT ({args[0]}) ---")
        print("-" * 40)
        label = result.stdout.strip() if result.stdout.strip() else "[NO OUTPUT — REDIRECTED TO FILE]"
        print(label)
        print("-" * 40)

    return result.stdout