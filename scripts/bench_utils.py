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

    print(f"[stage] qnodes: {len(selected)} nodes of type '{target_node_type}' (IDs {offset}–{offset+n_target-1})")


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
        # Always use forward traversal (-2): GraphStandardizer stores ALL edges
        # (including rev_* ones) as explicit forward entries in link.dat, so
        # reverse traversal (-3) is never needed and breaks OGB-style datasets.
        direction = "-2"
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

    print(f"[stage] rule: {rule_content.strip()}  (instance_id={instance_id})")


def compile_all_rules_for_cpp(
    rules: list,  # List[Tuple[str, int, str]]  — (metapath_str, instance_id, instance_node_type)
    g_hetero,
    data_dir: str,
    folder_name: str,
) -> int:
    """
    Compile ALL rules into a single global rules file matching the original
    paper's format.

    The original format groups rules by path pattern:
      <path_setup> -1 <last_edge> -5 <inst1> -5 <inst2> ... -4 -4

    This builds the path (N-1 edges), fires the variable rule (-1 + last edge),
    then chains instance rules (-5 <id>) that reuse the same accumulated ETypes.
    All on ONE line.

    Returns the number of rules written.
    """
    from src.utils import SchemaMatcher
    from collections import OrderedDict

    sorted_edges = sorted(list(g_hetero.edge_types))
    edge_map = {et: i for i, et in enumerate(sorted_edges)}

    # Compute global node ID offsets per type (alphabetical order)
    sorted_ntypes = sorted(g_hetero.node_types)
    ntype_offset = {}
    offset = 0
    for nt in sorted_ntypes:
        ntype_offset[nt] = offset
        offset += g_hetero[nt].num_nodes

    # Group rules by path pattern: {metapath_str: {"var": bool, "instances": [(global_id)]}}
    grouped = OrderedDict()
    for metapath_str, instance_id, inst_ntype in rules:
        if metapath_str not in grouped:
            grouped[metapath_str] = {"var": False, "instances": []}
        if instance_id == -1:
            grouped[metapath_str]["var"] = True
        else:
            # Convert type-local instance ID to global ID
            global_id = instance_id + ntype_offset.get(inst_ntype, 0)
            grouped[metapath_str]["instances"].append(global_id)

    all_parts = []
    n_rules = 0

    for metapath_str, group in grouped.items():
        path_list = [s.strip() for s in metapath_str.split(',')]
        if len(path_list) < 2:
            continue  # skip 1-hop (can't build path setup)

        try:
            eids = []
            for rel_str in path_list:
                matched_edge = SchemaMatcher.match(rel_str, g_hetero)
                eids.append(edge_map[matched_edge])
        except (KeyError, RuntimeError) as e:
            print(f"[stage] skipping path {metapath_str}: {e}")
            continue

        # Original paper format:
        #   <setup edges> -1 <dir> <last_edge> -5 <inst1> -5 <inst2> ... -4 -4
        #
        # The -1 (variable) ALWAYS fires first — it pushes the last edge
        # into ETypes so that NTypes.size() == len(path). Then -5 instances
        # chain on the same accumulated ETypes.
        parts = []

        # Setup: first N-1 edges
        for eid in eids[:-1]:
            parts.append("-2")
            parts.append(str(eid))

        # Variable trigger with last edge (always needed to build full ETypes)
        parts.append("-1")
        parts.append("-2")
        parts.append(str(eids[-1]))
        if group["var"]:
            n_rules += 1

        # Instance rules chain on same ETypes
        for inst_id in group["instances"]:
            parts.append("-5")
            parts.append(str(inst_id))
            n_rules += 1

        # Pop stack (one per hop)
        for _ in path_list:
            parts.append("-4")

        all_parts.extend(parts)

    rule_content = " ".join(all_parts) + "\n"

    file_limit = os.path.join(data_dir, f"cod-rules_{folder_name}.limit")
    file_dat   = os.path.join(data_dir, f"{folder_name}-cod-global-rules.dat")

    with open(file_limit, "w") as f:
        f.write(rule_content)
    with open(file_dat, "w") as f:
        f.write(rule_content)

    print(f"[stage] compiled {n_rules} rules into {file_dat}")
    return n_rules


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
    print(f"\n> {' '.join(args)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
    except subprocess.CalledProcessError as e:
        msg = f"C++ exited {e.returncode}: {' '.join(args)}"
        print(f"\n[ERROR] {msg}")
        if e.stderr: print(f"STDERR: {e.stderr.strip()}")
        raise RuntimeError(msg)
    except subprocess.TimeoutExpired:
        msg = f"C++ timed out after {timeout}s: {' '.join(args)}"
        print(f"\n[TIMEOUT] {msg}")
        raise RuntimeError(msg)

    if redirect_path:
        with open(redirect_path, 'w') as f:
            f.write(result.stdout)

    if print_output:
        out = result.stdout.strip()
        print(out if out else "[no output]")

    return result.stdout