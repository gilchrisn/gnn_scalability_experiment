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

        # Variable rules: -1 before last edge triggers the rule when the edge is pushed.
        # Instance rules: push all edges plainly, then chain -5 <id> after.
        # This matches the base paper's format (see misc/*.dat).
        if i == len(path_list) - 1 and instance_id == -1:
            parts.append("-1")

        parts.append(str(eid))

    if instance_id != -1:
        parts.extend(["-5", str(instance_id)])

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
    rules: list,  # List[Tuple[str, int]]  — (metapath_str, instance_id)
    g_hetero,
    data_dir: str,
    folder_name: str,
) -> int:
    """
    Compile ALL rules (variable + instance) into a single global rules file.

    The C++ binary reads one line from cod-rules_<folder>.limit per qnode.
    For centrality tasks (ExactD, GloD, etc.), it reads from
    <folder>-cod-global-rules.dat which has ALL rules on ONE line,
    separated by -4 (pop) opcodes.

    Returns the number of rules written.
    """
    from src.utils import SchemaMatcher

    sorted_edges = sorted(list(g_hetero.edge_types))
    edge_map = {et: i for i, et in enumerate(sorted_edges)}

    all_parts = []
    n_rules = 0

    for metapath_str, instance_id in rules:
        path_list = [s.strip() for s in metapath_str.split(',')]
        parts = []

        try:
            for i, rel_str in enumerate(path_list):
                direction = "-2"
                matched_edge = SchemaMatcher.match(rel_str, g_hetero)
                eid = edge_map[matched_edge]
                parts.append(direction)
                if i == len(path_list) - 1 and instance_id == -1:
                    parts.append("-1")
                parts.append(str(eid))
            if instance_id != -1:
                parts.extend(["-5", str(instance_id)])
            for _ in path_list:
                parts.append("-4")
            all_parts.extend(parts)
            n_rules += 1
        except (KeyError, RuntimeError) as e:
            print(f"[stage] skipping rule {metapath_str} (instance={instance_id}): {e}")
            continue

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

    Expected layout (hardcoded inside the C++ binary, written relative to cwd):
        global_res/<folder_name>/df1/   ← Degree centrality baselines
        global_res/<folder_name>/hf1/   ← H-Index centrality baselines

    The runner invokes the binary with cwd=<project_root>/staging, so these
    are created at ``staging/global_res/…`` on disk.

    Args:
        folder_name:  Dataset folder name (e.g. 'HGBn-ACM').
        project_root: Absolute path to the repository root.

    Returns:
        Tuple of (df1_dir, hf1_dir) absolute paths.
    """
    df1_dir = os.path.join(project_root, "staging", "global_res", folder_name, "df1")
    hf1_dir = os.path.join(project_root, "staging", "global_res", folder_name, "hf1")
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


# ---------------------------------------------------------------------------
# Unified edge counter (replaces the per-script `_count_edges` variants)
# ---------------------------------------------------------------------------
#
# Why this is centralised:
#   The three benchmark methods (Exact / KMV / MPRW) write their adjacency
#   files with three DIFFERENT conventions, which made the old "sum of NF-1"
#   counter return mutually-incomparable numbers and silently broke
#   cross-method density plots:
#
#     Exact (HUB/hg.cpp:25-46)
#         hidden_graph stored as std::vector<unsigned int> with push_back
#         and NO dedup → multi-edges. Paper-pair sharing 3 authors via the
#         meta-path appears 3 times in the adj. One-directional (only
#         active source nodes get a row).
#
#     KMV (sketch decode)
#         Bottom-k slots are deduped by hash value at merge time → set
#         semantics. At most k unique entries per source. One-directional.
#
#     MPRW (csrc/mprw_exec.cpp:349-358)
#         Sorted-vector dedup at every walk epoch → set semantics, AND
#         explicitly symmetrised at the end (for every (u,v), inserts
#         (v,u)). So sum-of-degrees is ≈ 2× unique-undirected-pair count.
#
# To make Edge_Count comparable across methods we collapse all three to
# the same canonical metric: |{ unordered {u, v} : v ∈ adj[u], u ≠ v }|.
# Self-loops are excluded because (a) MPRW already drops them, (b) Exact's
# hidden-graph rarely emits self-loops via the standard meta-path, and
# (c) they are never meaningful for the meta-path-induced density that
# downstream plots claim to measure.

def count_unique_undirected_edges(adj_path: str) -> int:
    """Count unique undirected non-self-loop edges in an adjacency-list file.

    File format: one line per source, "src nbr1 nbr2 ..." (whitespace-
    separated integers).

    The counter is uniform across Exact / KMV / MPRW outputs:
      - Exact's multi-edges → set-deduped to unique pairs.
      - KMV's set-edges → folded to undirected.
      - MPRW's pre-symmetrised adj → folded back to undirected (no double
        counting).

    Returns 0 if the file does not exist.
    """
    if not os.path.exists(adj_path):
        return 0
    pairs: set[tuple[int, int]] = set()
    with open(adj_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                src = int(parts[0])
            except ValueError:
                continue
            for nb in parts[1:]:
                try:
                    v = int(nb)
                except ValueError:
                    continue
                if v == src:
                    continue
                pairs.add((src, v) if src < v else (v, src))
    return len(pairs)


# awk one-liner that produces the same canonical count as
# count_unique_undirected_edges. Use this when the adj file lives on the
# WSL side and Python iteration would be too slow (OGB_MAG full-fraction
# adj files can be multi-GB). Plug it into the caller's _wsl() helper:
#
#     subprocess.run(_wsl(f"{AWK_UNIQUE_UNDIR_EDGES} {adj_wsl}"), ...)
#
# The associative array gives the same set semantics as the Python
# fallback above.
AWK_UNIQUE_UNDIR_EDGES = (
    "awk '{ src = $1; "
    "for (i = 2; i <= NF; i++) { v = $i; "
    "if (v == src) continue; "
    "key = (src+0 < v+0) ? src \"\\t\" v : v \"\\t\" src; "
    "pairs[key] = 1 } } "
    "END { print length(pairs)+0 }'"
)