"""
test_instance_metapath.py — Select, mirror, and validate top AnyBURL rules,
including ALL instance rule forms rejected by the existing parser.

Why a new script instead of modifying test_rule_inventory.py
-------------------------------------------------------------
The existing script has three structural gaps (Open/Closed Principle — extend,
don't modify):

  1. Parser silently rejects rules where the *first* body argument is a grounded
     node, e.g.  rel(paper_1638, X) <= ...   (Case C below).
  2. Rule selection uses separate variable/instance pools with independent caps
     instead of a single confidence-ranked top-N.
  3. Length-1 instance rules are excluded, even though they become length-2 after
     mirroring and are valid for materialization.

AnyBURL instance rule taxonomy handled here
-------------------------------------------
  Case A  rel(X, Y) <= ...              variable rule          instance_id = -1
  Case B  ... <= rel(X, paper_1284)     tail-grounded (v2)     instance_id = 1284
  Case C  ... <= rel(paper_1638, X)     head-grounded (v1)     instance_id = 1638
  Case D  mixed multi-atom paths combining any of the above

Mirroring (target-anchored)
--------------------------
  AnyBURL body paths often start at a non-target node (e.g. venue, paper).
  Before mirroring, the path is trimmed to begin at the first edge whose
  source is the target node type (e.g. "author" for DBLP).

  venue->paper->author->paper  -trim->  author->paper  -mirror->  APA
  paper->author->paper->venue  -trim->  author->paper->venue  -mirror->  APVPA

Pipeline
--------
  1. Stage dataset (PyGToCppAdapter + qnodes).
  2. Parse anyburl_rules.txt with extended parser.
  3. Select top-N rules by confidence (unified pool, no type separation).
  4. Mirror each rule.
  5. Compile to C++ bytecode → run hg_stats.
  6. Report: peer_size > 0 AND raw_edges > 0 → PASS.

Usage
-----
    python scripts/test_instance_metapath.py HGB_DBLP
    python scripts/test_instance_metapath.py HGB_DBLP --top-n 10
    python scripts/test_instance_metapath.py HGB_DBLP --min-conf 0.05
    python scripts/test_instance_metapath.py HGB_DBLP --skip-stage
    python scripts/test_instance_metapath.py HGB_DBLP --binary ./bin/graph_prep.exe
"""
from __future__ import annotations

import os
import re
import sys
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, run_cpp


# ===========================================================================
# Data model
# ===========================================================================

@dataclass
class Rule:
    """
    Parsed AnyBURL rule.

    Attributes:
        path:        Comma-separated relation names representing the body path,
                     e.g. "term_to_paper" or "paper_to_term,term_to_paper".
        confidence:  Rule confidence (parts[2] in the raw file).
        instance_id: -1 for variable rules; ≥ 0 for instance-anchored rules.
        raw_line:    Original text line for debugging.
    """
    path:        str
    confidence:  float
    instance_id: int  = -1
    raw_line:    str  = field(default="", repr=False)

    @property
    def rule_type(self) -> str:
        return "instance" if self.instance_id >= 0 else "variable"

    @property
    def length(self) -> int:
        return len(self.path.split(","))


@dataclass
class ValidationResult:
    rule:          Rule
    mirrored_path: str
    peer_size:     float = 0.0
    raw_edges:     float = 0.0
    density:       float = 0.0
    elapsed_s:     float = 0.0
    passed:        bool  = False
    skip_reason:   str   = ""


# ===========================================================================
# Extended AnyBURL parser
# ===========================================================================

def _extract_instance_id(node_str: str) -> int:
    """
    Extracts the integer suffix from a grounded AnyBURL node string.

    Examples:
        "paper_1284"  →  1284
        "conf_7"      →  7
    """
    return int(node_str.rsplit("_", 1)[-1])


def _is_instance(token: str) -> bool:
    """Returns True if the token is a grounded node (starts lowercase)."""
    return bool(token) and token[0].islower()


def _parse_line(line: str, min_conf: float) -> Optional[Rule]:
    """
    Parses a single AnyBURL output line into a Rule.

    Handles all four instance/variable combinations:

      Case A  rel(X, Y) <= rel(X, Y)             → variable rule
      Case B  rel(X, Y) <= rel(X, paper_1284)    → tail-grounded
      Case C  rel(X, Y) <= rel(paper_1638, X)    → head-grounded  ← previously rejected
      Case D  multi-atom combinations of B and C

    Returns None if the line is malformed, below min_conf, or disjointed.
    """
    line = line.strip()
    parts = line.split("\t")
    if len(parts) < 4:
        return None

    try:
        conf = float(parts[2])
    except ValueError:
        return None

    if conf < min_conf:
        return None

    rule_str = parts[3]

    try:
        body = rule_str.split(" <= ")[1]
    except IndexError:
        return None

    atoms      = body.split(", ")
    relations  = []
    current_var: Optional[str] = None
    instance_id: int = -1           # Will be set on first grounded node encountered

    for i, atom in enumerate(atoms):
        m = re.match(r'([a-zA-Z0-9_]+)\(([a-zA-Z0-9_]+),([a-zA-Z0-9_]+)\)', atom)
        if not m:
            return None

        rel, v1, v2 = m.groups()
        v1_inst = _is_instance(v1)
        v2_inst = _is_instance(v2)

        # ------------------------------------------------------------------
        # First atom — establish the traversal anchor
        # ------------------------------------------------------------------
        if i == 0:
            if v1_inst and v2_inst:
                # Both arguments grounded: uninterpretable traversal
                return None

            if v1_inst:
                # Case C: head-grounded  rel(paper_1638, X)
                # Anchor = paper_1638, walking variable = X
                instance_id = _extract_instance_id(v1)
                current_var = v2
                relations.append(rel)
                # If v2 is also a terminal instance we're done
                if v2_inst:
                    break
                continue

            # v1 is a variable
            current_var = v1

        # ------------------------------------------------------------------
        # Subsequent atoms — extend the path following current_var
        # ------------------------------------------------------------------
        if current_var == v1:
            relations.append(rel)
            if v2_inst:
                instance_id = _extract_instance_id(v2)
                break       # Instance rule: path terminates here
            current_var = v2

        elif current_var == v2:
            relations.append(f"rev_{rel}")
            if v1_inst:
                instance_id = _extract_instance_id(v1)
                break
            current_var = v1

        else:
            # Disjointed atoms — invalid path
            return None

    if not relations:
        return None

    # Variable rules must have length ≥ 2 to be non-trivial before mirroring.
    # Instance rules can be length 1 — they become length 2 after mirroring.
    if instance_id == -1 and len(relations) < 2:
        return None

    return Rule(
        path        = ",".join(relations),
        confidence  = conf,
        instance_id = instance_id,
        raw_line    = line,
    )


def parse_anyburl_file(filepath: str, min_conf: float = 0.0) -> List[Rule]:
    """
    Parses every line of an AnyBURL rules file and returns all valid Rule objects.

    Args:
        filepath: Path to anyburl_rules.txt.
        min_conf: Minimum confidence threshold; lines below are skipped.

    Returns:
        List of Rule objects (unsorted).
    """
    rules: List[Rule] = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            r = _parse_line(line, min_conf)
            if r is not None:
                rules.append(r)
    return rules


def select_top_n(rules: List[Rule], n: int) -> List[Rule]:
    """
    Returns the top-N rules by confidence from the *unified* pool.

    No separation between variable and instance rules — the highest-confidence
    rules win regardless of type.

    Args:
        rules: Parsed rules from parse_anyburl_file().
        n:     Maximum number of rules to return.

    Returns:
        List of at most n rules, sorted descending by confidence.
    """
    return sorted(rules, key=lambda r: r.confidence, reverse=True)[:n]


# ===========================================================================
# Mirroring: A-B-C  →  A-B-C-B-A
# ===========================================================================

def mirror_path(
    path_str:         str,
    g_hetero,
    target_node_type: Optional[str] = None,
) -> Optional[str]:
    """
    Appends the full reverse of `path_str` to itself, producing a symmetric path.

    If `target_node_type` is given, the path is first trimmed to begin at the
    *first edge whose source matches that type*.  This is essential when AnyBURL
    body paths start from a non-target node type (e.g. venue or paper), which
    would make the C++ engine's qnode traversal produce empty results.

    Example with target_node_type="author":
        "venue_to_paper,rev_author_to_paper,author_to_paper"
        → schema types: venue→paper, paper→author, author→paper
        → first edge with src="author" is index 2 ("author_to_paper")
        → trimmed path: "author_to_paper"
        → mirrored:     "author_to_paper,paper_to_author"   (APA)

        "paper_to_author,author_to_paper,rev_venue_to_paper"
        → schema types: paper→author, author→paper, paper→venue
        → first edge with src="author" is index 1 ("author_to_paper")
        → trimmed path: "author_to_paper,rev_venue_to_paper"
        → mirrored:     "author_to_paper,rev_venue_to_paper,venue_to_paper,paper_to_author"
                        (APVPA)

    For each edge in the reversed suffix:
      - Forward edge A→B  reverses to B→A  (named schema lookup; fallback "rev_<rel>").
      - Reverse edge "rev_X" reverses to the forward direction (drop the "rev_" prefix).

    Args:
        path_str:          Comma-separated relation names from the AnyBURL body.
        g_hetero:          Loaded PyG HeteroData (for schema lookup).
        target_node_type:  If set, trim path to start here before mirroring.

    Returns:
        Mirrored path string (starting at target_node_type when given), or None if:
        - any relation fails schema lookup
        - path is not contiguous
        - target_node_type never appears as an edge source in the path
        - trimmed path would be empty
    """
    from src.utils import SchemaMatcher

    rels     = path_str.split(",")
    resolved = []           # List of (rel_str, actual_src, actual_dst)

    for rel_str in rels:
        edge = SchemaMatcher.match(rel_str, g_hetero)
        if edge[0] == "node":
            return None     # Schema lookup failed

        schema_src, _, schema_dst = edge
        if rel_str.startswith("rev_"):
            # Physical traversal is reversed relative to the schema tuple
            actual_src, actual_dst = schema_dst, schema_src
        else:
            actual_src, actual_dst = schema_src, schema_dst

        resolved.append((rel_str, actual_src, actual_dst))

    # Contiguity check on actual traversal endpoints
    for i in range(len(resolved) - 1):
        if resolved[i][2] != resolved[i + 1][1]:
            return None

    # Trim to first edge starting at the target node type
    if target_node_type is not None:
        trim_idx = next(
            (i for i, (_, src, _) in enumerate(resolved) if src == target_node_type),
            None,
        )
        if trim_idx is None:
            return None     # target_node_type never appears as a traversal source
        resolved = resolved[trim_idx:]

    if not resolved:
        return None

    # Build reverse suffix
    suffix = []
    for rel_str, actual_src, actual_dst in reversed(resolved):
        if rel_str.startswith("rev_"):
            # Was traversing actual_dst → actual_src; flip = forward direction
            suffix.append(rel_str[4:])
        else:
            # Was traversing actual_src → actual_dst; flip = actual_dst → actual_src
            named = next(
                (r for s, r, d in g_hetero.edge_types if s == actual_dst and d == actual_src),
                None,
            )
            suffix.append(named if named else f"rev_{rel_str}")

    trimmed_path = ",".join(r for r, _, _ in resolved)
    return trimmed_path + "," + ",".join(suffix)


# ===========================================================================
# C++ validation
# ===========================================================================

def validate_rule(
    rule:        Rule,
    mirrored:    str,
    g_hetero,
    data_dir:    str,
    folder_name: str,
    binary:      str,
) -> ValidationResult:
    """
    Compiles `mirrored` to C++ bytecode, runs hg_stats, and parses the output.

    A rule PASSES if both peer_size > 0 AND raw_edges > 0.

    Args:
        rule:        The original (pre-mirror) Rule object.
        mirrored:    The mirrored path string to materialize.
        g_hetero:    Loaded PyG HeteroData.
        data_dir:    Directory containing staged C++ data files.
        folder_name: Dataset folder name (e.g. "HGBn-DBLP").
        binary:      Path to the graph_prep executable.

    Returns:
        ValidationResult with materialization statistics.
    """
    compile_rule_for_cpp(
        metapath_str = mirrored,
        g_hetero     = g_hetero,
        data_dir     = data_dir,
        folder_name  = folder_name,
        instance_id  = rule.instance_id,
    )

    t0     = time.perf_counter()
    stdout = run_cpp(binary, ["hg_stats", folder_name], print_output=False)
    elapsed = time.perf_counter() - t0

    edges_matches = re.findall(r"RAW_EDGES_E\*:\s*([0-9.eE+\-]+)", stdout)
    density_match = re.search(r"~dens:\s*([0-9.eE+\-]+)",           stdout)
    peer_match    = re.search(r"~\|peer\|:\s*([0-9.eE+\-]+)",       stdout)

    raw_edges = (
        sum(float(x) for x in edges_matches) / len(edges_matches)
        if edges_matches else 0.0
    )
    density   = float(density_match.group(1)) if density_match else 0.0
    peer_size = float(peer_match.group(1))    if peer_match    else 0.0

    return ValidationResult(
        rule          = rule,
        mirrored_path = mirrored,
        peer_size     = peer_size,
        raw_edges     = raw_edges,
        density       = density,
        elapsed_s     = elapsed,
        passed        = peer_size > 0 and raw_edges > 0,
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select, mirror (A-B-C → A-B-C-B-A), and validate top AnyBURL rules.\n"
            "Handles ALL instance rule forms, including head-grounded body atoms."
        )
    )
    parser.add_argument("dataset",       type=str,   help="Dataset key, e.g. HGB_DBLP")
    parser.add_argument("--top-n",       type=int,   default=10,
                        help="Number of top rules to test (unified confidence pool, default 10)")
    parser.add_argument("--min-conf",    type=float, default=0.0,
                        help="Minimum confidence threshold for rule inclusion")
    parser.add_argument("--binary",      type=str,   default=config.CPP_EXECUTABLE,
                        help="Path to the graph_prep binary")
    parser.add_argument("--skip-stage",  action="store_true",
                        help="Skip PyGToCppAdapter conversion (dataset already staged)")
    args = parser.parse_args()

    ds_name     = args.dataset
    folder_name = f"HGBn-{ds_name.split('_')[1]}"
    data_dir    = os.path.join(project_root, folder_name)

    base_data  = getattr(config, "DATA_DIR", os.path.join(project_root, "datasets"))
    mining_dir = os.path.join(base_data, f"mining_{ds_name}")
    raw_file   = os.path.join(mining_dir, "anyburl_rules.txt")

    print(f"\n{'='*62}")
    print(f"  Instance Metapath Test: {ds_name}")
    print(f"  Top-N: {args.top_n}  |  min-conf: {args.min_conf}")
    print(f"{'='*62}")

    # -----------------------------------------------------------------------
    # 1. Load graph
    # -----------------------------------------------------------------------
    from src.data import DatasetFactory
    from src.bridge import PyGToCppAdapter

    cfg = config.get_dataset_config(ds_name)
    print(f"\n[1/4] Loading graph ({cfg.source}/{cfg.dataset_name})...")
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    print(f"      Edge types  : {[et[1] for et in g_hetero.edge_types]}")
    print(f"      Target node : {cfg.target_node}")

    # -----------------------------------------------------------------------
    # 2. Stage C++ data
    # -----------------------------------------------------------------------
    if not args.skip_stage:
        print(f"\n[2/4] Staging C++ files → {data_dir}/")
        PyGToCppAdapter(data_dir).convert(g_hetero)
        generate_qnodes(data_dir, folder_name, target_node_type=cfg.target_node, g_hetero=g_hetero)
    else:
        print(f"\n[2/4] Staging skipped (--skip-stage).")

    # -----------------------------------------------------------------------
    # 3. Parse + select top-N (unified pool)
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Parsing {raw_file} ...")
    if not os.path.exists(raw_file):
        print(f"      [FATAL] File not found. Run AnyBURL mining first.")
        sys.exit(1)

    all_rules = parse_anyburl_file(raw_file, min_conf=args.min_conf)

    n_var       = sum(1 for r in all_rules if r.rule_type == "variable")
    n_inst_l1   = sum(1 for r in all_rules if r.rule_type == "instance" and r.length == 1)
    n_inst_long = sum(1 for r in all_rules if r.rule_type == "instance" and r.length > 1)

    print(f"      Total parsed          : {len(all_rules)}")
    print(f"        Variable            : {n_var}")
    print(f"        Instance (length=1) : {n_inst_l1}   ← mirrored → length-2, testable")
    print(f"        Instance (length>1) : {n_inst_long}")

    selected = select_top_n(all_rules, args.top_n)
    print(f"\n      Top-{args.top_n} by confidence (unified pool):")
    print(f"      {'Rank':>4}  {'Type':>8}  {'Conf':>7}  {'Len':>3}  {'Anchor':>8}  Path")
    print(f"      {'─'*70}")
    for rank, r in enumerate(selected, 1):
        anchor = f"@{r.instance_id}" if r.instance_id >= 0 else "variable"
        print(f"      {rank:>4}  {r.rule_type:>8}  {r.confidence:>7.4f}  "
              f"{r.length:>3}  {anchor:>8}  {r.path}")

    # -----------------------------------------------------------------------
    # 4. Mirror + validate
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Mirroring (A-B-C → A-B-C-B-A) + validating via hg_stats ...")
    results: List[ValidationResult] = []

    for rule in selected:
        tag = "[V]" if rule.rule_type == "variable" else f"[I@{rule.instance_id}]"
        print(f"\n  {tag} Original : {rule.path}")

        mirrored = mirror_path(rule.path, g_hetero, target_node_type=cfg.target_node)
        if mirrored is None:
            vr = ValidationResult(
                rule          = rule,
                mirrored_path = rule.path,
                skip_reason   = (
                    f"mirror failed: path never starts at '{cfg.target_node}' "
                    f"or schema lookup failed"
                ),
            )
            results.append(vr)
            print(f"       SKIP  → {vr.skip_reason}")
            continue

        # Show trimmed prefix if the original was longer than the mirrored base
        trimmed_base = mirrored.split(",")[: len(mirrored.split(",")) // 2]
        if ",".join(trimmed_base) != rule.path:
            print(f"       Trimmed : {','.join(trimmed_base)}  (trimmed to '{cfg.target_node}' start)")
        print(f"       Mirrored: {mirrored}")
        vr = validate_rule(rule, mirrored, g_hetero, data_dir, folder_name, args.binary)
        results.append(vr)

        status = "PASS ✔" if vr.passed else "FAIL ✘  → matching graph is EMPTY"
        print(f"       {status}")
        print(f"       peer={vr.peer_size:.0f}  |E*|={vr.raw_edges:.1f}  "
              f"dens={vr.density:.6f}  ({vr.elapsed_s:.2f}s)")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    passed  = [r for r in results if r.passed]
    failed  = [r for r in results if not r.passed and not r.skip_reason]
    skipped = [r for r in results if r.skip_reason]

    print(f"\n{'='*62}")
    print(f"  Summary")
    print(f"{'='*62}")
    print(f"  Tested   : {len(results) - len(skipped)}")
    print(f"  PASSED   : {len(passed)}")
    print(f"  FAILED   : {len(failed)}  (matching graph was empty)")
    print(f"  SKIPPED  : {len(skipped)}  (mirror failed)")

    if passed:
        print(f"\n  Non-empty matching graphs:")
        print(f"  {'Type':>12}  {'Conf':>6}  {'Peer':>8}  {'|E*|':>10}  "
              f"{'Dens':>10}  Mirrored Path")
        print(f"  {'─'*80}")
        for r in passed:
            anchor = f"var" if r.rule.rule_type == "variable" else f"@{r.rule.instance_id}"
            print(f"  {anchor:>12}  {r.rule.confidence:>6.4f}  "
                  f"{r.peer_size:>8.0f}  {r.raw_edges:>10.1f}  "
                  f"{r.density:>10.6f}  {r.mirrored_path}")

    if failed:
        print(f"\n  Empty graphs (investigate):")
        for r in failed:
            anchor = f"@{r.rule.instance_id}" if r.rule.instance_id >= 0 else "variable"
            print(f"    [{anchor}]  {r.mirrored_path}")

    if skipped:
        print(f"\n  Mirror failures:")
        for r in skipped:
            print(f"    {r.rule.path}")
            print(f"      → {r.skip_reason}")

    print()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()