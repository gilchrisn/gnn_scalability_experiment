"""
diagnose_instance_mode.py
=========================
Targeted diagnosis: do instance-mode rules (-5 flag) execute differently
from variable-mode rules (-1 flag) in the C++ engine?

QUESTIONS THIS SCRIPT ANSWERS
------------------------------
Q1. Does a known-good variable path (APA) produce peers?
    → Baseline sanity check. Must PASS or nothing else means anything.

Q2. Does the same path in instance mode (-5 <paper_id>) produce peers?
    → Tests whether instance mode ignores qnodes and anchors at the fixed node.

Q3. Does instance mode fail if the instance_id node type does NOT match
    the path's start type?
    → Detects whether wrong-type anchors silently return 0.

Q4. Does trimming (starting path mid-way through, at a non-start type) work
    in instance mode?
    → The fix applied to variable rules may be wrong for instance rules.

Q5. Does instance mode with a path that starts at 'paper' (the instance's
    type) and goes to 'author' produce peers?
    → Confirms what the correct path orientation should be for instance rules.

HOW TO USE
----------
Run AFTER staging HGB_DBLP (i.e. after test_rule_inventory.py or
test_instance_metapath.py has run once so HGBn-DBLP/ exists):

    python scripts/diagnose_instance_mode.py HGB_DBLP --skip-stage

Or with fresh staging:

    python scripts/diagnose_instance_mode.py HGB_DBLP

INTERPRETING RESULTS
--------------------
Each test prints: path, instance_id used, bytecode written, peer/edge counts.
Compare Q1 vs Q2-Q5 to understand instance mode semantics.
"""
from __future__ import annotations

import os
import re
import sys
import argparse
import time
from typing import Optional, Tuple

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, run_cpp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_and_measure(binary: str, folder_name: str) -> Tuple[float, float, float, float]:
    """Runs hg_stats and returns (peer_size, raw_edges, density, elapsed_s)."""
    t0     = time.perf_counter()
    stdout = run_cpp(binary, ["hg_stats", folder_name], print_output=False)
    elapsed = time.perf_counter() - t0

    edges_matches = re.findall(r"RAW_EDGES_E\*:\s*([0-9.]+)", stdout)
    density_match = re.search(r"~dens:\s*([0-9.]+)", stdout)
    peer_match    = re.search(r"~\|peer\|:\s*([0-9.]+)", stdout)

    raw_edges = (
        sum(float(x) for x in edges_matches) / len(edges_matches)
        if edges_matches else 0.0
    )
    density   = float(density_match.group(1)) if density_match else 0.0
    peer_size = float(peer_match.group(1))    if peer_match    else 0.0
    return peer_size, raw_edges, density, elapsed


def run_test(
    label:       str,
    path:        str,
    instance_id: int,
    g_hetero,
    data_dir:    str,
    folder_name: str,
    binary:      str,
) -> bool:
    """Compile + run one test case. Returns True (peer>0 and edges>0)."""
    mode = f"instance@{instance_id}" if instance_id >= 0 else "variable"
    print(f"\n{'─'*62}")
    print(f"  [{label}]")
    print(f"  path        : {path}")
    print(f"  mode        : {mode}")

    compile_rule_for_cpp(
        metapath_str = path,
        g_hetero     = g_hetero,
        data_dir     = data_dir,
        folder_name  = folder_name,
        instance_id  = instance_id,
    )

    peer, edges, dens, t = _run_and_measure(binary, folder_name)
    passed = peer > 0 and edges > 0
    status = "PASS ✔" if passed else "FAIL ✘"
    print(f"  peer={peer:.0f}  |E*|={edges:.1f}  dens={dens:.6f}  ({t:.2f}s)  → {status}")
    return passed


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

def run_diagnosis(
    g_hetero,
    data_dir:    str,
    folder_name: str,
    binary:      str,
) -> None:
    """
    Runs 5 targeted tests to characterise instance vs variable execution.

    We use DBLP edge types:
        author_to_paper  (author → paper)
        paper_to_author  (paper  → author)
        paper_to_term    (paper  → term)
        term_to_paper    (term   → paper)
        paper_to_venue   (paper  → venue)
        venue_to_paper   (venue  → paper)

    For instance tests we use paper node 0 as the anchor (always exists).
    """

    results = {}

    # ------------------------------------------------------------------
    # Q1. Baseline: variable APA path — must PASS or the engine is broken
    # ------------------------------------------------------------------
    results["Q1_var_APA"] = run_test(
        label       = "Q1  variable APA (baseline — must PASS)",
        path        = "author_to_paper,paper_to_author",
        instance_id = -1,
        g_hetero    = g_hetero,
        data_dir    = data_dir,
        folder_name = folder_name,
        binary      = binary,
    )

    # ------------------------------------------------------------------
    # Q2. Instance mode with the same APA path, anchored at paper_0
    #     The path starts at 'author' but the anchor is a paper node.
    #     If instance mode ignores qnodes and anchors at paper_0, then
    #     author_to_paper starting from a paper node makes no sense →
    #     we expect FAIL (wrong-type anchor).
    # ------------------------------------------------------------------
    results["Q2_inst_APA_wrong_anchor"] = run_test(
        label       = "Q2  instance mode: APA path, paper_0 anchor (type mismatch expected FAIL)",
        path        = "author_to_paper,paper_to_author",
        instance_id = 0,          # paper node 0
        g_hetero    = g_hetero,
        data_dir    = data_dir,
        folder_name = folder_name,
        binary      = binary,
    )

    # ------------------------------------------------------------------
    # Q3. Instance mode with paper→author path, anchored at paper_0
    #     Path starts at 'paper', anchor is paper_0 → type matches.
    #     If the engine traverses from paper_0 → authors, we expect PASS.
    # ------------------------------------------------------------------
    results["Q3_inst_paper_to_author"] = run_test(
        label       = "Q3  instance mode: paper_to_author (L=1), paper_0 anchor (type match)",
        path        = "paper_to_author",
        instance_id = 0,          # paper node 0
        g_hetero    = g_hetero,
        data_dir    = data_dir,
        folder_name = folder_name,
        binary      = binary,
    )

    # ------------------------------------------------------------------
    # Q4. Instance mode: paper → author → paper (PAP), paper_0 anchor
    #     This is what a mirrored instance rule becomes if we DON'T trim.
    #     Expect PASS if the engine starts at paper_0 and traverses PAP.
    # ------------------------------------------------------------------
    results["Q4_inst_PAP_no_trim"] = run_test(
        label       = "Q4  instance mode: PAP mirrored (no trim), paper_0 anchor",
        path        = "paper_to_author,author_to_paper",
        instance_id = 0,
        g_hetero    = g_hetero,
        data_dir    = data_dir,
        folder_name = folder_name,
        binary      = binary,
    )

    # ------------------------------------------------------------------
    # Q5. Variable mode: the same PAP path with no anchor
    #     Pure variable APA is already tested; PAP (starting at paper)
    #     should FAIL for variable mode because qnodes are author IDs.
    # ------------------------------------------------------------------
    results["Q5_var_PAP_starts_paper"] = run_test(
        label       = "Q5  variable mode: PAP (starts at paper, qnodes=author → expected FAIL)",
        path        = "paper_to_author,author_to_paper",
        instance_id = -1,
        g_hetero    = g_hetero,
        data_dir    = data_dir,
        folder_name = folder_name,
        binary      = binary,
    )

    # ------------------------------------------------------------------
    # Summary + interpretation
    # ------------------------------------------------------------------
    print(f"\n{'='*62}")
    print("  DIAGNOSIS SUMMARY")
    print(f"{'='*62}")

    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k:<40} {status}")

    print()
    print("  INTERPRETATION")
    print("  ─────────────────────────────────────────────────────")

    q1 = results["Q1_var_APA"]
    q2 = results["Q2_inst_APA_wrong_anchor"]
    q3 = results["Q3_inst_paper_to_author"]
    q4 = results["Q4_inst_PAP_no_trim"]
    q5 = results["Q5_var_PAP_starts_paper"]

    if not q1:
        print("  ⚠  Q1 FAILED — engine/staging is broken. Fix this first.")
        return

    print("  ✔  Q1 passed — engine and variable mode are working.")

    if q2:
        print("  ⚠  Q2 passed (unexpected) — instance mode may be using qnodes,")
        print("     not the anchor node, as the start. The anchor is decorative.")
    else:
        print("  ✔  Q2 failed (expected) — wrong-type anchor → 0 results.")
        print("     Instance mode DOES anchor execution at the fixed node type.")

    if q3:
        print("  ✔  Q3 passed — instance mode traverses from paper_0 outward.")
        print("     Path must START at the instance node type (paper).")
        print("     CONCLUSION: do NOT trim instance paths to target_node_type.")
    else:
        print("  ✘  Q3 failed — instance mode is not traversing from the anchor.")
        print("     The -5 flag may work differently than expected.")

    if q4 and q3:
        print("  ✔  Q4 passed — mirrored PAP (no trim) works with paper anchor.")
        print("     For instance rules: mirror WITHOUT trimming to author.")
    elif not q4:
        print("  ✘  Q4 failed — mirrored PAP doesn't work even with correct anchor.")

    if not q5:
        print("  ✔  Q5 failed (expected) — variable PAP fails because qnodes=author.")
        print("     Confirms: variable rules need trimming; instance rules do not.")
    else:
        print("  ⚠  Q5 passed (unexpected) — variable PAP starting at paper somehow works.")

    print()
    if q1 and not q2 and q3 and q4 and not q5:
        print("  VERDICT: Variable and instance modes are DIFFERENT.")
        print("    Variable rules: trim path to start at target_node_type (author).")
        print("    Instance rules: do NOT trim — path must start at instance node type.")
        print("    The current fix in test_instance_metapath.py incorrectly")
        print("    applies trimming to ALL rules. It must be conditioned on")
        print("    rule.rule_type == 'variable' only.")
    else:
        print("  VERDICT: Unexpected results — review individual test outputs above.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose variable vs instance execution mode in the C++ graph engine."
    )
    parser.add_argument("dataset", type=str, help="Dataset key, e.g. HGB_DBLP")
    parser.add_argument("--binary",     type=str, default=config.CPP_EXECUTABLE)
    parser.add_argument("--skip-stage", action="store_true",
                        help="Skip staging (use if HGBn-DBLP/ already exists)")
    args = parser.parse_args()

    ds_name     = args.dataset
    folder_name = f"HGBn-{ds_name.split('_')[1]}"
    data_dir    = os.path.join(project_root, folder_name)

    print(f"\n{'='*62}")
    print(f"  Instance Mode Diagnosis: {ds_name}")
    print(f"{'='*62}")

    from src.data import DatasetFactory
    from src.bridge import PyGToCppAdapter

    cfg = config.get_dataset_config(ds_name)
    print(f"\n[1/2] Loading graph...")
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    print(f"      Edge types   : {[et[1] for et in g_hetero.edge_types]}")
    print(f"      Target node  : {cfg.target_node}")

    if not args.skip_stage:
        print(f"\n[2/2] Staging C++ files → {data_dir}/")
        PyGToCppAdapter(data_dir).convert(g_hetero)
        generate_qnodes(data_dir, folder_name, target_node_type=cfg.target_node, g_hetero=g_hetero)
    else:
        print(f"\n[2/2] Staging skipped.")

    run_diagnosis(g_hetero, data_dir, folder_name, args.binary)


if __name__ == "__main__":
    main()