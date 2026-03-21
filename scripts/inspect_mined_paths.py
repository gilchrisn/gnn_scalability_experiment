"""
inspect_mined_paths.py — Load mined AnyBURL rules, apply mirror policy, count & report.

Policy
------
For every rule passing --min-conf:
  1. Parse the body as a chain of relation names (instance anchors stripped out).
  2. Mirror the chain: A-B-C  →  A-B-C-B-A
  3. Trim the mirrored path to begin at the target node type.
  4. Validate against the graph schema.
  5. Deduplicate.

Instance rules: the grounded anchor at the tail is DROPPED; the body path itself
is reused as a variable path.  This matches the user policy: "throw away the
instance, then mirror".

Usage
-----
    python scripts/inspect_mined_paths.py HGB_ACM

    python scripts/inspect_mined_paths.py HGB_DBLP --min-conf 0.05
    python scripts/inspect_mined_paths.py HGB_IMDB --min-conf 0.01 --top 20
    python scripts/inspect_mined_paths.py HGB_ACM --rules-file datasets/mining_HGB_ACM/anyburl_rules.txt
"""
from __future__ import annotations

import os
import sys
import argparse
from collections import Counter, defaultdict
from typing import Dict, List

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.bridge.anyburl import (
    build_schema,
    build_canonical,
    normalize_path,
    parse_rules,
    mirror,
    validate_and_trim,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect mined AnyBURL metapaths")
    parser.add_argument("dataset",       type=str, help="Dataset key, e.g. HGB_DBLP")
    parser.add_argument("--min-conf",    type=float, default=0.01,
                        help="Minimum confidence threshold (default 0.01)")
    parser.add_argument("--top",         type=int, default=20,
                        help="Number of top unique paths to print (default 20)")
    parser.add_argument("--rules-file",  type=str, default=None,
                        help="Explicit path to anyburl_rules.txt (auto-detected if omitted)")
    args = parser.parse_args()

    ds_name = args.dataset

    # ------------------------------------------------------------------
    # 1. Locate rules file
    # ------------------------------------------------------------------
    candidates = [
        args.rules_file,
        os.path.join(project_root, "datasets", ds_name, "anyburl_rules.txt"),
        os.path.join(project_root, "datasets", f"mining_{ds_name}", "anyburl_rules.txt"),
    ]
    rules_file = next((p for p in candidates if p and os.path.isfile(p)), None)
    if rules_file is None:
        print(f"[FATAL] No rules file found for {ds_name}. Tried:")
        for p in candidates[1:]:
            print(f"  {p}")
        sys.exit(1)

    print(f"\n[config]  dataset    = {ds_name}")
    print(f"[config]  rules_file = {rules_file}")
    print(f"[config]  min_conf   = {args.min_conf}")

    # ------------------------------------------------------------------
    # 2. Load graph (for schema)
    # ------------------------------------------------------------------
    from src.data import DatasetFactory
    cfg = config.get_dataset_config(ds_name)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    target = cfg.target_node
    print(f"[config]  target     = {target}")
    print(f"[config]  edge types = {[et[1] for et in g_hetero.edge_types]}")

    schema    = build_schema(g_hetero)
    canonical = build_canonical(g_hetero)

    # ------------------------------------------------------------------
    # 3. Parse rules
    # ------------------------------------------------------------------
    print(f"\n[1/4] Parsing {rules_file} ...")
    raw = parse_rules(rules_file, min_conf=0.0)   # parse all, filter below
    total_parsed = len(raw)
    print(f"      total parsed lines  : {total_parsed:,}")

    # Apply confidence filter, then normalize path names
    filtered_raw = [(conf, path, iid) for conf, path, iid in raw if conf >= args.min_conf]
    filtered = [(conf, normalize_path(path, schema, canonical), iid)
                for conf, path, iid in filtered_raw]
    n_var  = sum(1 for _, _, iid in filtered if iid == -1)
    n_inst = sum(1 for _, _, iid in filtered if iid >= 0)
    print(f"      after conf>={args.min_conf}        : {len(filtered):,}")
    print(f"        variable rules     : {n_var:,}")
    print(f"        instance rules     : {n_inst:,}  (anchor will be dropped)")

    # ------------------------------------------------------------------
    # 4. Mirror + validate + trim
    # ------------------------------------------------------------------
    print(f"\n[2/4] Mirroring paths ...")

    # Dedup raw paths first (keep highest conf per path)
    best_conf: Dict[str, float] = {}
    raw_count: Counter = Counter()
    for conf, path, _ in filtered:
        raw_count[path] += 1
        if conf > best_conf.get(path, -1):
            best_conf[path] = conf

    print(f"      unique raw paths     : {len(best_conf):,}")

    # Mirror each unique raw path, then validate+trim
    mirrored_map: Dict[str, List[str]] = defaultdict(list)  # mirrored → [raw paths]
    fail_schema   = 0
    fail_trim     = 0
    fail_symmetry = 0

    for raw_path in best_conf:
        m = normalize_path(mirror(raw_path), schema, canonical)
        rels = m.split(",")
        # Schema check
        if any(schema.get(r) is None for r in rels):
            fail_schema += 1
            continue
        # Trim + symmetry check (validate_and_trim returns None for both failures)
        validated = validate_and_trim(m, schema, target)
        if validated is None:
            # Distinguish: did it trim OK but fail symmetry, or never found target?
            rels_resolved = [(r, schema[r][0], schema[r][1]) for r in rels if schema.get(r)]
            idx = next((i for i, (_, s, _) in enumerate(rels_resolved) if s == target), None)
            if idx is None:
                fail_trim += 1
            else:
                fail_symmetry += 1
        else:
            mirrored_map[validated].append(raw_path)

    unique_mirrored = list(mirrored_map.keys())
    print(f"      unique mirrored paths: {len(unique_mirrored):,}")
    print(f"      failed (schema miss) : {fail_schema:,}")
    print(f"      failed (no target)   : {fail_trim:,}")
    print(f"      failed (asymmetric)  : {fail_symmetry:,}")

    # ------------------------------------------------------------------
    # 5. Statistics
    # ------------------------------------------------------------------
    print(f"\n[3/4] Statistics")

    lengths = [len(p.split(",")) for p in unique_mirrored]
    if lengths:
        len_counter = Counter(lengths)
        avg_len = sum(lengths) / len(lengths)
        print(f"\n  Length distribution (mirrored paths):")
        for l in sorted(len_counter):
            bar = "█" * min(len_counter[l], 60)
            print(f"    len={l:2d}  {len_counter[l]:6,}  {bar}")
        print(f"\n  avg length : {avg_len:.2f}")
        print(f"  min length : {min(lengths)}")
        print(f"  max length : {max(lengths)}")

    # Raw path length distribution (before mirror)
    raw_lengths = Counter(len(p.split(",")) for p in best_conf)
    print(f"\n  Raw path length distribution (before mirror):")
    for l in sorted(raw_lengths):
        bar = "█" * min(raw_lengths[l], 60)
        print(f"    len={l:2d}  {raw_lengths[l]:6,}  {bar}")

    # ------------------------------------------------------------------
    # 6. Top paths
    # ------------------------------------------------------------------
    print(f"\n[4/4] Top {args.top} unique mirrored paths (by source rule count)")

    # Sort by how many raw rules collapse to each mirrored path
    ranked = sorted(mirrored_map.items(), key=lambda kv: len(kv[1]), reverse=True)

    print(f"\n  {'#':>5}  {'len':>3}  {'n_src':>6}  {'best_conf':>9}  path")
    print(f"  {'─'*75}")
    for rank, (mpath, srcs) in enumerate(ranked[:args.top], 1):
        best = max(best_conf[s] for s in srcs)
        print(f"  {rank:>5}  {len(mpath.split(',')):>3}  {len(srcs):>6}  {best:>9.4f}  {mpath}")

    print(f"\n  Total unique valid mirrored paths: {len(unique_mirrored):,}")
    print()


if __name__ == "__main__":
    main()
