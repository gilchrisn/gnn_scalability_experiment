"""
Phase 1: Rule Mining Orchestrator.
Automated Rule Discovery with Strict Symmetry Mirroring and Stratified Sampling.
"""
import os
import json
import re
import time
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
import torch
from torch_geometric.data import HeteroData

from ..config import config
from ..data import DatasetFactory
from ..bridge import AnyBURLRunner
from ..utils import SchemaMatcher
from .base import AbstractExperimentPhase

class Phase1Mining(AbstractExperimentPhase):
    """
    Phase 1: Discovers, validates, and stratifies structural metapaths.

    Implements a robust curriculum generation strategy:
    1. Deep FOL search via AnyBURL.
    2. Head-Anchored parsing and Symmetry Mirroring.
    3. Empirical Data Gatekeeper (BFS validation).
    4. Stratified Length Bucketing to ensure structural diversity.
    5. Prefix Spawning and Cyclic Composition fallbacks for rigid topologies.
    """

    def execute(self) -> List[str]:
        print(f"\n>>> [Phase 1] Rule Mining & Mirroring: {self.cfg.dataset}")

        mining_dir = os.path.join(config.DATA_DIR, f"mining_{self.cfg.dataset}")
        result_file = os.path.join(mining_dir, "best_rules.json")
        self.target = self.dataset_cfg.target_node.lower()

        # 1. Caching Check
        if os.path.exists(result_file) and not self.cfg.force_remine:
            with open(result_file, 'r') as f:
                data = json.load(f)
                cached_rules = data.get('best_rules', [])
                if cached_rules:
                    print(f"    [Loaded] {len(cached_rules)} cached rules.")
                    return [r['rule'] for r in cached_rules]

        # 2. Setup Graph Schema Truth
        g_hetero, _ = DatasetFactory.get_data(
            self.dataset_cfg.source,
            self.dataset_cfg.dataset_name,
            self.dataset_cfg.target_node
        )

        runner = AnyBURLRunner(mining_dir, config.ANYBURL_JAR)

        # 3. Read Raw AnyBURL Rules
        rules_file = os.path.join(mining_dir, "anyburl_rules.txt")
        if not os.path.exists(rules_file):
            print("    [AnyBURL] Mining deep structural rules...")
            runner.export_for_mining(g_hetero)
            runner.run_mining(
                timeout=getattr(self.cfg, 'mining_timeout', 3600),
                max_length=4,
                num_threads=8
            )

        # 4. Initialize State for Funnel
        self.buckets: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
        self.seen_mirrored_paths = set()
        self.anchored_paths: List[Tuple[float, str]] = []

        print(f"    [Filtering] Validating cyclic paths starting with '{self.target}'...")
        start_t = time.perf_counter()

        with open(rules_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        # Extract, Parse, and Verify Standard Rules
        for line in all_lines:
            self._process_line_standard(line, g_hetero)

        # 5. Execute Strategy-Specific Logic
        valid_rules = []

        if self.cfg.mining_strategy == 'stratified':
            # Fallback A: Prefix Spawning
            if len(self.buckets) < 2:
                self._execute_prefix_fallback(g_hetero)

            # Fallback B: Dynamic Cyclic Composition
            target_buckets = set(self.cfg.stratified_buckets)
            if not target_buckets.issubset(self.buckets.keys()):
                self._compose_cyclic_paths(g_hetero, target_buckets)

            # Stratified Selection
            for length in sorted(self.cfg.stratified_buckets):
                if length in self.buckets:
                    paths = sorted(self.buckets[length], key=lambda x: x[0], reverse=True)
                    top_k = paths[:self.cfg.top_n_paths]
                    for conf, path in top_k:
                        valid_rules.append({"rule": path, "confidence": conf})
                    print(f"    [Bucket L={length}] Retained {len(top_k)} paths.")

        elif self.cfg.mining_strategy == 'top_k':
            # Classic Mode: Pool everything and take top N overall
            all_valid = []
            for paths in self.buckets.values():
                all_valid.extend(paths)
            all_valid.sort(key=lambda x: x[0], reverse=True)
            top_k = all_valid[:self.cfg.top_n_paths]
            valid_rules = [{"rule": p, "confidence": c} for c, p in top_k]
            print(f"    [Top K] Retained {len(valid_rules)} paths overall.")

        duration = time.perf_counter() - start_t
        print(f"    [Funnel] Processing complete in {duration:.2f}s.")

        if not valid_rules:
             raise RuntimeError(f"Mining failed: No valid rules survived for '{self.target}'.")

        # 6. Save Artifact
        artifact = {
            "dataset": self.cfg.dataset,
            "target_node": self.dataset_cfg.target_node,
            "best_rules": valid_rules
        }
        with open(result_file, 'w') as f:
            json.dump(artifact, f, indent=2)

        return [r['rule'] for r in valid_rules]

    def _compose_cyclic_paths(self, g_hetero: HeteroData, target_buckets: set) -> None:
        """
        Synthesizes missing required lengths by chaining verified bases.
        Subject to strict empirical data verification.
        """
        missing = target_buckets - self.buckets.keys()
        print(f"    [Fallback B] Synthesizing missing curriculum lengths: {missing}...")

        # Synthesize L=4 from L=2 + L=2
        if 4 in target_buckets and 2 in self.buckets and 4 not in self.buckets:
            for conf1, p1 in self.buckets[2]:
                for conf2, p2 in self.buckets[2]:
                    new_path = f"{p1},{p2}"
                    new_conf = conf1 * conf2
                    if new_path not in self.seen_mirrored_paths and self._verify_data_connectivity(g_hetero, new_path, self.target):
                        self.seen_mirrored_paths.add(new_path)
                        self.buckets[4].append((new_conf, new_path))

        # Synthesize L=6 from L=2 + L=4
        if 6 in target_buckets and 2 in self.buckets and 4 in self.buckets and 6 not in self.buckets:
            for conf1, p1 in self.buckets[2]:
                for conf2, p2 in self.buckets[4]:
                    new_path = f"{p1},{p2}"
                    new_conf = conf1 * conf2
                    if new_path not in self.seen_mirrored_paths and self._verify_data_connectivity(g_hetero, new_path, self.target):
                        self.seen_mirrored_paths.add(new_path)
                        self.buckets[6].append((new_conf, new_path))

        # Synthesize L=8 from L=4 + L=4
        if 8 in target_buckets and 4 in self.buckets and 8 not in self.buckets:
            for conf1, p1 in self.buckets[4]:
                for conf2, p2 in self.buckets[4]:
                    new_path = f"{p1},{p2}"
                    new_conf = conf1 * conf2
                    if new_path not in self.seen_mirrored_paths and self._verify_data_connectivity(g_hetero, new_path, self.target):
                        self.seen_mirrored_paths.add(new_path)
                        self.buckets[8].append((new_conf, new_path))

    def _process_line_standard(self, line: str, g_hetero: HeteroData) -> None:
        """Pushes a single rule through the standard validation funnel."""
        line = line.strip()
        parts = line.split("\t")
        if len(parts) < 4: return

        try: conf = float(parts[2])
        except ValueError: return

        if conf < getattr(self.cfg, 'min_confidence', 0.001): return

        rule_str = parts[3]

        # Reject Grounded Rules
        if re.search(r'[a-z]+_\d+', rule_str): return

        try: head, body = rule_str.split(" <= ")
        except ValueError: return

        # Head-Anchoring
        head_match = re.match(r'([a-zA-Z0-9_]+)\(([A-Z]),([A-Z])\)', head)
        if not head_match: return
        _, head_x, _ = head_match.groups()

        atoms = body.split(", ")
        relations = []
        current_var = head_x

        for atom in atoms:
            match = re.match(r'([a-zA-Z0-9_]+)\(([A-Z]),([A-Z])\)', atom)
            if not match: return
            rel, v1, v2 = match.groups()

            if current_var == v1:
                relations.append(rel)
                current_var = v2
            elif current_var == v2:
                relations.append(f"rev_{rel}")
                current_var = v1
            else:
                return  # Disjointed logic

        if not relations: return
        path_str = ",".join(relations)

        if not path_str.startswith(f"{self.target}_"): return

        # Retain state for Fallback A
        self.anchored_paths.append((conf, path_str))
        self._mirror_and_verify(path_str, conf, g_hetero)

    def _execute_prefix_fallback(self, g_hetero: HeteroData) -> None:
        """Executes the Prefix Spawning logic with supremum deduplication."""
        print(f"    [Fallback A Triggered] Insufficient structural diversity (|L| = {len(self.buckets)}).")
        print("    Executing Prefix Spawning...")

        prefix_map = {}
        for conf, path_str in self.anchored_paths:
            rels = path_str.split(',')
            for i in range(1, len(rels) + 1):
                prefix = ",".join(rels[:i])
                if prefix not in prefix_map or conf > prefix_map[prefix]:
                    prefix_map[prefix] = conf

        for prefix_str, conf in prefix_map.items():
            self._mirror_and_verify(prefix_str, conf, g_hetero)

    def _mirror_and_verify(self, path_str: str, conf: float, g_hetero: HeteroData) -> None:
        """Handles Schema Mirroring and BFS Verification."""
        mirrored_path = self._prove_and_mirror(path_str, g_hetero, self.target)
        if not mirrored_path: return

        if mirrored_path in self.seen_mirrored_paths: return
        self.seen_mirrored_paths.add(mirrored_path)

        if self._verify_data_connectivity(g_hetero, mirrored_path, self.target):
            length = len(mirrored_path.split(','))
            self.buckets[length].append((conf, mirrored_path))

    def _prove_and_mirror(self, path_str: str, g_hetero: HeteroData, target_node: str) -> Optional[str]:
        """Validates path contiguity and rigidly mirrors it to guarantee cyclicity."""
        rels = path_str.split(',')
        matched_path = []
        for r in rels:
            edge_tuple = SchemaMatcher.match(r, g_hetero)
            if edge_tuple[0] == 'node': return None
            matched_path.append(edge_tuple)

        for i in range(len(matched_path) - 1):
            if matched_path[i][2] != matched_path[i+1][0]: return None

        if matched_path[-1][2] == target_node: return path_str

        mirrored_rels = []
        for src, rel, dst in reversed(matched_path):
            rev_edge = next((r for s, r, d in g_hetero.edge_types if s == dst and d == src), None)
            if rev_edge: mirrored_rels.append(rev_edge)
            else: return None

        return path_str + "," + ",".join(mirrored_rels)

    def _verify_data_connectivity(self, g_hetero: HeteroData, path_str: str, target_node: str, num_samples: int = 200) -> bool:
        """Empirical Gatekeeper: Validates non-zero matrix instantiation using breadth-first traversal."""
        path_tuples = [SchemaMatcher.match(r.strip(), g_hetero) for r in path_str.split(',')]
        total_nodes = g_hetero[target_node].num_nodes
        if total_nodes == 0: return False

        curr_frontier = torch.randperm(total_nodes)[:num_samples]

        for src, rel, dst in path_tuples:
            edge_index = g_hetero[src, rel, dst].edge_index
            if edge_index.size(1) == 0: return False
            mask = torch.isin(edge_index[0], curr_frontier)
            curr_frontier = edge_index[1][mask].unique()
            if curr_frontier.numel() == 0: return False

        return True