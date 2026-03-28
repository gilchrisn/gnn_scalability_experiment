"""
AnyBURL implementation of the RuleMiner interface.
Handles graph serialization to triples, Java execution, and rule parsing.
"""
import os
import re
import subprocess
from typing import Optional, List, Tuple, Dict, Any, Set
from torch_geometric.data import HeteroData
from .base import RuleMiner


# ---------------------------------------------------------------------------
# Relation normalization utilities (module-level, no class required)
# ---------------------------------------------------------------------------

def build_canonical(g_hetero: HeteroData) -> Dict[Tuple[str, str], str]:
    """
    Build a lookup: (src_type, dst_type) → explicit relation name.

    Used to replace AnyBURL's synthetic ``rev_X`` tokens with the canonical
    edge name already stored explicitly in the HGB graph.  For example::

        rev_paper_to_author  →  author_to_paper
    """
    return {(src, dst): rel for src, rel, dst in g_hetero.edge_types}


def build_schema(g_hetero: HeteroData) -> Dict[str, Tuple[str, str]]:
    """
    Build a traversal lookup: relation_string → (actual_src_type, actual_dst_type).

    Stores both forward and ``rev_`` forms with their correct traversal
    directions so callers can look up and use the result directly.
    """
    lookup: Dict[str, Tuple[str, str]] = {}
    for src, rel, dst in g_hetero.edge_types:
        fwd_key = f"{src}_to_{dst}"
        lookup[rel]              = (src, dst)
        lookup[fwd_key]          = (src, dst)
        lookup[f"rev_{rel}"]     = (dst, src)
        lookup[f"rev_{fwd_key}"] = (dst, src)
    return lookup


def normalize_rel(rel: str,
                  schema: Dict[str, Tuple[str, str]],
                  canonical: Dict[Tuple[str, str], str]) -> str:
    """
    Normalize a single relation token produced by the AnyBURL parser.

    * **Rule 1** — cancel double reverse: ``rev_rev_X`` → ``X``
    * **Rule 2** — replace synthetic ``rev_X`` with the explicit canonical
      edge name when that direction already exists in the graph schema:
      ``rev_paper_to_author`` → ``author_to_paper``
    """
    if rel.startswith("rev_rev_"):
        rel = rel[8:]
    if rel.startswith("rev_"):
        sd = schema.get(rel)
        if sd is not None:
            explicit = canonical.get((sd[0], sd[1]))
            if explicit is not None:
                rel = explicit
    return rel


def normalize_path(path_str: str,
                   schema: Dict[str, Tuple[str, str]],
                   canonical: Dict[Tuple[str, str], str]) -> str:
    """Apply :func:`normalize_rel` to every relation in a comma-separated path."""
    return ",".join(normalize_rel(r, schema, canonical) for r in path_str.split(","))


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

def _is_instance(token: str) -> bool:
    """True if token is a grounded node (lowercase start + numeric suffix)."""
    return bool(token) and token[0].islower() and re.search(r'_\d+$', token) is not None


def parse_rules(filepath: str, min_conf: float) -> List[Tuple[float, str, int]]:
    """
    Parse an AnyBURL rule file.  Returns ``(conf, path_string, instance_id)`` tuples.

    ``path_string`` is a comma-separated chain of relation names.
    ``instance_id`` is ``-1`` for variable rules, or the grounded integer ID at
    the tail for instance-anchored rules.  Skips head-anchored rules (first body
    argument is grounded — no free query variable to start from).
    """
    results = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                conf = float(parts[2])
            except ValueError:
                continue
            if conf < min_conf:
                continue

            rule_str = parts[3]
            try:
                body = rule_str.split(" <= ")[1]
            except IndexError:
                continue

            atoms = body.split(", ")
            relations:   List[str] = []
            current_var: Optional[str] = None
            instance_id: int = -1

            ok = True
            for i, atom in enumerate(atoms):
                m = re.match(r'([a-zA-Z0-9_]+)\(([a-zA-Z0-9_]+),([a-zA-Z0-9_]+)\)', atom)
                if not m:
                    ok = False
                    break

                rel, v1, v2 = m.groups()
                v1_inst = _is_instance(v1)
                v2_inst = _is_instance(v2)

                if i == 0:
                    if v1_inst:
                        ok = False
                        break
                    current_var = v1

                if current_var == v1:
                    relations.append(rel)
                    if v2_inst:
                        instance_id = int(v2.rsplit("_", 1)[-1])
                        break
                    current_var = v2
                elif current_var == v2:
                    relations.append(f"rev_{rel}")
                    if v1_inst:
                        instance_id = int(v1.rsplit("_", 1)[-1])
                        break
                    current_var = v1
                else:
                    ok = False
                    break

            if ok and relations:
                results.append((conf, ",".join(relations), instance_id))

    return results


# ---------------------------------------------------------------------------
# Mirroring
# ---------------------------------------------------------------------------

def _rev_rel(rel: str) -> str:
    """Return the reverse of a relation name."""
    return rel[4:] if rel.startswith("rev_") else f"rev_{rel}"


def mirror(path_str: str) -> str:
    """
    Mirror path ``A-B-C`` into symmetric path ``A-B-C-B-A``.

    Hub queries require symmetric metapaths (start and end at the same node
    type).  AnyBURL mines one-directional chains; mirroring makes them valid.
    """
    rels = path_str.split(",")
    return ",".join(rels + [_rev_rel(r) for r in reversed(rels)])


# ---------------------------------------------------------------------------
# Schema validation + trimming
# ---------------------------------------------------------------------------

def validate_and_trim(
    path_str: str,
    schema:   Dict[str, Tuple[str, str]],
    target:   str,
) -> Optional[str]:
    """
    Resolve each edge against the schema, then trim the path to start at
    ``target`` node type.

    Returns the trimmed path string, or ``None`` if any edge is unknown, the
    path is non-contiguous, the target type never appears as a source, or the
    path does not end at target (not a valid symmetric hub-query path).
    """
    rels = path_str.split(",")
    resolved: List[Tuple[str, str, str]] = []

    for rel_str in rels:
        sd = schema.get(rel_str)
        if sd is None:
            return None
        resolved.append((rel_str, sd[0], sd[1]))

    for i in range(len(resolved) - 1):
        if resolved[i][2] != resolved[i + 1][1]:
            return None

    trim_idx = next(
        (i for i, (_, src, _) in enumerate(resolved) if src == target),
        None,
    )
    if trim_idx is None:
        return None

    trimmed = resolved[trim_idx:]
    if not trimmed:
        return None

    if trimmed[-1][2] != target:
        return None

    return ",".join(r for r, _, _ in trimmed)


# ---------------------------------------------------------------------------
# Full validated pipeline
# ---------------------------------------------------------------------------

def load_validated_metapaths(
    rules_file:  str,
    g_hetero:    HeteroData,
    target_node: str,
    min_conf:    float = 0.1,
    max_n:       int   = 500,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Full metapath loading pipeline: parse → normalize → mirror → validate → dedup.

    Args:
        rules_file:  Path to AnyBURL output file.
        g_hetero:    Loaded PyG HeteroData (used to build schema + canonical lookup).
        target_node: Node type that hub queries are issued from (e.g. ``'paper'``).
        min_conf:    Minimum AnyBURL confidence threshold.  Default ``0.1`` matches
                     the original paper.
        max_n:       Maximum number of validated metapaths to return.

    Returns:
        ``(metapaths, stats)`` where ``metapaths`` is a list of validated
        comma-separated path strings (at most ``max_n`` entries) and ``stats``
        is a dict with counts for logging and verification.
    """
    schema    = build_schema(g_hetero)
    canonical = build_canonical(g_hetero)

    # 1. Parse all rules (no confidence floor here — filter next)
    raw = parse_rules(rules_file, min_conf=0.0)

    # 2. Confidence filter + normalize relation names
    filtered = [
        (conf, normalize_path(path, schema, canonical), iid)
        for conf, path, iid in raw
        if conf >= min_conf
    ]

    # 3. Dedup raw paths, keeping highest confidence per path.
    #    Instance rules are treated as variable (anchor dropped — we care only
    #    about the path shape, not the grounded anchor).
    best_conf: Dict[str, float] = {}
    for conf, path, _ in filtered:
        if conf > best_conf.get(path, -1.0):
            best_conf[path] = conf

    # 4. Mirror + validate + trim; deduplicate final validated paths
    seen_validated: Set[str] = set()
    metapaths: List[str] = []
    fail_schema = fail_trim = fail_symmetry = 0

    for raw_path in best_conf:
        mirrored = normalize_path(mirror(raw_path), schema, canonical)
        rels = mirrored.split(",")

        if any(schema.get(r) is None for r in rels):
            fail_schema += 1
            continue

        validated = validate_and_trim(mirrored, schema, target_node)
        if validated is None:
            # Distinguish failure modes for stats
            rels_resolved = [(r, schema[r][0], schema[r][1]) for r in rels if schema.get(r)]
            idx = next(
                (i for i, (_, s, _) in enumerate(rels_resolved) if s == target_node), None
            )
            if idx is None:
                fail_trim += 1
            else:
                fail_symmetry += 1
            continue

        if validated not in seen_validated:
            seen_validated.add(validated)
            metapaths.append(validated)
            if len(metapaths) >= max_n:
                break

    stats: Dict[str, Any] = {
        "total_parsed":      len(raw),
        "after_conf_filter": len(filtered),
        "unique_raw_paths":  len(best_conf),
        "valid_mirrored":    len(seen_validated),
        "fail_schema":       fail_schema,
        "fail_trim":         fail_trim,
        "fail_symmetry":     fail_symmetry,
        "returned":          len(metapaths),
    }
    return metapaths, stats


def load_validated_rules(
    rules_file:  str,
    g_hetero:    HeteroData,
    target_node: str,
    min_conf:    float = 0.1,
    max_n:       int   = 500,
    max_hops:    int   = 4,
) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    """
    Load ALL validated rules (variable + instance) for the C++ binary.

    Unlike ``load_validated_metapaths`` which deduplicates to unique path patterns,
    this function preserves each instance rule as a separate entry — matching
    the original paper's protocol where each rule is scored independently.

    Returns:
        ``(rules, stats)`` where ``rules`` is a list of ``(metapath_str, instance_id)``
        tuples (instance_id=-1 for variable rules).
    """
    schema    = build_schema(g_hetero)
    canonical = build_canonical(g_hetero)

    raw = parse_rules(rules_file, min_conf=0.0)

    # Confidence filter + normalize
    filtered = [
        (conf, normalize_path(path, schema, canonical), iid)
        for conf, path, iid in raw
        if conf >= min_conf
    ]

    # Sort by confidence (highest first)
    filtered.sort(key=lambda x: -x[0])

    # Validate each rule (mirror variable rules, keep instance anchors)
    rules: List[Tuple[str, int]] = []
    seen: Set[str] = set()
    fail_schema = fail_trim = fail_symmetry = fail_hops = 0

    for conf, path, iid in filtered:
        # Mirror to make symmetric (only for variable rules — instance rules
        # already have the C++ binary handle directionality)
        if iid == -1:
            mirrored = normalize_path(mirror(path), schema, canonical)
        else:
            mirrored = path  # instance rules use the raw (un-mirrored) path

        rels = mirrored.split(",")

        if any(schema.get(r) is None for r in rels):
            fail_schema += 1
            continue

        # Variable rules: validate symmetry + trim to target
        if iid == -1:
            validated = validate_and_trim(mirrored, schema, target_node)
            if validated is None:
                rels_resolved = [(r, schema[r][0], schema[r][1]) for r in rels if schema.get(r)]
                idx = next((i for i, (_, s, _) in enumerate(rels_resolved) if s == target_node), None)
                if idx is None:
                    fail_trim += 1
                else:
                    fail_symmetry += 1
                continue
            # Hop filter
            if len(validated.split(",")) > max_hops:
                fail_hops += 1
                continue
            sig = f"var:{validated}"
            if sig not in seen:
                seen.add(sig)
                rules.append((validated, -1))
        else:
            # Instance rules: just check schema contiguity
            validated = validate_and_trim(mirrored, schema, target_node)
            if validated is None:
                # Try without trim for instance rules
                fail_trim += 1
                continue
            if len(validated.split(",")) > max_hops:
                fail_hops += 1
                continue
            sig = f"inst:{validated}:{iid}"
            if sig not in seen:
                seen.add(sig)
                rules.append((validated, iid))

        if len(rules) >= max_n:
            break

    stats = {
        "total_parsed":      len(raw),
        "after_conf_filter": len(filtered),
        "total_rules":       len(rules),
        "variable_rules":    sum(1 for _, iid in rules if iid == -1),
        "instance_rules":    sum(1 for _, iid in rules if iid != -1),
        "fail_schema":       fail_schema,
        "fail_trim":         fail_trim,
        "fail_symmetry":     fail_symmetry,
        "fail_hops":         fail_hops,
    }
    return rules, stats


class AnyBURLRunner(RuleMiner):
    """
    Interoperability layer for AnyBURL.
    """
    
    def __init__(self, data_dir: str, jar_path: str):
        self.data_dir = data_dir
        self.jar_path = jar_path
        
        os.makedirs(data_dir, exist_ok=True)
        
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"AnyBURL JAR not found: {jar_path}")
        
        self.triples_file = os.path.join(data_dir, "anyburl_triples.txt")
        self.rules_file = os.path.join(data_dir, "anyburl_rules.txt")
        self.config_file = os.path.join(data_dir, "config-learn.properties")

    def export_for_mining(self, g_hetero: HeteroData) -> None:
        """Serializes HeteroData edge indices into AnyBURL triple format."""
        if os.path.exists(self.triples_file) and os.path.getsize(self.triples_file) > 0:
             print(f"[AnyBURL] Using existing triples at {self.triples_file}")
             return

        print(f"[AnyBURL] Exporting graph to {self.triples_file}...")
        with open(self.triples_file, 'w') as f:
            for edge_type in g_hetero.edge_types:
                src_type, rel, dst_type = edge_type
                edges = g_hetero[edge_type].edge_index
                srcs = edges[0].tolist()
                dsts = edges[1].tolist()
                
                for u, v in zip(srcs, dsts):
                    # Canonical formatting: nodeType_nodeID
                    f.write(f"{src_type}_{u}\t{rel}\t{dst_type}_{v}\n")
        print(f"[AnyBURL] Export complete.")

    def run_mining(self, timeout: int, max_length: int = 4, num_threads: int = 1, seed: int = 42) -> None:
        """
        Generates configuration and executes the Java learning process.
        """
        if os.path.exists(self.rules_file) and os.path.getsize(self.rules_file) > 0:
            print(f"[AnyBURL] Rules exist; skipping mining.")
            return

        # Prepare config
        safe_triples = self.triples_file.replace(os.sep, '/')
        safe_rules = self.rules_file.replace(os.sep, '/')
        
        config_content = (
            f"PATH_TRAINING = {safe_triples}\n"
            f"PATH_OUTPUT = {safe_rules}\n"
            f"SNAPSHOTS_AT = {timeout}\n"
            f"WORKER_THREADS = {num_threads}\n"
            f"MAX_LENGTH_CYCLIC = {max_length}\n"
            f"ZERO_RULES_ACTIVE = false\n"
            f"THRESHOLD_CORRECT_PREDICTIONS = 2\n"
            f"THRESHOLD_CONFIDENCE = 0.0001\n"
            f"RANDOM_SEED = {seed}\n"
        )
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Execute
        print(f"[AnyBURL] Learning rules (timeout={timeout}s)...")
        cmd = ["java", "-Xmx12G", "-cp", self.jar_path, "de.unima.ki.anyburl.Learn", self.config_file]
        try:
            subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout + 300)
        except subprocess.TimeoutExpired:
            print("[AnyBURL] Subprocess reached secondary timeout (Expected behavior for snapshots).")

        # Rename snapshot to canonical output path
        expected_output = f"{self.rules_file}-{timeout}"
        if os.path.exists(expected_output):
            if os.path.exists(self.rules_file): os.remove(self.rules_file)
            os.rename(expected_output, self.rules_file)
        else:
            print(f"[AnyBURL] Warning: Expected snapshot {expected_output} not found.")

    def get_top_k_paths(self, k: int = 5, min_conf: float = 0.1) -> List[Tuple[float, str, int]]:
        """Parses output file to identify top-K generic OR instance paths."""
        if not os.path.exists(self.rules_file): return []
        
        valid_rules = []
        with open(self.rules_file, 'r') as f:
            for line in f:
                try:
                    res = self._parse_single_line(line, min_conf)
                    if res: valid_rules.append(res)
                except Exception: continue

        # Deduplicate and Sort
        valid_rules.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        unique = []
        for conf, path, instance_id in valid_rules:
            # A rule is uniquely identified by BOTH its path and its target instance
            rule_signature = f"{path}_{instance_id}"
            if rule_signature not in seen:
                unique.append((conf, path, instance_id))
                seen.add(rule_signature)
                if len(unique) >= k: break
        return unique

    def _parse_single_line(self, line: str, min_conf: float) -> Optional[Tuple[float, str, int]]:
        line = line.strip()
        parts = line.split("\t")
        if len(parts) < 4: return None
        
        try:
            conf = float(parts[2])
        except ValueError:
            return None
            
        if conf < min_conf: return None
        rule_str = parts[3]
        
        try:
            body = rule_str.split(" <= ")[1]
        except IndexError:
            return None
            
        atoms = body.split(", ")
        relations = []
        current_var = None
        
        for i, atom in enumerate(atoms):
            # Expanded Regex to capture both variables (X) and instances (paper_1998)
            match = re.match(r'([a-zA-Z0-9_]+)\(([a-zA-Z0-9_]+),([a-zA-Z0-9_]+)\)', atom)
            if not match: 
                return None
            
            rel, v1, v2 = match.groups()
            
            v1_is_inst = v1[0].islower()
            v2_is_inst = v2[0].islower()
            
            if i == 0:
                # The start of a valid query path must be a variable
                if v1_is_inst: return None 
                current_var = v1
                
            if current_var == v1:
                relations.append(rel)
                if v2_is_inst:
                    # Target Hit: Return Instance Rule
                    instance_id = int(v2.split('_')[-1])
                    return (conf, ",".join(relations), instance_id)
                else:
                    current_var = v2
                    
            elif current_var == v2:
                relations.append(f"rev_{rel}")
                if v1_is_inst:
                    # Target Hit: Return Instance Rule
                    instance_id = int(v1.split('_')[-1])
                    return (conf, ",".join(relations), instance_id)
                else:
                    current_var = v1
            else:
                return None
                
        # If the loop finishes with no instance target found, it is a VARIABLE RULE
        if len(relations) >= 2:
            return (conf, ",".join(relations), -1) 
        return None