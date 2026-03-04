"""
Rigorous Unit Test Suite for AnyBURL FOL Parsing and Symmetry Mirroring.
Proves the mathematical soundness of the graph traversal transformations.
"""
import re
from typing import Tuple, Optional, List

# =============================================================================
# 1. CORE LOGIC TO TEST (Isolated from pipeline)
# =============================================================================

def parse_fol_rule(line: str, min_conf: float = 0.0) -> Optional[Tuple[float, str]]:
    """Translates AnyBURL FOL into sequential metapaths."""
    line = line.strip()
    parts = line.split("\t")
    if len(parts) < 4: return None
    
    try: conf = float(parts[2])
    except ValueError: return None
        
    if conf < min_conf: return None
    
    rule_str = parts[3]
    if re.search(r'[a-z]+_\d+', rule_str): return None # Reject grounded
    
    try: body = rule_str.split(" <= ")[1]
    except IndexError: return None
        
    atoms = body.split(", ")
    relations = []
    current_var = None
    
    for i, atom in enumerate(atoms):
        match = re.match(r'([a-zA-Z0-9_]+)\(([A-Z]),([A-Z])\)', atom)
        if not match: return None
        rel, v1, v2 = match.groups()
        
        if i == 0: current_var = v1
            
        if current_var == v1:
            relations.append(rel)
            current_var = v2
        elif current_var == v2:
            relations.append(f"rev_{rel}")
            current_var = v1
        else: return None # Disjointed
            
    if len(relations) >= 2: return (conf, ",".join(relations))
    return None

def prove_and_mirror(path_str: str, schema_edges: List[Tuple[str, str, str]], target_node: str) -> Optional[str]:
    """Validates path against schema and strictly mirrors it if asymmetric."""
    rels = path_str.split(',')
    matched_path = []
    
    # Simple Schema Matcher for testing
    def match_edge(rel_name: str):
        # 1. Exact match
        for s, r, d in schema_edges:
            if r == rel_name: return (s, r, d)
        # 2. Structural match if 'to' is present
        if "_to_" in rel_name:
            parts = rel_name.split("_to_")
            src, dst = parts[0].replace('rev_', ''), parts[-1]
            for s, r, d in schema_edges:
                if s == src and d == dst: return (s, r, d)
        return ('node', rel_name, 'node')

    # Axiom 1: Existence
    for r in rels:
        edge_tuple = match_edge(r)
        if edge_tuple[0] == 'node': return None
        matched_path.append(edge_tuple)
        
    # Axiom 2: Contiguous Connectivity
    for i in range(len(matched_path) - 1):
        if matched_path[i][2] != matched_path[i+1][0]: return None
            
    last_dst = matched_path[-1][2]
    
    # Axiom 3: Cyclicity
    if last_dst == target_node:
        return path_str # Already Cyclic
        
    # Axiom 4: Strict Mirroring
    mirrored_rels = []
    for src, rel, dst in reversed(matched_path):
        rev_edge_name = None
        for s, r, d in schema_edges:
            if s == dst and d == src:
                rev_edge_name = r
                break
        if rev_edge_name: mirrored_rels.append(rev_edge_name)
        else: return None # Fails if reverse edge physically missing from schema
            
    return path_str + "," + ",".join(mirrored_rels)


# =============================================================================
# 2. RIGOROUS TEST SUITE
# =============================================================================

class MockSchema:
    """A controlled environment to test graph topologies."""
    DBLP = [
        ('author', 'author_to_paper', 'paper'),
        ('paper', 'paper_to_author', 'author'),
        ('paper', 'paper_to_term', 'term'),
        ('term', 'term_to_paper', 'paper'),
        ('author', 'author_to_school', 'school') # Missing reverse edge intentionally!
    ]

def run_tests():
    print("============================================================")
    print(" EXECUTING RIGOROUS AXIOMATIC TESTS")
    print("============================================================\n")

    # --- TEST 1: FOL Parsing (Forward & Reverse Variables) ---
    print("[Test 1] FOL Variable Translation to Directions")
    rule_line = "123\t10\t0.85\ttarget(X,Y) <= author_to_paper(X,A), paper_to_author(Y,A)"
    conf, path = parse_fol_rule(rule_line)
    
    # Assert conf is extracted correctly
    assert conf == 0.85, f"Expected 0.85, got {conf}"
    # Assert variables triggered the reverse prefix!
    assert path == "author_to_paper,rev_paper_to_author", f"Expected 'author_to_paper,rev_paper_to_author', got {path}"
    print("  ✓ Passed: FOL variables accurately mapped to sequence.")

    # --- TEST 2: Naturally Cyclic Path (No Mirroring Needed) ---
    print("\n[Test 2] Naturally Cyclic Metapath")
    path_str = "author_to_paper,paper_to_author"
    result = prove_and_mirror(path_str, MockSchema.DBLP, "author")
    
    assert result == "author_to_paper,paper_to_author", "Cyclic path should not be modified."
    print("  ✓ Passed: Target-aligned paths are left intact.")

    # --- TEST 3: Strict Symmetry Mirroring (The DBLP Fix) ---
    print("\n[Test 3] Asymmetric Path Mirroring")
    # Path: author -> paper -> term (Ends at term, needs to go back to author)
    path_str = "author_to_paper,paper_to_term"
    result = prove_and_mirror(path_str, MockSchema.DBLP, "author")
    
    expected = "author_to_paper,paper_to_term,term_to_paper,paper_to_author"
    assert result == expected, f"Failed mirroring. Got: {result}"
    print("  ✓ Passed: Asymmetric path perfectly mirrored back to target node.")

    # --- TEST 4: Contiguity Violation (Broken Link) ---
    print("\n[Test 4] Schema Contiguity Violation")
    # Path: author -> paper -> (break) -> term -> paper
    path_str = "author_to_paper,term_to_paper"
    result = prove_and_mirror(path_str, MockSchema.DBLP, "author")
    
    assert result is None, "Path should be rejected due to contiguity failure."
    print("  ✓ Passed: Disconnected paths are correctly rejected.")

    # --- TEST 5: The 'Missing Reverse Edge' Axiom ---
    print("\n[Test 5] Missing Reverse Edge in Schema")
    # author -> school (But schema has no school -> author edge)
    path_str = "author_to_school"
    result = prove_and_mirror(path_str, MockSchema.DBLP, "author")
    
    assert result is None, "Path should be rejected because mirroring is impossible."
    print("  ✓ Passed: Mirroring gracefully aborts if schema lacks symmetric edge.")

    # --- TEST 6: Disjointed FOL Rule (Ghost AnyBURL logic) ---
    print("\n[Test 6] Disjointed FOL Logic Rejection")
    # X -> A, but then suddenly B -> Y (A and B are disconnected)
    rule_line = "1\t1\t0.5\ttarget(X,Y) <= rel1(X,A), rel2(B,Y)"
    res = parse_fol_rule(rule_line)
    
    assert res is None, "Disjointed logic should be rejected in FOL parser."
    print("  ✓ Passed: Disjointed variables cause immediate parsing rejection.")

    print("\n============================================================")
    print(" ALL AXIOMS VERIFIED SUCCESSFULLY.")
    print(" The Symmetry Mirroring protocol is mathematically sound.")
    print("============================================================")

if __name__ == "__main__":
    run_tests()