import os
import sys
import subprocess
import shutil

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_BIN = os.path.join(ROOT_DIR, "bin", "graph_prep.exe")
TEMP_DIR = os.path.join(ROOT_DIR, "output", "test_complex")

if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

def run_cpp(mode, *args):
    cmd = [CPP_BIN, mode] + list(args)
    subprocess.run(cmd, capture_output=True, text=True)

def test_semantic_clusters():
    print(f"\n{'='*60}")
    print("TEST: SEMANTIC CLUSTERS (User -> Buy -> Item -> Cat)")
    print(f"{'='*60}")
    
    # --- 1. Define Graph Topology (DENSE IDs) ---
    # User Type (0): Nodes 0, 1, 2, 3
    # Item Type (1): Nodes 4, 5, 6, 7
    # Cat Type  (2): Nodes 8, 9
    
    # Mappings:
    # 0: Alice,   1: Bob,    2: Charlie, 3: Dave
    # 4: Laptop,  5: Mouse,  6: Apple,   7: Table
    # 8: Tech,    9: Fruit
    
    nodes = [
        "0\tAlice\t0", "1\tBob\t0", "2\tCharlie\t0", "3\tDave\t0",
        "4\tLaptop\t1", "5\tMouse\t1", "6\tApple\t1", "7\tTable\t1",
        "8\tTech\t2", "9\tFruit\t2"
    ]
    
    edges = [
        # Cluster 1: Tech Lovers (Alice & Bob)
        "0\t4\t10",  # Alice -> Laptop
        "4\t8\t20",  # Laptop -> Tech
        "1\t5\t10",  # Bob -> Mouse
        "5\t8\t20",  # Mouse -> Tech
        
        # Cluster 2: Fruit Lover (Charlie)
        "2\t6\t10",  # Charlie -> Apple
        "6\t9\t20",  # Apple -> Fruit
        
        # Cluster 3: Dead End (Dave)
        "3\t7\t10"   # Dave -> Table (No link to category)
    ]
    
    # --- 2. Write Data Files ---
    print(f"Generating {len(nodes)} nodes and {len(edges)} edges...")
    
    with open(os.path.join(TEMP_DIR, "node.dat"), "w") as f:
        f.write("\n".join(nodes))
        
    with open(os.path.join(TEMP_DIR, "link.dat"), "w") as f:
        f.write("\n".join(edges))
        
    with open(os.path.join(TEMP_DIR, "meta.dat"), "w") as f:
        # Total nodes MUST match the number of lines in node.dat
        f.write("Node Number is : 10\n")

    # --- 3. Define Rule ---
    # Path: User -(10)-> Item -(20)-> Category
    # Format: -2 10 -2 -1 20 -4 -4
    rule_str = "-2 10 -2 -1 20 -4 -4"
    rule_file = os.path.join(TEMP_DIR, "cluster.rule")
    with open(rule_file, "w") as f: f.write(rule_str)
    
    # --- 4. Run Engine ---
    print("Running C++ Engine...")
    out_file = os.path.join(TEMP_DIR, "results.txt")
    run_cpp("materialize", TEMP_DIR, rule_file, out_file)
    
    # --- 5. Validate Results ---
    print("\n" + "-"*30)
    print("VALIDATION")
    print("-"*30)
    
    if not os.path.exists(out_file):
        print("❌ FAIL: No output file created.")
        return

    results = {}
    with open(out_file, "r") as f:
        for line in f:
            parts = list(map(int, line.split()))
            if parts:
                results[parts[0]] = set(parts[1:])

    # CHECK 1: Alice (0) and Bob (1) should be peers (Tech)
    alice_peers = results.get(0, set())
    if 1 in alice_peers:
        print(f"✅ PASS: Alice's peers include Bob: {alice_peers}")
    else:
        print(f"❌ FAIL: Alice missed Bob! Found: {alice_peers}")

    # CHECK 2: Bob (1) should see Alice (0)
    bob_peers = results.get(1, set())
    if 0 in bob_peers:
        print(f"✅ PASS: Bob's peers include Alice: {bob_peers}")
    else:
        print(f"❌ FAIL: Bob missed Alice! Found: {bob_peers}")

    # CHECK 3: Charlie (2) should be isolated (Fruit)
    charlie_peers = results.get(2, set())
    # Note: Charlie is a peer of himself (0 distance). C++ includes self-loops.
    if charlie_peers == {2}: 
        print(f"✅ PASS: Charlie is isolated correctly: {charlie_peers}")
    elif 2 in charlie_peers and len(charlie_peers) == 1:
        print(f"✅ PASS: Charlie is isolated correctly: {charlie_peers}")
    else:
        print(f"❌ FAIL: Charlie has unexpected peers: {charlie_peers}")

    # CHECK 4: Dave (3) should have NO output (Dead end)
    dave_peers = results.get(3, set())
    if not dave_peers:
        print(f"✅ PASS: Dave (Dead End) produced no peers.")
    else:
        print(f"❌ FAIL: Dave found non-existent peers: {dave_peers}")

if __name__ == "__main__":
    test_semantic_clusters()