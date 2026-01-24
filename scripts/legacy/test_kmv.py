import os
import sys
import subprocess
import shutil

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_BIN = os.path.join(ROOT_DIR, "bin", "graph_prep.exe")
TEMP_DIR = os.path.join(ROOT_DIR, "output", "test_kmv")

if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

def run_cpp(mode, *args):
    cmd = [CPP_BIN, mode] + list(args)
    subprocess.run(cmd, capture_output=True, text=True)

def test_kmv_clusters():
    print(f"\n{'='*60}")
    print("TEST: KMV SKETCHING (User -> Buy -> Item -> Cat)")
    print(f"{'='*60}")
    
    # 1. Reuse the same topology setup (Dense IDs 0-9)
    nodes = [
        "0\tAlice\t0", "1\tBob\t0", "2\tCharlie\t0", "3\tDave\t0",
        "4\tLaptop\t1", "5\tMouse\t1", "6\tApple\t1", "7\tTable\t1",
        "8\tTech\t2", "9\tFruit\t2"
    ]
    edges = [
        "0\t4\t10", "4\t8\t20",  # Alice -> Laptop -> Tech
        "1\t5\t10", "5\t8\t20",  # Bob -> Mouse -> Tech
        "2\t6\t10", "6\t9\t20",  # Charlie -> Apple -> Fruit
        "3\t7\t10"               # Dave -> Table -> (Null)
    ]
    
    print(f"Generating Graph...")
    with open(os.path.join(TEMP_DIR, "node.dat"), "w") as f: f.write("\n".join(nodes))
    with open(os.path.join(TEMP_DIR, "link.dat"), "w") as f: f.write("\n".join(edges))
    with open(os.path.join(TEMP_DIR, "meta.dat"), "w") as f: f.write("Node Number is : 10\n")

    # 2. Correct Rule Format
    rule_str = "-2 10 -2 -1 20 -4 -4"
    rule_file = os.path.join(TEMP_DIR, "cluster.rule")
    with open(rule_file, "w") as f: f.write(rule_str)
    
    # --- 3. RUN SKETCHING ---
    print("Running C++ Engine (Mode: SKETCH, K=32)...")
    out_base = os.path.join(TEMP_DIR, "sketch_result.txt")
    run_cpp("sketch", TEMP_DIR, rule_file, out_base, "32", "1")
    
    actual_out = out_base.replace(".txt", "_0.txt")
    
    # --- 4. VALIDATION ---
    print("\n" + "-"*30)
    print("VALIDATION")
    print("-"*30)
    
    if not os.path.exists(actual_out):
        print(f"❌ FAIL: Output file {actual_out} not found.")
        return

    results = {}
    with open(actual_out, "r") as f:
        for line in f:
            parts = list(map(int, line.split()))
            if parts:
                results[parts[0]] = set(parts[1:])

    # CHECK 1: Alice (0) should find Bob (1)
    # Note: Sketch engine filters self-loops, so we expect {1} or {0,1}
    alice_peers = results.get(0, set())
    if 1 in alice_peers:
        print(f"✅ PASS: Alice found Bob! {alice_peers}")
    else:
        print(f"❌ FAIL: Alice missed Bob. Found: {alice_peers}")

    # CHECK 2: Charlie (2) should be isolated
    # For Sketch engine, isolated means EMPTY SET (self-loop filtered)
    charlie_peers = results.get(2, set())
    if len(charlie_peers) == 0:
        print(f"✅ PASS: Charlie has no external peers (Correct for Sketch).")
    elif charlie_peers == {2}:
        print(f"✅ PASS: Charlie found himself.")
    else:
        print(f"❌ FAIL: Charlie found unexpected peers: {charlie_peers}")

    # CHECK 3: Dave (3) should be empty
    dave_peers = results.get(3, set())
    if not dave_peers:
        print(f"✅ PASS: Dave (Dead End) produced no peers.")
    else:
        print(f"❌ FAIL: Dave found non-existent peers: {dave_peers}")

if __name__ == "__main__":
    test_kmv_clusters()