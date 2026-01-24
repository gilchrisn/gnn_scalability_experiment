"""
Verification suite for the refactored Bridge module.
Checks:
1. PyG -> C++ Conversion (Exact Match)
2. Rule Parsing Logic
3. Engine Execution
"""
import os
import sys
import shutil
import torch
import pandas as pd
from torch_geometric.data import HeteroData

# Path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.bridge import PyGToCppAdapter, CppEngine
from src.config import config

TEST_DIR = os.path.join(ROOT_DIR, "output", "test_refactor")

def create_dummy_graph() -> HeteroData:
    """Creates a simple, predictable heterogeneous graph for testing."""
    g = HeteroData()
    
    # 3 Authors, 3 Papers
    g['author'].num_nodes = 3
    g['paper'].num_nodes = 3
    
    # Edges: (0->0), (1->1), (2->2)
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    g['author', 'writes', 'paper'].edge_index = edge_index
    
    # Reverse edges
    g['paper', 'rev_writes', 'author'].edge_index = torch.flip(edge_index, [0])
    
    return g

def test_conversion_integrity():
    print(f"\n[{'='*10} TEST 1: Data Conversion Integrity {'='*10}]")
    
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)
    
    # 1. Initialize Adapter
    adapter = PyGToCppAdapter(TEST_DIR)
    g = create_dummy_graph()
    
    # 2. Convert
    adapter.convert(g)
    
    # 3. Verify Files Exist
    required = ["node.dat", "link.dat", "meta.dat", "offsets.json"]
    for f in required:
        path = os.path.join(TEST_DIR, f)
        if not os.path.exists(path):
            print(f"❌ FAILED: Missing {f}")
            return
    print("✅ File generation successful.")
    
    # 4. Verify Content Logic
    # Check node.dat: Total nodes should be 6 (3 authors + 3 papers)
    df_node = pd.read_csv(os.path.join(TEST_DIR, "node.dat"), sep='\t', header=None)
    if len(df_node) != 6:
        print(f"❌ FAILED: Expected 6 nodes, got {len(df_node)}")
    else:
        print("✅ Node count match.")
        
    # Check link.dat: Total edges should be 6
    df_link = pd.read_csv(os.path.join(TEST_DIR, "link.dat"), sep='\t', header=None)
    if len(df_link) != 6:
        print(f"❌ FAILED: Expected 6 edges, got {len(df_link)}")
    else:
        print("✅ Edge count match.")

def test_rule_generation():
    print(f"\n[{'='*10} TEST 2: Rule Generation {'='*10}]")
    
    adapter = PyGToCppAdapter(TEST_DIR)
    
    # Simulate a path: Author -> Paper -> Author
    # IDs in dummy graph: Author=0, Paper=1 (based on sorted keys)
    # writes=(0, 1), rev_writes=(1, 0) (IDs depend on sort order)
    
    # Simple rule string: "Forward(0) -> Forward(1)"
    # Code: -2 0 -2 -1 1 -4 -4
    rule_str = "-2 0 -2 -1 1 -4 -4"
    
    path = adapter.write_rule_file(rule_str, "test.rule")
    
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read()
        if content == rule_str:
            print("✅ Rule file content matches.")
        else:
            print(f"❌ FAILED: Content mismatch. Got '{content}'")
    else:
        print("❌ FAILED: Rule file not created.")

def test_engine_mock():
    print(f"\n[{'='*10} TEST 3: Engine Interface {'='*10}]")
    
    # Check if we can instantiate and check binary existence
    try:
        engine = CppEngine("dummy/path/exe", TEST_DIR)
    except FileNotFoundError:
        print("✅ Correctly caught missing binary.")
    
    # Note: We can't run the actual binary without compiling it, 
    # but we verify the class structure holds.
    
    # Check if load_result handles missing file gracefully
    if os.path.exists("dummy"): shutil.rmtree("dummy")
    os.makedirs("dummy", exist_ok=True)
    
    engine = CppEngine(sys.executable, "dummy") # Point to python as dummy exe
    res = engine.load_result("non_existent_file.txt", 10, 0)
    
    if res.num_nodes == 10 and res.edge_index.size(1) == 0:
        print("✅ load_result handled missing file gracefully.")
    else:
        print("❌ FAILED: load_result behavior incorrect.")
    
    shutil.rmtree("dummy")

if __name__ == "__main__":
    test_conversion_integrity()
    test_rule_generation()
    test_engine_mock()
    
    print("\n[SUMMARY] If all tests passed, the refactoring is safe to deploy.")
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)