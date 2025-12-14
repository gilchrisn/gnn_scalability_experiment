import sys
import os
import time
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DatasetFactory
from src.adapter import PyGToCppAdapter
from src import config

def test_materialization(dataset_name, target_config):
    print(f"\n{'='*60}")
    print(f"TESTING MATERIALIZATION: {dataset_name}")
    print(f"{'='*60}")

    # 1. Load Data
    print("1. Loading Data...")
    g, info = DatasetFactory.get_data(
        source_type=target_config['source'],
        dataset_name=target_config['dataset_name'],
        target_ntype=target_config['target_node']
    )

    # 2. Run Adapter
    print("2. Running Adapter...")
    adapter = PyGToCppAdapter(config.TEMP_DIR)
    adapter.convert(g)

    # 3. Generate Rule
    print("3. Generating Rule String...")
    # We must use the exact keys present in the graph
    metapath = target_config['metapath']
    
    try:
        rule_str = adapter.generate_cpp_rule(metapath)
        print(f"   Generated Rule: {rule_str}")
    except ValueError as e:
        print(f"   [Error] Metapath mismatch: {e}")
        print("   (Make sure config.TRAIN_METAPATH matches the actual edge types in the graph)")
        return

    rule_path = adapter.write_rule_file(rule_str)

    # 4. Run C++ Materialization
    print("4. Running C++ Executable...")
    output_file = os.path.join(config.TEMP_DIR, "materialized_adj.txt")
    
    cmd = [
        config.CPP_EXECUTABLE,
        "materialize",
        config.TEMP_DIR, # Input dir (contains node.dat, link.dat)
        rule_path,       # Rule file
        output_file      # Output file
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        duration = time.time() - start_time
        print(f"   [C++ Output] {result.stdout.strip()}")
        print(f"   SUCCESS: C++ finished in {duration:.4f}s")
        
        # 5. Verify Output
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"   Output File Created: {output_file} ({size_mb:.2f} MB)")
            
            # Peek at first few lines
            with open(output_file, 'r') as f:
                head = [next(f).strip() for _ in range(3)]
            print(f"   Sample Content: {head}")
        else:
            print("   [FAIL] Output file was not created!")

    except subprocess.CalledProcessError as e:
        print(f"   [FAIL] C++ Error:\n{e.stderr}")
    except FileNotFoundError:
        print(f"   [FAIL] Executable not found at {config.CPP_EXECUTABLE}")

if __name__ == "__main__":
    # Test with HGB_DBLP first (It's small and fast)
    cfg = config.DATASET_CONFIGS['HGB_DBLP']
    test_materialization('HGB_DBLP', cfg)