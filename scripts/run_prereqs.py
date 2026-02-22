import os
import sys
import shutil
import subprocess
import pandas as pd
import random
import time
import re
import torch
from typing import List, Dict, Tuple

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.experiments.config import ExperimentConfig
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.experiments.mining import Phase1Mining
from src.utils import SchemaMatcher

# --- Configuration ---
# The 4 HGB datasets you mentioned
TARGET_DATASETS = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "HGB_Freebase"]

# Parameters for Journal Experiments
LAMBDA_VALUES = ["0.01", "0.02", "0.03", "0.04", "0.05"]
K_VALUES = [2, 4, 8, 16, 32]
DEFAULT_K_DEGREE = "32"
DEFAULT_K_HINDEX = "4"
DEFAULT_LAMBDA = "0.05"
FIXED_BETA = "0.1"  # Pruning parameter for PerD+

class JournalBenchmarkOrchestrator:
    def __init__(self, dataset_key: str):
        self.dataset_key = dataset_key
        # Map config key (HGB_ACM) to C++ folder name (HGBn-ACM)
        self.folder_name = f"HGBn-{dataset_key.split('_')[1]}"
        self.binary_path = config.CPP_EXECUTABLE
        
        # Directories
        self.data_dir = os.path.join(project_root, self.folder_name)
        self.res_dir = os.path.join(project_root, "global_res", self.folder_name)
        self.df1_dir = os.path.join(self.res_dir, "df1")
        self.hf1_dir = os.path.join(self.res_dir, "hf1")

        # Runtime State
        self.g_hetero = None
        self.metapath_str = None
        self.edge_map = {} # Maps edge tuple -> Integer ID

    def _ensure_cpp_binary(self):
        if not os.path.exists(self.binary_path):
            # Fallback for Linux/Windows differences
            alt_path = os.path.join(project_root, "bin", "graph_prep")
            if os.path.exists(alt_path):
                self.binary_path = alt_path
            else:
                raise FileNotFoundError(f"C++ binary not found at {self.binary_path}. Compile it first!")

    def run(self) -> List[Dict]:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {self.dataset_key} -> {self.folder_name}")
        print(f"{'='*60}")
        
        self._ensure_cpp_binary()
        
        # 1. Mining Phase (Get the Rule)
        self._step_mining()
        
        # 2. Data Staging (PyG -> .dat files)
        self._step_staging()
        
        # 3. Rule Compilation (String -> C++ Int Code)
        self._step_compile_rule()
        
        # 4. Prerequisite: Ground Truth Generation
        self._step_ground_truth()
        
        # 5. Run Experiments
        results = []
        results.extend(self._exp_degree_effectiveness())
        results.extend(self._exp_hindex_effectiveness())
        results.extend(self._exp_scalability())
        
        return results

    def _step_mining(self):
        """Runs AnyBURL (Phase 1) to get the best metapath."""
        print("\n>>> [Step 1] Mining Metapaths...")
        cfg = ExperimentConfig(dataset=self.dataset_key)
        miner = Phase1Mining(cfg)
        self.metapath_str = miner.execute()
        print(f"    Selected Rule: {self.metapath_str}")

    def _step_staging(self):
        """Converts HeteroData to C++ TSV format."""
        print("\n>>> [Step 2] Staging Data...")
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.df1_dir, exist_ok=True)
        os.makedirs(self.hf1_dir, exist_ok=True)

        # Load Graph
        cfg = config.get_dataset_config(self.dataset_key)
        self.g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        # Determine Edge ID Mapping (Deterministic Sort)
        sorted_edges = sorted(list(self.g_hetero.edge_types))
        self.edge_map = {et: i for i, et in enumerate(sorted_edges)}
        
        # Run Adapter
        adapter = PyGToCppAdapter(self.data_dir)
        adapter.convert(self.g_hetero)
        
        # Generate Query Nodes (for PerD experiments)
        self._generate_qnodes()

    def _generate_qnodes(self):
        """Selects 100 random nodes for personalized queries."""
        node_file = os.path.join(self.data_dir, "node.dat")
        valid_ids = []
        
        # Read valid IDs from node.dat (first 5000 lines to save time)
        with open(node_file, 'r') as f:
            for i, line in enumerate(f):
                if i > 5000: break
                parts = line.strip().split('\t')
                if parts: valid_ids.append(parts[0])
                
        # Sample
        if not valid_ids: valid_ids = ["0"]
        selected = random.sample(valid_ids, min(100, len(valid_ids)))
        
        qnode_path = os.path.join(self.data_dir, f"qnodes_{self.folder_name}.dat")
        with open(qnode_path, 'w') as f:
            f.write("\n".join(selected))
        print(f"    Generated query nodes at: {qnode_path}")

    def _step_compile_rule(self):
        """Compiles 'paper_to_author,author_to_paper' into '-2 0 -2 1...'."""
        print("\n>>> [Step 3] Compiling Rule for C++ Engine...")
        
        # Parse string to edge tuples
        path_list = [SchemaMatcher.match(s.strip(), self.g_hetero) for s in self.metapath_str.split(',')]
        
        # Convert to IDs
        try:
            rule_ids = [self.edge_map[edge] for edge in path_list]
        except KeyError as e:
            print(f"    [Error] Mined rule contains edge {e} not found in sorted schema.")
            raise

        # Generate Stack Machine Bytecode
        # Format: -2 <ID> (Forward) ... -1 <LastID> (Trigger) -5 -1 -4 (Pop) ...
        parts = []
        for i, eid in enumerate(rule_ids):
            parts.append("-2")
            if i == len(rule_ids) - 1:
                parts.append("-1") # Trigger variable mode on last edge
            parts.append(str(eid))
        
        # Cleanup stack (Pop for every push)
        parts.extend(["-5", "-1"])
        for _ in rule_ids: parts.append("-4")
        
        rule_content = " ".join(parts)
        
        # Write to specific filename required by C++: {folder}/{folder}-cod-global-rules.dat
        filename = f"{self.folder_name}-cod-global-rules.dat"
        file_path = os.path.join(self.data_dir, filename)
        
        with open(file_path, "w") as f:
            f.write(rule_content)
            
        print(f"    Written Rule: {rule_content}")
        print(f"    Path: {file_path}")

    def _step_ground_truth(self):
        """Runs the 4 Exact commands to generate ground truth files."""
        print("\n>>> [Step 4] Generating Ground Truth (ExactD/H)...")
        
        # We need to run this for EVERY Lambda we plan to test
        # However, to save time, we usually just run for the max lambda if the logic permits,
        # but the C++ tool creates specific filenames like 'hg_global_r0.05.res'.
        # So we iterate all target lambdas.
        
        for lam in LAMBDA_VALUES:
            # Check if exists
            target_file = os.path.join(self.df1_dir, f"hg_global_r{lam}.res")
            if os.path.exists(target_file):
                continue
                
            print(f"    Generating for Lambda={lam}...")
            
            # 1. ExactD+ (Inclusive)
            self._run_cpp(["ExactD+", self.folder_name, lam, "0"], 
                          stdout_path=os.path.join(self.df1_dir, f"hg_global_r{lam}.res"))
            
            # 2. ExactD (Strict)
            self._run_cpp(["ExactD", self.folder_name, lam, "0"], 
                          stdout_path=os.path.join(self.df1_dir, f"hg_global_greater_r{lam}.res"))
                          
            # 3. ExactH+ (Inclusive)
            self._run_cpp(["ExactH+", self.folder_name, lam, "0"], 
                          stdout_path=os.path.join(self.hf1_dir, f"hg_global_r{lam}.res"))
            
            # 4. ExactH (Strict)
            self._run_cpp(["ExactH", self.folder_name, lam, "0"], 
                          stdout_path=os.path.join(self.hf1_dir, f"hg_global_greater_r{lam}.res"))

    def _exp_degree_effectiveness(self):
        """Experiment 1: Degree Centrality (GloD / PerD)."""
        print("\n>>> [Exp 1] Degree Centrality...")
        res = []
        
        # A. Varying Lambda (Fixed K)
        for lam in LAMBDA_VALUES:
            # GloD
            out = self._run_cpp(["GloD", self.folder_name, lam, "0", DEFAULT_K_DEGREE])
            f1, time_val = self._parse_metrics(out)
            res.append(self._rec("Degree_Lambda", "GloD", lam, DEFAULT_K_DEGREE, f1, time_val))
            
            # PerD
            out = self._run_cpp(["PerD", self.folder_name, lam, FIXED_BETA, DEFAULT_K_DEGREE])
            _, time_val = self._parse_metrics(out)
            res.append(self._rec("Degree_Lambda", "PerD", lam, DEFAULT_K_DEGREE, -1, time_val)) # PerD outputs accuracy in some versions, but time is key
            
        # B. Varying K (Fixed Lambda)
        for k in K_VALUES:
            k_str = str(k)
            # GloD
            out = self._run_cpp(["GloD", self.folder_name, DEFAULT_LAMBDA, "0", k_str])
            f1, time_val = self._parse_metrics(out)
            res.append(self._rec("Degree_K", "GloD", DEFAULT_LAMBDA, k_str, f1, time_val))
            
        return res

    def _exp_hindex_effectiveness(self):
        """Experiment 2: H-Index Centrality (GloH / PerH)."""
        print("\n>>> [Exp 2] H-Index Centrality...")
        res = []
        
        # A. Varying Lambda
        for lam in LAMBDA_VALUES:
            # GloH
            out = self._run_cpp(["GloH", self.folder_name, lam, "0", DEFAULT_K_HINDEX])
            f1, time_val = self._parse_metrics(out)
            res.append(self._rec("HIndex_Lambda", "GloH", lam, DEFAULT_K_HINDEX, f1, time_val))
            
        # B. Varying K
        for k in K_VALUES:
            k_str = str(k)
            # GloH
            out = self._run_cpp(["GloH", self.folder_name, DEFAULT_LAMBDA, "0", k_str])
            f1, time_val = self._parse_metrics(out)
            res.append(self._rec("HIndex_K", "GloH", DEFAULT_LAMBDA, k_str, f1, time_val))
            
        return res

    def _exp_scalability(self):
        """Experiment 3: Efficiency vs Path Length."""
        print("\n>>> [Exp 3] Scalability...")
        res = []
        
        # 1. Split rules by length (creates .dat1, .dat2, etc)
        self._run_cpp(["lensplit", self.folder_name])
        
        # 2. Test lengths
        lengths = ["dat1", "dat2", "dat3"]
        for ext in lengths:
            length_val = ext[-1]
            # Check if file exists
            rule_path = os.path.join(self.data_dir, f"{self.folder_name}-cod-global-rules.{ext}")
            if not os.path.exists(rule_path): continue
            
            # Run Scale test
            out = self._run_cpp(["scale", self.folder_name, "0.05", "d", f".{ext}"])
            _, time_val = self._parse_metrics(out)
            res.append(self._rec("Scalability", "Sketch", "0.05", "32", -1, time_val, extra=length_val))
            
        return res

    def _run_cpp(self, args: List[str], stdout_path: str = None) -> str:
        """Executes the binary and optionally writes stdout to file."""
        cmd = [self.binary_path] + args
        # print(f"    Exec: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = res.stdout
            
            if stdout_path:
                with open(stdout_path, "w") as f:
                    f.write(output)
            
            return output
        except subprocess.CalledProcessError as e:
            print(f"    [CRASH] Command failed: {e}")
            print(f"    STDERR: {e.stderr}")
            return ""

    def _parse_metrics(self, output: str) -> Tuple[float, float]:
        """Extracts goodness (F1) and time from stdout."""
        goodness = 0.0
        time_val = 0.0
        for line in output.split('\n'):
            if "goodness:" in line:
                try: goodness = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                except: pass
            if "time:" in line:
                try: time_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                except: pass
        return goodness, time_val

    def _rec(self, exp, method, lam, k, f1, t, extra=""):
        return {
            "Dataset": self.dataset_key,
            "Experiment": exp,
            "Method": method,
            "Lambda": lam,
            "K": k,
            "F1_Score": f1,
            "Time": t,
            "Extra": extra
        }

def main():
    all_results = []
    
    print("STARTING JOURNAL EXTENSION BENCHMARKS")
    print(f"Binary: {config.CPP_EXECUTABLE}")
    
    for ds in TARGET_DATASETS:
        try:
            orch = JournalBenchmarkOrchestrator(ds)
            ds_results = orch.run()
            all_results.extend(ds_results)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed on {ds}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save Results
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(project_root, "journal_benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n\n[DONE] Results saved to: {csv_path}")
        print(df.head())
    else:
        print("\n[DONE] No results generated.")

if __name__ == "__main__":
    main()