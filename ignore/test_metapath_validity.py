"""
Hypothesis Verification Suite for C++ Graph Engine Failure Modes.

This script creates a controlled synthetic environment to scientifically prove 
the conditions required for the C++ binary (`graph_prep`) to successfully execution.

BACKGROUND:
-----------
During the HGB dataset experiments, we observed "Silent Failures" where the C++ engine 
ran successfully but returned 0 results (`rule_count:0`), despite the provided metapaths 
being valid in the graph schema (e.g., `Movie -> Director -> Movie`).

HYPOTHESIS:
-----------
The C++ Engine will find peers IF AND ONLY IF three conditions are met simultaneously:
1. **Schema Validity**: The edge type IDs in the rule exist in the graph.
2. **Data Connectivity**: The specific edges in `link.dat` actually form the requested cycle 
   (i.e., the path is not just theoretically possible, but populated with data).
3. **Syntax Correctness**: The rule file format strictly adheres to the `-1` flag placement logic.

TEST MATRIX & FINDINGS:
-----------------------
This script generates a toy graph with 4 node types (A, B, C, D) to simulate 4 scenarios:

1. **The "Golden" Path (A->B->C)**: 
   - Schema: Valid. 
   - Data: Dense (Edges exist). 
   - Syntax: Correct.
   - **Result**: SUCCESS (Peers found).

2. **The "Ghost" Path (A->B->D)**: 
   - Schema: Valid (Type B connects to Type D in theory).
   - Data: **EMPTY** (No edges written to `link.dat` for this type).
   - **Result**: FAILURE (0 Peers).
   - **Significance**: This replicates the issue seen with `HGB_DBLP`. AnyBURL mined a rule 
     that was valid in the schema, but the specific subset of data used had no instances 
     of that connection, causing the C++ engine to exit early without error.

3. **The "Impossible" Path (A->???)**:
   - Schema: **INVALID** (Edge ID 99 does not exist).
   - **Result**: FAILURE (0 Peers/Crash).

4. **The "Syntax Error"**:
   - Condition: Valid Data + Valid Schema, but the `-1` flag is misplaced.
   - **Result**: FAILURE (0 Peers).
   - **Significance**: This confirms why our initial manual fixes failed until we aligned 
     the Python writer with the exact state machine logic in `ReadAnyBURLRules.cpp`.

CONCLUSION:
-----------
The "Meta Path Filtering" issue corresponds to **Case 2**. It is not enough for a rule 
to be "valid" (exist in schema). It must be **active** (exist in data). 
The pipeline's "Self-Healing" mechanism solves this by verifying data connectivity 
in Python before invoking C++.
"""

import os
import sys
import subprocess
import pandas as pd
import shutil
import logging

# --- PATH SETUP ---
if os.getcwd().endswith("scripts"):
    os.chdir("..")
ROOT_DIR = os.path.abspath(os.getcwd())
sys.path.append(ROOT_DIR)

from src.config import config

# --- CONFIGURATION ---
# Use a simple top-level folder name to ensure C++ path construction works correctly
# C++ logic: path = folder + "/" + folder + "-cod-global-rules.dat"
DATASET_NAME = "temp_test_validity"
TEST_DIR = os.path.join(ROOT_DIR, DATASET_NAME)
LOG_FILE = "validity_test.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", # Clean format for readability
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

class ValidityTester:
    def __init__(self):
        self.binary = self._resolve_binary()
        if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)
        os.makedirs(TEST_DIR, exist_ok=True)
        
    def _resolve_binary(self):
        binary = config.CPP_EXECUTABLE
        if not os.path.exists(binary):
            binary = os.path.join(ROOT_DIR, "bin", "graph_prep.exe" if os.name == 'nt' else "graph_prep")
        return binary

    def setup_synthetic_data(self):
        """
        Creates a Controlled Synthetic Graph.
        
        Topology:
        1. A -> B (Type 0): Dense connection.
        2. B -> C (Type 1): Dense connection.
        3. B -> D (Type 2): SCHEMA ONLY (No actual edges in link.dat).
        
        This allows us to test "Schema Valid but Data Invalid" paths.
        """
        logger.info(f"--- SETUP: Creating Synthetic Graph in {TEST_DIR} ---")
        
        # 1. create node.dat
        # IDs: 0-9 (Type A), 10-19 (Type B), 20-29 (Type C), 30-39 (Type D)
        nodes = []
        for i in range(10): nodes.append(f"{i}\tA_{i}\t0")      # Type A
        for i in range(10, 20): nodes.append(f"{i}\tB_{i}\t1")  # Type B
        for i in range(20, 30): nodes.append(f"{i}\tC_{i}\t2")  # Type C
        for i in range(30, 40): nodes.append(f"{i}\tD_{i}\t3")  # Type D
        
        with open(os.path.join(TEST_DIR, "node.dat"), "w") as f:
            f.write("\n".join(nodes))
            
        # 2. create link.dat
        links = []
        # Type 0 (A->B): 0->10, 1->11...
        for i in range(10):
            links.append(f"{i}\t{i+10}\t0")
            
        # Type 1 (B->C): 10->20, 11->21...
        for i in range(10):
            links.append(f"{i+10}\t{i+20}\t1")
            
        # Type 2 (B->D): INTENTIONALLY EMPTY (No links added)
        
        with open(os.path.join(TEST_DIR, "link.dat"), "w") as f:
            f.write("\n".join(links))
            
        # 3. create meta.dat
        # Header must be exactly 17 chars long
        with open(os.path.join(TEST_DIR, "meta.dat"), "w") as f:
            f.write("Node Number is : 40\n")
            
        logger.info("[OK] Synthetic Data Created.")
        logger.info("   Nodes: 40 (Types 0,1,2,3)")
        logger.info("   Edges: Type 0 (10 edges), Type 1 (10 edges), Type 2 (0 edges)")

    def run_cpp_simulation(self, rule_str, test_name):
        """Runs the C++ binary with a specific rule string."""
        
        # Name the rule file exactly how C++ expects it:
        # {DATASET_NAME}/{DATASET_NAME}-cod-global-rules.dat
        rule_filename = f"{DATASET_NAME}-cod-global-rules.dat"
        rule_path = os.path.join(TEST_DIR, rule_filename)
        
        with open(rule_path, "w") as f:
            f.write(rule_str)
            
        # Run ExactD+ (fastest check)
        cmd = [self.binary, "ExactD+", DATASET_NAME, "0.1", "0"]
        
        logger.info(f"\n>>> Running Test: {test_name}")
        logger.info(f"    Rule: {rule_str}")
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            output = res.stdout
            
            # Analyze Output
            if "rule_count:1" in output:
                logger.info("    [RESULT] SUCCESS (C++ found peers)")
                return True
            elif "rule_count:0" in output:
                logger.info("    [RESULT] FAILURE (C++ found 0 peers)")
                return False
            else:
                logger.info("    [RESULT] CRASH/ERROR")
                logger.info(f"    Stderr: {res.stderr}")
                return False
        except Exception as e:
            logger.error(f"    Execution failed: {e}")
            return False

    def test_all_assumptions(self):
        """
        Matrix of tests to prove IFF conditions.
        """
        
        # --- TEST 1: The "Golden" Path ---
        # Path: A -> B -> C
        # Edge IDs: 0 then 1.
        # Format: Correct (-2 0 -2 -1 1 ...)
        # Expectation: TRUE
        rule_valid = "-2 0 -2 -1 1 -5 -1 -4 -4"
        res1 = self.run_cpp_simulation(rule_valid, "1. Valid Schema + Valid Data + Valid Format")
        
        # --- TEST 2: The "Ghost" Path (Data Invalidity) ---
        # Path: A -> B -> D
        # Edge IDs: 0 then 2.
        # Schema: Valid (Type 2 exists in theory).
        # Data: Empty (We wrote 0 edges for Type 2).
        # Expectation: FALSE
        rule_empty = "-2 0 -2 -1 2 -5 -1 -4 -4"
        res2 = self.run_cpp_simulation(rule_empty, "2. Valid Schema + EMPTY Data")
        
        # --- TEST 3: The "Broken" Path (Schema Invalidity) ---
        # Path: A -> ???
        # Edge ID: 99 (Does not exist).
        # Expectation: FALSE (Crash or 0)
        rule_schema_bad = "-2 0 -2 -1 99 -5 -1 -4 -4"
        res3 = self.run_cpp_simulation(rule_schema_bad, "3. INVALID Schema (Edge 99)")
        
        # --- TEST 4: The "Syntax Error" (Format Invalidity) ---
        # Path: A -> B -> C (Same as Test 1)
        # Format: BAD (Put -1 at the end instead of before last ID)
        # Old broken format: -2 0 -2 1 -1 ...
        # Expectation: FALSE (C++ state machine fails)
        rule_syntax_bad = "-2 0 -2 1 -1 -5 -1 -4 -4"
        res4 = self.run_cpp_simulation(rule_syntax_bad, "4. Valid Data + BAD Rule Syntax")

        # --- SUMMARY ---
        logger.info("\n" + "="*40)
        logger.info("HYPOTHESIS VERIFICATION TABLE")
        logger.info("="*40)
        logger.info(f"1. Everything Correct  -> {'PASS' if res1 else 'FAIL'} (Expected: PASS)")
        logger.info(f"2. Data Empty          -> {'FAIL' if not res2 else 'PASS'} (Expected: FAIL)")
        logger.info(f"3. Schema Invalid      -> {'FAIL' if not res3 else 'PASS'} (Expected: FAIL)")
        logger.info(f"4. Syntax Invalid      -> {'FAIL' if not res4 else 'PASS'} (Expected: FAIL)")
        
        if res1 and not res2 and not res3 and not res4:
            logger.info("\nCONCLUSION: The pipeline works IF AND ONLY IF Schema, Data, and Syntax are correct.")
            logger.info("The 'Meta Path Filtering' issue you suspected corresponds to Case 2 (Data Empty).")
        else:
            logger.error("\nCONCLUSION: Hypothesis rejected. Behavior is unexpected.")

if __name__ == "__main__":
    tester = ValidityTester()
    tester.setup_synthetic_data()
    tester.test_all_assumptions()