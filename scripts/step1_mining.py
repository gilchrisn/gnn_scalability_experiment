"""
Phase 1: Rule Mining Orchestrator.
Mines logical rules (metapaths) using AnyBURL and caches the best candidates.

Adheres to SRP: Focuses solely on discovery and persistence of rules.
"""
import os
import sys
import json
import logging
from typing import List, Dict, Optional

# Path setup to ensure visibility of src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data import DatasetFactory
from src.bridge import AnyBURLRunner

# --- Configuration ---
TARGET_DATASETS = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "HGB_Freebase"]
MINING_TIMEOUT = 3600  # Seconds
MIN_CONFIDENCE = 0.01

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MiningOrchestrator:
    """
    Manages the lifecycle of rule mining for a specific dataset.
    """
    
    def __init__(self, dataset_key: str):
        self.dataset_key = dataset_key
        self.cfg = config.get_dataset_config(dataset_key)
        self.work_dir = os.path.join(config.DATA_DIR, f"mining_{dataset_key}")
        self.result_file = os.path.join(self.work_dir, "best_rules.json")
        
    def run(self) -> None:
        """Executes the mining pipeline."""
        logger.info(f"[{self.dataset_key}] Starting Mining Pipeline...")
        
        # 1. Load Data
        logger.info(f"[{self.dataset_key}] Loading Graph ({self.cfg.source})...")
        g, _ = DatasetFactory.get_data(self.cfg.source, self.cfg.dataset_name, self.cfg.target_node)
        
        # 2. Initialize Miner (Bridge Pattern)
        runner = AnyBURLRunner(self.work_dir, config.ANYBURL_JAR)
        
        # 3. Export Data for External Engine
        runner.export_for_mining(g)
        
        # 4. Run Mining Process
        runner.run_mining(
            timeout=MINING_TIMEOUT, 
            max_length=4, 
            num_threads=4
        )
        
        # 5. Parse and Cache Results
        top_rules = runner.get_top_k_paths(k=1, min_conf=MIN_CONFIDENCE)
        
        if top_rules:
            self._save_results(top_rules)
            logger.info(f"[{self.dataset_key}] Success. Best Rule: {top_rules[0][1]}")
        else:
            logger.warning(f"[{self.dataset_key}] No valid rules found.")

    def _save_results(self, rules: List) -> None:
        """Persists mining results to disk for Step 3."""
        data = {
            "dataset": self.dataset_key,
            "target_node": self.cfg.target_node,
            "best_rule_str": rules[0][1],
            "confidence": rules[0][0]
        }
        with open(self.result_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"[{self.dataset_key}] Cached results to {self.result_file}")

def main():
    print("=== Phase 1: Mining & Caching ===")
    for ds in TARGET_DATASETS:
        try:
            orch = MiningOrchestrator(ds)
            orch.run()
        except Exception as e:
            logger.error(f"Failed to mine {ds}: {e}", exc_info=True)

if __name__ == "__main__":
    main()