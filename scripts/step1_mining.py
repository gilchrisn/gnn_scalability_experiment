import os
import sys
import time
import textwrap
import pandas as pd
from typing import List, Tuple

# Path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.config import config
from src.data import DatasetFactory
from src.bridge import AnyBURLRunner

# --- CONFIGURATION ---
TARGET_DATASETS = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "HGB_Freebase"]
MINING_TIMEOUT = 3600  # 1 Hour (Set to 60 for debug)
REPORT_FILE = os.path.join(ROOT_DIR, "output", "MINING_REPORT.md")

class MiningAuditor:
    def __init__(self):
        self.buffer = []
        self.add_header("Phase 1: Deep Rule Mining Audit")
        
    def add_header(self, text):
        self.buffer.append(f"# {text}\n")
        self.buffer.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def add_section(self, title, content):
        self.buffer.append(f"## {title}\n")
        self.buffer.append(content + "\n\n")

    def save(self):
        with open(REPORT_FILE, "w") as f:
            f.writelines(self.buffer)
        print(f"\n[Report] Saved to {REPORT_FILE}")

def analyze_rule_quality(rule_file: str) -> dict:
    """
    Parses raw rule file to calculate quality metrics.
    """
    if not os.path.exists(rule_file):
        return {"error": "File not found"}

    stats = {
        "total_rules": 0,
        "generic_rules": 0,  # Rules with variables (X, Y)
        "grounded_rules": 0, # Rules with constants (person_0)
        "cyclic_rules": 0,   # Useful for GNNs
        "top_rules": []
    }

    with open(rule_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            stats["total_rules"] += 1
            
            parts = line.split('\t')
            if len(parts) < 4: continue
            
            rule_content = parts[3]
            confidence = float(parts[2])
            
            # Check Generality (Logic: Constants usually start with lowercase or have underscores specific to data)
            # AnyBURL convention: Variables are A, B, C, X, Y. Constants are usually lower case.
            # Robust check: If rule body has 'X' and head has 'Y' (or vice versa) and NO specific entities.
            
            is_generic = True
            # Simple heuristic: Grounded rules usually mention specific IDs often containing numbers or underscores
            # Better check: AnyBURL uses UpperCase for vars, LowerCase/Numbers for constants
            
            head = rule_content.split(" <= ")[0]
            if "(" in head:
                args = head.split("(")[1].replace(")", "").split(",")
                for arg in args:
                    if not arg[0].isupper(): 
                        is_generic = False
                        break
            
            if is_generic:
                stats["generic_rules"] += 1
                if "Cyclic" in line or len(stats["top_rules"]) < 10:
                    # Save high confidence generic rules
                    if len(stats["top_rules"]) < 5:
                        stats["top_rules"].append(f"`{rule_content}` (Conf: {confidence})")
            else:
                stats["grounded_rules"] += 1

    # Metrics
    total = max(1, stats["total_rules"])
    stats["generality_ratio"] = stats["generic_rules"] / total
    return stats

def run_mining_phase():
    auditor = MiningAuditor()
    
    for ds in TARGET_DATASETS:
        print(f"\n=== Processing {ds} ===")
        
        # 1. Setup
        cfg = config.get_dataset_config(ds)
        work_dir = os.path.join(config.DATA_DIR, f"mining_{ds}")
        os.makedirs(work_dir, exist_ok=True)
        
        # 2. Load & Export
        print("  Loading Graph...")
        g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        runner = AnyBURLRunner(work_dir, config.ANYBURL_JAR)
        runner.export_graph(g)
        
        # 3. Configure (HEAVY RUN)
        config_path = os.path.join(work_dir, "config-learn.properties")
        safe_triples = runner.triples_file.replace(os.sep, '/')
        safe_rules = runner.rules_file.replace(os.sep, '/')
        
        # Note: SNAPSHOTS_AT controls runtime.
        custom_config = textwrap.dedent(f"""
            PATH_TRAINING = {safe_triples}
            PATH_OUTPUT = {safe_rules}
            SNAPSHOTS_AT = {MINING_TIMEOUT}
            WORKER_THREADS = 4
            MAX_LENGTH_CYCLIC = 4
            THRESHOLD_CORRECT_PREDICTIONS = 2
            THRESHOLD_CONFIDENCE = 0.0001
            ZERO_RULES_ACTIVE = false
        """).strip()
        
        with open(config_path, 'w') as f:
            f.write(custom_config)
            
        # 4. Run
        print(f"  Mining for {MINING_TIMEOUT} seconds...")
        runner.run_mining(timeout=MINING_TIMEOUT, max_length=4, num_threads=4)
        
        # 5. Audit
        print("  Auditing Results...")
        quality = analyze_rule_quality(runner.rules_file)
        
        # 6. Report
        report_text = f"""
- **Total Rules Found:** {quality['total_rules']}
- **Generic Rules:** {quality['generic_rules']}
- **Grounded (Noise):** {quality['grounded_rules']}
- **Generality Ratio:** {quality['generality_ratio']:.4f} (Target > 0.1)

**Top 5 Generic Rules:**
""" + "\n".join([f"- {r}" for r in quality['top_rules']])

        auditor.add_section(f"Dataset: {ds}", report_text)
        
    auditor.save()

if __name__ == "__main__":
    run_mining_phase()