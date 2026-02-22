import os
import time
import copy
import pandas as pd
from typing import Dict, List, Optional
import torch

from src.backend.base import BackendFactory
from src.data.factory import DatasetFactory
from src.utils import SchemaMatcher
from .config import ExperimentConfig
from .mining import Phase1Mining
from .training import Phase2FoundationTraining
from .robustness import Phase3Robustness
from .depth import Phase4Depth
from ..config import config

class MasterExperimentOrchestrator:
    """
    Facade to execute the full experimental protocol V2.1.
    Implements Idempotent Execution and Graceful Degradation for OOM failures.
    """
    
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.results_dir = config.RESULTS_DIR

    def _is_rule_fully_evaluated(self, rule_str: str) -> bool:
        """
        Inventory Reconciliation: Checks if a rule has already been fully processed.
        
        Args:
            rule_str: The metapath rule string to check.
            
        Returns:
            True if all expected K-values for this rule and model architecture 
            exist in the robustness logs, False otherwise.
        """
        rob_file = os.path.join(self.results_dir, f"robustness_{self.cfg.dataset}_{self.cfg.model_arch}.csv")
        
        if not os.path.exists(rob_file):
            return False
            
        try:
            df = pd.read_csv(rob_file)
            # Filter by Model Architecture and Rule
            mask = (df['ModelArch'] == self.cfg.model_arch) & (df['Rule'] == rule_str)
            existing_k_values = set(df[mask]['K'].dropna().unique())
            
            # 0 represents the Exact run baseline
            expected_k_values = set([0] + self.cfg.k_values)
            
            return expected_k_values.issubset(existing_k_values)
        except Exception as e:
            print(f"    [Reconciliation Error] Could not parse log: {e}")
            return False

    def run_all(self) -> None:
        start_global = time.perf_counter()
        print("="*60)
        print(f"STARTING MASTER PROTOCOL V2.1: {self.cfg.dataset}")
        print("="*60)

        # 1. Load Global State ONCE
        dataset_cfg = config.get_dataset_config(self.cfg.dataset)
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg.source, dataset_cfg.dataset_name, dataset_cfg.target_node
        )

        try:
            miner = Phase1Mining(self.cfg)
            best_rules = miner.execute()

            print(f"\n>>> [Phase 2] Building Model Zoo (Depths: {self.cfg.model_depths})")
            self._train_specific_model(self.cfg.standard_depth)
            for depth in self.cfg.model_depths:
                if depth != self.cfg.standard_depth:
                    self._train_specific_model(depth)

            # 2. Rule-Level Execution with Graceful Degradation
            for i, rule in enumerate(best_rules, 1):
                print(f"\n{'*'*40}")
                print(f"EVALUATING METAPATH {i}/{len(best_rules)}: {rule}")
                print(f"{'*'*40}")

                if self._is_rule_fully_evaluated(rule):
                    print("    [Skip] Inventory Reconciliation: Rule already fully evaluated.")
                    continue

                path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in rule.split(',')]
                backend = BackendFactory.create('cpp',
                    executable_path=config.CPP_EXECUTABLE,
                    temp_dir=config.TEMP_DIR,
                    device=config.DEVICE
                )
                backend.initialize(g_hetero, path_list, info)

                # --- GRACEFUL DEGRADATION BLOCK ---
                print("    [Orchestrator] Materializing EXACT graph...")
                g_exact = None
                exact_time = None
                
                t_start_exact = time.perf_counter()
                try:
                    g_exact = backend.materialize_exact()
                    exact_time = backend.get_prep_time() or (time.perf_counter() - t_start_exact)
                    
                    if g_exact.edge_index.size(1) == 0:
                        print(f"    [WARNING] Rule '{rule}' produced 0 edges! Skipping.")
                        backend.cleanup()
                        continue
                except (MemoryError, RuntimeError) as e:
                    if "memory" in str(e).lower() or "alloc" in str(e).lower():
                        print(f"    [CRITICAL] EXACT materialization OOM. Triggering Graceful Degradation.")
                        g_exact = None
                        exact_time = -1.0
                    else:
                        raise e  # Rethrow non-memory related fatal errors
                # ----------------------------------

                print("    [Orchestrator] Materializing KMV graphs...")
                g_kmv_dict = {}
                kmv_times = {}
                
                for k in self.cfg.k_values:
                    t_start_kmv = time.perf_counter()
                    g_kmv_dict[k] = backend.materialize_kmv(k=k, seed=self.cfg.seed)
                    kmv_times[k] = backend.get_prep_time() or (time.perf_counter() - t_start_kmv)

                # 3. Inject State into Consumers
                p3_cfg = copy.deepcopy(self.cfg)
                p3_cfg.current_model_name = f"{self.cfg.model_arch}_L{self.cfg.standard_depth}"
                
                robustness = Phase3Robustness(
                    cfg=p3_cfg, rule_str=rule, g_exact=g_exact, exact_time=exact_time,
                    g_kmv_dict=g_kmv_dict, kmv_times=kmv_times, info=info
                )
                robustness.execute()

                depth_study = Phase4Depth(
                    cfg=self.cfg, best_rule=rule, g_exact=g_exact,
                    g_kmv=g_kmv_dict[self.cfg.depth_study_k], info=info
                )
                depth_study.execute()

                backend.cleanup()

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Experiment aborted: {e}")
            import traceback
            traceback.print_exc()

        total_time = time.perf_counter() - start_global
        print(f"\n[DONE] Total Protocol Time: {total_time:.2f}s")

    def _train_specific_model(self, depth: int) -> None:
        step_cfg = copy.deepcopy(self.cfg)
        step_cfg.current_model_name = f"{self.cfg.model_arch}_L{depth}"
        trainer = Phase2FoundationTraining(step_cfg, num_layers=depth)
        trainer.execute()