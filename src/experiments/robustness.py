import os
import json
import torch
import pandas as pd
from typing import Dict, Any, Optional
from torch_geometric.data import Data

from ..config import config
from ..models import get_model
from ..analysis import CosineFidelityMetric, PredictionAgreementMetric
from .base import AbstractExperimentPhase
from .config import ExperimentConfig

class Phase3Robustness(AbstractExperimentPhase):
    """
    Phase 3: Robustness Study (Fidelity vs K).
    Implements Graceful Degradation for OOM baselines and Incremental Persistence.
    """
    VOID_FLAG = "VOID_OOM"

    def __init__(self, 
                 cfg: ExperimentConfig, 
                 rule_str: str, 
                 g_exact: Optional[Data], 
                 exact_time: Optional[float], 
                 g_kmv_dict: Dict[int, Data], 
                 kmv_times: Dict[int, float], 
                 info: Dict[str, Any]):
        super().__init__(cfg)
        self.rule_str = rule_str
        self.g_exact = g_exact
        self.exact_time = exact_time
        self.g_kmv_dict = g_kmv_dict
        self.kmv_times = kmv_times
        self.info = info
        self.out_file = os.path.join(self.output_dir, f"robustness_{self.cfg.dataset}_{self.cfg.model_arch}.csv")

    def _save_incremental(self, record: Dict[str, Any]) -> None:
        """Appends a single record to the CSV atomically."""
        df = pd.DataFrame([record])
        header = not os.path.exists(self.out_file)
        df.to_csv(self.out_file, mode='a', header=header, index=False)

    def execute(self) -> None:
        print(f"\n>>> [Phase 3] Robustness Study (Exact vs KMV)")
        print(f"    Rule: {self.rule_str}")

        model, mapper_cfg = self._load_frozen_model(self.info)
        metric_cos = CosineFidelityMetric()
        metric_agree = PredictionAgreementMetric()
        test_mask = self.info['masks']['test']
        target_dim = mapper_cfg['global_max_dim']

        z_exact = None
        acc_exact = self.VOID_FLAG
        num_edges_exact = self.VOID_FLAG
        
        # --- A. Exact Run Evaluation (with Degradation check) ---
        if self.g_exact is not None:
            num_nodes = self.g_exact.num_nodes
            num_edges_exact = self.g_exact.edge_index.size(1)
            print(f"    [Oracle] Nodes: {num_nodes} | Exact Edges: {num_edges_exact}")
            
            try:
                z_exact = self._inference(model, self.g_exact, target_dim)
                acc_exact = self._compute_accuracy(z_exact, self.info['labels'], test_mask)
                print(f"    -> Exact Time: {self.exact_time:.4f}s | Acc: {acc_exact:.4f}")
            except (RuntimeError, MemoryError) as e:
                print("    [OOM ERROR] Exact graph inference failed due to memory limits.")
                z_exact = None
                acc_exact = self.VOID_FLAG
        else:
            print("    [Oracle] SKIPPED (Prior Materialization OOM).")
            # We must derive num_nodes from a KMV graph since Exact is None
            num_nodes = next(iter(self.g_kmv_dict.values())).num_nodes

        # Persist Exact Baseline Row
        self._save_incremental({
            "Dataset": self.cfg.dataset, "ModelArch": self.cfg.model_arch, 
            "Method": "Exact", "K": 0, "Rule": self.rule_str,
            "NumNodes": num_nodes, "ExactEdges": num_edges_exact, "KMVEdges": num_edges_exact,
            "Time": self.exact_time if self.exact_time != -1.0 else self.VOID_FLAG, 
            "Accuracy": acc_exact, "Fidelity": 1.0, 
            "Agreement": 1.0, "Speedup": 1.0
        })

        # --- B. Approximation Loop ---
        for k in self.cfg.k_values:
            g_kmv = self.g_kmv_dict[k]
            t_kmv = self.kmv_times[k]
            num_edges_kmv = g_kmv.edge_index.size(1)

            z_kmv = self._inference(model, g_kmv, target_dim)
            acc_kmv = self._compute_accuracy(z_kmv, self.info['labels'], test_mask)

            # Dependent metrics fall back to VOID_FLAG if Oracle failed
            if z_exact is not None:
                fid_score = metric_cos.calculate(z_exact[test_mask], z_kmv[test_mask])
                agree_score = metric_agree.calculate(z_exact[test_mask], z_kmv[test_mask])
                speedup = self.exact_time / t_kmv if t_kmv > 0 else 0
            else:
                fid_score = self.VOID_FLAG
                agree_score = self.VOID_FLAG
                speedup = self.VOID_FLAG

            print(f"        -> [K={k}] Time: {t_kmv:.4f}s | Acc: {acc_kmv:.4f} | Fid: {fid_score if isinstance(fid_score, str) else f'{fid_score:.4f}'}")

            self._save_incremental({
                "Dataset": self.cfg.dataset, "ModelArch": self.cfg.model_arch,
                "Method": "KMV", "Rule": self.rule_str, "K": k, 
                "NumNodes": num_nodes, "ExactEdges": num_edges_exact, "KMVEdges": num_edges_kmv,
                "Time": t_kmv, "Accuracy": acc_kmv, "Fidelity": fid_score,
                "Agreement": agree_score, "Speedup": speedup
            })

    def _inference(self, model: torch.nn.Module, graph: Data, target_dim: int) -> torch.Tensor:
        with torch.no_grad():
            x = graph.x.to(config.DEVICE)
            edge_index = graph.edge_index.to(config.DEVICE)
            if x.size(1) != target_dim:
                x = torch.nn.functional.pad(x, (0, target_dim - x.size(1)))
            return model(x, edge_index)

    def _compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
        pred = logits[mask].argmax(dim=1)
        true = labels[mask].to(logits.device)
        return (pred == true).float().mean().item()

    def _load_frozen_model(self, info: Dict[str, Any]) -> tuple:
        model_name = self.cfg.current_model_name
        model_path = config.get_model_path(self.cfg.dataset, model_name)
        mapper_path = model_path.replace('.pt', '_mapper.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Frozen model not found: {model_path}. Run Phase 2 first.")

        with open(mapper_path, 'r') as f:
            mapper_cfg = json.load(f)

        model = get_model(
            self.cfg.model_arch, 
            mapper_cfg['global_max_dim'], 
            info['num_classes'], 
            config.HIDDEN_DIM,
            num_layers=self.cfg.standard_depth
        )
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
        model.to(config.DEVICE).eval()
        return model, mapper_cfg