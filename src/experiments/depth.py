import os
import json
import torch
import pandas as pd
from typing import Dict, Any, Optional

from ..config import config
from ..models import get_model
from ..analysis import CosineFidelityMetric, DirichletEnergyMetric, MeanAverageDistanceMetric
from .base import AbstractExperimentPhase
from .config import ExperimentConfig
from torch_geometric.data import Data

class Phase4Depth(AbstractExperimentPhase):
    """
    Phase 4: Depth vs. Bias Accumulation.
    Implements Graceful Degradation for Exact structures and Incremental Appends.
    """
    VOID_FLAG = "VOID_OOM"

    def __init__(self,
                 cfg: ExperimentConfig,
                 best_rule: str,
                 g_exact: Optional[Data],
                 g_kmv: Data,
                 info: Dict[str, Any]):
        super().__init__(cfg)
        self.best_rule = best_rule
        self.g_exact = g_exact
        self.g_kmv = g_kmv
        self.info = info
        self.out_file = os.path.join(self.output_dir, f"depth_bias_{self.cfg.dataset}_{self.cfg.model_arch}.csv")

    def _save_incremental(self, record: Dict[str, Any]) -> None:
        df = pd.DataFrame([record])
        header = not os.path.exists(self.out_file)
        df.to_csv(self.out_file, mode='a', header=header, index=False)

    def execute(self) -> None:
        print(f"\n>>> [Phase 4] Depth Bias Study (Models L{min(self.cfg.model_depths)}-L{max(self.cfg.model_depths)})")
        print(f"    Fixed Rule: {self.best_rule}")
        print(f"    Fixed K: {self.cfg.depth_study_k}")

        metric_cos = CosineFidelityMetric()
        metric_dir = DirichletEnergyMetric()
        metric_mad = MeanAverageDistanceMetric()
        test_mask = self.info['masks']['test']
        num_nodes = self.g_kmv.num_nodes # Safe fallback if exact is None

        for depth in self.cfg.model_depths:
            model_name = f"{self.cfg.model_arch}_L{depth}"
            print(f"\n    --- Testing Depth L{depth} ({model_name}) ---")

            try:
                model, mapper_cfg = self._load_model(model_name, self.info, depth)
                target_dim = mapper_cfg['global_max_dim']

                # KMV Inference (Always Runs)
                z_kmv = self._inference(model, self.g_kmv, target_dim)
                dir_kmv = metric_dir.calculate(z_kmv, self.g_kmv.edge_index)
                mad_kmv = metric_mad.calculate(z_kmv)
                kmv_edges = self.g_kmv.edge_index.size(1)

                # Exact Inference (Subject to Graceful Degradation)
                if self.g_exact is not None:
                    try:
                        z_exact = self._inference(model, self.g_exact, target_dim)
                        fidelity = metric_cos.calculate(z_exact[test_mask], z_kmv[test_mask])
                        dir_exact = metric_dir.calculate(z_exact, self.g_exact.edge_index)
                        mad_exact = metric_mad.calculate(z_exact)
                        exact_edges = self.g_exact.edge_index.size(1)
                    except (RuntimeError, MemoryError):
                        print(f"    [OOM ERROR] Depth {depth} Exact inference failed.")
                        fidelity, dir_exact, mad_exact, exact_edges = self.VOID_FLAG, self.VOID_FLAG, self.VOID_FLAG, self.VOID_FLAG
                else:
                    fidelity, dir_exact, mad_exact, exact_edges = self.VOID_FLAG, self.VOID_FLAG, self.VOID_FLAG, self.VOID_FLAG

                print(f"    -> Fidelity: {fidelity if isinstance(fidelity, str) else f'{fidelity:.4f}'}")
                print(f"    -> Dirichlet (Exact/KMV): {dir_exact if isinstance(dir_exact, str) else f'{dir_exact:.4f}'} / {dir_kmv:.4f}")

                self._save_incremental({
                    "Dataset": self.cfg.dataset, "ModelArch": self.cfg.model_arch,
                    "Rule": self.best_rule, "Depth": depth, "K": self.cfg.depth_study_k,
                    "NumNodes": num_nodes,
                    "ExactEdges": exact_edges,
                    "KMVEdges": kmv_edges,
                    "Fidelity": fidelity, "Dirichlet_Exact": dir_exact,
                    "Dirichlet_KMV": dir_kmv, "MAD_Exact": mad_exact, "MAD_KMV": mad_kmv
                })

                del model
                torch.cuda.empty_cache()

            except FileNotFoundError:
                print(f"    [Skip] Model {model_name} not found. Did Phase 2 complete?")

    def _load_model(self, model_name: str, info: Dict[str, Any], num_layers: int):
        """Loads a specific depth model from the Model Zoo."""
        model_path = config.get_model_path(self.cfg.dataset, model_name)
        mapper_path = model_path.replace('.pt', '_mapper.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
            
        with open(mapper_path, 'r') as f:
            mapper_cfg = json.load(f)
            
        model = get_model(
            self.cfg.model_arch, 
            mapper_cfg['global_max_dim'], 
            info['num_classes'], 
            config.HIDDEN_DIM,
            num_layers=num_layers
        )
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.to(config.DEVICE).eval()
        
        return model, mapper_cfg

    def _inference(self, model, graph, target_dim):
        """Batched inference helper."""
        with torch.no_grad():
            x = graph.x.to(config.DEVICE)
            edge_index = graph.edge_index.to(config.DEVICE)
            
            # Diagonal Padding Check
            if x.size(1) != target_dim:
                x = torch.nn.functional.pad(x, (0, target_dim - x.size(1)))
                
            return model(x, edge_index)