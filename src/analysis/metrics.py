"""
Metric calculation strategies for graph comparison and representation quality.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from typing import Optional, Any
from torch_geometric.utils import to_dense_adj

# --- Comparison Metrics (Binary) ---

class MetricStrategy(ABC):
    """
    Strategy interface for comparing two sets of embeddings (A vs B).
    """
    @abstractmethod
    def calculate(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        pass

class CosineFidelityMetric(MetricStrategy):
    """
    Calculates average Cosine Similarity between corresponding nodes.
    """
    def calculate(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        if emb_a.shape != emb_b.shape:
            raise ValueError(f"Shape mismatch: {emb_a.shape} vs {emb_b.shape}")
            
        a_norm = F.normalize(emb_a, p=2, dim=1)
        b_norm = F.normalize(emb_b, p=2, dim=1)
        similarity = (a_norm * b_norm).sum(dim=1)
        return similarity.mean().item()

class PredictionAgreementMetric(MetricStrategy):
    """
    Calculates percentage of nodes where predicted class is identical.
    """
    def calculate(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
        pred_a = logits_a.argmax(dim=1)
        pred_b = logits_b.argmax(dim=1)
        agreement = (pred_a == pred_b).float().mean()
        return agreement.item()

# --- Representation Quality Metrics (Unary) ---

class RepresentationMetric(ABC):
    """
    Strategy interface for analyzing the quality of a single embedding set.
    Used for diagnosing oversmoothing and collapse.
    """
    @abstractmethod
    def calculate(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> float:
        """
        Args:
            x: Node embeddings [N, D]
            edge_index: Graph connectivity (required for Dirichlet)
        """
        pass

class DirichletEnergyMetric(RepresentationMetric):
    """
    Calculates Dirichlet Energy to quantify oversmoothing.

    E(X) = (1 / |E|) * Σ_{(i,j) ∈ E} ||x_i - x_j||^2

    Uses the direct edge-difference formula rather than the Laplacian trace
    form, which requires a symmetric (undirected) graph to guarantee
    non-negative values.  The materialized graphs here are directed, so the
    Laplacian trace can produce negative results due to asymmetric degree
    normalization.  The edge-difference formula is always >= 0.
    """
    def calculate(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> float:
        if edge_index is None:
            raise ValueError("Dirichlet Energy requires edge_index.")
        if edge_index.size(1) == 0:
            return 0.0

        src, dst = edge_index[0], edge_index[1]
        diff   = x[src] - x[dst]               # [|E|, D]
        energy = (diff * diff).sum(dim=1).mean()  # mean over edges, sum over dims
        return energy.item()

class MeanAverageDistanceMetric(RepresentationMetric):
    """
    Calculates Mean Average Distance (MAD) to quantify Feature Collapse.
    
    MAD(X) = Mean of Cosine Distances between all pairs of nodes.
    Low MAD -> All nodes have same representation (Collapse).
    """
    def calculate(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> float:
        # Note: Full pairwise calculation is O(N^2). 
        # For large graphs, we sample random pairs if N > 10,000 to avoid OOM.
        
        num_nodes = x.size(0)
        
        if num_nodes > 10000:
            # Approximation Strategy for Scalability
            perm = torch.randperm(num_nodes, device=x.device)[:5000]
            x_sub = x[perm]
        else:
            x_sub = x

        # Normalize for Cosine Distance
        x_norm = F.normalize(x_sub, p=2, dim=1)
        
        # Cosine Similarity Matrix: S = X @ X^T
        sim_matrix = torch.mm(x_norm, x_norm.t())
        
        # Cosine Distance: D = 1 - S
        dist_matrix = 1.0 - sim_matrix
        
        # Exclude self-loops (distance 0) from mean
        # Off-diagonal elements only
        n_sub = x_sub.size(0)
        sum_dist = dist_matrix.sum()
        # MAD = Sum / (N^2 - N)
        mad = sum_dist / (n_sub * (n_sub - 1))
        
        return mad.item()