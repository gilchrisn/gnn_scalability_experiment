"""
Metric calculation strategies for graph comparison.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from typing import Dict, Any

class MetricStrategy(ABC):
    """Abstract base class for comparison metrics."""
    
    @abstractmethod
    def calculate(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        """
        Calculates a similarity score between two embedding sets.
        
        Args:
            emb_a: Baseline embeddings (Exact).
            emb_b: Comparison embeddings (Approximation/KMV).
            
        Returns:
            Scalar metric value.
        """
        pass

class CosineFidelityMetric(MetricStrategy):
    """
    Calculates the average Cosine Similarity between corresponding nodes.
    Measures how much the approximation shifted the representation vector.
    """
    
    def calculate(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        """
        Computes mean cosine similarity.
        
        Formula: 1/N * Sum((A . B) / (|A|*|B|))
        """
        if emb_a.shape != emb_b.shape:
            raise ValueError(f"Shape mismatch: {emb_a.shape} vs {emb_b.shape}")
            
        # Normalize for cosine similarity
        a_norm = F.normalize(emb_a, p=2, dim=1)
        b_norm = F.normalize(emb_b, p=2, dim=1)
        
        # Element-wise dot product followed by sum (efficient cosine sim)
        similarity = (a_norm * b_norm).sum(dim=1)
        return similarity.mean().item()

class PredictionAgreementMetric(MetricStrategy):
    """
    Calculates the percentage of nodes where the predicted class remained the same.
    """
    
    def calculate(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
        pred_a = logits_a.argmax(dim=1)
        pred_b = logits_b.argmax(dim=1)
        
        agreement = (pred_a == pred_b).float().mean()
        return agreement.item()