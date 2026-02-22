import os
from abc import ABC, abstractmethod
from typing import Any
from ..config import config
from .config import ExperimentConfig

class AbstractExperimentPhase(ABC):
    """
    Template for a single experiment phase.
    Follows Single Responsibility Principle (SRP).
    """
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.dataset_cfg = config.get_dataset_config(cfg.dataset)
        self.output_dir = config.RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def execute(self) -> Any:
        """Executes the specific phase logic."""
        pass