from typing import Dict, Type
from .base import GraphSampler
from .algorithms import SnowballSampler

class SamplerFactory:
    """
    Factory for creating graph samplers.
    """
    _registry: Dict[str, Type[GraphSampler]] = {
        'snowball': SnowballSampler
    }

    @classmethod
    def register(cls, name: str, sampler_cls: Type[GraphSampler]):
        cls._registry[name] = sampler_cls

    @classmethod
    def create(cls, name: str) -> GraphSampler:
        if name not in cls._registry:
            raise ValueError(f"Unknown sampler: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]()