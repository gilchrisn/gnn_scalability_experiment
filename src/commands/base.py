from abc import ABC, abstractmethod
from argparse import Namespace

class BaseCommand(ABC):
    """Abstract base class for all CLI commands."""
    
    @abstractmethod
    def execute(self, args: Namespace) -> None:
        """Executes the command logic."""
        pass
    
    @staticmethod
    @abstractmethod
    def register_subparser(subparsers) -> None:
        """Registers arguments for this command."""
        pass