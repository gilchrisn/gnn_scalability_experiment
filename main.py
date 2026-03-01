import argparse
import sys
import torch

# Import global configuration
from src.config import config

# Import all command implementations
from src.commands import (
    TrainCommand,
    BenchmarkCommand,
    MineCommand,
    AnalyzeCommand,
    CompareCommand,
    TrainFidelityCommand,
    VisualizeFidelityCommand,
    MLPBaselineCommand,
)

def main():
    """
    Main entry point for the GNN Scalability Framework.
    Uses the Command Pattern to dispatch tasks based on CLI arguments.
    """

    # 1. Setup Top-Level Parser
    parser = argparse.ArgumentParser(
        description="GNN Scalability Engine: Exact vs. Approximate Materialization Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global arguments (apply to all commands)
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Override hardware device (e.g., "cpu", "cuda:0")'
    )

    # Initialize Subparsers
    subparsers = parser.add_subparsers(dest='command', required=True, help='Action to perform')

    # 2. Register Subcommands
    # Core Production Commands
    BenchmarkCommand.register_subparser(subparsers)
    TrainCommand.register_subparser(subparsers)
    MineCommand.register_subparser(subparsers)

    # Diagnostic Commands
    AnalyzeCommand.register_subparser(subparsers)
    CompareCommand.register_subparser(subparsers)

    # Journal Experiment Commands
    TrainFidelityCommand.register_subparser(subparsers)
    VisualizeFidelityCommand.register_subparser(subparsers)

    # Baseline / Validation Commands
    MLPBaselineCommand.register_subparser(subparsers)

    # 3. Parse Arguments
    args = parser.parse_args()

    # 4. Global Configuration Setup
    if args.device:
        print(f"[System] Overriding device to: {args.device}")
        config.DEVICE = torch.device(args.device)

    # 5. Command Dispatch
    command_registry = {
        'benchmark':          BenchmarkCommand(),
        'train':              TrainCommand(),
        'mine':               MineCommand(),
        'analyze':            AnalyzeCommand(),
        'compare':            CompareCommand(),
        'train_fidelity':     TrainFidelityCommand(),
        'visualize_fidelity': VisualizeFidelityCommand(),
        'mlp_baseline':       MLPBaselineCommand(),
    }

    if args.command in command_registry:
        try:
            command_registry[args.command].execute(args)
        except KeyboardInterrupt:
            print("\n[System] Operation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()