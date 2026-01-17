from argparse import Namespace
from .base import BaseCommand
from ..diagnostics import EntropyAnalyzer, RankAnalyzer, DistortionAnalyzer

class AnalyzeCommand(BaseCommand):
    """
    Unified entry point for graph diagnostics.
    Supports: Entropy, Effective Rank, and Semantic Distortion.
    """

    def execute(self, args: Namespace) -> None:
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC ANALYSIS: {args.type.upper()}")
        print(f"Dataset: {args.dataset}")
        print(f"{'='*60}\n")
        
        if args.type == 'entropy':
            analyzer = EntropyAnalyzer(args.dataset, args.model, args.metapath, args.k)
            analyzer.run()
            
        elif args.type == 'rank':
            analyzer = RankAnalyzer(args.dataset, args.metapath)
            analyzer.run()
            
        elif args.type == 'distortion':
            analyzer = DistortionAnalyzer(args.dataset, args.metapath, args.k)
            analyzer.run()
            
        else:
            print(f"Unknown analysis type: {args.type}")

    @staticmethod
    def register_subparser(subparsers) -> None:
        parser = subparsers.add_parser('analyze', help='Run diagnostic analysis')
        parser.add_argument('type', choices=['entropy', 'rank', 'distortion'])
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--metapath', type=str, required=True)
        parser.add_argument('--model', type=str, default='GAT', help='Required for entropy')
        parser.add_argument('--k', type=int, default=4, help='K-value for distortion/entropy')