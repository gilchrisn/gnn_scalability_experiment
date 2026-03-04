"""
End-to-End Pipeline Comparison (C++ Backend, CPU-forced)
=========================================================
Compares two completely independent pipelines, both using the C++ backend:

  Oracle : graph_prep materialize → Train(static) → graph_prep materialize → Inference
  KMV    : graph_prep sketch(K,L) → Train(cycling) → graph_prep sketch(K,1) → Inference

Key design invariants:
  1. Exact materialisation is NEVER called inside the KMV pipeline.
  2. The Python backend is NEVER used.
  3. ALL computation is forced to CPU — torch, lightning, everything.
  4. Quality diagnostics are logged per run for post-hoc validation.

Usage:
    python scripts/run_e2e_comparison.py \
        --dataset HGB_ACM \
        --metapath "paper_to_author,author_to_paper" \
        --model SAGE \
        --epochs 100 \
        --k-values 2 4 8 16 32 64 128 \
        --l 5
"""
import argparse
import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.data import Data, HeteroData

# ---------------------------------------------------------------------------
# Force CPU globally before anything else imports torch
# ---------------------------------------------------------------------------
torch.set_num_threads(os.cpu_count() or 4)

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.backend import BackendFactory
from src.callbacks import GraphCyclingCallback
from src.config import config
from src.data import DatasetFactory
from src.lit_model_full import LitFullBatchGNN
from src.models import get_model
from src.utils import SchemaMatcher

# Force CPU on config immediately
config.DEVICE = torch.device('cpu')


# ---------------------------------------------------------------------------
# Quality logger
# ---------------------------------------------------------------------------

class QualityLog:
    """
    Collects diagnostics during a pipeline run for post-hoc validation.
    Written to a sidecar .log file next to the CSV.

    Sections:
      GRAPH   — edge count, node count, density, isolated nodes
      TRAIN   — per-epoch loss and accuracy curves
      RESULT  — final test accuracy and timing breakdown
    """

    def __init__(self, label: str):
        self.label   = label
        self.entries: List[str] = []

    def log(self, msg: str) -> None:
        line = f"[{self.label}] {msg}"
        print(line)
        self.entries.append(line)

    def log_graph(self, g: Data, tag: str = "") -> None:
        N = g.num_nodes
        E = g.edge_index.size(1)
        density = E / (N * (N - 1)) if N > 1 else 0.0
        # Isolated nodes = nodes with no edges
        if E > 0:
            connected = g.edge_index.unique().numel()
            isolated  = N - connected
        else:
            isolated = N
        self.log(f"  Graph{' '+tag if tag else ''}: "
                 f"N={N:,}  E={E:,}  density={density:.6f}  "
                 f"isolated={isolated:,} ({100*isolated/N:.1f}%)")

    def log_train_epoch(self, epoch: int, train_loss: float,
                        val_loss: float, val_acc: float) -> None:
        self.entries.append(
            f"  epoch={epoch:03d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(self.entries))
            f.write('\n')


# ---------------------------------------------------------------------------
# Lightning callback that feeds epoch metrics into QualityLog
# ---------------------------------------------------------------------------

class EpochLoggerCallback(pl.Callback):
    """Captures per-epoch metrics and writes them into a QualityLog."""

    def __init__(self, quality_log: QualityLog):
        self.qlog = quality_log
        self._train_loss: float = float('nan')

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logged = trainer.callback_metrics
        self._train_loss = float(logged.get('train_loss', float('nan')))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logged    = trainer.callback_metrics
        val_loss  = float(logged.get('val_loss', float('nan')))
        val_acc   = float(logged.get('val_acc',  float('nan')))
        self.qlog.log_train_epoch(trainer.current_epoch,
                                  self._train_loss, val_loss, val_acc)


# ---------------------------------------------------------------------------
# Result DTO
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Standardised result record for one pipeline run."""
    dataset:    str
    metapath:   str
    model_arch: str
    method:     str
    k:          Any
    l:          Any
    mat_time:   float
    train_time: float
    total_time: float
    test_acc:   float
    num_edges:  int
    num_nodes:  int
    epochs_run: int
    # Quality diagnostics
    graph_density:    float = 0.0
    isolated_nodes:   int   = 0
    final_val_loss:   float = float('nan')
    final_val_acc:    float = float('nan')


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_cpp_backend(g_hetero: HeteroData,
                        path_list: List,
                        info: Dict[str, Any],
                        num_sketches: int = 1):
    backend = BackendFactory.create(
        'cpp',
        executable_path=config.CPP_EXECUTABLE,
        temp_dir=config.TEMP_DIR,
        device=torch.device('cpu'),   # always CPU
        num_sketches=num_sketches,
    )
    backend.initialize(g_hetero, path_list, info)
    return backend


def _build_trainer(epochs: int,
                   extra_callbacks: Optional[List] = None) -> pl.Trainer:
    callbacks = [EarlyStopping(monitor='val_loss', patience=15, mode='min')]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    return pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",      # hard CPU
        devices=1,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=1,
        callbacks=callbacks,
    )


def _graph_diagnostics(g: Data) -> Dict[str, Any]:
    N = g.num_nodes
    E = g.edge_index.size(1)
    density  = E / (N * (N - 1)) if N > 1 else 0.0
    isolated = (N - g.edge_index.unique().numel()) if E > 0 else N
    return {'density': density, 'isolated': int(isolated)}


# ---------------------------------------------------------------------------
# Oracle Pipeline
# ---------------------------------------------------------------------------

class OraclePipeline:
    def __init__(self, model_arch: str, epochs: int):
        self.model_arch = model_arch
        self.epochs     = epochs

    def run(self, dataset: str, metapath_str: str, path_list: List,
            g_hetero: HeteroData, info: Dict[str, Any],
            qlog: QualityLog, log_path: str) -> PipelineResult:

        qlog.log("=" * 55)
        qlog.log("ORACLE PIPELINE (C++ Exact, CPU)")
        qlog.log("=" * 55)

        backend = _build_cpp_backend(g_hetero, path_list, info, num_sketches=1)

        # 1. Train graph
        t0 = time.perf_counter()
        g_train = backend.materialize_exact()
        t_mat_train = time.perf_counter() - t0

        if g_train.edge_index.size(1) == 0:
            raise RuntimeError("Exact materialisation produced 0 edges — check metapath/rule file.")

        diag = _graph_diagnostics(g_train)
        qlog.log(f"Train materialisation: {t_mat_train:.3f}s")
        qlog.log_graph(g_train, tag="[train/exact]")

        # 2. Train
        in_dim  = g_train.x.shape[1]
        model   = get_model(self.model_arch, in_dim, info['num_classes'], config.HIDDEN_DIM)
        lit     = LitFullBatchGNN(model, g_train, lr=config.LEARNING_RATE)
        epoch_cb = EpochLoggerCallback(qlog)
        trainer  = _build_trainer(self.epochs, extra_callbacks=[epoch_cb])

        qlog.log(f"Training: {self.model_arch}, in_dim={in_dim}, "
                 f"hidden={config.HIDDEN_DIM}, classes={info['num_classes']}")
        qlog.log("--- per-epoch log ---")

        t0 = time.perf_counter()
        trainer.fit(lit)
        t_train    = time.perf_counter() - t0
        epochs_run = trainer.current_epoch + 1

        final_metrics = trainer.callback_metrics
        final_val_loss = float(final_metrics.get('val_loss', float('nan')))
        final_val_acc  = float(final_metrics.get('val_acc',  float('nan')))

        qlog.log(f"Training done: {epochs_run} epochs, {t_train:.3f}s")
        qlog.log(f"Final val_loss={final_val_loss:.4f}  val_acc={final_val_acc:.4f}")

        del g_train
        gc.collect()

        # 3. Inference graph
        t0 = time.perf_counter()
        g_test = backend.materialize_exact()
        t_mat_test = time.perf_counter() - t0
        qlog.log(f"Inference materialisation: {t_mat_test:.3f}s")
        qlog.log_graph(g_test, tag="[test/exact]")

        # 4. Test
        lit.update_graph(g_test)
        results    = trainer.test(lit, verbose=False)[0]
        test_acc   = float(results['test_acc'])
        total_mat  = t_mat_train + t_mat_test

        qlog.log(f"TEST ACCURACY: {test_acc:.4f}")
        qlog.log(f"Mat time: {total_mat:.3f}s  Train time: {t_train:.3f}s  "
                 f"Total: {total_mat + t_train:.3f}s")
        qlog.write(log_path)

        backend.cleanup()

        return PipelineResult(
            dataset=dataset, metapath=metapath_str, model_arch=self.model_arch,
            method='Exact', k='N/A', l='N/A',
            mat_time=total_mat, train_time=t_train,
            total_time=total_mat + t_train,
            test_acc=test_acc,
            num_edges=g_test.edge_index.size(1),
            num_nodes=g_test.num_nodes,
            epochs_run=epochs_run,
            graph_density=diag['density'],
            isolated_nodes=diag['isolated'],
            final_val_loss=final_val_loss,
            final_val_acc=final_val_acc,
        )


# ---------------------------------------------------------------------------
# KMV Pipeline
# ---------------------------------------------------------------------------

class KMVPipeline:
    def __init__(self, model_arch: str, epochs: int, k: int, l: int = 5):
        self.model_arch = model_arch
        self.epochs     = epochs
        self.k          = k
        self.l          = l

    def run(self, dataset: str, metapath_str: str, path_list: List,
            g_hetero: HeteroData, info: Dict[str, Any],
            qlog: QualityLog, log_path: str) -> PipelineResult:

        qlog.log("=" * 55)
        qlog.log(f"KMV PIPELINE K={self.k} L={self.l} (C++, CPU)")
        qlog.log("=" * 55)

        # 1. Ensemble sketches for training
        train_backend = _build_cpp_backend(g_hetero, path_list, info,
                                           num_sketches=self.l)

        t0 = time.perf_counter()
        graphs = train_backend.materialize_kmv_ensemble(k=self.k)
        t_mat_train = time.perf_counter() - t0

        if not graphs:
            raise RuntimeError(
                f"C++ backend returned 0 graphs for K={self.k}, L={self.l}."
            )

        qlog.log(f"Ensemble materialisation: {t_mat_train:.3f}s | {len(graphs)} sketches")
        for i, g in enumerate(graphs):
            qlog.log_graph(g, tag=f"[sketch {i}]")

        # Warn if sketches are extremely sparse
        avg_edges = sum(g.edge_index.size(1) for g in graphs) / len(graphs)
        qlog.log(f"Average sketch edges: {avg_edges:.0f}")

        # 2. Train with GraphCyclingCallback
        g_initial  = graphs[0]
        cycling_cb = GraphCyclingCallback(graphs)
        in_dim     = g_initial.x.shape[1]
        model      = get_model(self.model_arch, in_dim, info['num_classes'],
                               config.HIDDEN_DIM)
        lit        = LitFullBatchGNN(model, g_initial, lr=config.LEARNING_RATE)
        epoch_cb   = EpochLoggerCallback(qlog)
        trainer    = _build_trainer(self.epochs, extra_callbacks=[cycling_cb, epoch_cb])

        qlog.log(f"Training: {self.model_arch}, in_dim={in_dim}, "
                 f"hidden={config.HIDDEN_DIM}, classes={info['num_classes']}")
        qlog.log("--- per-epoch log ---")

        diag = _graph_diagnostics(g_initial)

        t0 = time.perf_counter()
        trainer.fit(lit)
        t_train    = time.perf_counter() - t0
        epochs_run = trainer.current_epoch + 1

        final_metrics  = trainer.callback_metrics
        final_val_loss = float(final_metrics.get('val_loss', float('nan')))
        final_val_acc  = float(final_metrics.get('val_acc',  float('nan')))

        qlog.log(f"Training done: {epochs_run} epochs, {t_train:.3f}s")
        qlog.log(f"Final val_loss={final_val_loss:.4f}  val_acc={final_val_acc:.4f}")

        train_backend.cleanup()
        del graphs, g_initial
        gc.collect()

        # 3. Fresh inference sketch
        infer_backend = _build_cpp_backend(g_hetero, path_list, info,
                                           num_sketches=1)

        t0 = time.perf_counter()
        g_test = infer_backend.materialize_kmv(k=self.k)
        t_mat_test = time.perf_counter() - t0
        qlog.log(f"Inference materialisation: {t_mat_test:.3f}s")
        qlog.log_graph(g_test, tag=f"[test/kmv K={self.k}]")

        # 4. Test
        lit.update_graph(g_test)
        results   = trainer.test(lit, verbose=False)[0]
        test_acc  = float(results['test_acc'])
        total_mat = t_mat_train + t_mat_test

        qlog.log(f"TEST ACCURACY: {test_acc:.4f}")
        qlog.log(f"Mat time: {total_mat:.3f}s  Train time: {t_train:.3f}s  "
                 f"Total: {total_mat + t_train:.3f}s")
        qlog.write(log_path)

        infer_backend.cleanup()

        return PipelineResult(
            dataset=dataset, metapath=metapath_str, model_arch=self.model_arch,
            method='KMV', k=self.k, l=self.l,
            mat_time=total_mat, train_time=t_train,
            total_time=total_mat + t_train,
            test_acc=test_acc,
            num_edges=g_test.edge_index.size(1),
            num_nodes=g_test.num_nodes,
            epochs_run=epochs_run,
            graph_density=diag['density'],
            isolated_nodes=diag['isolated'],
            final_val_loss=final_val_loss,
            final_val_acc=final_val_acc,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class E2EComparisonOrchestrator:
    def __init__(self, dataset: str, metapath_str: str, model_arch: str,
                 epochs: int, k_values: List[int], l: int, output_csv: str):
        self.dataset      = dataset
        self.metapath_str = metapath_str
        self.model_arch   = model_arch
        self.epochs       = epochs
        self.k_values     = k_values
        self.l            = l
        self.output_csv   = output_csv
        # Sidecar log alongside the CSV
        base     = os.path.splitext(output_csv)[0]
        mp_short = metapath_str.replace(',', '_').replace(' ', '')[:40]
        self.log_path = f"{base}_{dataset}_{mp_short}.log"

    def run(self) -> pd.DataFrame:
        print(f"\n{'='*60}")
        print(f"E2E COMPARISON  (CPU-forced, C++ backend)")
        print(f"Dataset  : {self.dataset}")
        print(f"Metapath : {self.metapath_str}")
        print(f"Model    : {self.model_arch}")
        print(f"K values : {self.k_values}  L={self.l}")
        print(f"Epochs   : {self.epochs}")
        print(f"Device   : cpu (forced)")
        print(f"Binary   : {config.CPP_EXECUTABLE}")
        print(f"{'='*60}")

        if not os.path.exists(config.CPP_EXECUTABLE):
            raise FileNotFoundError(
                f"C++ binary not found: {config.CPP_EXECUTABLE}\n"
                "Run: make   (or see README)"
            )

        dataset_cfg = config.get_dataset_config(self.dataset)
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg.source, dataset_cfg.dataset_name, dataset_cfg.target_node
        )
        path_list = [
            SchemaMatcher.match(s.strip(), g_hetero)
            for s in self.metapath_str.split(',')
        ]

        results: List[PipelineResult] = []

        # Oracle
        try:
            qlog = QualityLog(f"{self.dataset}/Exact")
            oracle = OraclePipeline(self.model_arch, self.epochs)
            r = oracle.run(self.dataset, self.metapath_str, path_list,
                           g_hetero, info, qlog, self.log_path)
            results.append(r)
            self._save_incremental(r)
        except (MemoryError, RuntimeError) as exc:
            print(f"\n[Oracle] FAILED: {exc}")
            traceback.print_exc()

        # KMV for each K
        for k in self.k_values:
            try:
                qlog = QualityLog(f"{self.dataset}/KMV-K{k}")
                kmv  = KMVPipeline(self.model_arch, self.epochs, k, self.l)
                r    = kmv.run(self.dataset, self.metapath_str, path_list,
                               g_hetero, info, qlog, self.log_path)
                results.append(r)
                self._save_incremental(r)
            except Exception as exc:
                print(f"\n[KMV K={k}] FAILED: {exc}")
                traceback.print_exc()

        df = pd.DataFrame([vars(r) for r in results])
        self._print_summary(df)
        return df

    def _save_incremental(self, result: PipelineResult) -> None:
        row    = pd.DataFrame([vars(result)])
        header = not os.path.exists(self.output_csv)
        row.to_csv(self.output_csv, mode='a', header=header, index=False)

    @staticmethod
    def _print_summary(df: pd.DataFrame) -> None:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        cols = ['method', 'k', 'test_acc', 'final_val_acc',
                'mat_time', 'train_time', 'num_edges', 'isolated_nodes', 'epochs_run']
        available = [c for c in cols if c in df.columns]
        print(df[available].to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E Oracle vs KMV comparison (CPU-forced, C++ backend)"
    )
    parser.add_argument('--dataset',    type=str, required=True)
    parser.add_argument('--metapath',   type=str, required=True)
    parser.add_argument('--model',      type=str, default='SAGE',
                        choices=['GCN', 'SAGE', 'GAT'])
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--k-values',   type=int, nargs='+',
                        default=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument('--l',          type=int, default=5)
    parser.add_argument('--output-csv', type=str,
                        default=os.path.join(
                            project_root, 'output', 'results', 'e2e_comparison.csv'))
    args = parser.parse_args()

    # CPU forced — no --cpu flag needed, it's always on
    config.DEVICE = torch.device('cpu')

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    E2EComparisonOrchestrator(
        dataset=args.dataset,
        metapath_str=args.metapath,
        model_arch=args.model,
        epochs=args.epochs,
        k_values=args.k_values,
        l=args.l,
        output_csv=args.output_csv,
    ).run()

    print(f"\n[Done] Results: {args.output_csv}")


if __name__ == "__main__":
    main()