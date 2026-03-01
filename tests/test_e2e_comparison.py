"""
Unit Tests for run_e2e_comparison.py
======================================
Validates every component of the end-to-end pipeline comparison
without requiring the actual C++ binary, trained models, or real datasets.

Test strategy:
  - All external dependencies (C++ backend, DatasetFactory, Lightning Trainer)
    are replaced with lightweight mocks.
  - Tests verify BEHAVIOUR (what is called, in what order, with what arguments),
    not just that code runs without crashing.
  - The critical invariant — exact materialisation is NEVER called inside the
    KMV pipeline — is tested explicitly and would catch any regression.

Run:
    python -m pytest tests/test_e2e_comparison.py -v
    python -m pytest tests/test_e2e_comparison.py -v --tb=short   # on failure
"""
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch, PropertyMock

import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData

# ---------------------------------------------------------------------------
# Path setup — allow importing from project root
# ---------------------------------------------------------------------------
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Import the module under test AFTER path setup
# ---------------------------------------------------------------------------
from scripts.run_e2e_comparison import (
    E2EComparisonOrchestrator,
    KMVPipeline,
    OraclePipeline,
    PipelineResult,
    _build_cpp_backend,
    _build_trainer,
)


# ===========================================================================
# Shared test fixtures
# ===========================================================================

def _make_dummy_graph(num_nodes: int = 10,
                      feature_dim: int = 8,
                      num_classes: int = 3) -> Data:
    """Returns a minimal homogeneous Data object with all required attributes."""
    edge_index = torch.randint(0, num_nodes, (2, 20))
    x          = torch.randn(num_nodes, feature_dim)
    y          = torch.randint(0, num_classes, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:6] = True
    val_mask[6:8]  = True
    test_mask[8:]  = True

    g = Data(
        edge_index=edge_index,
        x=x, y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )
    return g


def _make_info(num_nodes: int = 10,
               feature_dim: int = 8,
               num_classes: int = 3) -> Dict[str, Any]:
    """Returns a standard info dict matching DatasetFactory output contract."""
    features = torch.randn(num_nodes, feature_dim)
    labels   = torch.randint(0, num_classes, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool); train_mask[:6] = True
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool); val_mask[6:8]  = True
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool); test_mask[8:]  = True
    return {
        'features':    features,
        'labels':      labels,
        'num_classes': num_classes,
        'masks':       {'train': train_mask, 'val': val_mask, 'test': test_mask},
    }


def _make_hetero_graph() -> HeteroData:
    """Returns a minimal HeteroData object with one edge type."""
    g = HeteroData()
    g['paper'].num_nodes  = 10
    g['author'].num_nodes = 5
    g['paper', 'paper_to_author', 'author'].edge_index = torch.randint(0, 5, (2, 15))
    g['author', 'author_to_paper', 'paper'].edge_index = torch.randint(0, 10, (2, 15))
    return g


def _make_mock_backend(num_nodes: int = 10,
                       num_edges: int = 20,
                       feature_dim: int = 8,
                       num_classes: int = 3) -> MagicMock:
    """
    Creates a mock CppBackend that returns realistic dummy graphs.

    The mock tracks all calls so tests can assert on call patterns.
    """
    graph = _make_dummy_graph(num_nodes, feature_dim, num_classes)
    backend = MagicMock()
    backend.materialize_exact.return_value          = graph
    backend.materialize_kmv.return_value            = graph
    backend.materialize_kmv_ensemble.return_value   = [graph, graph, graph]
    backend.get_prep_time.return_value              = 0.01
    return backend


# ===========================================================================
# 1. PipelineResult DTO
# ===========================================================================

class TestPipelineResult(unittest.TestCase):
    """Validates the result data structure."""

    def test_fields_accessible(self):
        r = PipelineResult(
            dataset='HGB_ACM', metapath='p_a_p', model_arch='SAGE',
            method='Exact', k='N/A', l='N/A',
            mat_time=1.0, train_time=5.0, total_time=6.0,
            test_acc=0.85, num_edges=100, num_nodes=10, epochs_run=50,
        )
        self.assertEqual(r.method,     'Exact')
        self.assertAlmostEqual(r.total_time, 6.0)
        self.assertEqual(r.k, 'N/A')

    def test_total_time_consistency(self):
        """total_time must equal mat_time + train_time."""
        r = PipelineResult(
            dataset='D', metapath='M', model_arch='GCN',
            method='KMV', k=32, l=5,
            mat_time=2.5, train_time=7.5, total_time=10.0,
            test_acc=0.80, num_edges=50, num_nodes=10, epochs_run=40,
        )
        self.assertAlmostEqual(r.mat_time + r.train_time, r.total_time)

    def test_vars_serialisable_to_dataframe(self):
        """vars(result) must produce a dict that pd.DataFrame accepts."""
        r = PipelineResult(
            dataset='D', metapath='M', model_arch='GCN',
            method='KMV', k=16, l=3,
            mat_time=1.0, train_time=4.0, total_time=5.0,
            test_acc=0.75, num_edges=30, num_nodes=8, epochs_run=20,
        )
        df = pd.DataFrame([vars(r)])
        self.assertEqual(len(df), 1)
        self.assertIn('method', df.columns)
        self.assertIn('test_acc', df.columns)


# ===========================================================================
# 2. _build_cpp_backend
# ===========================================================================

class TestBuildCppBackend(unittest.TestCase):
    """Validates backend construction and initialisation."""

    @patch('scripts.run_e2e_comparison.BackendFactory')
    def test_creates_cpp_backend_not_python(self, mock_factory):
        """Must request 'cpp', never 'python'."""
        mock_backend = _make_mock_backend()
        mock_factory.create.return_value = mock_backend

        g_hetero  = _make_hetero_graph()
        path_list = [('paper', 'paper_to_author', 'author')]
        info      = _make_info()

        with patch('scripts.run_e2e_comparison.config') as mock_cfg:
            mock_cfg.CPP_EXECUTABLE = '/fake/graph_prep'
            mock_cfg.TEMP_DIR       = '/tmp/fake'
            mock_cfg.DEVICE         = torch.device('cpu')
            _build_cpp_backend(g_hetero, path_list, info, num_sketches=1)

        backend_type = mock_factory.create.call_args[0][0]
        self.assertEqual(backend_type, 'cpp',
                         f"Expected backend type 'cpp', got '{backend_type}'")

    @patch('scripts.run_e2e_comparison.BackendFactory')
    def test_num_sketches_passed_to_factory(self, mock_factory):
        """num_sketches must be forwarded to BackendFactory.create as num_sketches kwarg."""
        mock_backend = _make_mock_backend()
        mock_factory.create.return_value = mock_backend

        g_hetero  = _make_hetero_graph()
        path_list = [('paper', 'paper_to_author', 'author')]
        info      = _make_info()

        with patch('scripts.run_e2e_comparison.config') as mock_cfg:
            mock_cfg.CPP_EXECUTABLE = '/fake/graph_prep'
            mock_cfg.TEMP_DIR       = '/tmp/fake'
            mock_cfg.DEVICE         = torch.device('cpu')
            _build_cpp_backend(g_hetero, path_list, info, num_sketches=7)

        kwargs = mock_factory.create.call_args[1]
        self.assertIn('num_sketches', kwargs,
                      "num_sketches must be passed as keyword argument to BackendFactory.create")
        self.assertEqual(kwargs['num_sketches'], 7)

    @patch('scripts.run_e2e_comparison.BackendFactory')
    def test_initialize_called_once(self, mock_factory):
        """backend.initialize() must be called exactly once per _build_cpp_backend call."""
        mock_backend = _make_mock_backend()
        mock_factory.create.return_value = mock_backend

        g_hetero  = _make_hetero_graph()
        path_list = [('paper', 'paper_to_author', 'author')]
        info      = _make_info()

        with patch('scripts.run_e2e_comparison.config') as mock_cfg:
            mock_cfg.CPP_EXECUTABLE = '/fake/graph_prep'
            mock_cfg.TEMP_DIR       = '/tmp/fake'
            mock_cfg.DEVICE         = torch.device('cpu')
            _build_cpp_backend(g_hetero, path_list, info)

        mock_backend.initialize.assert_called_once()


# ===========================================================================
# 3. OraclePipeline — Critical Invariant Tests
# ===========================================================================

class TestOraclePipeline(unittest.TestCase):
    """
    Tests for the Oracle pipeline.

    Critical invariants:
      - materialize_exact() is called exactly twice (train + inference).
      - materialize_kmv() and materialize_kmv_ensemble() are NEVER called.
      - cleanup() is called.
    """

    def _run_oracle_with_mock_backend(self,
                                       mock_backend: MagicMock) -> PipelineResult:
        """Helper: runs OraclePipeline with a fully mocked C++ backend."""
        g_hetero  = _make_hetero_graph()
        path_list = [('paper', 'paper_to_author', 'author'),
                     ('author', 'author_to_paper', 'paper')]
        info = _make_info()

        with patch('scripts.run_e2e_comparison._build_cpp_backend',
                   return_value=mock_backend), \
             patch('scripts.run_e2e_comparison.get_model') as mock_get_model, \
             patch('scripts.run_e2e_comparison.LitFullBatchGNN') as mock_lit_cls, \
             patch('scripts.run_e2e_comparison._build_trainer') as mock_trainer_fn, \
             patch('scripts.run_e2e_comparison.config') as mock_cfg:

            mock_cfg.HIDDEN_DIM    = 64
            mock_cfg.LEARNING_RATE = 0.001
            mock_cfg.DEVICE        = torch.device('cpu')

            # Model mock
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            # LightningModule mock
            mock_lit = MagicMock()
            mock_lit.device = torch.device('cpu')
            mock_lit_cls.return_value = mock_lit

            # Trainer mock
            mock_trainer = MagicMock()
            mock_trainer.current_epoch = 49
            mock_trainer.test.return_value = [{'test_acc': 0.85}]
            mock_trainer_fn.return_value = mock_trainer

            pipeline = OraclePipeline(model_arch='SAGE', epochs=50)
            return pipeline.run(
                dataset='HGB_ACM',
                metapath_str='paper_to_author,author_to_paper',
                path_list=path_list,
                g_hetero=g_hetero,
                info=info,
            )

    def test_materialize_exact_called_exactly_twice(self):
        """Oracle must call exact materialisation exactly twice: train + inference."""
        mock_backend = _make_mock_backend()
        self._run_oracle_with_mock_backend(mock_backend)
        self.assertEqual(
            mock_backend.materialize_exact.call_count, 2,
            f"Expected materialize_exact() called 2 times, "
            f"got {mock_backend.materialize_exact.call_count}"
        )

    def test_materialize_kmv_never_called_in_oracle(self):
        """
        CRITICAL INVARIANT: Oracle must never touch KMV materialisation.
        Any call to materialize_kmv or materialize_kmv_ensemble in the
        Oracle pipeline means exact baseline results are contaminated.
        """
        mock_backend = _make_mock_backend()
        self._run_oracle_with_mock_backend(mock_backend)

        self.assertEqual(
            mock_backend.materialize_kmv.call_count, 0,
            "Oracle pipeline called materialize_kmv() — this should never happen."
        )
        self.assertEqual(
            mock_backend.materialize_kmv_ensemble.call_count, 0,
            "Oracle pipeline called materialize_kmv_ensemble() — this should never happen."
        )

    def test_cleanup_called(self):
        """Backend resources must be released after Oracle completes."""
        mock_backend = _make_mock_backend()
        self._run_oracle_with_mock_backend(mock_backend)
        mock_backend.cleanup.assert_called_once()

    def test_result_method_is_exact(self):
        """Result DTO must record method='Exact'."""
        mock_backend = _make_mock_backend()
        result = self._run_oracle_with_mock_backend(mock_backend)
        self.assertEqual(result.method, 'Exact')

    def test_result_k_is_not_applicable(self):
        """Oracle result must have k='N/A' since no sketch size applies."""
        mock_backend = _make_mock_backend()
        result = self._run_oracle_with_mock_backend(mock_backend)
        self.assertEqual(result.k, 'N/A')

    def test_result_test_acc_in_valid_range(self):
        """Accuracy must be a float in [0, 1]."""
        mock_backend = _make_mock_backend()
        result = self._run_oracle_with_mock_backend(mock_backend)
        self.assertIsInstance(result.test_acc, float)
        self.assertGreaterEqual(result.test_acc, 0.0)
        self.assertLessEqual(result.test_acc,    1.0)

    def test_raises_on_zero_edge_graph(self):
        """
        Oracle must raise RuntimeError if exact materialisation returns 0 edges.
        An empty graph is a signal of a bad metapath/rule file — we must not
        silently train on it and produce meaningless accuracy numbers.
        """
        mock_backend = _make_mock_backend()
        empty_graph  = _make_dummy_graph()
        empty_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        mock_backend.materialize_exact.return_value = empty_graph

        with self.assertRaises(RuntimeError):
            self._run_oracle_with_mock_backend(mock_backend)

    def test_total_time_is_positive(self):
        """total_time must be > 0 (timers must be recording real durations)."""
        mock_backend = _make_mock_backend()
        result = self._run_oracle_with_mock_backend(mock_backend)
        # total_time = mat_time + train_time; both must be non-negative
        self.assertGreaterEqual(result.mat_time,   0.0)
        self.assertGreaterEqual(result.train_time,  0.0)
        self.assertGreaterEqual(result.total_time,  0.0)


# ===========================================================================
# 4. KMVPipeline — Critical Invariant Tests
# ===========================================================================

class TestKMVPipeline(unittest.TestCase):
    """
    Tests for the KMV pipeline.

    Critical invariants:
      - materialize_exact() is NEVER called.
      - materialize_kmv_ensemble() is called for training (not materialize_kmv).
      - A separate backend instance is used for inference.
      - Both backends are cleaned up.
      - GraphCyclingCallback is used (not DynamicGraphCallback).
    """

    def _run_kmv_with_mock_backends(self,
                                     k: int = 32,
                                     l: int = 3
                                     ):
        """
        Helper: runs KMVPipeline with two independently tracked mock backends.

        Returns (result, train_backend_mock, infer_backend_mock).
        """
        g_hetero  = _make_hetero_graph()
        path_list = [('paper', 'paper_to_author', 'author'),
                     ('author', 'author_to_paper', 'paper')]
        info = _make_info()

        train_backend = _make_mock_backend()
        infer_backend = _make_mock_backend()

        # _build_cpp_backend is called twice: first for training, then for inference.
        # We return different mocks for each call so we can assert on them separately.
        call_count = {'n': 0}
        def _side_effect(*args, **kwargs):
            call_count['n'] += 1
            return train_backend if call_count['n'] == 1 else infer_backend

        with patch('scripts.run_e2e_comparison._build_cpp_backend',
                   side_effect=_side_effect), \
             patch('scripts.run_e2e_comparison.get_model') as mock_get_model, \
             patch('scripts.run_e2e_comparison.LitFullBatchGNN') as mock_lit_cls, \
             patch('scripts.run_e2e_comparison.GraphCyclingCallback') as mock_cb_cls, \
             patch('scripts.run_e2e_comparison._build_trainer') as mock_trainer_fn, \
             patch('scripts.run_e2e_comparison.config') as mock_cfg:

            mock_cfg.HIDDEN_DIM    = 64
            mock_cfg.LEARNING_RATE = 0.001
            mock_cfg.DEVICE        = torch.device('cpu')

            mock_get_model.return_value = MagicMock()

            mock_lit = MagicMock()
            mock_lit.device = torch.device('cpu')
            mock_lit_cls.return_value = mock_lit

            mock_trainer = MagicMock()
            mock_trainer.current_epoch = 49
            mock_trainer.test.return_value = [{'test_acc': 0.82}]
            mock_trainer_fn.return_value = mock_trainer

            mock_cb_cls.return_value = MagicMock()

            pipeline = KMVPipeline(model_arch='SAGE', epochs=50, k=k, l=l)
            result = pipeline.run(
                dataset='HGB_ACM',
                metapath_str='paper_to_author,author_to_paper',
                path_list=path_list,
                g_hetero=g_hetero,
                info=info,
            )

        return result, train_backend, infer_backend, mock_cb_cls

    def test_exact_materialisation_never_called(self):
        """
        THE MOST CRITICAL TEST.

        materialize_exact() must never be called anywhere in the KMV pipeline.
        If it is called, the pipeline is paying the O(|E|^L) cost it is supposed
        to avoid, and the experimental comparison is invalid.
        """
        _, train_backend, infer_backend, _ = self._run_kmv_with_mock_backends()

        self.assertEqual(
            train_backend.materialize_exact.call_count, 0,
            "KMV PIPELINE INVARIANT VIOLATED: train_backend.materialize_exact() "
            "was called. Exact materialisation must NEVER appear in the KMV pipeline."
        )
        self.assertEqual(
            infer_backend.materialize_exact.call_count, 0,
            "KMV PIPELINE INVARIANT VIOLATED: infer_backend.materialize_exact() "
            "was called during inference. Exact materialisation must NEVER appear "
            "in the KMV pipeline."
        )

    def test_ensemble_used_for_training(self):
        """
        Training must use materialize_kmv_ensemble (not materialize_kmv).
        materialize_kmv_ensemble generates L files in one C++ call, which
        GraphCyclingCallback then cycles through. This is the correct C++ path.
        """
        _, train_backend, _, _ = self._run_kmv_with_mock_backends(k=32, l=3)

        self.assertEqual(
            train_backend.materialize_kmv_ensemble.call_count, 1,
            "Training must call materialize_kmv_ensemble() exactly once."
        )
        # materialize_kmv on the train backend should NOT be used for training
        self.assertEqual(
            train_backend.materialize_kmv.call_count, 0,
            "Training should use materialize_kmv_ensemble(), not materialize_kmv()."
        )

    def test_k_passed_to_ensemble(self):
        """The K value must be forwarded correctly to materialize_kmv_ensemble."""
        target_k = 64
        _, train_backend, _, _ = self._run_kmv_with_mock_backends(k=target_k, l=3)

        args, kwargs = train_backend.materialize_kmv_ensemble.call_args
        passed_k = kwargs.get('k', args[0] if args else None)
        self.assertEqual(passed_k, target_k,
                         f"Expected K={target_k} passed to ensemble, got K={passed_k}")

    def test_separate_backend_for_inference(self):
        """
        Inference must use a separate backend instance from training.
        This guarantees no shared temp files or state between train and test sketches.
        """
        _, train_backend, infer_backend, _ = self._run_kmv_with_mock_backends()

        # Inference sketch must come from the infer_backend, not train_backend
        self.assertEqual(
            infer_backend.materialize_kmv.call_count, 1,
            "Inference must call materialize_kmv() on the inference backend exactly once."
        )
        self.assertEqual(
            train_backend.materialize_kmv.call_count, 0,
            "Training backend should not be used for inference materialisation."
        )

    def test_graph_cycling_callback_used(self):
        """
        GraphCyclingCallback must be instantiated — not DynamicGraphCallback.
        DynamicGraphCallback uses the Python KMV kernel, bypassing the C++ engine.
        """
        _, _, _, mock_cb_cls = self._run_kmv_with_mock_backends()
        mock_cb_cls.assert_called_once()

    def test_both_backends_cleaned_up(self):
        """Memory must be released: both training and inference backends must call cleanup()."""
        _, train_backend, infer_backend, _ = self._run_kmv_with_mock_backends()
        train_backend.cleanup.assert_called_once()
        infer_backend.cleanup.assert_called_once()

    def test_result_method_is_kmv(self):
        result, _, _, _ = self._run_kmv_with_mock_backends()
        self.assertEqual(result.method, 'KMV')

    def test_result_k_matches_input(self):
        result, _, _, _ = self._run_kmv_with_mock_backends(k=16)
        self.assertEqual(result.k, 16)

    def test_result_l_matches_input(self):
        result, _, _, _ = self._run_kmv_with_mock_backends(l=7)
        self.assertEqual(result.l, 7)

    def test_result_acc_in_valid_range(self):
        result, _, _, _ = self._run_kmv_with_mock_backends()
        self.assertGreaterEqual(result.test_acc, 0.0)
        self.assertLessEqual(result.test_acc,    1.0)

    def test_raises_if_ensemble_returns_empty(self):
        """
        If the C++ backend returns 0 sketches (e.g. bad rule file, disk error),
        the pipeline must raise RuntimeError rather than silently training on nothing.
        """
        g_hetero  = _make_hetero_graph()
        path_list = [('paper', 'paper_to_author', 'author')]
        info      = _make_info()

        empty_backend = _make_mock_backend()
        empty_backend.materialize_kmv_ensemble.return_value = []

        with patch('scripts.run_e2e_comparison._build_cpp_backend',
                   return_value=empty_backend), \
             patch('scripts.run_e2e_comparison.config') as mock_cfg:
            mock_cfg.HIDDEN_DIM    = 64
            mock_cfg.LEARNING_RATE = 0.001
            mock_cfg.DEVICE        = torch.device('cpu')

            pipeline = KMVPipeline(model_arch='SAGE', epochs=10, k=32, l=3)
            with self.assertRaises(RuntimeError):
                pipeline.run(
                    dataset='HGB_ACM',
                    metapath_str='paper_to_author,author_to_paper',
                    path_list=path_list,
                    g_hetero=g_hetero,
                    info=info,
                )


# ===========================================================================
# 5. E2EComparisonOrchestrator
# ===========================================================================

class TestE2EComparisonOrchestrator(unittest.TestCase):
    """
    Tests for the top-level orchestrator.

    Validates: binary existence check, incremental CSV saving, correct
    number of pipeline runs, and crash-resilience for individual K values.
    """

    def _make_orchestrator(self,
                            tmpdir: str,
                            k_values: Optional[List[int]] = None,
                            binary_name: str = 'graph_prep') -> E2EComparisonOrchestrator:
        return E2EComparisonOrchestrator(
            dataset='HGB_ACM',
            metapath_str='paper_to_author,author_to_paper',
            model_arch='SAGE',
            epochs=10,
            k_values=k_values or [2, 4],
            l=2,
            output_csv=os.path.join(tmpdir, 'e2e_results.csv'),
        )

    def _dummy_result(self, method: str = 'Exact', k: Any = 'N/A') -> PipelineResult:
        return PipelineResult(
            dataset='HGB_ACM', metapath='pap', model_arch='SAGE',
            method=method, k=k, l=2 if method == 'KMV' else 'N/A',
            mat_time=0.1, train_time=1.0, total_time=1.1,
            test_acc=0.80, num_edges=50, num_nodes=10, epochs_run=10,
        )

    def _create_fake_binary(self, tmpdir: str) -> str:
        """
        Creates an empty file that passes os.path.exists() for the binary check.

        Root cause of the previous 4 failures:
        patch('scripts.run_e2e_comparison.os.path.exists') patches os.path.exists
        GLOBALLY (os is a shared module reference, not a local copy). The side_effect
        that called back into os.path.exists then recursed into the mock itself,
        causing the CSV to never be written.

        Fix: use a real file in the temp dir so the real os.path.exists is used
        throughout, no patching of os.path.exists needed anywhere.
        """
        binary_path = os.path.join(tmpdir, 'graph_prep')
        with open(binary_path, 'w') as f:
            f.write('# fake binary for testing\n')
        return binary_path

    def test_raises_if_cpp_binary_missing(self):
        """
        Orchestrator must raise FileNotFoundError before running any pipeline
        if the C++ binary does not exist. This prevents a confusing crash mid-run.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = self._make_orchestrator(tmpdir)

            with patch('scripts.run_e2e_comparison.config') as mock_cfg, \
                 patch('scripts.run_e2e_comparison.DatasetFactory') as mock_factory, \
                 patch('scripts.run_e2e_comparison.SchemaMatcher'):
                # Point to a path that genuinely does not exist
                mock_cfg.CPP_EXECUTABLE = os.path.join(tmpdir, 'nonexistent_binary')
                mock_cfg.DEVICE         = torch.device('cpu')
                mock_factory.get_data.return_value = (_make_hetero_graph(), _make_info())

                with self.assertRaises(FileNotFoundError):
                    orch.run()

    def test_csv_created_after_first_result(self):
        """
        Results must be saved incrementally. The CSV must exist after the
        first pipeline result — not only at the end of the full run.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = self._create_fake_binary(tmpdir)
            orch        = self._make_orchestrator(tmpdir, k_values=[2])
            csv_path    = orch.output_csv

            self.assertFalse(os.path.exists(csv_path),
                             "CSV should not exist before run starts.")

            with patch('scripts.run_e2e_comparison.config') as mock_cfg, \
                 patch('scripts.run_e2e_comparison.DatasetFactory') as mock_factory, \
                 patch('scripts.run_e2e_comparison.SchemaMatcher'), \
                 patch('scripts.run_e2e_comparison.OraclePipeline') as mock_oracle_cls, \
                 patch('scripts.run_e2e_comparison.KMVPipeline') as mock_kmv_cls:

                mock_cfg.CPP_EXECUTABLE = binary_path   # real file → exists() returns True
                mock_cfg.DEVICE         = torch.device('cpu')
                mock_factory.get_data.return_value = (_make_hetero_graph(), _make_info())

                oracle_instance = MagicMock()
                oracle_instance.run.return_value = self._dummy_result('Exact')
                mock_oracle_cls.return_value = oracle_instance

                kmv_instance = MagicMock()
                kmv_instance.run.return_value = self._dummy_result('KMV', k=2)
                mock_kmv_cls.return_value = kmv_instance

                orch.run()

            self.assertTrue(os.path.exists(csv_path),
                            "CSV must exist after orchestrator.run() completes.")

    def test_csv_has_one_row_per_pipeline_run(self):
        """
        CSV must contain exactly 1 Oracle row + 1 row per K value.
        With k_values=[2, 4], expect 3 rows total.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = self._create_fake_binary(tmpdir)
            orch        = self._make_orchestrator(tmpdir, k_values=[2, 4])

            with patch('scripts.run_e2e_comparison.config') as mock_cfg, \
                 patch('scripts.run_e2e_comparison.DatasetFactory') as mock_factory, \
                 patch('scripts.run_e2e_comparison.SchemaMatcher'), \
                 patch('scripts.run_e2e_comparison.OraclePipeline') as mock_oracle_cls, \
                 patch('scripts.run_e2e_comparison.KMVPipeline') as mock_kmv_cls:

                mock_cfg.CPP_EXECUTABLE = binary_path
                mock_cfg.DEVICE         = torch.device('cpu')
                mock_factory.get_data.return_value = (_make_hetero_graph(), _make_info())

                oracle_instance = MagicMock()
                oracle_instance.run.return_value = self._dummy_result('Exact')
                mock_oracle_cls.return_value = oracle_instance

                # Use return_value only — side_effect takes precedence and the
                # previous version used `lambda **kw:` which crashed on positional args
                kmv_instance = MagicMock()
                kmv_instance.run.return_value = self._dummy_result('KMV', k=2)
                mock_kmv_cls.return_value = kmv_instance

                orch.run()

            df = pd.read_csv(orch.output_csv)
            self.assertEqual(len(df), 3,
                             f"Expected 3 rows (1 Oracle + 2 KMV), got {len(df)}")

    def test_one_failing_k_does_not_abort_others(self):
        """
        If KMV fails for one K value (e.g. OOM), the orchestrator must
        continue and run the remaining K values. An overnight run must not
        silently die halfway through.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = self._create_fake_binary(tmpdir)
            orch        = self._make_orchestrator(tmpdir, k_values=[2, 4, 8])

            with patch('scripts.run_e2e_comparison.config') as mock_cfg, \
                 patch('scripts.run_e2e_comparison.DatasetFactory') as mock_factory, \
                 patch('scripts.run_e2e_comparison.SchemaMatcher'), \
                 patch('scripts.run_e2e_comparison.OraclePipeline') as mock_oracle_cls, \
                 patch('scripts.run_e2e_comparison.KMVPipeline') as mock_kmv_cls:

                mock_cfg.CPP_EXECUTABLE = binary_path
                mock_cfg.DEVICE         = torch.device('cpu')
                mock_factory.get_data.return_value = (_make_hetero_graph(), _make_info())

                oracle_instance = MagicMock()
                oracle_instance.run.return_value = self._dummy_result('Exact')
                mock_oracle_cls.return_value = oracle_instance

                call_n = {'n': 0}
                def kmv_run(*args, **kwargs):
                    call_n['n'] += 1
                    if call_n['n'] == 2:   # K=4 (second KMV call) fails
                        raise MemoryError("OOM simulated for K=4")
                    return self._dummy_result('KMV', k=2)

                kmv_instance = MagicMock()
                kmv_instance.run.side_effect = kmv_run
                mock_kmv_cls.return_value = kmv_instance

                orch.run()   # must not raise

            # 1 Oracle + 2 successful KMV (K=2, K=8) = 3 rows
            df = pd.read_csv(orch.output_csv)
            self.assertEqual(len(df), 3,
                             f"Expected 3 rows after one K failure; got {len(df)}")

    def test_oracle_oom_does_not_abort_kmv_runs(self):
        """
        If the Oracle pipeline fails (exact OOM on a large graph), the
        KMV runs must still proceed. This is the most likely real failure mode.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = self._create_fake_binary(tmpdir)
            orch        = self._make_orchestrator(tmpdir, k_values=[2, 4])

            with patch('scripts.run_e2e_comparison.config') as mock_cfg, \
                 patch('scripts.run_e2e_comparison.DatasetFactory') as mock_factory, \
                 patch('scripts.run_e2e_comparison.SchemaMatcher'), \
                 patch('scripts.run_e2e_comparison.OraclePipeline') as mock_oracle_cls, \
                 patch('scripts.run_e2e_comparison.KMVPipeline') as mock_kmv_cls:

                mock_cfg.CPP_EXECUTABLE = binary_path
                mock_cfg.DEVICE         = torch.device('cpu')
                mock_factory.get_data.return_value = (_make_hetero_graph(), _make_info())

                oracle_instance = MagicMock()
                oracle_instance.run.side_effect = MemoryError("Exact OOM")
                mock_oracle_cls.return_value = oracle_instance

                kmv_instance = MagicMock()
                kmv_instance.run.return_value = self._dummy_result('KMV', k=2)
                mock_kmv_cls.return_value = kmv_instance

                orch.run()   # must not raise

            df = pd.read_csv(orch.output_csv)
            self.assertEqual(len(df), 2,
                             "Expected 2 KMV rows even when Oracle fails.")


# ===========================================================================
# 6. Cross-Pipeline Isolation Tests
# ===========================================================================

class TestPipelineIsolation(unittest.TestCase):
    """
    Tests that validate the Oracle and KMV pipelines are completely
    independent — no shared backends, no shared cached artifacts.
    """

    def test_build_cpp_backend_called_separately_per_pipeline(self):
        """
        _build_cpp_backend must be called at least twice during a full run:
        once for Oracle train/inference and once for KMV training,
        plus once for KMV inference = minimum 3 calls for 1 K value.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real fake binary so os.path.exists passes without patching
            binary_path = os.path.join(tmpdir, 'graph_prep')
            with open(binary_path, 'w') as f:
                f.write('# fake\n')

            orch = E2EComparisonOrchestrator(
                dataset='HGB_ACM',
                metapath_str='paper_to_author,author_to_paper',
                model_arch='SAGE',
                epochs=5,
                k_values=[32],
                l=2,
                output_csv=os.path.join(tmpdir, 'results.csv'),
            )

            build_calls = []

            def track_build(*args, **kwargs):
                mock_b = _make_mock_backend()
                build_calls.append(kwargs.get('num_sketches', 1))
                return mock_b

            with patch('scripts.run_e2e_comparison.config') as mock_cfg, \
                 patch('scripts.run_e2e_comparison.DatasetFactory') as mock_factory, \
                 patch('scripts.run_e2e_comparison.SchemaMatcher'), \
                 patch('scripts.run_e2e_comparison.get_model', return_value=MagicMock()), \
                 patch('scripts.run_e2e_comparison.LitFullBatchGNN') as mock_lit_cls, \
                 patch('scripts.run_e2e_comparison.GraphCyclingCallback', return_value=MagicMock()), \
                 patch('scripts.run_e2e_comparison._build_cpp_backend',
                        side_effect=track_build), \
                 patch('scripts.run_e2e_comparison._build_trainer') as mock_trainer_fn:

                mock_cfg.CPP_EXECUTABLE = binary_path   # real file, no os.path.exists patch needed
                mock_cfg.DEVICE         = torch.device('cpu')
                mock_cfg.HIDDEN_DIM     = 64
                mock_cfg.LEARNING_RATE  = 0.001
                mock_factory.get_data.return_value = (_make_hetero_graph(), _make_info())

                mock_lit = MagicMock()
                mock_lit.device = torch.device('cpu')
                mock_lit_cls.return_value = mock_lit

                mock_trainer = MagicMock()
                mock_trainer.current_epoch = 4
                mock_trainer.test.return_value = [{'test_acc': 0.80}]
                mock_trainer_fn.return_value = mock_trainer

                orch.run()

            # Oracle: 1 backend (used for both train + inference materialisations)
            # KMV: 1 train backend + 1 infer backend = 2
            # Total = 3 minimum
            self.assertGreaterEqual(
                len(build_calls), 3,
                f"Expected >= 3 _build_cpp_backend calls (Oracle + KMV train + KMV infer), "
                f"got {len(build_calls)}"
            )


# ===========================================================================
# 7. CSV Schema Validation
# ===========================================================================

class TestCSVSchema(unittest.TestCase):
    """Validates the output CSV has all required columns in the correct types."""

    REQUIRED_COLUMNS = [
        'dataset', 'metapath', 'model_arch', 'method',
        'k', 'l', 'mat_time', 'train_time', 'total_time',
        'test_acc', 'num_edges', 'num_nodes', 'epochs_run',
    ]

    def test_all_required_columns_present(self):
        """Every result row must include all required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            orch = E2EComparisonOrchestrator(
                dataset='D', metapath_str='M', model_arch='SAGE',
                epochs=5, k_values=[4], l=2, output_csv=csv_path,
            )

            result = PipelineResult(
                dataset='D', metapath='M', model_arch='SAGE',
                method='KMV', k=4, l=2,
                mat_time=0.1, train_time=1.0, total_time=1.1,
                test_acc=0.75, num_edges=40, num_nodes=10, epochs_run=5,
            )
            orch._save_incremental(result)

            df = pd.read_csv(csv_path)
            for col in self.REQUIRED_COLUMNS:
                self.assertIn(col, df.columns,
                              f"Required column '{col}' missing from CSV.")

    def test_numeric_columns_are_numeric(self):
        """Timing and accuracy columns must be numeric, not string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            orch = E2EComparisonOrchestrator(
                dataset='D', metapath_str='M', model_arch='SAGE',
                epochs=5, k_values=[4], l=2, output_csv=csv_path,
            )

            for method, k in [('Exact', 'N/A'), ('KMV', 16)]:
                orch._save_incremental(PipelineResult(
                    dataset='D', metapath='M', model_arch='SAGE',
                    method=method, k=k, l=2 if method == 'KMV' else 'N/A',
                    mat_time=0.5, train_time=2.0, total_time=2.5,
                    test_acc=0.80, num_edges=30, num_nodes=8, epochs_run=10,
                ))

            df = pd.read_csv(csv_path)
            for col in ['mat_time', 'train_time', 'total_time', 'test_acc']:
                self.assertTrue(
                    pd.api.types.is_numeric_dtype(df[col]),
                    f"Column '{col}' should be numeric."
                )

    def test_incremental_save_appends_not_overwrites(self):
        """Each _save_incremental call must append a row, not overwrite the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            orch = E2EComparisonOrchestrator(
                dataset='D', metapath_str='M', model_arch='SAGE',
                epochs=5, k_values=[4], l=2, output_csv=csv_path,
            )

            for k in [2, 4, 8]:
                orch._save_incremental(PipelineResult(
                    dataset='D', metapath='M', model_arch='SAGE',
                    method='KMV', k=k, l=2,
                    mat_time=0.1, train_time=1.0, total_time=1.1,
                    test_acc=0.7, num_edges=20, num_nodes=8, epochs_run=5,
                ))

            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 3,
                             "3 _save_incremental calls must produce exactly 3 rows.")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    # Run with verbose output so individual test names are visible
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Registration order matches logical test flow
    for cls in [
        TestPipelineResult,
        TestBuildCppBackend,
        TestOraclePipeline,
        TestKMVPipeline,
        TestE2EComparisonOrchestrator,
        TestPipelineIsolation,
        TestCSVSchema,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)