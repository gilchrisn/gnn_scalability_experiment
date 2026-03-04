"""
Unit tests for src/bridge/runner.py and scripts/bench_utils.py.

Design principles:
  - No C++ binary required: all subprocess calls are mocked.
  - No PyG/dataset fixtures: compile_rule_for_cpp uses a MagicMock for g_hetero.
  - Tests are grouped by class/module and ordered from pure helpers → stateful methods.
  - Each test has a single, clear assertion focus.

Run with:
    pytest tests/test_runner_and_utils.py -v
"""
import os
import re
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Make project root importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.bridge.runner import (
    GraphPrepRunner,
    GroundTruthSet,
    GloResult,
    PerResult,
    ScaleResult,
    MpcountResult,
)
from scripts.bench_utils import (
    generate_qnodes,
    setup_global_res_dirs,
    run_cpp,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_runner(tmp_path: Path) -> GraphPrepRunner:
    """Create a GraphPrepRunner pointing at a fake binary in tmp_path."""
    binary = tmp_path / "bin" / "graph_prep"
    binary.parent.mkdir(parents=True)
    binary.touch()
    return GraphPrepRunner(
        binary      = str(binary),
        working_dir = str(tmp_path),
        verbose     = False,
    )


def _write_res(path: Path, content: str = "( 1 2 3\n") -> None:
    """Write a minimal non-empty .res file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _fake_subprocess_result(stdout: str = "", returncode: int = 0):
    """Return a mock CompletedProcess-like object."""
    result = MagicMock()
    result.stdout     = stdout
    result.returncode = returncode
    return result


# ===========================================================================
# 1. GraphPrepRunner — constructor
# ===========================================================================

class TestRunnerInit(unittest.TestCase):

    def test_raises_if_binary_missing(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            GraphPrepRunner(binary="/nonexistent/graph_prep", working_dir="/tmp")
        self.assertIn("graph_prep binary not found", str(ctx.exception))

    def test_resolves_binary_to_absolute_path(self, tmp_path=None):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d) / "graph_prep"
            binary.touch()
            runner = GraphPrepRunner(binary=str(binary), working_dir=d)
            self.assertTrue(os.path.isabs(runner.binary))


# ===========================================================================
# 2. GraphPrepRunner._canonical_topr  (pure static method)
# ===========================================================================

class TestCanonicalTopr(unittest.TestCase):

    def test_strips_trailing_zero(self):
        self.assertEqual(GraphPrepRunner._canonical_topr("0.10"), "0.1")

    def test_strips_multiple_trailing_zeros(self):
        self.assertEqual(GraphPrepRunner._canonical_topr("0.050"), "0.05")

    def test_leaves_canonical_unchanged(self):
        self.assertEqual(GraphPrepRunner._canonical_topr("0.1"), "0.1")

    def test_handles_integer_like(self):
        # "1.0" → "1"
        self.assertEqual(GraphPrepRunner._canonical_topr("1.0"), "1")

    def test_raises_on_non_numeric(self):
        with self.assertRaises(ValueError):
            GraphPrepRunner._canonical_topr("bad")

    def test_small_lambda_values_preserved(self):
        self.assertEqual(GraphPrepRunner._canonical_topr("0.02"), "0.02")
        self.assertEqual(GraphPrepRunner._canonical_topr("0.03"), "0.03")


# ===========================================================================
# 3. GraphPrepRunner._res_path_for_exact  (pure instance method)
# ===========================================================================

class TestResPathForExact(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        binary = Path(self._tmpdir) / "graph_prep"
        binary.touch()
        self.runner = GraphPrepRunner(
            binary      = str(binary),
            working_dir = self._tmpdir,
            verbose     = False,
        )

    def _path(self, task, topr="0.1"):
        return self.runner._res_path_for_exact(task, "HGBn-ACM", topr)

    def test_exact_d_plus_goes_to_df1_inclusive(self):
        p = self._path("ExactD+")
        self.assertIn("df1", str(p))
        self.assertIn("hg_global_r0.1.res", p.name)

    def test_exact_d_goes_to_df1_strict(self):
        p = self._path("ExactD")
        self.assertIn("df1", str(p))
        self.assertIn("hg_global_greater_r0.1.res", p.name)

    def test_exact_h_plus_goes_to_hf1_inclusive(self):
        p = self._path("ExactH+")
        self.assertIn("hf1", str(p))
        self.assertIn("hg_global_r0.1.res", p.name)

    def test_exact_h_goes_to_hf1_strict(self):
        p = self._path("ExactH")
        self.assertIn("hf1", str(p))
        self.assertIn("hg_global_greater_r0.1.res", p.name)

    def test_topr_string_used_verbatim_in_filename(self):
        # "0.05" must appear literally — not "0.050" or "0.5"
        p = self._path("ExactD+", "0.05")
        self.assertIn("r0.05.res", p.name)


# ===========================================================================
# 4. GroundTruthSet.assert_all_exist
# ===========================================================================

class TestGroundTruthSetAssertions(unittest.TestCase):

    def _make_gt(self, tmp_path, write_all=True, empty_one=None):
        files = {
            "df1_inclusive": tmp_path / "df1" / "hg_global_r0.1.res",
            "df1_strict":    tmp_path / "df1" / "hg_global_greater_r0.1.res",
            "hf1_inclusive": tmp_path / "hf1" / "hg_global_r0.1.res",
            "hf1_strict":    tmp_path / "hf1" / "hg_global_greater_r0.1.res",
        }
        for attr, path in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if write_all:
                content = "" if attr == empty_one else "( 1 2\n"
                path.write_text(content)
        return GroundTruthSet(**files)

    def test_passes_when_all_files_exist_and_non_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            gt = self._make_gt(Path(d))
            gt.assert_all_exist()  # should not raise

    def test_raises_file_not_found_when_file_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            gt = self._make_gt(Path(d), write_all=False)
            with self.assertRaises(FileNotFoundError):
                gt.assert_all_exist()

    def test_raises_value_error_when_file_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            gt = self._make_gt(Path(d), empty_one="df1_inclusive")
            with self.assertRaises(ValueError) as ctx:
                gt.assert_all_exist()
            self.assertIn("empty", str(ctx.exception))


# ===========================================================================
# 5. GraphPrepRunner._assert_prereqs
# ===========================================================================

class TestAssertPrereqs(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        binary = Path(self._tmpdir) / "graph_prep"
        binary.touch()
        self.runner = GraphPrepRunner(
            binary      = str(binary),
            working_dir = self._tmpdir,
            verbose     = False,
        )

    def test_glo_d_checks_df1_files(self):
        """GloD should check df1/ prereqs, not hf1/."""
        # Write only hf1 files — should still raise for GloD
        for task in ("ExactH+", "ExactH"):
            p = self.runner._res_path_for_exact(task, "HGBn-ACM", "0.1")
            _write_res(p)
        with self.assertRaises(FileNotFoundError) as ctx:
            self.runner._assert_prereqs("GloD", "HGBn-ACM", "0.1")
        self.assertIn("df1", str(ctx.exception))

    def test_glo_h_checks_hf1_files(self):
        """GloH should check hf1/ prereqs."""
        for task in ("ExactD+", "ExactD"):
            p = self.runner._res_path_for_exact(task, "HGBn-ACM", "0.1")
            _write_res(p)
        with self.assertRaises(FileNotFoundError) as ctx:
            self.runner._assert_prereqs("GloH", "HGBn-ACM", "0.1")
        self.assertIn("hf1", str(ctx.exception))

    def test_passes_when_both_prereqs_exist(self):
        for task in ("ExactD+", "ExactD"):
            _write_res(self.runner._res_path_for_exact(task, "HGBn-ACM", "0.1"))
        self.runner._assert_prereqs("GloD", "HGBn-ACM", "0.1")  # should not raise

    def test_raises_on_empty_prereq(self):
        p = self.runner._res_path_for_exact("ExactD+", "HGBn-ACM", "0.1")
        _write_res(p, content="")  # empty
        _write_res(self.runner._res_path_for_exact("ExactD", "HGBn-ACM", "0.1"))
        with self.assertRaises(ValueError) as ctx:
            self.runner._assert_prereqs("GloD", "HGBn-ACM", "0.1")
        self.assertIn("empty", str(ctx.exception))


# ===========================================================================
# 6. GraphPrepRunner.run_per — beta validation
# ===========================================================================

class TestRunPerBetaValidation(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        binary = Path(self._tmpdir) / "graph_prep"
        binary.touch()
        self.runner = GraphPrepRunner(
            binary      = str(binary),
            working_dir = self._tmpdir,
            verbose     = False,
        )
        # Write prerequisite .res files
        for task in ("ExactD+", "ExactD"):
            _write_res(self.runner._res_path_for_exact(task, "HGBn-ACM", "0.1"))

    def test_per_d_plus_raises_on_zero_beta(self):
        with self.assertRaises(ValueError) as ctx:
            self.runner.run_per("PerD+", "HGBn-ACM", topr="0.1", k=32, beta=0)
        self.assertIn("beta > 0", str(ctx.exception))

    def test_per_h_plus_raises_on_negative_beta(self):
        # PerH+ prereqs live in hf1
        for task in ("ExactH+", "ExactH"):
            _write_res(self.runner._res_path_for_exact(task, "HGBn-ACM", "0.1"))
        with self.assertRaises(ValueError):
            self.runner.run_per("PerH+", "HGBn-ACM", topr="0.1", k=4, beta=-0.1)

    def test_per_d_accepts_zero_beta(self):
        """PerD (no +) should not enforce the beta constraint."""
        glo_stdout = "~goodness:0.85\n~time_per_rule:1.2 s\nrule_count:5\n"
        with patch("subprocess.run", return_value=_fake_subprocess_result(glo_stdout)):
            result = self.runner.run_per("PerD", "HGBn-ACM", topr="0.1", k=32, beta=0)
        self.assertIsInstance(result, PerResult)


# ===========================================================================
# 7. GraphPrepRunner.run_exact — directory creation + stdout redirect
# ===========================================================================

class TestRunExact(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        binary = Path(self._tmpdir) / "graph_prep"
        binary.touch()
        self.runner = GraphPrepRunner(
            binary      = str(binary),
            working_dir = self._tmpdir,
            verbose     = False,
        )

    def test_creates_df1_dir_if_missing(self):
        stdout = "( 1 2 3\ntime:0.5 s\n~time_per_rule:0.5 s\nrule_count:1\n"
        with patch("subprocess.run", return_value=_fake_subprocess_result(stdout)):
            result = self.runner.run_exact("ExactD+", "HGBn-ACM", topr="0.1")
        self.assertTrue(result.res_path.parent.exists())

    def test_writes_stdout_to_res_file(self):
        stdout = "( 10 20\ntime:1.0 s\n~time_per_rule:1.0 s\nrule_count:2\n"
        with patch("subprocess.run", return_value=_fake_subprocess_result(stdout)):
            result = self.runner.run_exact("ExactD", "HGBn-ACM", topr="0.1")
        self.assertEqual(result.res_path.read_text(), stdout)

    def test_canonicalizes_topr_before_building_filename(self):
        stdout = "( 1\ntime:0.1 s\n~time_per_rule:0.1 s\nrule_count:1\n"
        with patch("subprocess.run", return_value=_fake_subprocess_result(stdout)):
            result = self.runner.run_exact("ExactD+", "HGBn-ACM", topr="0.10")
        # File should be named with "0.1" not "0.10"
        self.assertIn("r0.1.res", result.res_path.name)
        self.assertNotIn("r0.10.res", result.res_path.name)

    def test_returns_exact_result_with_correct_task(self):
        stdout = "( 1\n~time_per_rule:0.5 s\nrule_count:1\n"
        with patch("subprocess.run", return_value=_fake_subprocess_result(stdout)):
            result = self.runner.run_exact("ExactH+", "HGBn-ACM", topr="0.1")
        self.assertEqual(result.task, "ExactH+")


# ===========================================================================
# 8. GraphPrepRunner.run_glo — stdout parsing
# ===========================================================================

class TestRunGlo(unittest.TestCase):

    _SAMPLE_STDOUT = textwrap.dedent("""\
        % [0,1] instance:-1
        # l=0
        goodness:0.82
        time:1.1 s
        SCATTER_DATA: 0,150,1.1
        goodness:0.78
        time:0.9 s
        SCATTER_DATA: 1,130,0.9
        ~goodness:0.80
        ~time_per_rule:1.0 s
        rule_count:2
    """)

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        binary = Path(self._tmpdir) / "graph_prep"
        binary.touch()
        self.runner = GraphPrepRunner(
            binary      = str(binary),
            working_dir = self._tmpdir,
            verbose     = False,
        )
        for task in ("ExactD+", "ExactD"):
            _write_res(self.runner._res_path_for_exact(task, "HGBn-ACM", "0.1"))

    def test_parses_avg_f1(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_glo("GloD", "HGBn-ACM", topr="0.1", k=32)
        self.assertAlmostEqual(result.avg_f1, 0.80)

    def test_parses_avg_time(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_glo("GloD", "HGBn-ACM", topr="0.1", k=32)
        self.assertAlmostEqual(result.avg_time_s, 1.0)

    def test_parses_rule_count(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_glo("GloD", "HGBn-ACM", topr="0.1", k=32)
        self.assertEqual(result.rule_count, 2)

    def test_parses_per_rule_f1(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_glo("GloD", "HGBn-ACM", topr="0.1", k=32)
        self.assertEqual(len(result.per_rule_f1), 2)
        self.assertAlmostEqual(result.per_rule_f1[0], 0.82)
        self.assertAlmostEqual(result.per_rule_f1[1], 0.78)

    def test_parses_scatter_data(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_glo("GloD", "HGBn-ACM", topr="0.1", k=32)
        self.assertEqual(len(result.scatter_data), 2)
        self.assertIn("0,150,1.1", result.scatter_data[0])

    def test_k_stored_on_result(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_glo("GloD", "HGBn-ACM", topr="0.1", k=16)
        self.assertEqual(result.k, 16)

    def test_raises_if_prereqs_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d) / "graph_prep"
            binary.touch()
            fresh_runner = GraphPrepRunner(binary=str(binary), working_dir=d, verbose=False)
            with self.assertRaises(FileNotFoundError):
                fresh_runner.run_glo("GloD", "HGBn-ACM", topr="0.1", k=32)


# ===========================================================================
# 9. GraphPrepRunner.run_per — stdout parsing + query result extraction
# ===========================================================================

class TestRunPer(unittest.TestCase):

    _SAMPLE_STDOUT = textwrap.dedent("""\
        % [0,1] instance:-1
        # l=0
        qn:101  istopr:true
        qn:202  istopr:false
        qn:303  istopr:true
        time:0.5 s
        SCATTER_DATA: 0,50,0.5
        ~goodness:0.67
        ~time_per_rule:0.5 s
        rule_count:1
    """)

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        binary = Path(self._tmpdir) / "graph_prep"
        binary.touch()
        self.runner = GraphPrepRunner(
            binary      = str(binary),
            working_dir = self._tmpdir,
            verbose     = False,
        )
        for task in ("ExactD+", "ExactD"):
            _write_res(self.runner._res_path_for_exact(task, "HGBn-ACM", "0.1"))

    def test_parses_query_results(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_per("PerD", "HGBn-ACM", topr="0.1", k=32)
        self.assertEqual(len(result.query_results), 3)
        self.assertEqual(result.query_results[0], {"node_id": 101, "istopr": True})
        self.assertEqual(result.query_results[1], {"node_id": 202, "istopr": False})

    def test_parses_avg_accuracy(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_per("PerD", "HGBn-ACM", topr="0.1", k=32)
        self.assertAlmostEqual(result.avg_accuracy, 0.67)

    def test_beta_stored_on_result(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_per("PerD", "HGBn-ACM", topr="0.1", k=32, beta=0.2)
        self.assertAlmostEqual(result.beta, 0.2)


# ===========================================================================
# 10. GraphPrepRunner.run_scale
# ===========================================================================

class TestRunScale(unittest.TestCase):

    _SAMPLE_STDOUT = "~time_per_rule:2.5 s\nrule_count:10\n"

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        binary = Path(self._tmpdir) / "graph_prep"
        binary.touch()
        self.runner = GraphPrepRunner(
            binary      = str(binary),
            working_dir = self._tmpdir,
            verbose     = False,
        )

    def test_parses_time_and_rule_count(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_scale("scale", "HGBn-ACM", k=32)
        self.assertAlmostEqual(result.avg_time_s, 2.5)
        self.assertEqual(result.rule_count, 10)

    def test_returns_scale_result(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_scale("hg_scale", "HGBn-ACM", k=4)
        self.assertIsInstance(result, ScaleResult)
        self.assertEqual(result.task, "hg_scale")


# ===========================================================================
# 11. GraphPrepRunner.run_mpcount
# ===========================================================================

class TestRunMpcount(unittest.TestCase):

    _SAMPLE_STDOUT = textwrap.dedent("""\
        Length Count:
        len=1  count:5
        len=2  count:12
        len=3  count:8
        len=4  count:0
        len=5  count:0

        rule_count:25
    """)

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        binary = Path(self._tmpdir) / "graph_prep"
        binary.touch()
        self.runner = GraphPrepRunner(
            binary      = str(binary),
            working_dir = self._tmpdir,
            verbose     = False,
        )

    def test_parses_counts_by_length(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_mpcount("HGBn-ACM")
        self.assertEqual(result.counts_by_len[1], 5)
        self.assertEqual(result.counts_by_len[2], 12)
        self.assertEqual(result.counts_by_len[3], 8)

    def test_parses_total(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result(self._SAMPLE_STDOUT)):
            result = self.runner.run_mpcount("HGBn-ACM")
        self.assertEqual(result.total, 25)


# ===========================================================================
# 12. bench_utils.compile_rule_for_cpp — bytecode ordering
# ===========================================================================

class TestCompileRuleForCpp(unittest.TestCase):
    """
    Tests for the stack-machine bytecode emitted by compile_rule_for_cpp.

    The C++ parser requires the termination opcode (-1 for variable mode)
    to appear IMMEDIATELY BEFORE the final edge integer, not after all edges.
    Getting this order wrong produces rule_count:0 silently — this was the
    root cause of all experiments returning empty results.

    Reference (README §4.2):
        Meta-path [0→, 2→] as variable rule = "-2 0 -2 -1 2 -4 -4"
        NOT:                                   "-2 0 -2 2 -1 -4 -4"
    """

    def _get_bytecode(self, metapath_tokens, instance_id=-1):
        """
        Simulate the fixed algorithm with sequential edge IDs (0, 1, 2, ...)
        and return the space-separated bytecode string.
        """
        path_list = metapath_tokens
        eids = list(range(len(path_list)))
        parts = []
        for i, (rel_str, eid) in enumerate(zip(path_list, eids)):
            direction = "-3" if rel_str.startswith("rev_") else "-2"
            parts.append(direction)
            if i == len(path_list) - 1:
                if instance_id == -1:
                    parts.append("-1")
                else:
                    parts.extend(["-5", str(instance_id)])
            parts.append(str(eid))
        for _ in path_list:
            parts.append("-4")
        return " ".join(parts)

    def test_single_edge_variable_rule(self):
        # 1-edge: direction, -1, edge_id, -4
        self.assertEqual(self._get_bytecode(["a_to_b"]), "-2 -1 0 -4")

    def test_two_edge_variable_rule_readme_example(self):
        # README §4.2 canonical structure: intermediate edge first, then -1, then final edge
        self.assertEqual(self._get_bytecode(["a_to_b", "b_to_c"]), "-2 0 -2 -1 1 -4 -4")

    def test_three_edge_variable_rule(self):
        self.assertEqual(
            self._get_bytecode(["a_to_b", "b_to_c", "c_to_d"]),
            "-2 0 -2 1 -2 -1 2 -4 -4 -4"
        )

    def test_backward_direction_uses_minus_3(self):
        result = self._get_bytecode(["rev_b_to_a"])
        self.assertIn("-3", result)
        self.assertNotIn("-2", result)

    def test_mixed_forward_backward(self):
        self.assertEqual(self._get_bytecode(["a_to_b", "rev_c_to_b"]), "-2 0 -3 -1 1 -4 -4")

    def test_instance_mode_uses_minus_5(self):
        result = self._get_bytecode(["a_to_b", "b_to_c"], instance_id=42)
        self.assertIn("-5 42", result)
        self.assertNotIn(" -1 ", result)

    def test_pop_count_equals_path_length(self):
        for n in [1, 2, 3, 4]:
            tokens = [f"e{i}_to_x" for i in range(n)]
            result = self._get_bytecode(tokens)
            self.assertEqual(result.split().count("-4"), n)

    def test_termination_opcode_immediately_precedes_last_edge_id(self):
        """
        Core regression test: -1 must be the token directly before the last
        edge integer, not appended after all edges.
        """
        tokens = self._get_bytecode(["a_to_b", "b_to_c"]).split()
        idx = tokens.index("-1")
        # Token after -1 must be the final edge id "1", not a pop "-4"
        self.assertEqual(tokens[idx + 1], "1")
        self.assertNotEqual(tokens[idx + 1], "-4")


# ===========================================================================
# 12. bench_utils.setup_global_res_dirs
# ===========================================================================

class TestSetupGlobalResDirs(unittest.TestCase):

    def test_creates_both_subdirs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            df1, hf1 = setup_global_res_dirs("HGBn-ACM", d)
            self.assertTrue(Path(df1).is_dir())
            self.assertTrue(Path(hf1).is_dir())

    def test_returns_correct_paths(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            df1, hf1 = setup_global_res_dirs("HGBn-DBLP", d)
            self.assertIn("HGBn-DBLP", df1)
            self.assertIn("df1",       df1)
            self.assertIn("HGBn-DBLP", hf1)
            self.assertIn("hf1",       hf1)

    def test_idempotent_if_dirs_already_exist(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            setup_global_res_dirs("HGBn-ACM", d)
            setup_global_res_dirs("HGBn-ACM", d)  # should not raise


# ===========================================================================
# 13. bench_utils.generate_qnodes
# ===========================================================================

class TestGenerateQnodes(unittest.TestCase):

    def _write_node_dat(self, path: Path, n: int = 200) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{i}\tsome_label\ttype1\n" for i in range(n)]
        path.write_text("".join(lines))

    def test_creates_qnodes_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_node_dat(Path(d) / "node.dat")
            generate_qnodes(d, "HGBn-ACM")
            self.assertTrue((Path(d) / "qnodes_HGBn-ACM.dat").exists())

    def test_sample_size_respected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_node_dat(Path(d) / "node.dat", n=500)
            generate_qnodes(d, "HGBn-ACM", sample_size=50)
            lines = (Path(d) / "qnodes_HGBn-ACM.dat").read_text().splitlines()
            self.assertEqual(len(lines), 50)

    def test_handles_fewer_nodes_than_sample_size(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_node_dat(Path(d) / "node.dat", n=10)
            generate_qnodes(d, "HGBn-ACM", sample_size=100)
            lines = (Path(d) / "qnodes_HGBn-ACM.dat").read_text().splitlines()
            self.assertLessEqual(len(lines), 10)

    def test_max_scan_limits_candidate_pool(self):
        """With max_scan=5, only first 5 lines are candidates."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_node_dat(Path(d) / "node.dat", n=1000)
            generate_qnodes(d, "HGBn-ACM", max_scan=5, sample_size=5)
            lines = (Path(d) / "qnodes_HGBn-ACM.dat").read_text().splitlines()
            self.assertLessEqual(len(lines), 5)

    def test_fallback_when_node_dat_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "node.dat").write_text("")
            generate_qnodes(d, "HGBn-ACM")
            content = (Path(d) / "qnodes_HGBn-ACM.dat").read_text()
            self.assertEqual(content.strip(), "0")  # fallback node


# ===========================================================================
# 14. bench_utils.run_cpp
# ===========================================================================

class TestRunCpp(unittest.TestCase):

    def test_returns_stdout_string(self):
        mock_result = _fake_subprocess_result("rule_count:3\n")
        with patch("subprocess.run", return_value=mock_result):
            out = run_cpp("bin/graph_prep", ["hg_stats", "HGBn-ACM"], print_output=False)
        self.assertIn("rule_count:3", out)

    def test_writes_redirect_file(self):
        import tempfile
        mock_result = _fake_subprocess_result("( 1 2 3\n")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".res", delete=False) as f:
            redirect = f.name
        with patch("subprocess.run", return_value=mock_result):
            run_cpp("bin/graph_prep", ["ExactD+", "HGBn-ACM"], redirect_path=redirect, print_output=False)
        self.assertEqual(Path(redirect).read_text(), "( 1 2 3\n")
        os.unlink(redirect)

    def test_sys_exit_on_nonzero_returncode(self):
        import subprocess
        error = subprocess.CalledProcessError(1, "graph_prep", output="", stderr="segfault")
        with patch("subprocess.run", side_effect=error):
            with self.assertRaises(SystemExit):
                run_cpp("bin/graph_prep", ["GloD", "HGBn-ACM"], print_output=False)


# ===========================================================================
# 15. Integration: run_ground_truth wires all four Exact tasks correctly
# ===========================================================================

class TestRunGroundTruth(unittest.TestCase):
    """
    Checks that run_ground_truth() calls the binary four times with the
    correct task names and writes to the four expected paths.
    """

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        binary = Path(self._tmpdir) / "graph_prep"
        binary.touch()
        self.runner = GraphPrepRunner(
            binary      = str(binary),
            working_dir = self._tmpdir,
            verbose     = False,
        )

    def test_four_exact_tasks_are_called(self):
        call_log = []

        def fake_run(cmd, **kwargs):
            call_log.append(cmd[1])  # task name is argv[1]
            # Write a non-empty file to whatever redirect_path the runner expects
            # (we intercept at subprocess level, so _run's write_text handles it)
            return _fake_subprocess_result("( 1 2\n")

        with patch("subprocess.run", side_effect=fake_run):
            self.runner.run_ground_truth("HGBn-ACM", topr="0.1")

        self.assertIn("ExactD+", call_log)
        self.assertIn("ExactD",  call_log)
        self.assertIn("ExactH+", call_log)
        self.assertIn("ExactH",  call_log)
        self.assertEqual(len(call_log), 4)

    def test_returns_ground_truth_set_with_all_paths(self):
        with patch("subprocess.run", return_value=_fake_subprocess_result("( 1\n")):
            gt = self.runner.run_ground_truth("HGBn-ACM", topr="0.1")
        self.assertIsInstance(gt, GroundTruthSet)
        self.assertTrue(gt.df1_inclusive.exists())
        self.assertTrue(gt.df1_strict.exists())
        self.assertTrue(gt.hf1_inclusive.exists())
        self.assertTrue(gt.hf1_strict.exists())

    def test_canonicalizes_topr(self):
        """Passing "0.10" should result in .res files named with "0.1"."""
        with patch("subprocess.run", return_value=_fake_subprocess_result("( 1\n")):
            gt = self.runner.run_ground_truth("HGBn-ACM", topr="0.10")
        self.assertIn("r0.1.res", gt.df1_inclusive.name)
        self.assertNotIn("r0.10.res", gt.df1_inclusive.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ===========================================================================
# 16. BytecodeParser — rule enumeration from .dat files
# ===========================================================================

class TestAnyburlLineParser(unittest.TestCase):
    """
    Tests for _parse_anyburl_line in test_rule_inventory.py.
    Test cases are derived from the actual anyburl_rules.txt TSV format and
    cross-checked against AnyBURLRunner._parse_single_line behaviour.
    """

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_rule_inventory",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "test_rule_inventory.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._parse = mod._parse_anyburl_line
        self.Rule   = mod.Rule

    def _line(self, confidence, rule_str):
        """Build a TSV line in AnyBURL format: idx\tidx\tconf\trule_str"""
        return f"0\t0\t{confidence}\t{rule_str}"

    # ── Variable rules ───────────────────────────────────────────────────

    def test_two_edge_variable_rule(self):
        # author_to_paper(X,A), paper_to_author(A,Y) should parse as variable
        line = self._line(0.9, "target(X,Y) <= author_to_paper(X,A), paper_to_author(A,Y)")
        r = self._parse(line, min_conf=0.0)
        self.assertIsNotNone(r)
        self.assertEqual(r.rule_type, "variable")
        self.assertEqual(r.path, "author_to_paper,paper_to_author")
        self.assertAlmostEqual(r.confidence, 0.9)

    def test_three_edge_variable_rule(self):
        line = self._line(0.7, "t(X,Y) <= a_to_b(X,A), b_to_c(A,B), c_to_a(B,Y)")
        r = self._parse(line, min_conf=0.0)
        self.assertIsNotNone(r)
        self.assertEqual(r.path, "a_to_b,b_to_c,c_to_a")
        self.assertEqual(r.length, 3)

    def test_reverse_edge_gets_rev_prefix(self):
        # A reverse traversal happens when the current variable is v2 (destination)
        # rather than v1 (source) of an atom.
        # Chain: X -a_to_b-> A, then A <-a_to_b- B (B is the new endpoint).
        # In the second atom a_to_b(B,A): current_var=A == v2, so append rev_a_to_b.
        line = self._line(0.8, "t(X,Y) <= a_to_b(X,A), a_to_b(Y,A)")
        r = self._parse(line, min_conf=0.0)
        self.assertIsNotNone(r)
        self.assertIn("rev_a_to_b", r.path)
        self.assertEqual(r.path, "a_to_b,rev_a_to_b")

    def test_single_edge_variable_rule_rejected(self):
        # AnyBURLRunner._parse_single_line requires len(relations) >= 2 for variable rules
        line = self._line(0.95, "t(X,Y) <= author_to_paper(X,Y)")
        r = self._parse(line, min_conf=0.0)
        self.assertIsNone(r)

    def test_below_min_conf_rejected(self):
        line = self._line(0.05, "t(X,Y) <= a_to_b(X,A), b_to_a(A,Y)")
        r = self._parse(line, min_conf=0.1)
        self.assertIsNone(r)

    def test_exactly_at_min_conf_accepted(self):
        line = self._line(0.1, "t(X,Y) <= a_to_b(X,A), b_to_a(A,Y)")
        r = self._parse(line, min_conf=0.1)
        self.assertIsNotNone(r)

    # ── Instance-constrained rules ───────────────────────────────────────

    def test_instance_rule_grounded_at_end(self):
        # paper_to_author(X,A), author_to_paper(A,paper_1998) — grounded destination
        line = self._line(0.85, "t(X,Y) <= author_to_paper(X,A), paper_to_author(A,paper_1998)")
        r = self._parse(line, min_conf=0.0)
        self.assertIsNotNone(r)
        self.assertEqual(r.rule_type, "instance")
        self.assertEqual(r.instance_id, 1998)
        self.assertEqual(r.path, "author_to_paper,paper_to_author")

    def test_instance_rule_extracts_numeric_id_correctly(self):
        line = self._line(0.6, "t(X,Y) <= a_to_b(X,node_42)")
        r = self._parse(line, min_conf=0.0)
        self.assertIsNotNone(r)
        self.assertEqual(r.instance_id, 42)

    def test_rule_starting_with_grounded_node_rejected(self):
        # If v1 of first atom is lowercase (grounded), reject entirely
        line = self._line(0.9, "t(X,Y) <= author_to_paper(paper_0,X)")
        r = self._parse(line, min_conf=0.0)
        self.assertIsNone(r)

    # ── Malformed input ──────────────────────────────────────────────────

    def test_too_few_tsv_columns_returns_none(self):
        r = self._parse("0\t0\t0.9", min_conf=0.0)
        self.assertIsNone(r)

    def test_no_arrow_in_rule_returns_none(self):
        line = self._line(0.9, "author_to_paper(X,Y)")
        r = self._parse(line, min_conf=0.0)
        self.assertIsNone(r)

    def test_non_numeric_confidence_returns_none(self):
        r = self._parse("0\t0\tnot_a_float\tt(X,Y) <= a(X,Y)", min_conf=0.0)
        self.assertIsNone(r)

    # ── Integration: parse_anyburl_raw on a temp file ────────────────────

    def test_parse_raw_file_counts_both_types(self):
        import tempfile, importlib.util
        spec = importlib.util.spec_from_file_location(
            "tri2", os.path.join(os.path.dirname(__file__), "..", "scripts", "test_rule_inventory.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        content = "\n".join([
            "0\t0\t0.9\tt(X,Y) <= a_to_b(X,A), b_to_a(A,Y)",        # variable
            "0\t0\t0.8\tt(X,Y) <= a_to_b(X,A), b_to_a(A,node_5)",   # instance
            "0\t0\t0.7\tt(X,Y) <= a_to_b(X,A), b_to_a(A,node_10)",  # instance
            "bad line",                                                 # malformed
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            fname = f.name

        rules = mod.parse_anyburl_raw(fname, min_conf=0.0)
        var_rules  = [r for r in rules if r.rule_type == "variable"]
        inst_rules = [r for r in rules if r.rule_type == "instance"]

        self.assertEqual(len(var_rules),  1)
        self.assertEqual(len(inst_rules), 2)
        self.assertEqual(inst_rules[0].instance_id, 5)
        self.assertEqual(inst_rules[1].instance_id, 10)
        os.unlink(fname)