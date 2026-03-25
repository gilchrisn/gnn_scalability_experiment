"""
GraphPrepRunner — Typed Python wrappers for graph_prep experiment commands.

Why this exists
---------------
The graph_prep binary has a number of footguns that are invisible when building
argument lists manually:

  1. argv[5] simultaneously sets both K (sketch width) AND SEED (RNG seed).
     There is no way to set them independently — this class documents that
     constraint explicitly.

  2. `topr` is used verbatim in filenames.  "0.1" and "0.10" produce different
     .res files, and the Glo*/Per* tasks silently read empty ground truth if the
     string doesn't match exactly what was used during ExactD/H generation.

  3. PerD/PerD+ read from different ground-truth directories (hf1 vs df1)
     despite having almost identical CLI signatures.

  4. PerD+/PerH+ crash with Beta=0 (std::vector out-of-range).

  5. Glo*/Per* produce nonsense F1 scores if the four .res prerequisite files
     don't exist and are non-empty.  No error is thrown — output just looks
     like F1=0.

This class enforces all of those contracts at call time instead of at 2am.

Usage
-----
    runner = GraphPrepRunner(
        binary      = "bin/graph_prep",
        working_dir = "/path/to/project/root",
    )

    # Step 1: generate ground truth (returns Paths to the written .res files)
    gt = runner.run_exact("HGBn-ACM", topr="0.1")

    # Step 2: run approximate methods against that ground truth
    glo_result = runner.run_glo("GloD", "HGBn-ACM", topr="0.1", k=32)
    per_result = runner.run_per("PerD+", "HGBn-ACM", topr="0.1", k=32, beta=0.1)
"""
from __future__ import annotations

import os
import re
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Return-value dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExactResult:
    """Paths written by an ExactD/ExactD+/ExactH/ExactH+ run."""
    task:          str
    res_path:      Path   # The .res file that was written (used by Glo*/Per*)
    stdout:        str    # Raw C++ stdout for debugging


@dataclass
class GroundTruthSet:
    """
    All four .res files required before any Glo*/Per* method can run on a
    given (dataset, topr) combination.

    Attributes:
        df1_inclusive:  ExactD+  → global_res/<ds>/df1/hg_global_r<topr>.res
        df1_strict:     ExactD   → global_res/<ds>/df1/hg_global_greater_r<topr>.res
        hf1_inclusive:  ExactH+  → global_res/<ds>/hf1/hg_global_r<topr>.res
        hf1_strict:     ExactH   → global_res/<ds>/hf1/hg_global_greater_r<topr>.res
    """
    df1_inclusive: Path
    df1_strict:    Path
    hf1_inclusive: Path
    hf1_strict:    Path

    def assert_all_exist(self) -> None:
        """Raise FileNotFoundError if any .res file is missing or empty."""
        for attr in ("df1_inclusive", "df1_strict", "hf1_inclusive", "hf1_strict"):
            p: Path = getattr(self, attr)
            if not p.exists():
                raise FileNotFoundError(
                    f"Ground truth file missing: {p}\n"
                    "Run runner.run_ground_truth() for this dataset/topr first."
                )
            if p.stat().st_size == 0:
                raise ValueError(
                    f"Ground truth file exists but is empty: {p}\n"
                    "The ExactD/H run may have produced no output (peer_size=0)."
                )


@dataclass
class GloResult:
    """Parsed output from a GloD or GloH run."""
    task:             str
    dataset:          str
    topr:             str
    k:                int          # Also used as SEED (K_AND_SEED constraint)
    avg_f1:           float        # ~goodness:
    avg_time_s:       float        # ~time_per_rule:
    rule_count:       int
    per_rule_f1:      List[float]  # goodness: lines (no tilde)
    scatter_data:     List[str]    # Raw SCATTER_DATA: CSV strings
    stdout:           str


@dataclass
class PerResult:
    """Parsed output from a PerD, PerH, PerD+, or PerH+ run."""
    task:             str
    dataset:          str
    topr:             str
    k:                int
    beta:             float
    avg_accuracy:     float        # ~goodness:
    avg_time_s:       float        # ~time_per_rule:
    rule_count:       int
    query_results:    List[Dict]   # List of {node_id, istopr} dicts
    scatter_data:     List[str]
    stdout:           str


@dataclass
class ScaleResult:
    """Parsed output from scale / personalized_scale / hg_scale."""
    task:             str
    dataset:          str
    k:                int
    avg_time_s:       float        # ~time_per_rule:
    rule_count:       int
    stdout:           str


@dataclass
class MpcountResult:
    """Parsed output from mpcount."""
    dataset:          str
    counts_by_len:    Dict[int, int]   # {length: count}
    total:            int


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

# Task literals for static type-checking
ExactTask = Literal["ExactD", "ExactD+", "ExactH", "ExactH+"]
GloTask   = Literal["GloD", "GloH"]
PerTask   = Literal["PerD", "PerH", "PerD+", "PerH+"]
ScaleTask = Literal["scale", "personalized_scale", "hg_scale"]


class GraphPrepRunner:
    """
    Typed Python interface to the graph_prep C++ binary.

    All argument construction, filename derivation, and precondition checking
    is handled here so callers don't have to reason about the binary's quirks.

    Args:
        binary:      Path to the compiled graph_prep executable.
        working_dir: The directory from which the binary should be invoked
                     (must contain the dataset folders and global_res/).
        timeout:     Per-command subprocess timeout in seconds (default 20 min).
        verbose:     If True, echoes C++ stdout to the terminal in real time.
    """

    # Tasks whose ground truth lives in df1/
    _DF1_TASKS = {"GloD", "PerD", "PerD+"}
    # Tasks whose ground truth lives in hf1/
    _HF1_TASKS = {"GloH", "PerH", "PerH+"}
    # Tasks that require Beta > 0 (crash with Beta=0 via std::vector out-of-range)
    _BETA_REQUIRED = {"PerD+", "PerH+"}

    def __init__(
        self,
        binary:      str,
        working_dir: str,
        timeout:     int  = 1200,
        verbose:     bool = True,
    ) -> None:
        self.binary      = os.path.abspath(binary)
        self.working_dir = os.path.abspath(working_dir)
        self.timeout     = timeout
        self.verbose     = verbose

        if not os.path.isfile(self.binary):
            raise FileNotFoundError(
                f"graph_prep binary not found at: {self.binary}\n"
                "Compile with: g++ -o bin/graph_prep main.cpp param.cpp -O3 -std=c++11"
            )

    # ------------------------------------------------------------------
    # Public API: Ground truth generation
    # ------------------------------------------------------------------

    def run_exact(
        self,
        task:    ExactTask,
        dataset: str,
        topr:    str = "0.1",
    ) -> ExactResult:
        """
        Run one ExactD / ExactD+ / ExactH / ExactH+ task and redirect stdout
        to the correct .res file.

        The topr string is canonicalized (trailing zeros stripped) to prevent
        the silent filename mismatch bug where "0.10" and "0.1" produce
        different file names.

        Args:
            task:    One of "ExactD", "ExactD+", "ExactH", "ExactH+".
            dataset: Dataset folder name (e.g. "HGBn-ACM").
            topr:    Top-R threshold as a decimal string (default "0.1").

        Returns:
            ExactResult with the path of the .res file written.
        """
        topr = self._canonical_topr(topr)
        res_path = self._res_path_for_exact(task, dataset, topr)

        # Ensure directory exists (binary never creates it)
        res_path.parent.mkdir(parents=True, exist_ok=True)

        stdout = self._run(
            args          = [task, dataset, topr, "0"],
            redirect_path = str(res_path),
        )
        return ExactResult(task=task, res_path=res_path, stdout=stdout)

    def run_ground_truth(
        self,
        dataset: str,
        topr:    str = "0.1",
    ) -> GroundTruthSet:
        """
        Convenience: run all four Exact tasks in dependency order and return
        the complete GroundTruthSet.  This is the only safe way to ensure
        Glo*/Per* tasks have valid input.

        Execution order: ExactD+ → ExactD → ExactH+ → ExactH
        """
        topr = self._canonical_topr(topr)
        print(f"\n[ground truth] {dataset}  topr={topr}")

        d_plus  = self.run_exact("ExactD+", dataset, topr)
        d_strict = self.run_exact("ExactD",  dataset, topr)
        h_plus  = self.run_exact("ExactH+", dataset, topr)
        h_strict = self.run_exact("ExactH",  dataset, topr)

        gt = GroundTruthSet(
            df1_inclusive = d_plus.res_path,
            df1_strict    = d_strict.res_path,
            hf1_inclusive = h_plus.res_path,
            hf1_strict    = h_strict.res_path,
        )
        gt.assert_all_exist()
        return gt

    # ------------------------------------------------------------------
    # Public API: Approximate global methods
    # ------------------------------------------------------------------

    def run_glo(
        self,
        task:    GloTask,
        dataset: str,
        topr:    str = "0.1",
        k:       int = 32,
    ) -> GloResult:
        """
        Run GloD or GloH against pre-existing ground truth.

        NOTE: The C++ binary uses argv[5] as BOTH K (sketch width) AND SEED
        (RNG seed) simultaneously.  You cannot set them independently.
        Passing k=32 means K=32 and SEED=32.

        Args:
            task:    "GloD" or "GloH".
            dataset: Dataset folder name.
            topr:    Top-R threshold (must match what was used for ExactD/H).
            k:       Sketch width (= RNG seed, see NOTE above).

        Raises:
            FileNotFoundError: If the required .res ground truth files are missing.
            ValueError:        If any .res ground truth file is empty.
        """
        topr = self._canonical_topr(topr)
        self._assert_prereqs(task, dataset, topr)

        stdout = self._run(args=[task, dataset, topr, "0", str(k)])
        return GloResult(
            task         = task,
            dataset      = dataset,
            topr         = topr,
            k            = k,
            avg_f1       = self._parse_float(stdout, r"~goodness:\s*([0-9.]+)"),
            avg_time_s   = self._parse_float(stdout, r"~time_per_rule:\s*([0-9.]+)"),
            rule_count   = self._parse_int(stdout,   r"rule_count:\s*(\d+)"),
            per_rule_f1  = [float(m) for m in re.findall(r"(?<!~)goodness:\s*([0-9.]+)", stdout)],
            scatter_data = re.findall(r"SCATTER_DATA:\s*(.+)", stdout),
            stdout       = stdout,
        )

    # ------------------------------------------------------------------
    # Public API: Approximate personalized methods
    # ------------------------------------------------------------------

    def run_per(
        self,
        task:    PerTask,
        dataset: str,
        topr:    str   = "0.1",
        k:       int   = 32,
        beta:    float = 0.1,
    ) -> PerResult:
        """
        Run PerD / PerH / PerD+ / PerH+ against pre-existing ground truth.

        NOTE: Same K=SEED constraint as run_glo().

        Args:
            task:    One of "PerD", "PerH", "PerD+", "PerH+".
            dataset: Dataset folder name.
            topr:    Top-R threshold.
            k:       Sketch width (= RNG seed).
            beta:    Pruning tolerance.  MUST be > 0 for PerD+/PerH+ or the
                     binary crashes with std::vector out-of-range.

        Raises:
            ValueError: If task is PerD+/PerH+ and beta <= 0.
        """
        topr = self._canonical_topr(topr)

        if task in self._BETA_REQUIRED and beta <= 0:
            raise ValueError(
                f"{task} requires beta > 0 (beta={beta} will crash the binary with "
                "std::vector out-of-range).  Pass beta=0.1 or higher."
            )

        self._assert_prereqs(task, dataset, topr)

        stdout = self._run(args=[task, dataset, topr, str(beta), str(k)])

        # Parse per-query results:  "qn:<id>  istopr:<true|false>"
        query_results = [
            {"node_id": int(m[0]), "istopr": m[1] == "true"}
            for m in re.findall(r"qn:(\d+)\s+istopr:(true|false)", stdout)
        ]

        return PerResult(
            task          = task,
            dataset       = dataset,
            topr          = topr,
            k             = k,
            beta          = beta,
            avg_accuracy  = self._parse_float(stdout, r"~goodness:\s*([0-9.]+)"),
            avg_time_s    = self._parse_float(stdout, r"~time_per_rule:\s*([0-9.]+)"),
            rule_count    = self._parse_int(stdout,   r"rule_count:\s*(\d+)"),
            query_results = query_results,
            scatter_data  = re.findall(r"SCATTER_DATA:\s*(.+)", stdout),
            stdout        = stdout,
        )

    # ------------------------------------------------------------------
    # Public API: Scalability benchmarks
    # ------------------------------------------------------------------

    def run_scale(
        self,
        task:    ScaleTask,
        dataset: str,
        topr:    str = "0.1",
        k:       int = 32,
    ) -> ScaleResult:
        """
        Run one of the three scalability benchmarks (timing only, no accuracy).

        Stops after PATHCOUNT=10 rules (hardcoded in the binary).

        Args:
            task:    "scale", "personalized_scale", or "hg_scale".
            dataset: Dataset folder name.
            topr:    Top-R threshold string.
            k:       Sketch width (= RNG seed).
        """
        topr   = self._canonical_topr(topr)
        stdout = self._run(args=[task, dataset, topr, "0", str(k)])
        return ScaleResult(
            task       = task,
            dataset    = dataset,
            k          = k,
            avg_time_s = self._parse_float(stdout, r"~time_per_rule:\s*([0-9.]+)"),
            rule_count = self._parse_int(stdout,   r"rule_count:\s*(\d+)"),
            stdout     = stdout,
        )

    # ------------------------------------------------------------------
    # Public API: Utility commands
    # ------------------------------------------------------------------

    def run_lensplit(self, dataset: str) -> None:
        """
        Split the master .dat rule file into three sub-files by rule length.

        Writes:
            <dataset>/<dataset>-cod-global-rules.dat1  (length-1 rules)
            <dataset>/<dataset>-cod-global-rules.dat2  (length-2 rules)
            <dataset>/<dataset>-cod-global-rules.dat3  (length-3 rules)

        WARNING: Rules with 4+ edges are silently discarded with no warning.
        """
        self._run(args=["lensplit", dataset])

    def run_mpcount(self, dataset: str) -> MpcountResult:
        """
        Count meta-path rules in the .limit rule file, grouped by path length.

        Reads:  <dataset>/<dataset>-cod-global-rules.limit
        NOTE:   Uses the .limit file, NOT the .dat file.
        """
        stdout = self._run(args=["mpcount", dataset])
        counts: Dict[int, int] = {}
        for length, count in re.findall(r"len=(\d+)\s+count:(\d+)", stdout):
            counts[int(length)] = int(count)
        total = self._parse_int(stdout, r"rule_count:\s*(\d+)")
        return MpcountResult(dataset=dataset, counts_by_len=counts, total=total)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        args:          List[str],
        redirect_path: Optional[str] = None,
    ) -> str:
        """Execute the binary and return stdout.  Raises RuntimeError on crash/timeout."""
        import time as _time
        cmd = [self.binary] + args
        print(f"\n> {' '.join(args)}  (timeout={self.timeout}s)")
        if redirect_path:
            print(f"  -> {redirect_path}")

        t0 = _time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output = True,
                text           = True,
                check          = True,
                timeout        = self.timeout,
                cwd            = self.working_dir,
            )
        except subprocess.CalledProcessError as e:
            elapsed = _time.perf_counter() - t0
            msg = f"graph_prep exited with code {e.returncode} after {elapsed:.1f}s: {' '.join(args)}"
            print(f"\n[ERROR] {msg}")
            if e.stderr: print(f"STDERR: {e.stderr.strip()}")
            raise RuntimeError(msg)
        except subprocess.TimeoutExpired:
            print(f"\n[TIMEOUT] {' '.join(args)} — killed after {self.timeout}s")
            raise RuntimeError(f"graph_prep timed out after {self.timeout}s: {' '.join(args)}")

        elapsed = _time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s")

        if redirect_path:
            Path(redirect_path).write_text(result.stdout)

        if self.verbose and not redirect_path:
            out = result.stdout.strip()
            print(out if out else "[no output]")

        return result.stdout

    def _res_path_for_exact(self, task: str, dataset: str, topr: str) -> Path:
        """
        Derive the exact .res filename the Glo*/Per* tasks will look for.

        Mapping (from §5.5–5.8 of the README):
            ExactD+  → df1/hg_global_r<topr>.res
            ExactD   → df1/hg_global_greater_r<topr>.res
            ExactH+  → hf1/hg_global_r<topr>.res
            ExactH   → hf1/hg_global_greater_r<topr>.res
        """
        subfolder = "df1" if task in ("ExactD", "ExactD+") else "hf1"
        filename  = (
            f"hg_global_r{topr}.res"         if task.endswith("+")
            else f"hg_global_greater_r{topr}.res"
        )
        return Path(self.working_dir) / "global_res" / dataset / subfolder / filename

    def _assert_prereqs(self, task: str, dataset: str, topr: str) -> None:
        """
        Verify that the .res files required by this task exist and are non-empty.
        This prevents the silent "empty ground truth → F1=0" failure mode.
        """
        if task in self._DF1_TASKS:
            needed = [
                self._res_path_for_exact("ExactD+", dataset, topr),
                self._res_path_for_exact("ExactD",  dataset, topr),
            ]
        else:
            needed = [
                self._res_path_for_exact("ExactH+", dataset, topr),
                self._res_path_for_exact("ExactH",  dataset, topr),
            ]

        for p in needed:
            if not p.exists():
                raise FileNotFoundError(
                    f"[{task}] Missing prerequisite file: {p}\n"
                    f"Run runner.run_ground_truth('{dataset}', '{topr}') first."
                )
            if p.stat().st_size == 0:
                raise ValueError(
                    f"[{task}] Prerequisite file is empty: {p}\n"
                    "The Exact run may have produced no output (check peer_size > 0)."
                )

    @staticmethod
    def _canonical_topr(topr: str) -> str:
        """
        Strip trailing zeros to prevent the silent filename mismatch bug.

        "0.10" and "0.1" produce *different* .res filenames because topr is
        used verbatim in string concatenation inside the C++ binary.

        >>> GraphPrepRunner._canonical_topr("0.10")
        '0.1'
        >>> GraphPrepRunner._canonical_topr("0.050")
        '0.05'
        >>> GraphPrepRunner._canonical_topr("0.1")
        '0.1'
        """
        try:
            # float() then back to string strips trailing zeros
            return str(float(topr)).rstrip("0").rstrip(".")  # handles "1.0" → "1"
        except ValueError:
            raise ValueError(
                f"Invalid topr value: {topr!r}. Expected a decimal string like '0.1'."
            )

    @staticmethod
    def _parse_float(stdout: str, pattern: str, default: float = 0.0) -> float:
        m = re.search(pattern, stdout)
        return float(m.group(1)) if m else default

    @staticmethod
    def _parse_int(stdout: str, pattern: str, default: int = 0) -> int:
        m = re.search(pattern, stdout)
        return int(m.group(1)) if m else default