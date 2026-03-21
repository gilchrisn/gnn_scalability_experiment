"""
BoolAP bridge — converter and runner for BoolAPCore experiments.

Provides an Adapter (BoolAPConverter) that converts a PyG HeteroData graph
into the three-file input format expected by BoolAPCoreD / BoolAPCoreG, and
a Runner (BoolAPRunner) that executes the compiled binary and returns parsed
timing results.

Both classes mirror the conventions of PyGToCppAdapter and GraphPrepRunner:
  - Same global node-ID scheme (node types concatenated in sorted order)
  - Same edge-type sort order and 1-indexed integer mapping

References:
    Guo et al., "Efficient Core Decomposition over Large Heterogeneous
    Information Networks", ICDE 2024.
    Source: parallel-k-P-core-decomposition-code/
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MS_TO_S: float = 1e-3

_RE_TOTAL  = re.compile(r"running time of BoolAPCore?\([DG]\):\s*([0-9.]+)ms", re.IGNORECASE)
_RE_BUILD  = re.compile(r"running time for building Gp:\s*([0-9.]+)ms")
_RE_DECOMP = re.compile(r"running time for core decomposition:\s*([0-9.]+)ms")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BoolAPFiles:
    """Paths of the three files written by BoolAPConverter.convert()."""

    hin_path:      Path
    metapath_path: Path
    vertices_path: Path


@dataclass
class BoolAPResult:
    """Parsed timing output from a BoolAPCoreD or BoolAPCoreG run."""

    total_time_s:    float  # Full algorithm wall-clock time (seconds)
    build_gp_time_s: float  # Time to build projected graph Gp (seconds)
    decomp_time_s:   float  # Time for core decomposition only (seconds)
    stdout:          str    # Raw binary output (for debugging)


# ---------------------------------------------------------------------------
# BoolAPConverter  (Adapter pattern)
# ---------------------------------------------------------------------------

class BoolAPConverter:
    """
    Converts a PyG HeteroData graph to BoolAP's three-file input format.

    Uses the same global node-ID scheme as PyGToCppAdapter (node types
    concatenated in sorted alphabetical order, 0-indexed) and the same
    edge-type sort order as bench_utils.compile_rule_for_cpp (1-indexed).

    The HIN file lists every directed edge from every relation type.
    The metapath file encodes the metapath as 1-indexed edge-type integers
    (always positive, since HGB stores explicit reverse relations).
    The vertices file records the node-type integer for every vertex.

    Args:
        out_dir: Directory where output files will be written.
    """

    def __init__(self, out_dir: str) -> None:
        self._out_dir = Path(out_dir)

    def convert(
        self,
        g_hetero: HeteroData,
        metapath_str: str,
        prefix: str,
    ) -> BoolAPFiles:
        """
        Write HIN, metapath, and vertices files for a BoolAP run.

        Args:
            g_hetero:     Loaded PyG heterogeneous graph.
            metapath_str: Comma-separated relation names
                          (e.g. ``'author_to_paper,paper_to_author'``).
            prefix:       File-name prefix (typically the dataset folder name).

        Returns:
            BoolAPFiles with absolute paths to the three written files.

        Raises:
            RuntimeError: If any relation in *metapath_str* cannot be
                          resolved against the graph schema.
        """
        self._out_dir.mkdir(parents=True, exist_ok=True)

        sorted_node_types: List[str]              = sorted(g_hetero.node_types)
        sorted_edge_types: List[Tuple[str, str, str]] = sorted(g_hetero.edge_types)

        offsets      = self._build_offsets(g_hetero, sorted_node_types)
        edge_type_map = {et: idx + 1 for idx, et in enumerate(sorted_edge_types)}
        total_nodes  = sum(g_hetero[nt].num_nodes for nt in sorted_node_types)

        hin_path      = self._out_dir / f"hin_{prefix}.txt"
        metapath_path = self._out_dir / f"metapath_{prefix}.txt"
        vertices_path = self._out_dir / f"vertices_{prefix}.txt"

        self._write_hin(
            g_hetero, sorted_edge_types, offsets, edge_type_map,
            total_nodes, hin_path,
        )
        self._write_metapath(g_hetero, metapath_str, edge_type_map, metapath_path)
        self._write_vertices(g_hetero, sorted_node_types, vertices_path)

        return BoolAPFiles(
            hin_path=hin_path,
            metapath_path=metapath_path,
            vertices_path=vertices_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_offsets(
        g_hetero: HeteroData,
        sorted_node_types: List[str],
    ) -> Dict[str, int]:
        """Return cumulative global-ID offsets per node type."""
        offsets: Dict[str, int] = {}
        cursor = 0
        for nt in sorted_node_types:
            offsets[nt] = cursor
            cursor += g_hetero[nt].num_nodes
        return offsets

    @staticmethod
    def _write_hin(
        g_hetero: HeteroData,
        sorted_edge_types: List[Tuple[str, str, str]],
        offsets: Dict[str, int],
        edge_type_map: Dict[Tuple[str, str, str], int],
        total_nodes: int,
        path: Path,
    ) -> None:
        """Write HIN file: header line followed by one directed edge per line."""
        total_edges = sum(
            g_hetero[et].edge_index.shape[1] for et in sorted_edge_types
        )
        num_types = len(sorted_edge_types)

        with open(path, "w") as fh:
            fh.write(f"{total_nodes} {total_edges} {num_types}\n")
            for src_type, rel, dst_type in sorted_edge_types:
                et      = (src_type, rel, dst_type)
                et_int  = edge_type_map[et]
                eidx    = g_hetero[et].edge_index
                s_off   = offsets[src_type]
                d_off   = offsets[dst_type]
                for i in range(eidx.shape[1]):
                    src = int(eidx[0, i]) + s_off
                    dst = int(eidx[1, i]) + d_off
                    fh.write(f"{src} {dst} {et_int}\n")

    @staticmethod
    def _write_metapath(
        g_hetero: HeteroData,
        metapath_str: str,
        edge_type_map: Dict[Tuple[str, str, str], int],
        path: Path,
    ) -> None:
        """Write metapath file: path length then space-separated type integers."""
        from src.utils import SchemaMatcher

        relations = [r.strip() for r in metapath_str.split(",")]
        type_ints: List[int] = []

        for rel_str in relations:
            matched = SchemaMatcher.match(rel_str, g_hetero)
            if matched not in edge_type_map:
                raise RuntimeError(
                    f"Relation '{rel_str}' resolved to {matched!r} which is "
                    f"not present in the graph schema."
                )
            type_ints.append(edge_type_map[matched])

        with open(path, "w") as fh:
            fh.write(f"{len(relations)}\n")
            fh.write(" ".join(str(t) for t in type_ints) + "\n")

    @staticmethod
    def _write_vertices(
        g_hetero: HeteroData,
        sorted_node_types: List[str],
        path: Path,
    ) -> None:
        """Write vertices file: one node-type integer per vertex line."""
        with open(path, "w") as fh:
            for type_idx, nt in enumerate(sorted_node_types):
                for _ in range(g_hetero[nt].num_nodes):
                    fh.write(f"{type_idx}\n")


# ---------------------------------------------------------------------------
# BoolAPRunner
# ---------------------------------------------------------------------------

class BoolAPRunner:
    """
    Subprocess wrapper for BoolAPCoreD and BoolAPCoreG binaries.

    On Windows the binary must be compiled inside WSL.  This class
    automatically converts file paths to WSL-compatible ``/mnt/<drive>/...``
    format and prepends ``wsl`` to the command when running on Windows.

    Args:
        binary_path: Path to the compiled BoolAPCoreD or BoolAPCoreG
                     executable (Linux path if calling from WSL directly,
                     or Windows path if this class handles the WSL prefix).
    """

    def __init__(self, binary_path: str, num_threads: int = 1) -> None:
        # Strip "wsl " prefix if passed by mistake — _build_cmd adds it automatically
        if binary_path.startswith("wsl "):
            binary_path = binary_path[4:].strip()
        self._binary_path = binary_path
        self._num_threads  = num_threads  # OMP_NUM_THREADS; default 1 matches base paper

    def run(
        self,
        files: BoolAPFiles,
        timeout: int = 1200,
    ) -> BoolAPResult:
        """
        Execute the BoolAP binary and return parsed timing results.

        Args:
            files:   Output of BoolAPConverter.convert() for this metapath.
            timeout: Subprocess timeout in seconds (default 1200).

        Returns:
            BoolAPResult with wall-clock timings broken down by phase.

        Raises:
            FileNotFoundError: If the binary does not exist.
            subprocess.TimeoutExpired: If the run exceeds *timeout* seconds.
            RuntimeError: If the binary exits non-zero or required timing
                          lines are absent from stdout.
        """
        if not os.path.exists(self._binary_path):
            raise FileNotFoundError(
                f"BoolAP binary not found: {self._binary_path}"
            )

        cmd = self._build_cmd(files.hin_path, files.metapath_path)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if proc.returncode != 0:
            raise RuntimeError(
                f"BoolAP exited {proc.returncode}.\n"
                f"STDERR: {proc.stderr.strip()}\n"
                f"STDOUT: {proc.stdout.strip()}"
            )

        return self._parse(proc.stdout)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_cmd(self, hin_path: Path, metapath_path: Path) -> List[str]:
        """Construct the subprocess command, adding WSL wrapper on Windows.

        OMP_NUM_THREADS is set via ``env`` so the binary runs single-threaded
        by default, matching the base paper's single-threaded experimental setup.
        Pass ``num_threads > 1`` only for ablation / sensitivity studies.
        """
        omp_prefix = f"OMP_NUM_THREADS={self._num_threads}"
        if sys.platform == "win32":
            return [
                "wsl", "env", omp_prefix,
                self._to_wsl_path(self._binary_path),
                self._to_wsl_path(str(hin_path)),
                self._to_wsl_path(str(metapath_path)),
            ]
        return ["env", omp_prefix,
                self._binary_path, str(hin_path), str(metapath_path)]

    @staticmethod
    def _to_wsl_path(windows_path: str) -> str:
        """Convert a Windows absolute path to a WSL ``/mnt/<drive>/...`` path."""
        p = Path(windows_path).resolve()
        drive = p.drive.rstrip(":").lower()
        rest  = str(p.relative_to(p.anchor)).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    @staticmethod
    def _parse(stdout: str) -> BoolAPResult:
        """Extract timing floats from BoolAP stdout."""
        total_ms  = BoolAPRunner._require_float(_RE_TOTAL,  stdout, "total runtime")
        build_ms  = BoolAPRunner._require_float(_RE_BUILD,  stdout, "Gp build time")
        decomp_ms = BoolAPRunner._require_float(_RE_DECOMP, stdout, "core decomp time")
        return BoolAPResult(
            total_time_s=total_ms   * _MS_TO_S,
            build_gp_time_s=build_ms  * _MS_TO_S,
            decomp_time_s=decomp_ms * _MS_TO_S,
            stdout=stdout,
        )

    @staticmethod
    def _require_float(pattern: re.Pattern, text: str, label: str) -> float:
        """Parse a required float from stdout; raises RuntimeError if absent."""
        m = pattern.search(text)
        if m is None:
            raise RuntimeError(
                f"BoolAP stdout missing expected token for '{label}'.\n"
                f"Raw output (first 500 chars):\n{text[:500]}"
            )
        return float(m.group(1))
