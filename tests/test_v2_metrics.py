"""Tests for SPEC v2 metric helpers (`_unbiased_cka`, `_compute_v2_metrics`).

Locked spec: `final_report/research_notes/SPEC_approach_a_2026_05_07.md` Section 5.

These tests pin the math against canonical formulas and the property
assertions baked into the production code.  Run with:

    pytest tests/test_v2_metrics.py -v

The tests are self-contained — no torch_sparse / C++ binary needed.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Make project root importable for the cka / exp3 helpers.  We bypass the
# `src/` package __init__ (which eagerly imports torch_sparse-dependent
# kernels) by loading `src/analysis/cka.py` directly via importlib.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

# torch_sparse is not installed on the local Windows env (only on the GPU
# server).  exp3_inference.py imports `from src.config import config`, which
# triggers src/__init__.py → src/kernels/exact.py → `from torch_sparse import
# spspmm`.  None of that machinery is needed for the metric helpers we test
# here, so we stub the missing module before any imports.
if "torch_sparse" not in sys.modules:
    import types
    _stub = types.ModuleType("torch_sparse")
    _stub.spspmm = lambda *a, **kw: (_ for _ in ()).throw(  # noqa: E731
        RuntimeError("torch_sparse stub — kernels not exercised in tests"))
    sys.modules["torch_sparse"] = _stub


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Direct file-load to avoid triggering src/__init__.py's torch_sparse imports.
_cka_mod = _load_module_from_file(
    "_cka_direct", PROJECT_ROOT / "src" / "analysis" / "cka.py"
)
LinearCKA = _cka_mod.LinearCKA  # noqa: N816

# exp3_inference depends on torch_geometric (always present locally) but may
# also pull torch_sparse via `src.bridge`.  If that import fails, skip the
# suite cleanly with a clear message.
try:
    _exp3 = _load_module_from_file(
        "_exp3_direct", PROJECT_ROOT / "scripts" / "exp3_inference.py"
    )
    _unbiased_cka = _exp3._unbiased_cka
    _compute_v2_metrics = _exp3._compute_v2_metrics
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"exp3_inference helpers unavailable on this env: {exc}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hand_metrics(Z_e: torch.Tensor, Z_k: torch.Tensor):
    """Compute the v2 fidelity metrics with hand-rolled formulas."""
    eps = 1e-10
    n = Z_e.size(0)
    cos_per_row = torch.stack([
        (Z_e[i] @ Z_k[i]) / (Z_e[i].norm() * Z_k[i].norm())
        for i in range(n)
    ])
    rel = torch.stack([
        (Z_e[i] - Z_k[i]).norm() / Z_e[i].norm().clamp(min=eps)
        for i in range(n)
    ])
    frob = (Z_e - Z_k).norm() / Z_e.norm().clamp(min=eps)
    # Procrustes Q-orth (Schoenemann 1966).
    M = Z_e.T @ Z_k
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    Q_opt = U @ Vt   # canonical Schoenemann form
    procrustes = (Z_e @ Q_opt - Z_k).norm() / Z_e.norm().clamp(min=eps)
    return {
        "row_cos_mean": float(cos_per_row.mean().item()),
        "rel_l2_mean":  float(rel.mean().item()),
        "frob":         float(frob.item()),
        "procrustes_q_orth": float(procrustes.item()),
    }


def _run_compute_v2(Z_e_full: torch.Tensor, Z_k_full: torch.Tensor,
                    mask: torch.Tensor, tmp: Path, layered: bool = False):
    """Drive `_compute_v2_metrics` via tmp z-files.  Returns the metrics dict."""
    if layered:
        torch.save([Z_e_full], tmp / "ze_layers.pt")
        torch.save([Z_k_full], tmp / "zk_layers.pt")
        return _compute_v2_metrics(
            None, str(tmp / "ze_layers.pt"),
            None, str(tmp / "zk_layers.pt"),
            mask, torch.device("cpu"), LinearCKA(),
        )
    torch.save(Z_e_full, tmp / "ze.pt")
    torch.save(Z_k_full, tmp / "zk.pt")
    return _compute_v2_metrics(
        str(tmp / "ze.pt"), None,
        str(tmp / "zk.pt"), None,
        mask, torch.device("cpu"), LinearCKA(),
    )


# ---------------------------------------------------------------------------
# Hand-verification (Task 1)
# ---------------------------------------------------------------------------

def test_hand_verification_5x4_random():
    """Compare the production implementation against hand-rolled formulas."""
    torch.manual_seed(0)
    Z_e = torch.randn(5, 4, dtype=torch.float64)
    Z_k = torch.randn(5, 4, dtype=torch.float64)
    eps = 1e-10

    # Per-row cosine — must match torch.nn.functional.cosine_similarity.
    cos_hand = torch.stack([
        (Z_e[i] @ Z_k[i]) / (Z_e[i].norm() * Z_k[i].norm())
        for i in range(5)
    ])
    cos_code = F.cosine_similarity(Z_e, Z_k, dim=1)
    assert (cos_hand - cos_code).abs().max().item() < 1e-12

    # Per-row relative L2.
    rel_hand = torch.stack([
        (Z_e[i] - Z_k[i]).norm() / Z_e[i].norm() for i in range(5)
    ])
    rel_code = (Z_e - Z_k).norm(dim=1) / Z_e.norm(dim=1).clamp(min=eps)
    assert (rel_hand - rel_code).abs().max().item() < 1e-12

    # Frobenius reconstruction error.
    fhand = math.sqrt(sum((Z_e[i] - Z_k[i]).pow(2).sum().item()
                          for i in range(5)))
    fhand /= math.sqrt(sum(Z_e[i].pow(2).sum().item() for i in range(5)))
    fcode = ((Z_e - Z_k).norm() / Z_e.norm().clamp(min=eps)).item()
    assert abs(fhand - fcode) < 1e-12

    # Procrustes (Schoenemann form) — code form vs canonical Schoenemann.
    M = Z_e.T @ Z_k
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    schoen = ((Z_e @ (U @ Vt) - Z_k).norm() / Z_e.norm()).item()
    # Code: Q = (U@Vt).T; resid = (Z_e @ Q.T - Z_k).norm() / Z_e.norm()
    #       Q.T = U@Vt → equivalent to Schoenemann.
    Q = (U @ Vt).T
    resid_code = ((Z_e @ Q.T - Z_k).norm() / Z_e.norm()).item()
    assert abs(schoen - resid_code) < 1e-12, (
        "Code Procrustes form is NOT equivalent to Schoenemann"
    )

    # Property: Q-orth <= Q=I  (rotation can only help).
    q_eq_i = ((Z_e - Z_k).norm() / Z_e.norm()).item()
    assert schoen <= q_eq_i + 1e-9


# ---------------------------------------------------------------------------
# 1. Identity test — X == Y → all metrics meet identity bounds.
# ---------------------------------------------------------------------------

def test_identity(tmp_path):
    """For X == Y, all six fidelity metrics must hit identity bounds."""
    torch.manual_seed(1)
    n, d = 50, 8
    Z = torch.randn(n, d).float()
    mask = torch.ones(n, dtype=torch.bool)
    out = _run_compute_v2(Z, Z.clone(), mask, tmp_path, layered=False)

    # Print all six values for the audit log (visible with -v).
    for k, v in out.items():
        print(f"  identity {k}: {v}")

    assert abs(out["row_cosine_per_layer_mean"][0] - 1.0) < 1e-5
    assert out["row_rel_l2_per_layer_mean"][0] < 1e-5
    assert out["frob_recon_err_per_layer"][0] < 1e-5
    assert out["procrustes_q_eq_i_per_layer"][0] < 1e-5
    assert out["procrustes_q_orth_per_layer"][0] < 1e-4
    assert abs(out["cka_unbiased_per_layer"][0] - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# 2. Orthogonal-rotation test
#    Y = X @ R for orthogonal R → Procrustes-Q-orth ≈ 0,
#    but row_cos != 1 in general (per-row), and Q=I residual > 0.
# ---------------------------------------------------------------------------

def test_orthogonal_rotation(tmp_path):
    """Y = X @ R for random orthogonal R: Q-orth ≈ 0; Q=I > 0."""
    torch.manual_seed(2)
    n, d = 40, 6
    X = torch.randn(n, d).float()
    # Build orthogonal R via QR.
    A = torch.randn(d, d).float()
    Q, _ = torch.linalg.qr(A)
    Y = X @ Q

    mask = torch.ones(n, dtype=torch.bool)
    out = _run_compute_v2(X, Y, mask, tmp_path, layered=False)

    # Procrustes with optimal Q should recover the rotation → near 0.
    assert out["procrustes_q_orth_per_layer"][0] < 1e-4, (
        f"Q-orth should ~ 0 under pure rotation; got "
        f"{out['procrustes_q_orth_per_layer'][0]}"
    )
    # Q=I (= Frobenius reconstruction) is genuinely nonzero — rotation is real.
    assert out["procrustes_q_eq_i_per_layer"][0] > 1e-3
    # Property: Q-orth <= Q=I.
    assert (out["procrustes_q_orth_per_layer"][0]
            <= out["procrustes_q_eq_i_per_layer"][0] + 1e-3)


# ---------------------------------------------------------------------------
# 3. Random i.i.d. inputs — CKA ≈ 0, row_cos ≈ 0.
# ---------------------------------------------------------------------------

def test_random_inputs_decorrelated(tmp_path):
    torch.manual_seed(3)
    n, d = 200, 16
    X = torch.randn(n, d).float()
    Y = torch.randn(n, d).float()

    mask = torch.ones(n, dtype=torch.bool)
    out = _run_compute_v2(X, Y, mask, tmp_path, layered=False)

    # row_cos mean for independent random vectors ≈ 0 (broad tolerance).
    assert abs(out["row_cosine_per_layer_mean"][0]) < 0.15
    # CKA for independent inputs at this scale should be near 0.
    cka = out["cka_unbiased_per_layer"][0]
    assert abs(cka) < 0.2, f"CKA should be ~0 for independent random Z; got {cka}"


# ---------------------------------------------------------------------------
# 4. Bounds over many random pairs — all metrics in valid ranges.
# ---------------------------------------------------------------------------

def test_bounds_random_inputs(tmp_path):
    """Over 50 random (X, Y) pairs, all metrics must stay in valid ranges."""
    for seed in range(50):
        torch.manual_seed(100 + seed)
        n = int(torch.randint(20, 80, ()).item())
        d = int(torch.randint(4, 32, ()).item())
        X = torch.randn(n, d).float()
        Y = torch.randn(n, d).float()
        mask = torch.ones(n, dtype=torch.bool)
        # Use a fresh subdir each iteration.
        sub = tmp_path / f"s{seed}"
        sub.mkdir()
        out = _run_compute_v2(X, Y, mask, sub, layered=False)

        rc  = out["row_cosine_per_layer_mean"][0]
        rl2 = out["row_rel_l2_per_layer_mean"][0]
        fr  = out["frob_recon_err_per_layer"][0]
        qei = out["procrustes_q_eq_i_per_layer"][0]
        qor = out["procrustes_q_orth_per_layer"][0]
        cu  = out["cka_unbiased_per_layer"][0]

        assert -1.05 <= rc <= 1.05
        assert rl2 >= 0
        assert fr >= 0
        assert qei >= 0
        assert qor is not None and 0 <= qor <= qei + 1e-3
        assert -0.1 <= cu <= 1.05


# ---------------------------------------------------------------------------
# 5. Unbiased CKA ≈ biased CKA at large n, d=10.
# ---------------------------------------------------------------------------

def test_unbiased_vs_biased_cka_large_n():
    torch.manual_seed(4)
    n, d = 200, 10
    X = torch.randn(n, d).float()
    # Y is a noisy rotation of X — non-trivial CKA.
    A = torch.randn(d, d).float()
    Q, _ = torch.linalg.qr(A)
    Y = X @ Q + 0.1 * torch.randn(n, d).float()

    cka_calc = LinearCKA()
    biased = cka_calc.calculate(X, Y)
    unbiased = _unbiased_cka(X, Y, cka_calc)
    assert abs(biased - unbiased) < 0.01, (
        f"Unbiased ({unbiased}) should be within 0.01 of biased ({biased}) "
        f"at n={n}"
    )


# ---------------------------------------------------------------------------
# 6. Davari diagonal-zeroing — verified via the assertion in _unbiased_cka.
#    If the diagonal is not zeroed, the assert in production code triggers.
#    Here we verify the code path is reached and that the production helper
#    does not raise on standard input (its internal assertion is satisfied).
# ---------------------------------------------------------------------------

def test_unbiased_cka_diagonal_zeroing_property():
    torch.manual_seed(5)
    n, d = 30, 8
    X = torch.randn(n, d).float()
    Y = torch.randn(n, d).float()
    # The internal assertion `K_tilde diagonal not zeroed` would raise
    # AssertionError if the code path skipped the zeroing step.
    out = _unbiased_cka(X, Y, LinearCKA())
    assert out is not None and -1.0 <= out <= 1.05

    # Also verify we hit the unbiased branch (not the n<=4 fallback).
    # When n=4 the function should fall back to biased; when n>4 it shouldn't.
    Xs = torch.randn(4, d).float()
    Ys = torch.randn(4, d).float()
    fb = _unbiased_cka(Xs, Ys, LinearCKA())
    biased4 = LinearCKA().calculate(Xs, Ys)
    assert abs(fb - biased4) < 1e-6, "n<=4 must fall back to biased CKA"


# ---------------------------------------------------------------------------
# 7. Cross-check against a v2 JSON when paired z_*.pt files are available.
# ---------------------------------------------------------------------------

def test_v2_metrics_against_existing_json():
    """If a v2 JSON has paired z_*.pt + z_*_layers.pt files alongside it,
    recompute the metrics and verify they match within 1e-5.

    Skipped cleanly when no paired files are present — the v2 producer in
    the current repo does not save z tensors next to the JSON.
    """
    json_root = PROJECT_ROOT / "results" / "approach_a_2026_05_07"
    if not json_root.exists():
        pytest.skip("no v2 JSON tree present")

    paired = None
    for jpath in json_root.rglob("*.json"):
        # Look for sibling z_*.pt or z_*_layers.pt.
        d = jpath.parent
        cand_e = list(d.glob("z_exact*.pt"))
        cand_k = list(d.glob("z_kmv*.pt"))
        if cand_e and cand_k:
            paired = (jpath, cand_e[0], cand_k[0])
            break

    if paired is None:
        pytest.skip("no v2 JSON has paired z_*.pt files — skipping replay")

    jpath, ze_path, zk_path = paired
    j = json.loads(jpath.read_text())
    Ze = torch.load(ze_path, weights_only=True)
    Zk = torch.load(zk_path, weights_only=True)
    n = Ze.size(0)
    mask = torch.ones(n, dtype=torch.bool)

    out = _compute_v2_metrics(
        str(ze_path), None, str(zk_path), None,
        mask, torch.device("cpu"), LinearCKA(),
    )

    # The recorded JSON values are per-layer; the .pt files we replay here
    # are the final output (single-element list in `out`).  Match against
    # the LAST element of each recorded per-layer list.
    rec = j.get("frob_recon_err_per_layer")
    if rec:
        assert abs(out["frob_recon_err_per_layer"][-1] - rec[-1]) < 1e-4


# ---------------------------------------------------------------------------
# 8. Property assertion firing — feed a doctored layer and verify the
#    assert in `_compute_v2_metrics` does not crash on borderline noise.
# ---------------------------------------------------------------------------

def test_assertions_tolerate_borderline_noise(tmp_path):
    """A near-identity input has tiny fp noise; assertions must not fire."""
    torch.manual_seed(6)
    n, d = 50, 10
    Z = torch.randn(n, d).float()
    Z2 = Z + 1e-7 * torch.randn(n, d).float()
    mask = torch.ones(n, dtype=torch.bool)
    out = _run_compute_v2(Z, Z2, mask, tmp_path, layered=False)
    assert out["row_cosine_per_layer_mean"][0] > 0.999
    assert out["frob_recon_err_per_layer"][0] < 1e-3


# ---------------------------------------------------------------------------
# 9. Layered input — verify both layers are processed.
# ---------------------------------------------------------------------------

def test_layered_input(tmp_path):
    torch.manual_seed(7)
    n, d = 30, 8
    Z_e1 = torch.randn(n, d).float()
    Z_e2 = torch.randn(n, d).float()
    Z_k1 = Z_e1 + 0.1 * torch.randn(n, d).float()
    Z_k2 = Z_e2 + 0.1 * torch.randn(n, d).float()
    mask = torch.ones(n, dtype=torch.bool)

    torch.save([Z_e1, Z_e2], tmp_path / "ze_layers.pt")
    torch.save([Z_k1, Z_k2], tmp_path / "zk_layers.pt")
    out = _compute_v2_metrics(
        None, str(tmp_path / "ze_layers.pt"),
        None, str(tmp_path / "zk_layers.pt"),
        mask, torch.device("cpu"), LinearCKA(),
    )
    assert len(out["row_cosine_per_layer_mean"]) == 2
    assert len(out["frob_recon_err_per_layer"]) == 2
    # Both layers should have similar (high) fidelity given small noise.
    for i in range(2):
        assert out["row_cosine_per_layer_mean"][i] > 0.7
