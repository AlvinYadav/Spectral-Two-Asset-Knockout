"""Each basis solves the pricing equation in the sense claimed: pointwise (Mathieu,
Bessel) or Galerkin (Fourier)."""

from __future__ import annotations

import numpy as np
import pytest

from twoasset.solvers._optimal import build_solver
from twoasset.solvers.weak_residual import (
    per_mode_operator_residual,
    fourier_galerkin_self_consistency,
)

STRONG_CONFIGS = ["default", "tight_barrier", "mahalanobis"]
WEAK_CONFIGS = ["dnt", "spread", "worst_of"]

_STRONG_MAX = 1e-4
_WEAK_MIN = 1e-3
_EPS = 2e-3


@pytest.mark.parametrize("cfg_name", STRONG_CONFIGS)
def test_mathieu_bessel_modes_satisfy_operator_pointwise(cfg_name):
    """Every retained Mathieu/Bessel mode satisfies -Delta_z Psi = k_sq Psi
    pointwise, down to the finite-difference floor."""
    solver, _m, _p, _d, basis = build_solver(cfg_name)
    res = per_mode_operator_residual(solver, n_modes=6, n_points=20, eps=_EPS)
    assert res["n_modes"] >= 4, f"{cfg_name}: only {res['n_modes']} modes measured"
    assert res["operator"] == "isotropic_laplacian_z"
    assert res["worst"] < _STRONG_MAX, (
        f"{cfg_name}: worst per-mode pointwise residual {res['worst']:.3e} "
        f"exceeds {_STRONG_MAX:.0e}; the expansion is not a strong solution"
    )


@pytest.mark.parametrize("cfg_name", WEAK_CONFIGS)
def test_fourier_modes_do_not_satisfy_operator_pointwise(cfg_name):
    """Fourier modes do NOT satisfy the anisotropic generator pointwise."""
    solver, _m, _p, _d, basis = build_solver(cfg_name)
    res = per_mode_operator_residual(solver, n_modes=6, n_points=20, eps=_EPS)
    assert res["n_modes"] >= 4, f"{cfg_name}: only {res['n_modes']} modes measured"
    assert res["operator"] == "anisotropic_generator_x"
    assert res["median"] > _WEAK_MIN, (
        f"{cfg_name}: median per-mode pointwise residual {res['median']:.3e} is "
        f"below {_WEAK_MIN:.0e}. The Fourier expansion is documented as a weak "
        f"solution; a small pointwise residual here would contradict that."
    )


@pytest.mark.parametrize("cfg_name", ["default", "mahalanobis"])
def test_strong_residual_is_finite_difference_truncation(cfg_name):
    """The Mathieu/Bessel residual is truncation error, not a real defect."""
    solver, _m, _p, _d, _b = build_solver(cfg_name)
    coarse = per_mode_operator_residual(solver, n_modes=4, n_points=16, eps=4e-3)
    fine = per_mode_operator_residual(solver, n_modes=4, n_points=16, eps=1e-3)
    ratio = coarse["median"] / fine["median"]
    assert 10.0 < ratio < 22.0, (
        f"{cfg_name}: refinement ratio {ratio:.2f} is not the ~16 expected for "
        f"a second-order stencil ({coarse['median']:.3e} -> {fine['median']:.3e}). "
        f"A ratio near 1 would mean the residual is structural, i.e. the "
        f"expansion is not a strong solution after all."
    )


@pytest.mark.parametrize("cfg_name", WEAK_CONFIGS)
def test_weak_residual_is_structural_not_truncation(cfg_name):
    """The Fourier residual does not move when the step is refined."""
    solver, _m, _p, _d, _b = build_solver(cfg_name)
    coarse = per_mode_operator_residual(solver, n_modes=4, n_points=16, eps=4e-3)
    fine = per_mode_operator_residual(solver, n_modes=4, n_points=16, eps=1e-3)
    ratio = coarse["median"] / fine["median"]
    assert 0.7 < ratio < 1.4, (
        f"{cfg_name}: refinement ratio {ratio:.2f} is not the ~1 expected of a "
        f"structural residual ({coarse['median']:.3e} -> {fine['median']:.3e})"
    )


@pytest.mark.parametrize("cfg_name", ["spread", "worst_of"])
def test_fourier_is_an_exact_weak_solution(cfg_name):
    """Fourier fails the strong test but passes the weak one exactly."""
    _solver, market, _p, _d, basis = build_solver(cfg_name)
    result = fourier_galerkin_self_consistency(basis, market)
    assert result["max_residual"] < 1e-8, (
        f"{cfg_name}: Galerkin eigen-relation residual "
        f"{result['max_residual']:.3e} over {result['n_checked']} modes"
    )


def test_separation_is_orders_of_magnitude(capsys):
    """The pointwise residual separates the curvilinear bases from the Fourier basis
    by orders of magnitude."""
    strong, weak = {}, {}
    for cfg in STRONG_CONFIGS:
        solver, _m, _p, _d, _b = build_solver(cfg)
        strong[cfg] = per_mode_operator_residual(
            solver, n_modes=6, n_points=20, eps=_EPS)["median"]
    for cfg in WEAK_CONFIGS:
        solver, _m, _p, _d, _b = build_solver(cfg)
        weak[cfg] = per_mode_operator_residual(
            solver, n_modes=6, n_points=20, eps=_EPS)["median"]

    print("\n  per-mode pointwise operator residual (median over modes)")
    for cfg, v in strong.items():
        print(f"    {cfg:<16} strong  {v:.3e}")
    for cfg, v in weak.items():
        print(f"    {cfg:<16} weak    {v:.3e}")
    gap = min(weak.values()) / max(strong.values())
    print(f"    separation: {gap:.0f}x")

    assert max(strong.values()) < min(weak.values()), (
        "strong and weak configurations overlap: "
        f"worst strong {max(strong.values()):.3e} vs "
        f"best weak {min(weak.values()):.3e}"
    )
    assert gap > 100, f"separation only {gap:.0f}x, expected orders of magnitude"
