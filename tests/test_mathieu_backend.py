"""Checks of the tridiagonal Mathieu functions: differential equations,
orthonormality, signs."""

from __future__ import annotations

import numpy as np
import pytest

from twoasset.runlog import assert_gate

from twoasset.mathieu_backend import (
    MathieuAngularTridiagonal,
    angular_value_deriv2,
    radial_value_deriv2,
    _locate,
    gate_G1_angular_ode_residual,
    gate_G2_angular_orthonormality,
    gate_G3_radial_ode_residual,
    gate_G4_low_q_scipy_crosscheck,
)

Q_GRID = (0.1, 0.5, 1.0, 5.0, 25.0, 50.0, 75.0, 99.5, 150.0, 400.0)


@pytest.mark.parametrize("q", Q_GRID)
def test_G1_angular_ode_residual(q):
    """ce''/se'' + (a - 2q cos2nu)*ce/se = 0 to < 1e-11, every mode."""
    g = gate_G1_angular_ode_residual(q, n_max=8)
    assert g["passed"], (
        f"G1 failed at q={q}: worst={g['worst']}"
    )


@pytest.mark.parametrize("q", Q_GRID)
def test_G2_angular_orthonormality(q):
    """integral ce_m*ce_n dnu = pi*delta_mn to < 1e-12, within class."""
    g = gate_G2_angular_orthonormality(q, n_max=8)
    assert g["passed"], (
        f"G2 failed at q={q}: worst={g['worst']}"
    )


@pytest.mark.parametrize("q", Q_GRID)
def test_G3_radial_ode_residual(q):
    """The radial functions satisfy y'' - (a - 2q cosh 2mu) y = 0 to 1e-10."""
    g = gate_G3_radial_ode_residual(q, n_max=8, mu0=1.2)
    assert g["passed"], (
        f"G3 failed at q={q}: worst={g['worst']}"
    )


def test_G4_low_q_scipy_crosscheck(runlog):
    """At low q, values and first derivatives agree with SciPy to 1e-8."""
    g = gate_G4_low_q_scipy_crosscheck()
    assert_gate(runlog, "G4_low_q_scipy_crosscheck", g["worst_val"], 1e-8,
                cmp="<=", gate_class="S", worst=g["worst"])


def test_no_degradation_with_q():
    """The angular and radial equation residuals stay small over the whole q range."""
    g1_worst = [gate_G1_angular_ode_residual(q, n_max=6)["worst"]["residual_rel"]
               for q in Q_GRID]
    g3_worst = [gate_G3_radial_ode_residual(q, n_max=6, mu0=1.0)["worst"]["gate_val"]
               for q in Q_GRID]
    assert max(g1_worst) < 1e-10, f"G1 residuals: {g1_worst}"
    assert max(g3_worst) < 1e-9, f"G3 residuals: {g3_worst}"


def test_truncation_K_grows_and_converges():
    """The number of Fourier coefficients converges well below its cap."""
    for n_max in (4, 8, 12):
        s = MathieuAngularTridiagonal(q=50.0, n_max=n_max)
        assert s.K >= max(2 * n_max + 16, 32)
        assert s.K <= 4096


def test_sign_convention():
    """Sign convention: ce_n(0) > 0 and se_n'(0) > 0."""
    s = MathieuAngularTridiagonal(q=10.0, n_max=8)
    for n in range(9):
        val, _, _ = angular_value_deriv2(s, n, "even", 0.0)
        assert val > 0, f"ce_{n}(0) = {val} is not > 0"
        if n >= 1:
            _, dval, _ = angular_value_deriv2(s, n, "odd", 0.0)
            assert dval > 0, f"se_{n}'(0) = {dval} is not > 0"
