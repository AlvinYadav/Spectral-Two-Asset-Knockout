"""
Finite differences in whitened coordinates and one-solve grid Greeks.

Fast versions of the solver gates: the closed-form disc (Z1), the M-matrix sign
pattern and spectrum (Z2), the spot node, Lagrange exactness, and the one-solve
x-space path reading the same solution as `fdm_price`, and the regression of the ported
solver against the stored levels of the 28 Oct 2026 gate run. The production-contract
gates (Z3–Z6) and the full Z1/Z2 grids run in `validation/paper1_zfdm.py`.
"""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
from scipy.special import j0, jn_zeros

from ._evidence import require_artefact
from twoasset.cli import assemble
from twoasset.solvers.fdm import fdm_price
from twoasset.solvers.fdm_grid_greeks import fdm_x_onesolve, local_derivatives
from twoasset.solvers.fdm_z import (assemble_operator, build_grid, covariance,
                                    fdm_z_solve, time_step, whitening_matrix)

ROOT = Path(__file__).resolve().parent.parent
J01 = jn_zeros(0, 1)[0]


def _cfg(name):
    return assemble(tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text()))


def _disc_error(m, Nt, b, r, T=1.0):
    """Centre error against V = exp(-b·z - (|b|²/2 + r)τ - j01²τ/2) J0(j01|z|)."""
    g = build_grid(np.eye(2), 1.0, m)
    A = assemble_operator(g, b, r)
    ii, jj = np.nonzero(g.inside)
    z1, z2 = (ii - g.K1) * g.h, (jj - g.K2) * g.h
    bz = b[0] * z1 + b[1] * z2
    rr = np.hypot(z1, z2)
    V, _, _ = time_step(A, np.exp(-bz) * j0(J01 * rr), T, Nt)
    exact = np.exp(-bz - (0.5 * (b[0]**2 + b[1]**2) + r) * T - 0.5 * J01**2 * T) * j0(J01 * rr)
    return abs(V[g.idx[g.K1, g.K2]] - exact[g.idx[g.K1, g.K2]]), np.max(np.abs(V - exact))


@pytest.mark.parametrize("b, r", [((0.0, 0.0), 0.0), ((0.3, -0.2), 0.05)])
def test_closed_form_disc_is_second_order(b, r):
    e = [_disc_error(m, Nt, b, r) for m, Nt in ((20, 25), (40, 50), (80, 100))]
    for k in (0, 1):
        p = np.log2(e[1][k] / e[2][k])
        assert 1.8 <= p <= 2.2, f"order {p:.3f} ({'centre' if k == 0 else 'max'})"


@pytest.mark.parametrize("name", ["default", "tight_barrier", "mahalanobis"])
def test_operator_is_m_matrix_with_stable_spectrum(name):
    market, _payoff, domain, _b, _ = _cfg(name)
    s = fdm_z_solve(market, domain, lambda S1, S2: 0.0 * S1, m=15, N_time=10)
    A = s.A.toarray()
    off = A - np.diag(np.diag(A))
    assert off.min() >= 0.0
    assert np.all(A.sum(axis=1) <= -market.r + 1e-9)
    assert np.max(np.linalg.eigvals(A).real) < 0.0


def test_spot_is_a_node_and_mahalanobis_disc_is_whitening_invariant():
    market, payoff, domain, _b, _ = _cfg("mahalanobis")
    sc = fdm_z_solve(market, domain, payoff, m=30, N_time=40, whitening="cholesky")
    se = fdm_z_solve(market, domain, payoff, m=30, N_time=40, whitening="eigen")
    # zᵀz < R² in both whitenings: the same disc grid, and the spot is its centre node
    assert sc.grid.n_unknowns == se.grid.n_unknowns
    assert sc.grid.idx[sc.grid.K1, sc.grid.K2] >= 0
    assert abs(sc.price() - se.price()) < 0.02 * sc.price()


def test_lagrange_derivatives_exact_for_cubics():
    h1, h2 = 0.1, 0.07
    I, J = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    X, Y = I * h1, J * h2
    U = 1 + 2 * X - Y + X**2 * Y - 0.5 * Y**3 + X**3
    inside = np.ones_like(U, dtype=bool)
    fi, fj = 5.37, 6.81
    x, y = fi * h1, fj * h2
    v, g, H = local_derivatives(U, inside, fi, fj, h1, h2)
    assert np.isclose(v, 1 + 2 * x - y + x**2 * y - 0.5 * y**3 + x**3)
    assert np.allclose(g, [2 + 2 * x * y + 3 * x**2, -1 + x**2 - 1.5 * y**2])
    assert np.allclose(H, [[2 * y + 6 * x, 2 * x], [2 * x, -3 * y]])


@pytest.mark.parametrize("name", ["default", "dnt"])
def test_x_onesolve_reads_the_fdm_price_solution(name):
    market, payoff, domain, _b, _ = _cfg(name)
    one = fdm_x_onesolve(market, domain, payoff, N=40, N_time=50)
    assert one.price() == fdm_price(market, domain, payoff=payoff, N=40, N_time=50)["price"]
    S1 = market.S1_0 * 1.01
    assert one.price(S1=S1) == fdm_price(market, domain, payoff=payoff, N=40, N_time=50,
                                         S1_target=S1)["price"]


def test_cholesky_whitening_solves_the_x_space_equation():
    # Same contract, two whitenings with different grids: prices agree to grid error.
    market, payoff, domain, _b, _ = _cfg("default")
    pc = fdm_z_solve(market, domain, payoff, m=60, N_time=100).price()
    pe = fdm_z_solve(market, domain, payoff, m=60, N_time=100, whitening="eigen").price()
    W = whitening_matrix(covariance(market.sigma1, market.sigma2, market.rho))
    assert np.allclose(W @ covariance(market.sigma1, market.sigma2, market.rho) @ W.T,
                       np.eye(2))
    assert abs(pc - pe) < 1e-3 * pc


@pytest.mark.parametrize("name", ["default", "tight_barrier", "mahalanobis"])
def test_reproduces_the_stored_gate_levels(name, request):
    # The m = 40 and 80 levels must equal those of the stored gate run (bit for bit on
    # the machine that wrote them, with the MMD_AT_PLUS_A ordering).
    path = ROOT / "validation" / "artifacts" / "paper1_zfdm_levels.json"
    require_artefact(path, request=request, cost_note="about 15 minutes",
                     regen_command="PYTHONPATH=. .venv/bin/python3 validation/paper1_zfdm.py levels")
    market, payoff, domain, _b, _ = _cfg(name)
    for stored in json.loads(path.read_text())[f"{name}/cholesky"]["levels"][:2]:
        g = fdm_z_solve(market, domain, payoff, m=stored["m"], N_time=stored["N_time"]).greeks()
        for k in ("price", "delta1", "delta2", "gamma1", "gamma2", "gamma12", "theta",
                  "theta_bdf2"):
            assert abs(g[k] - stored[k]) <= 1e-10 * max(abs(stored[k]), 1e-4), (k, stored["m"])
