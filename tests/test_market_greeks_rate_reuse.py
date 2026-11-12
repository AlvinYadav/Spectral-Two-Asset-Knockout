"""Rho on the reused eigenbasis, and the contractual covariance of a Mahalanobis barrier.

The rate leaves Σ, the domain and the eigenpairs unchanged, so `_market_greeks` prices
the two rate bumps on a copy of the fitted basis instead of re-solving. The result must
equal a full re-solve, and the base solver must be left untouched. The finite-difference
solver accepts a Mahalanobis barrier fixed at a contractual covariance
(`Sigma_contract`); without one it must behave exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pytest

from twoasset.core.domain import MahalanobisEllipse
from twoasset.solvers._optimal import build_solver
from twoasset.solvers.fdm import _barrier_inverse_cov, fdm_price
from twoasset.solvers.fdm_greeks import _bumped_market
from twoasset.solvers.spectral import SpectralSolver


def _rho_by_resolve(solver, rate_bump=1e-4):
    """The previous implementation: a fresh basis fitted at each bumped rate."""
    def price_at(r):
        b = solver.basis
        fitted = {"modes", "market", "domain", "c_focal", "mu_0", "A_semi", "B_semi"}
        fresh = type(b)(**{f.name: getattr(b, f.name) for f in fields(b)
                           if not f.name.startswith("_") and f.name not in fitted
                           and isinstance(getattr(b, f.name), (int, float, str, bool))})
        s = SpectralSolver(market=_bumped_market(solver.market, r=r),
                           payoff=solver.payoff, domain=solver.domain, basis=fresh,
                           smooth=solver.smooth, smooth_tau_frac=solver.smooth_tau_frac,
                           smooth_N_grid=solver.smooth_N_grid,
                           smooth_N_steps=solver.smooth_N_steps,
                           projection_method=solver.projection_method)
        s.fit(); s.project()
        return s.price()
    r = solver.market.r
    return (price_at(r + rate_bump) - price_at(r - rate_bump)) / (2 * rate_bump)


@pytest.mark.parametrize("cfg", ["mahalanobis", "dnt"])
def test_rho_on_reused_basis_equals_resolve(cfg):
    solver = build_solver(cfg)[0]
    p0 = solver.price()
    g = solver._market_greeks(solver.market.S1_0, solver.market.S2_0, 0.0)
    assert solver.price() == p0, "rate bumps changed the base solver's coefficients"
    ref = _rho_by_resolve(solver)
    assert abs(g["rho"] - ref) <= 1e-10 * max(abs(ref), 1.0), (g["rho"], ref)


@dataclass
class _ContractMahalanobis(MahalanobisEllipse):
    Sigma_contract: np.ndarray = None


def test_barrier_inverse_cov_default_and_contract():
    solver = build_solver("mahalanobis")[0]
    m, d = solver.market, solver.domain
    assert np.array_equal(_barrier_inverse_cov(m, d), np.linalg.inv(m.Sigma))
    fixed = _ContractMahalanobis(R=d.R, Sigma_contract=m.Sigma.copy())
    bumped = _bumped_market(m, sigma1=m.sigma1 + 0.01)
    assert np.array_equal(_barrier_inverse_cov(bumped, fixed), np.linalg.inv(m.Sigma))
    assert not np.array_equal(_barrier_inverse_cov(bumped, d), np.linalg.inv(m.Sigma))


def test_contract_barrier_at_base_market_is_bit_identical():
    solver = build_solver("mahalanobis")[0]
    m, d, g = solver.market, solver.domain, solver.payoff
    fixed = _ContractMahalanobis(R=d.R, Sigma_contract=m.Sigma.copy())
    assert (fdm_price(m, d, g, N=60, N_time=60)["price"]
            == fdm_price(m, fixed, g, N=60, N_time=60)["price"])
