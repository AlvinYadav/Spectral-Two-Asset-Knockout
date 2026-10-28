"""Regression test: the six production prices against stored reference values."""

from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import tomllib

ROOT = Path(__file__).parent.parent
GOLDEN_PATH = ROOT / "tests" / "golden" / "level0_prices.json"

OPTIMAL = {
    "default":       {"smooth_tau": 0.000, "N_modes": 25},
    "tight_barrier": {"smooth_tau": 0.000, "N_modes": 25},
    "mahalanobis":   {"smooth_tau": 0.000, "N_modes": 25},
    "dnt":           {"smooth_tau": 0.000, "N_modes": 20},
    "spread":        {"smooth_tau": 0.000, "N_modes": 50},
    "worst_of":      {"smooth_tau": 0.000, "N_modes": 32},
}


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _run_config(cfg_name: str, optimal: dict):
    from twoasset.cli import assemble
    from twoasset.solvers.spectral import SpectralSolver
    from twoasset.solvers.validate import pde_residual

    cfg_path = ROOT / "configs" / f"{cfg_name}.toml"
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    N_modes = optimal["N_modes"]
    N_quad = max(120, N_modes * 4)
    cfg2 = copy.deepcopy(cfg)
    cfg2.setdefault("spectral", {})["N_modes"] = N_modes
    cfg2["spectral"]["N_quad"] = N_quad

    smooth_tau = optimal["smooth_tau"]
    from twoasset.cli import assemble_solver
    solver, _ = assemble_solver(
        cfg2,
        smooth=(smooth_tau > 0),
        smooth_tau_frac=max(smooth_tau, 1e-12),
    )
    m = solver.market
    solver.fit()
    solver.project()
    price = solver.price(m.S1_0, m.S2_0, 0.0)
    g = solver.greeks()
    res, scale = pde_residual(
        m, m.S1_0, m.S2_0,
        V=g["price"], delta1=g["delta1"], delta2=g["delta2"],
        gamma11=g["gamma1"], gamma22=g["gamma2"],
        gamma12=g["gamma12"], theta=g["theta"],
    )
    rel = abs(res) / scale if scale > 0 else 0.0
    extras = dict(greeks=g, pde_residual=res, pde_scale=scale,
                  solver=solver)
    return price, rel, m, extras


# ── isinstance check ──────────────────────────────────────────────────────────

def test_gbmmarket_is_market():
    from twoasset.core.market import GBMMarket
    from twoasset.core.market_base import Market

    m = GBMMarket(S1_0=100, S2_0=100, sigma1=0.2, sigma2=0.3, rho=0.5,
                  r=0.05, T=1.0)
    assert isinstance(m, Market), "GBMMarket must be a subclass of Market"


# ── Coordinate/gauge methods inherited correctly ──────────────────────────────

def test_inherited_coordinate_methods():
    """spot_to_x, x_to_spot, x_to_z, z_to_x, gauge_factor still work."""
    from twoasset.core.market import GBMMarket

    m = GBMMarket(S1_0=100, S2_0=100, sigma1=0.2, sigma2=0.3, rho=0.5,
                  r=0.05, T=1.0)

    # spot_to_x / x_to_spot round-trip
    x1, x2 = m.spot_to_x(110.0, 95.0)
    S1r, S2r = m.x_to_spot(x1, x2)
    assert abs(S1r - 110.0) < 1e-10
    assert abs(S2r - 95.0) < 1e-10

    # x_to_z / z_to_x round-trip
    z1, z2 = m.x_to_z(x1, x2)
    x1r, x2r = m.z_to_x(z1, z2)
    assert abs(x1r - x1) < 1e-12
    assert abs(x2r - x2) < 1e-12

    # gauge_factor > 0
    gf = m.gauge_factor(100.0, 100.0, 1.0)
    assert gf > 0


@pytest.mark.parametrize("cfg_name", list(OPTIMAL.keys()))
def test_price_unchanged(cfg_name, golden):
    price, _, _, _ = _run_config(cfg_name, OPTIMAL[cfg_name])
    golden_price = golden[cfg_name]["price"]
    # Allow floating-point reproducibility tolerance (1 ULP ~ 1e-14 relative)
    assert abs(price - golden_price) < 1e-10 * max(abs(golden_price), 1), (
        f"{cfg_name}: price={price:.10g} golden={golden_price:.10g} "
        f"diff={abs(price - golden_price):.2e}"
    )


@pytest.mark.parametrize("cfg_name", list(OPTIMAL.keys()))
def test_pde_residual_unchanged(cfg_name, golden):
    _, rel, _, _ = _run_config(cfg_name, OPTIMAL[cfg_name])
    golden_rel = golden[cfg_name]["pde_relative"]
    # Residuals can vary slightly due to FD step noise; allow 5% relative change.
    assert abs(rel - golden_rel) < 0.05 * max(golden_rel, 1e-9), (
        f"{cfg_name}: pde_rel={rel:.4e} golden={golden_rel:.4e}"
    )
