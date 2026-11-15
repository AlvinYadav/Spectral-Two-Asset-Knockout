"""
Eigenvalue-cutoff truncation of the Mathieu and Bessel bases.

The production curved-barrier bases keep every mode with lambda^2 < Lambda,
Lambda = lambda_1^2 + (2 / tau_min) ln(1 / eps), tau_min = T/4, eps = 1e-6,
instead of an index box (n <= 8, five or six radial roots). Gates, with the
labels I (invariant), R (regression), D (diagnostic):

  I  every retained Mathieu mode satisfies the turning-point condition
     a_n(q) < 2 q cosh 2 mu_0, and the order above the largest scanned one has
     no root below the scan limit (stopping rule);
  I  zero count: the m-th root of a family has m - 1 interior zeros of its
     radial function on (0, mu_0), counted at 2,000 points;
  D  Weyl count: modes found against area Lambda / (4 pi) - perimeter
     sqrt(Lambda) / (4 pi), flagged beyond max(3, 3%);
  I  weighted Gram matrix equal to the identity to 1e-10 on the production
     quadrature grid;
  I  the special-function checks of Table A.2 extended to every retained order:
     angular equation residual below 1e-10 relative, radial equation residual
     below 1e-10 of max over mu of |M|, boundary value below 1e-10 of the same,
     angular orthonormality to 1e-12;
  R  spectrum of the Euclidean basket against an independent implementation
     (lambda_1^2 and the mode counts below six cutoffs);
  R  prices and spot Greeks at tau = T unchanged from the index rule to well
     below the printed digits;
  I  the truncation bound holds against a basis cut at twice the cutoff.

The short-maturity gate (tau = T/4 against finite differences at the thirteen
spots of Section 6.1) reads `validation/artifacts/paper1_zfdm_short.json`.
"""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from tests._evidence import require_artefact
from twoasset.cli import assemble, assemble_solver
from twoasset.mathieu_backend import (
    MathieuAngularTridiagonal, _locate, angular_value_deriv2, radial_value_deriv2,
)

ROOT = Path(__file__).parent.parent
CURVED = ("default", "tight_barrier", "mahalanobis")
MATHIEU = ("default", "tight_barrier")


def _cfg(name):
    with open(ROOT / "configs" / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def fitted():
    out = {}
    for name in CURVED:
        market, payoff, domain, basis, kind = assemble(_cfg(name))
        assert basis.truncation == "eigenvalue", name
        basis.fit(market, domain)
        out[name] = basis
    return out


def _weyl(basis, lam):
    if hasattr(basis, "A_semi"):
        a, b = max(basis.A_semi, basis.B_semi), min(basis.A_semi, basis.B_semi)
        area = np.pi * a * b
        h = ((a - b) / (a + b)) ** 2
        perimeter = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    else:
        area, perimeter = np.pi * basis.R ** 2, 2 * np.pi * basis.R
    return area * lam / (4 * np.pi) - perimeter * np.sqrt(lam) / (4 * np.pi)


def test_cutoff_definition(fitted):
    """Lambda = lambda_1^2 + (2 / tau_min) ln(1 / eps) with tau_min = T/4, and
    every retained mode lies below it."""
    for name, b in fitted.items():
        info = b.truncation_info
        lam1 = min(md.k_sq for md in b.modes)
        assert info["lambda2_first"] == pytest.approx(lam1, rel=1e-9), name
        T = b.market.T
        expect = lam1 + (2 / (0.25 * T)) * np.log(1e6)
        assert info["lambda2_cut"] == pytest.approx(expect, rel=1e-9), name
        assert max(md.k_sq for md in b.modes) < info["lambda2_cut"], name


def test_euclidean_spectrum_matches_independent_implementation(fitted):
    """R: lambda_1^2 and the counts below six cutoffs (independent implementation)."""
    b = fitted["default"]
    k = np.array(sorted(md.k_sq for md in b.modes))
    assert k[0] == pytest.approx(1.4836873439, abs=1e-9)
    for cut, count in ((28.1, 28), (45, 46), (60, 65), (80, 84), (100, 109), (110, 121)):
        assert int((k < cut).sum()) == count, cut


def test_turning_point_and_stopping_rule(fitted):
    """I: a_n(q) < 2 q cosh 2 mu_0 at every retained root; at the scan limit
    both families of order n_max + 1 lie above it."""
    for name in MATHIEU:
        b = fitted[name]
        two_cosh = 2 * np.cosh(2 * b.mu_0)
        for md in b.modes:
            n, q, parity = md.meta["n"], md.meta["q"], md.meta["parity"]
            cname, j = _locate(n, parity)
            a = MathieuAngularTridiagonal(q=q, n_max=n).a_value(cname, j)
            assert a < two_cosh * q, (name, n, parity, q)
        info = b.truncation_info
        q_hi, n_stop = info["q_scan_max"], info["n_max"] + 1
        s = MathieuAngularTridiagonal(q=q_hi, n_max=n_stop)
        for parity in ("even", "odd"):
            cname, j = _locate(n_stop, parity)
            assert s.a_value(cname, j) >= two_cosh * q_hi, (name, parity)
        assert info["rejected_turning_point"] == 0, name
        assert info["brent_failures"] == 0, name


def test_zero_count(fitted):
    """I: the m-th root of each family has m - 1 interior radial zeros."""
    for name in MATHIEU:
        rows = fitted[name].radial_zero_counts(2000)
        bad = [r for r in rows if not r["ok"]]
        assert not bad, (name, bad[:5])


def test_weyl_count(fitted):
    """D: number of modes within max(3, 3%) of the two-term Weyl law."""
    for name, b in fitted.items():
        lam = b.truncation_info["lambda2_cut"]
        w = _weyl(b, lam)
        assert abs(len(b.modes) - w) <= max(3.0, 0.03 * w), (name, len(b.modes), w)


def test_gram_identity(fitted):
    """I: weighted Gram matrix on the production grid is the identity."""
    for name, b in fitted.items():
        G = b.gram_matrix
        assert np.abs(G - np.eye(len(G))).max() < 1e-10, name
        assert b.mode_ledger["n_dedup_dropped"] == 0, name
        assert b.mode_ledger["n_downstream_low_norm"] == 0, name


def test_function_checks_every_retained_order(fitted):
    """I: Table A.2 checks at every retained root and order."""
    nu = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    nu_fine = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    dnu = 2 * np.pi / len(nu_fine)
    for name in MATHIEU:
        b = fitted[name]
        mu = np.linspace(1e-3, b.mu_0, 60)
        for md in b.modes:
            n, q, parity = md.meta["n"], md.meta["q"], md.meta["parity"]
            s = MathieuAngularTridiagonal(q=q, n_max=n)
            cname, j = _locate(n, parity)
            a = s.a_value(cname, j)
            v, _, d2 = angular_value_deriv2(s, n, parity, nu)
            ang = np.max(np.abs(d2 + (a - 2 * q * np.cos(2 * nu)) * v)) / np.max(np.abs(v))
            assert ang < 1e-10, (name, n, parity, q, ang)
            R, _, Rdd = radial_value_deriv2(s, n, parity, mu, q)
            scale = np.max(np.abs(R))
            rad = np.max(np.abs(Rdd - (a - 2 * q * np.cosh(2 * mu)) * R)) / scale
            assert rad < 1e-10, (name, n, parity, q, rad)
            assert abs(R[-1]) / scale < 1e-10, (name, n, parity, q, R[-1] / scale)
            # orthonormality of the angular functions of this parity at this q
            orders = range(0 if parity == "even" else 1, n + 1)
            vals = np.array([angular_value_deriv2(s, k, parity, nu_fine)[0] for k in orders])
            G = vals @ vals.T * dnu / np.pi
            assert np.abs(G - np.eye(len(G))).max() < 1e-12, (name, n, parity, q)


def test_modes_contributing_at_centre(fitted):
    """R: modes nonzero at the valuation spot (Section 6.1): only the cosine
    order-0 Bessel modes, and on the ellipse the even-order ce modes."""
    expect = {"default": 34, "tight_barrier": 13, "mahalanobis": 5}
    for name, b in fitted.items():
        m = b.market
        v = np.array([abs(b.psi_at_spot(md, m.S1_0, m.S2_0)) / md.norm for md in b.modes])
        assert int((v > 1e-10 * v.max()).sum()) == expect[name], name


def _roots(basis):
    return tuple((md.meta["n"], md.meta["m"], md.meta["parity"], md.meta["q"])
                 for md in basis.modes)


def _fit_cfg(name, *, T=None, cache=True):
    cfg = copy.deepcopy(_cfg(name))
    if T is not None:
        cfg["financial"]["T"] = T
    market, payoff, domain, basis, kind = assemble(cfg)
    basis.root_cache = cache
    basis.fit(market, domain)
    return basis


def test_root_memo_serves_and_extends_bit_identically():
    """I: the shape-keyed memo returns exactly the roots of a fresh scan when it
    filters (a smaller radius or a longer maturity) and when it extends a stored
    scan (a larger radius or a shorter maturity)."""
    from twoasset.bases import mathieu as M
    fresh = {(n, T): _roots(_fit_cfg(n, T=T, cache=False))
             for n, T in (("default", None), ("tight_barrier", None), ("default", 0.5))}
    M.clear_root_cache()
    try:
        b = _fit_cfg("tight_barrier")
        assert b.truncation_info["memo"] == "scanned"
        b = _fit_cfg("default")                          # larger q_max: extends
        assert b.truncation_info["memo"] == "extended"
        assert _roots(b) == fresh[("default", None)]
        b = _fit_cfg("tight_barrier")                    # smaller q_max: filters
        assert b.truncation_info["memo"] == "hit" and b._root_cache_hit
        assert _roots(b) == fresh[("tight_barrier", None)]
        b = _fit_cfg("default", T=0.5)                   # shorter maturity: extends
        assert b.truncation_info["memo"] == "extended"
        assert _roots(b) == fresh[("default", 0.5)]
        b = _fit_cfg("default")                          # served from the larger scan
        assert b.truncation_info["memo"] == "hit"
        assert _roots(b) == fresh[("default", None)]
    finally:
        M.clear_root_cache()


def test_vectorised_radial_evaluator_matches_loop(fitted):
    """I: the eigenvalue path's NumPy radial series equals the term-by-term
    series to 1e-13 at every production mode (value and two derivatives)."""
    from twoasset.mathieu_backend import radial_value_deriv2, radial_value_deriv2_vec
    for name in MATHIEU:
        b = fitted[name]
        mu = np.linspace(1e-3, b.mu_0, 50)
        for md in b.modes:
            n, q, parity = md.meta["n"], md.meta["q"], md.meta["parity"]
            s = b._angular_solver(q, n)
            a = radial_value_deriv2(s, n, parity, mu, q)
            v = radial_value_deriv2_vec(s, n, parity, mu, q)
            for x, y in zip(a, v):
                assert np.max(np.abs(x - y)) <= 1e-13 * np.max(np.abs(x)), (name, n, parity, q)


def _solver(name, truncation):
    cfg = copy.deepcopy(_cfg(name))
    cfg["numerical"]["truncation"] = truncation
    s, _ = assemble_solver(cfg)
    s.project()
    return s


@pytest.mark.parametrize("name", CURVED)
def test_regression_at_maturity(name):
    """R: price and spot Greeks at tau = T agree with the index rule far below
    the printed digits (prices to 1e-7 relative, Greeks to 1e-5)."""
    new, old = _solver(name, "eigenvalue"), _solver(name, "index")
    g_new, g_old = new.greeks(), old.greeks()
    assert abs(g_new["price"] / g_old["price"] - 1) < 1e-7
    for k in ("delta1", "delta2", "theta", "gamma1", "gamma2", "gamma12"):
        assert abs(g_new[k] - g_old[k]) <= 1e-5 * abs(g_old[k]), (k, g_new[k], g_old[k])


def test_truncation_bound_holds():
    """I: |V_N - V_ref| <= bound for the Euclidean basket at three spots and
    tau = T/4, T/2, where V_ref keeps every mode below 2 Lambda on the same
    quadrature grid (the extra modes are the truncation error up to a tail
    far below the bound)."""
    cfg = copy.deepcopy(_cfg("default"))
    cfg["numerical"].update(N_mu_quad=100, N_nu_quad=192)
    s, _ = assemble_solver(cfg)
    s.project()
    lam = s.basis.truncation_info["lambda2_cut"]
    cfg["numerical"]["lambda2_max"] = 2 * lam
    ref, _ = assemble_solver(cfg)
    ref.project()
    m = s.market
    R = 0.5
    spots = [(m.S1_0, m.S2_0)]
    for frac, deg in ((0.6, 50.0), (0.9, 80.0)):
        x = frac * R * np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        spots.append((m.S1_0 * np.exp(x[0]), m.S2_0 * np.exp(x[1])))
    for tau in (0.25 * m.T, 0.5 * m.T):
        for S1, S2 in spots:
            err = abs(s.price(S1, S2, m.T - tau) - ref.price(S1, S2, m.T - tau))
            bound = s.truncation_bound(S1, S2, tau)
            assert err <= bound, (tau, S1, S2, err, bound)


def test_short_maturity_against_finite_differences(request):
    """R: at tau = T/4, at the centre and the twelve off-centre spots, the
    eigenvalue-cutoff prices with the quadrature doubled are within 1e-4 of the
    z-space finite-difference reference on all three curved contracts, and so are
    the production prices of the tight and Mahalanobis baskets. The production
    Euclidean basket (50 x 96 nodes) is up to 4e-4 off near the barrier: that is
    its quadrature, not the cutoff (production keeps 50 x 96), so it is recorded
    but not gated here."""
    path = ROOT / "validation" / "artifacts" / "paper1_zfdm_short.json"
    require_artefact(path, regen_command="python validation/paper1_zfdm.py short",
                     request=request)
    data = json.loads(path.read_text())
    for name in CURVED:
        labels = ["production, quadrature x2"]
        if name != "default":
            labels.append("production")
        for label in labels:
            rows = [r for r in data[name]["spectral"] if r["label"] == label]
            assert len(rows) == 1, (name, label)
            assert rows[0].get("truncation") == "eigenvalue", (name, label)
            for sp in rows[0]["spots"]:
                assert abs(sp["spectral"] - sp["V_R"]) <= 1e-4 * abs(sp["V_R"]), (
                    name, label, sp)
