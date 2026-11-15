"""Cost, residual and mode-convergence evidence (Tables 6.3 to 6.5)."""
from __future__ import annotations

import copy
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import twoasset.bases.mathieu as mathieu_basis
from twoasset.cli import assemble_solver
from twoasset.core.domain import (EuclideanCircle,
                                  MahalanobisEllipse, Rectangle)
from twoasset.runlog import build_provenance
from twoasset.solvers._optimal import (ALL_CONFIGS,
                                       ARTIFACTS_DIR, OPTIMAL)
from twoasset.solvers.fdm import fdm_price
from twoasset.solvers.weak_residual import (
    per_mode_operator_residual)

OUT = Path(ARTIFACTS_DIR) / "paper1_evidence.json"
GOLDEN = json.loads((ROOT / "tests" / "golden" / "level0_prices.json").read_text())

REPEATS_HOT = 200
REPEATS_TIER = 50
N_FRESH = 12


class _NoRootMemo(dict):
    """Stands in for the process-wide Mathieu root memo during the cost section."""

    def get(self, key, default=None):
        return default

    def __setitem__(self, key, value):
        pass


def _pc():
    return time.perf_counter()


def _median_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = _pc()
        fn()
        ts.append(_pc() - t0)
    return float(np.median(ts))


def build(name, *, financial=None, n_modes=None, numerical=None):
    """Assemble a production solver, optionally overriding market inputs, the Fourier
    mode count or the quadrature."""
    cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
    cfg = copy.deepcopy(cfg)
    if financial:
        cfg["financial"].update(financial)
    if numerical:
        cfg.setdefault("numerical", {}).update(numerical)
    opt = OPTIMAL[name]
    N = n_modes if n_modes is not None else opt["N_modes"]
    cfg.setdefault("spectral", {})["N_modes"] = N
    cfg["spectral"]["N_quad"] = max(120, N * 4)
    solver, _kind = assemble_solver(
        cfg,
        smooth=(opt["smooth_tau"] > 0),
        smooth_tau_frac=max(opt["smooth_tau"], 1e-12),
    )
    return cfg, solver


def _ring(solver, radius, n, phase):
    """n spots at log-distance `radius` from the valuation spot, all inside the
    survival region and distinct from every other ring."""
    m = solver.market
    pts = []
    for k in range(n):
        th = phase + 2.0 * np.pi * k / n
        S1 = m.S1_0 * np.exp(radius * np.cos(th))
        S2 = m.S2_0 * np.exp(radius * np.sin(th))
        if solver._inside(S1, S2):
            pts.append((float(S1), float(S2)))
    return pts


def _batch_spots(solver, n_side=50):
    """A uniform n_side x n_side grid over the bounding box of the survival region in
    log-return space, restricted to interior points."""
    m, d = solver.market, solver.domain
    if isinstance(d, Rectangle):
        lo1, hi1, lo2, hi2 = d.x1_lo, d.x1_hi, d.x2_lo, d.x2_hi
    elif isinstance(d, EuclideanCircle):
        lo1 = lo2 = -d.R
        hi1 = hi2 = d.R
    elif isinstance(d, MahalanobisEllipse):
        half = d.R * np.sqrt(np.linalg.eigvalsh(m.Sigma).max())
        lo1 = lo2 = -half
        hi1 = hi2 = half
    else:
        raise TypeError(type(d).__name__)
    x1 = np.linspace(lo1, hi1, n_side + 2)[1:-1]
    x2 = np.linspace(lo2, hi2, n_side + 2)[1:-1]
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    S1 = m.S1_0 * np.exp(X1.ravel())
    S2 = m.S2_0 * np.exp(X2.ravel())
    keep = np.array([solver._inside(a, b) for a, b in zip(S1, S2)])
    return S1[keep], S2[keep]


def _batch_price(solver, S1, S2):
    """Vectorised valuation: V = gauge * (A @ (c * decay)), where A is the basis's
    own design matrix A[i,k] = psi_k(S_i)/||psi_k||."""
    m, basis = solver.market, solver.basis
    tau_eff = m.T - solver.smooth_tau
    w = np.array([md.coeff * md.decay(tau_eff) for md in basis.modes])
    A = basis.design_matrix_at_spot(S1, S2)
    gauge = m.gauge_factor(S1, S2, m.T)
    return gauge * (A @ w)


# ═══════════════════════════════════════════════════════════════════════
#  cost
# ═══════════════════════════════════════════════════════════════════════

def cost_one(name):
    print(f"[cost] {name}", flush=True)
    cfg, solver = build(name)
    m = solver.market
    S1, S2 = m.S1_0, m.S2_0

    t0 = _pc(); solver.fit(); t_fit = _pc() - t0
    t0 = _pc(); solver.project(); t_proj = _pc() - t0

    t0 = _pc(); price = solver.price(); t_price_first = _pc() - t0
    gold = GOLDEN[name]["price"]
    # The golden records the index truncation; under the eigenvalue cutoff the price
    # agrees with it far below the printed digits (tests/test_eigenvalue_cutoff.py).
    tol = 1e-7 if getattr(solver.basis, "truncation", "index") == "eigenvalue" else 1e-10
    if abs(price - gold) > tol * max(abs(gold), 1.0):
        raise SystemExit(f"{name}: price {price!r} does not reproduce golden {gold!r}")

    t0 = _pc(); g_spot = solver.greeks(); t_greeks_first_at_spot = _pc() - t0
    t_price_hot = _median_time(lambda: solver.price(S1, S2), REPEATS_HOT)
    t_greeks_hot = _median_time(lambda: solver.greeks(S1, S2), REPEATS_TIER)

    ring_a = _ring(solver, 0.04, N_FRESH, 0.1)
    ring_b = _ring(solver, 0.07, N_FRESH, 0.37)
    t_price_fresh = []
    for (a, b) in ring_a:
        t0 = _pc(); solver.price(a, b); t_price_fresh.append(_pc() - t0)
    t_greeks_fresh = []
    for (a, b) in ring_b:
        t0 = _pc(); solver.greeks(a, b); t_greeks_fresh.append(_pc() - t0)

    bS1, bS2 = _batch_spots(solver)
    t0 = _pc(); vb = _batch_price(solver, bS1, bS2); t_batch = _pc() - t0
    check_idx = np.linspace(0, len(bS1) - 1, 5).astype(int)
    batch_check = max(abs(vb[i] - solver.price(bS1[i], bS2[i]))
                      / max(abs(vb[i]), 1e-12) for i in check_idx)

    t0 = _pc(); g_full = solver.greeks(S1, S2, full=True); t_greeks_full = _pc() - t0

    f = cfg.get("fdm", {})
    t0 = _pc()
    fres = fdm_price(m, solver.domain, payoff=solver.payoff, N=320, N_time=500)
    t_fdm = _pc() - t0

    row = dict(
        config=name,
        basis=type(solver.basis).__name__,
        n_modes=int(solver.n_modes()),
        a12=float(m.a12),
        price=float(price),
        cold_fit_s=t_fit,
        cold_project_s=t_proj,
        cold_total_s=t_fit + t_proj,
        price_first_at_spot_s=t_price_first,
        greeks_first_at_spot_s=t_greeks_first_at_spot,
        price_hot_s=t_price_hot,
        greeks_hot_s=t_greeks_hot,
        price_fresh_median_s=float(np.median(t_price_fresh)),
        price_fresh_max_s=float(np.max(t_price_fresh)),
        n_price_fresh=len(t_price_fresh),
        greeks_fresh_median_s=float(np.median(t_greeks_fresh)),
        greeks_fresh_max_s=float(np.max(t_greeks_fresh)),
        n_greeks_fresh=len(t_greeks_fresh),
        batch_points=int(len(bS1)),
        batch_total_s=t_batch,
        batch_per_point_s=t_batch / max(len(bS1), 1),
        batch_vs_scalar_max_rel=float(batch_check),
        greeks_full_s=t_greeks_full,
        greeks=({k: float(v) for k, v in g_spot.items()}),
        greeks_full={k: float(v) for k, v in g_full.items()},
        fdm_N=320, fdm_N_time=500,
        fdm_price=float(fres["price"]),
        fdm_wall_s=t_fdm,
        fdm_config_grid=f,
    )
    print(f"   cold {row['cold_total_s']:.3f}s  price@fresh {1e3*row['price_fresh_median_s']:.2f}ms  "
          f"greeks@fresh {1e3*row['greeks_fresh_median_s']:.2f}ms  hot {1e3*t_price_hot:.3f}ms/"
          f"{1e3*t_greeks_hot:.3f}ms  batch {1e6*row['batch_per_point_s']:.1f}us/pt  "
          f"full {t_greeks_full:.2f}s  fdm {t_fdm:.2f}s", flush=True)
    return row, solver


# ═══════════════════════════════════════════════════════════════════════
#  strong / weak
# ═══════════════════════════════════════════════════════════════════════

def strong_one(name, solver=None):
    if solver is None:
        _cfg, solver = build(name)
        solver.fit(); solver.project()
    base = per_mode_operator_residual(solver, n_modes=6, n_points=20, eps=2e-3)
    coarse = per_mode_operator_residual(solver, n_modes=4, n_points=16, eps=4e-3)
    fine = per_mode_operator_residual(solver, n_modes=4, n_points=16, eps=1e-3)
    out = dict(
        config=name, operator=base["operator"], a12=float(solver.market.a12),
        median=base["median"], worst=base["worst"], n_modes=base["n_modes"],
        coarse_median=coarse["median"], fine_median=fine["median"],
        refinement_ratio=coarse["median"] / fine["median"],
    )
    print(f"[strong] {name:14s} median {out['median']:.3e}  ratio {out['refinement_ratio']:.2f}",
          flush=True)
    return out


def rho_sweep_modes(rhos=(0.5, 0.25), Ns=(10, 15, 20, 25, 30)):
    """The Fourier per-mode residual as the mode count grows: it decreases, so it is
    NOT independent of N, but it stays structural (refinement ratio ~1) at every
    N."""
    rows = []
    for rho in rhos:
        for N in Ns:
            _cfg, solver = build("dnt", financial={"rho": rho}, n_modes=N)
            solver.fit(); solver.project()
            r = strong_one("dnt", solver)
            r.update(rho=rho, N_modes=N, n_active=int(solver.n_modes()))
            rows.append(r)
    return rows


def rho_sweep(rhos=(0.5, 0.25, 0.0)):
    rows = []
    for rho in rhos:
        _cfg, solver = build("dnt", financial={"rho": rho})
        solver.fit(); solver.project()
        r = strong_one("dnt", solver)
        r["rho"] = rho
        r["price"] = float(solver.price())
        rows.append(r)
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  mode-count convergence
# ═══════════════════════════════════════════════════════════════════════

FOURIER_SWEEP = {
    "dnt": (6, 8, 10, 12, 16, 20, 28, 40),
    "worst_of": (10, 14, 20, 28, 40),
    "spread": (20, 30, 40, 50, 60),
}

BESSEL_SWEEP = ((6, 6), (8, 6), (10, 6), (8, 8))


def converge():
    out = dict(fourier={}, bessel=[])
    for name, Ns in FOURIER_SWEEP.items():
        rows = []
        for N in Ns:
            _cfg, solver = build(name, n_modes=N)
            t0 = _pc(); solver.fit(); solver.project(); t = _pc() - t0
            p = float(solver.price())
            rows.append(dict(N_modes=N, n_active=int(solver.n_modes()), price=p, cold_s=t))
            print(f"[converge] {name:9s} N={N:3d} modes={solver.n_modes():5d} "
                  f"price={p:.8f} cold={t:.2f}s", flush=True)
        out["fourier"][name] = rows
    for (na, mr) in BESSEL_SWEEP:
        _cfg, solver = build("mahalanobis",
                             numerical={"N_angular_max": na, "M_radial_max": mr})
        t0 = _pc(); solver.fit(); solver.project(); t = _pc() - t0
        p = float(solver.price())
        out["bessel"].append(dict(N_angular_max=na, M_radial_max=mr,
                                  n_modes=int(solver.n_modes()), price=p, cold_s=t))
        print(f"[converge] mahalanobis {na}x{mr} modes={solver.n_modes()} price={p:.14f}",
              flush=True)
    return out


def _merge(old, new):
    """Replace the rows of `old` whose config is in `new`, keeping the order."""
    by = {r["config"]: r for r in new}
    out = [by.pop(r["config"], r) for r in old]
    return out + list(by.values())


def main(argv):
    names = [a for a in argv if a in ALL_CONFIGS] or list(ALL_CONFIGS)
    parts = {a for a in argv if a not in ALL_CONFIGS} or {"cost", "strong", "converge"}
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    solvers = {}
    if "cost" in parts:
        mathieu_basis._ROOT_CACHE = _NoRootMemo()
        rows = []
        for name in names:
            row, solver = cost_one(name)
            rows.append(row)
            solvers[name] = solver
        data["cost"] = _merge(data.get("cost", []), rows)
    if "strong" in parts:
        data["strong"] = _merge(data.get("strong", []),
                                [strong_one(n, solvers.get(n)) for n in names])
        if names == list(ALL_CONFIGS):
            data["rho_sweep"] = rho_sweep()
    if "converge" in parts:
        data["converge"] = converge()
    if "modes" in parts:
        data["rho_sweep_modes"] = rho_sweep_modes()
    data["provenance"] = build_provenance()
    data["notes"] = [
        "fresh = first valuation at a spot never evaluated before on this "
        "solved basis; hot = repeated valuation at a visited spot (for the "
        "Mathieu basis this hits a per-(mu, nu) memo of the special-function "
        "values; Bessel and Fourier have no such memo).",
        "greeks = price, delta1, delta2, theta, gamma1, gamma2, gamma12. "
        "Deltas and theta are analytic; gammas central-difference the "
        "analytic gradient with a log-price step of 1e-4, which evaluates "
        "the gradient at four further spots.",
        "greeks_full adds vega1, vega2 and rho by central differences over "
        "six re-solved markets (vol bump 0.01, rate bump 1e-4).",
        "batch = vectorised valuation over the interior points of a 50 x 50 "
        "grid, through the basis's design matrix; checked against the "
        "scalar path at five points.",
        "fdm = one monolithic Crank-Nicolson solve, 4 Rannacher steps, "
        "Shortley-Weller boundary, N = 320, 500 time steps.",
    ]
    OUT.write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
