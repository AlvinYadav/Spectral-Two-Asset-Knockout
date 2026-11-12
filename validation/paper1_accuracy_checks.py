"""Accuracy checks that separate the sources of error in the spectral prices.

Each section writes its own artefact, so the sections can run in separate
processes:

  quad       the Table 6.6 truncation pair with the radial quadrature held
             fixed (the 102-mode column of projection_comparison used 60
             Gauss-Legendre nodes instead of 50), and every production
             contract with its quadrature grid doubled;
  fquad      the trapezoidal quadrature on the rectangle refined up to
             eight times its production size;
  rho0       the spread call at rho = 0, where the Galerkin matrix is
             diagonal, against finite differences at 400 to 2500 modes;
  margrabe   the spread call with K = 0 on the truncated domain against
             Margrabe's closed form, which measures the domain truncation;
  offcentre  the three curved-barrier contracts at off-centre spots against
             three-grid Richardson finite differences from full-grid solves;
  holdouts   a third finite-difference grid (N = 640) for the sealed
             holdouts B and C, read against their sealed spectral prices.
             The spectral prices are NOT re-evaluated; they are read from
             validation/holdout_evaluation_r4.json;
  timestep   the N = 320 finite-difference solution of the curved-barrier
             contracts with 500, 1000 and 2000 time steps;
  vega_fixed finite-difference vegas of the Mahalanobis basket with the
             barrier moving with the bumped covariance (as in Table 6.2) and
             with the barrier fixed at the base covariance;
  conv       the six contracts with one numerical setting refined at a time
             (on the curved barriers angular order, radial roots, each direction
             of the quadrature and the root scan; on the rectangles the modes
             per side and the quadrature; on all, the gamma step and the bumps
             of the market Greeks), at the centre, 60% and 90% of the way to
             the barrier, and at tau = T and T/4;
  conv_fd    three-grid Richardson finite differences for the spots of
             `offcentre`, the references of `conv`: at tau = T/4 on the curved
             barriers, whose tau = T references come from `offcentre`, and at
             both maturities on the rectangles, with deltas and gammas from a
             cubic spline of each grid;
  conv_angular  the Euclidean basket with angular orders up to 10, 12 and 14
             against the conv_fd and offcentre references (run conv_fd and
             offcentre for `default` first);
  conv_modes the rectangles with growing modes per side against the conv_fd
             references, which show how their gammas converge (run conv_fd first).

The offcentre, holdouts, rho0, margrabe and conv_fd sections solve finite
differences at N = 640, which needs several GB of memory per solve; run them in
sequence rather than in parallel processes.

Run (repo root):
    PYTHONPATH=. python validation/paper1_accuracy_checks.py <section> [...]
Writes:
    validation/artifacts/paper1_accuracy_<section>.json
"""
from __future__ import annotations

import copy
import json
import sys
import time
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twoasset.cli import assemble_solver                      # noqa: E402
from twoasset.core.domain import (EuclideanCircle,            # noqa: E402
                                  MahalanobisEllipse, Rectangle)
from twoasset.projection_comparison import _build_solver      # noqa: E402
from twoasset.runlog import build_provenance                  # noqa: E402
from twoasset.solvers._optimal import ARTIFACTS_DIR, OPTIMAL  # noqa: E402
from twoasset.solvers.fdm import fdm_price                    # noqa: E402
from twoasset.solvers.fdm_greeks import _bumped_market        # noqa: E402
from twoasset.solvers.spectral import SpectralSolver, _bump_market  # noqa: E402

GRIDS = ((160, 250), (320, 500), (640, 1000))


def _write(section, payload):
    payload["provenance"] = build_provenance(section=section)
    out = Path(ARTIFACTS_DIR) / f"paper1_accuracy_{section}.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out}", flush=True)


def build(name, *, overrides=None, n_modes=None, n_quad=None):
    """Assemble a production solver as paper1_evidence.build does, with
    optional per-section overrides of the TOML configuration."""
    cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
    cfg = copy.deepcopy(cfg)
    for section, values in (overrides or {}).items():
        cfg.setdefault(section, {}).update(values)
    opt = OPTIMAL[name]
    N = n_modes if n_modes is not None else opt["N_modes"]
    cfg.setdefault("spectral", {})["N_modes"] = N
    cfg["spectral"]["N_quad"] = max(120, N * 4) if n_quad is None else n_quad
    solver, _ = assemble_solver(
        cfg, smooth=(opt["smooth_tau"] > 0),
        smooth_tau_frac=max(opt["smooth_tau"], 1e-12))
    return solver


def richardson(v160, v320, v640):
    """Observed order and extrapolated value from three grids, as in
    Section 5.1. Non-monotone triples get no order and keep V_640."""
    d1, d2 = v320 - v160, v640 - v320
    if d1 * d2 > 0 and abs(d2) < abs(d1):
        p = float(np.log2(d1 / d2))
        vr = v640 + d2 / (2.0 ** p - 1.0)
    else:
        p, vr = float("nan"), v640
    u = abs(vr - v640) if np.isfinite(p) else abs(d2)
    return dict(p=p, V_R=float(vr), u_FD=float(u))


def _bilinear(U, x1v, x2v, x1, x2):
    """Bilinear interpolation on a full FD grid, zero outside the domain
    (the convention of fdm._interp2d)."""
    G = np.nan_to_num(U, nan=0.0)
    n = len(x1v)
    fi = (x1 - x1v[0]) / (x1v[1] - x1v[0])
    fj = (x2 - x2v[0]) / (x2v[1] - x2v[0])
    i = min(max(int(np.floor(fi)), 0), n - 2)
    j = min(max(int(np.floor(fj)), 0), n - 2)
    a, b = min(max(fi - i, 0.0), 1.0), min(max(fj - j, 0.0), 1.0)
    return float((1 - a) * (1 - b) * G[i, j] + a * (1 - b) * G[i + 1, j]
                 + (1 - a) * b * G[i, j + 1] + a * b * G[i + 1, j + 1])


# ═══════════════════════════════════════════════════════════════════════
#  quad
# ═══════════════════════════════════════════════════════════════════════

def _table66_price(cfg_name, n_ang, m_rad, q_max, n_mu):
    solver = _build_solver(cfg_name, n_ang, m_rad, q_max)
    solver.basis.N_mu_quad = n_mu
    solver.projection_method = "galerkin"
    solver.smooth = False
    solver.smooth_tau_frac = 1e-12
    solver.fit()
    solver.project()
    return dict(config=cfg_name, M_radial_max=m_rad, N_mu_quad=n_mu,
                n_modes=len(solver.basis.modes), price=float(solver.price()))


def section_quad():
    rows = []
    for cfg_name in ("default", "tight_barrier", "holdout_ellipse"):
        for m_rad, q_max, n_mu in ((5, 350.0, 50), (6, 400.0, 60),
                                   (6, 400.0, 50), (5, 350.0, 60)):
            r = _table66_price(cfg_name, 8, m_rad, q_max, n_mu)
            print(r, flush=True)
            rows.append(r)

    doubled = []
    for name in ("default", "tight_barrier", "mahalanobis",
                 "dnt", "spread", "worst_of"):
        base = build(name)
        p0 = float(base.price())
        if name in ("default", "tight_barrier"):
            s = build(name, overrides={"numerical": dict(
                N_mu_quad=100, N_nu_quad=192)})
        elif name == "mahalanobis":
            s = build(name, overrides={"numerical": dict(
                N_r_quad=80, N_theta_quad=192)})
        else:
            s = build(name)
            s.basis.N_quad = 2 * s.basis.N_quad
        p1 = float(s.price())
        doubled.append(dict(config=name, price=p0, price_doubled=p1,
                            rel_change=(p1 - p0) / p0))
        print(doubled[-1], flush=True)
    _write("quad", dict(table66=rows, doubled_quadrature=doubled))


def section_fquad():
    """Trapezoidal quadrature refinement on the rectangle, at the production
    mode counts: N_quad = 1, 2, 4 and 8 times the production value. The
    spread call is covered by the doubling in `quad` and by `conv`."""
    rows = []
    for name in ("dnt", "worst_of"):
        base = build(name).basis.N_quad
        for mult in (1, 2, 4, 8):
            s = build(name)
            s.basis.N_quad = base * mult
            rows.append(dict(config=name, N_quad=base * mult,
                             price=float(s.price())))
            print(rows[-1], flush=True)
    _write("fquad", dict(rows=rows))


# ═══════════════════════════════════════════════════════════════════════
#  rho0 and margrabe
# ═══════════════════════════════════════════════════════════════════════

def _fd_triple(solver):
    vals = [fdm_price(solver.market, solver.domain, solver.payoff,
                      N=N, N_time=Nt)["price"] for N, Nt in GRIDS]
    return dict(V160=vals[0], V320=vals[1], V640=vals[2], **richardson(*vals))


def section_rho0():
    over = {"financial": dict(rho=0.0)}
    fd = _fd_triple(build("spread", overrides=over))
    print("FD", fd, flush=True)
    sweep = []
    for n in (20, 30, 40, 50):
        p = float(build("spread", overrides=over, n_modes=n).price())
        sweep.append(dict(modes=n * n, price=p,
                          rel_err=(p - fd["V_R"]) / fd["V_R"]))
        print(sweep[-1], flush=True)
    _write("rho0", dict(fd=fd, sweep=sweep))


def section_margrabe():
    over = {"spread": dict(K_spread=0.0)}
    s = build("spread", overrides=over)
    m = s.market
    vol = np.sqrt(m.sigma1 ** 2 + m.sigma2 ** 2
                  - 2 * m.rho * m.sigma1 * m.sigma2)
    d1 = (np.log(m.S1_0 / m.S2_0) + 0.5 * vol ** 2 * m.T) / (vol * np.sqrt(m.T))
    exact = float(m.S1_0 * norm.cdf(d1) - m.S2_0 * norm.cdf(d1 - vol * np.sqrt(m.T)))
    spec = float(s.price())
    fd = _fd_triple(s)
    out = dict(margrabe=exact, spectral=spec, fd=fd,
               fd_vs_margrabe=(fd["V_R"] - exact) / exact,
               spectral_vs_margrabe=(spec - exact) / exact,
               spectral_vs_fd=(spec - fd["V_R"]) / fd["V_R"])
    print(out, flush=True)
    _write("margrabe", out)


# ═══════════════════════════════════════════════════════════════════════
#  offcentre
# ═══════════════════════════════════════════════════════════════════════

FRACTIONS = (0.3, 0.6, 0.9)
ANGLES_DEG = (20.0, 110.0, 200.0, 290.0)


def _boundary_radius(solver, u):
    d = solver.domain
    if isinstance(d, EuclideanCircle):
        return d.R
    if isinstance(d, MahalanobisEllipse):
        return d.R / np.sqrt(u @ np.linalg.solve(solver.market.Sigma, u))
    if isinstance(d, Rectangle):
        return min((d.x1_hi if u[0] > 0 else d.x1_lo) / u[0],
                   (d.x2_hi if u[1] > 0 else d.x2_lo) / u[1])
    raise TypeError(type(d).__name__)


def _spots(solver):
    """Valuation spot, then FRACTIONS of the distance to the barrier along
    each direction in ANGLES_DEG (rotated per fraction so no two rings
    share a ray), in log-return coordinates."""
    pts = [(0.0, 0.0, 0.0, 0.0)]
    for k, f in enumerate(FRACTIONS):
        for a in ANGLES_DEG:
            th = np.deg2rad(a + 30.0 * k)
            u = np.array([np.cos(th), np.sin(th)])
            x = f * _boundary_radius(solver, u) * u
            pts.append((float(x[0]), float(x[1]), f, float(np.rad2deg(th))))
    return pts


def section_offcentre(names=("default", "tight_barrier", "mahalanobis")):
    out = {}
    for name in names:
        solver = build(name)
        m = solver.market
        spots = _spots(solver)
        grids = []
        for N, Nt in GRIDS:
            t0 = time.perf_counter()
            r = fdm_price(m, solver.domain, solver.payoff, N=N, N_time=Nt,
                          return_full=True)
            print(f"[{name}] FD N={N} {time.perf_counter() - t0:.1f}s",
                  flush=True)
            vals = [_bilinear(r["u_grid"], r["x1v"], r["x2v"], x1, x2)
                    for x1, x2, _, _ in spots]
            assert abs(vals[0] - r["price"]) < 1e-12, "interpolation mismatch"
            grids.append(vals)
            del r
        rows = []
        for k, (x1, x2, f, th) in enumerate(spots):
            S1, S2 = m.S1_0 * np.exp(x1), m.S2_0 * np.exp(x2)
            spec = float(solver.price(S1, S2))
            ref = richardson(grids[0][k], grids[1][k], grids[2][k])
            rows.append(dict(x1=x1, x2=x2, fraction=f, angle_deg=th,
                             S1=float(S1), S2=float(S2), spectral=spec,
                             V320=grids[1][k], **ref,
                             diff=spec - ref["V_R"],
                             diff_pct=100 * (spec - ref["V_R"]) / ref["V_R"]))
            print(rows[-1], flush=True)
        out[name] = rows
    _write("offcentre_" + "_".join(names) if len(names) == 1 else "offcentre",
           dict(fractions=FRACTIONS, angles_deg=ANGLES_DEG, configs=out))


# ═══════════════════════════════════════════════════════════════════════
#  holdouts
# ═══════════════════════════════════════════════════════════════════════

def section_holdouts():
    sealed = json.loads((ROOT / "validation" /
                         "holdout_evaluation_r4.json").read_text())["configs"]
    out = {}
    for name in ("holdout_ellipse_2026b", "holdout_ellipse_2026c"):
        cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
        solver, _ = assemble_solver(cfg)
        m, d, g = solver.market, solver.domain, solver.payoff
        rec = sealed[name]
        v160, v320 = rec["fdm"]["half"], rec["fdm"]["fine"]
        # Determinism check: the N = 320 solve reproduces the sealed record.
        v320_now = fdm_price(m, d, g, N=320, N_time=500)["price"]
        t0 = time.perf_counter()
        v640 = fdm_price(m, d, g, N=640, N_time=1000)["price"]
        ref = richardson(v160, v320, v640)
        spec = rec["price"]["spectral"]
        out[name] = dict(spectral_sealed=spec, V160=v160, V320=v320,
                         V320_recomputed=v320_now, V640=v640, **ref,
                         diff=spec - ref["V_R"],
                         diff_pct=100 * (spec - ref["V_R"]) / ref["V_R"],
                         fd640_s=time.perf_counter() - t0)
        print(name, out[name], flush=True)
    _write("holdouts", out)


# ═══════════════════════════════════════════════════════════════════════
#  timestep
# ═══════════════════════════════════════════════════════════════════════

def section_timestep(names=("default", "tight_barrier", "mahalanobis")):
    """The N = 320 solution with 500, 1000 and 2000 time steps, which separates
    the time-stepping error from the spatial error of the reference."""
    out = {}
    for name in names:
        solver = build(name)
        m, d, g = solver.market, solver.domain, solver.payoff
        vals = {}
        for Nt in (500, 1000, 2000):
            t0 = time.perf_counter()
            vals[Nt] = fdm_price(m, d, g, N=320, N_time=Nt)["price"]
            print(f"[{name}] FD N=320 steps={Nt} {vals[Nt]:.10f} "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
        d1, d2 = vals[500] - vals[1000], vals[1000] - vals[2000]
        p = float(np.log2(d1 / d2)) if d1 * d2 > 0 else float("nan")
        out[name] = dict(V500=vals[500], V1000=vals[1000], V2000=vals[2000],
                         p_time=p,
                         rel_change_500_to_2000=(vals[2000] - vals[500]) / vals[500])
        print(name, out[name], flush=True)
    _write("timestep", out)


# ═══════════════════════════════════════════════════════════════════════
#  vega_fixed
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _ContractMahalanobis(MahalanobisEllipse):
    """Mahalanobis barrier fixed at a contractual covariance, which a bumped market
    does not move. Finite differences only: after whitening by the bumped
    covariance the barrier is no longer a disc, so the Bessel basis does not apply."""
    Sigma_contract: np.ndarray = None


def section_vega_fixed(name="mahalanobis", vol_bump=0.01):
    """Finite-difference vegas with the barrier moving with the bumped covariance, as
    in Table 6.2, and fixed at the base covariance, on the grids N = 160 and 320."""
    solver = build(name)
    m, g = solver.market, solver.payoff
    moving = solver.domain
    fixed = _ContractMahalanobis(R=moving.R, monitoring_dt=moving.monitoring_dt,
                                 Sigma_contract=m.Sigma.copy())
    out = {}
    for N, Nt in GRIDS[:2]:
        base = {label: fdm_price(m, dom, g, N=N, N_time=Nt)["price"]
                for label, dom in (("moving", moving), ("fixed", fixed))}
        assert base["moving"] == base["fixed"], "barriers differ at the base market"
        row = dict(price=base["moving"])
        for label, dom in (("moving", moving), ("fixed", fixed)):
            for i, key in ((1, "sigma1"), (2, "sigma2")):
                s0 = getattr(m, key)
                up = fdm_price(_bumped_market(m, **{key: s0 + vol_bump}), dom, g,
                               N=N, N_time=Nt)["price"]
                dn = fdm_price(_bumped_market(m, **{key: s0 - vol_bump}), dom, g,
                               N=N, N_time=Nt)["price"]
                row[f"vega{i}_{label}"] = (up - dn) / (2 * vol_bump)
        print(f"[{name}] N={N}", row, flush=True)
        out[f"N{N}"] = row
    _write("vega_fixed", dict(config=name, vol_bump=vol_bump, grids=out))


# ═══════════════════════════════════════════════════════════════════════
#  conv and conv_fd
# ═══════════════════════════════════════════════════════════════════════

CURVED = ("default", "tight_barrier", "mahalanobis")
RECTANGLES = ("dnt", "worst_of", "spread")
CONV_SPOTS = (0, 5, 9)       # indices into _spots: centre, 60% at 50 deg, 90% at 80 deg
CONV_TAUS = (1.0, 0.25)
SPOT_KEYS = ("price", "delta1", "delta2", "gamma1", "gamma2", "gamma12", "theta")
MARKET_KEYS = ("vega1", "vega2", "rho")
RECT_MODES = dict(dnt=(15, 30), worst_of=(24, 48), spread=(40, 60))


def _conv_variants(name):
    """Numerical settings refined or coarsened one at a time from production, as
    keyword arguments of `build`. A sixth Mathieu radial root needs the scan range
    raised from 350 to 400 (at the same 15 points per unit q), or four orders lose
    their sixth root. On a rectangle the quadrature stays at its production size
    when the modes change."""
    if name in RECTANGLES:
        n, q = OPTIMAL[name]["N_modes"], max(120, 4 * OPTIMAL[name]["N_modes"])
        return {**{f"modes {m}": dict(n_modes=m, n_quad=q) for m in RECT_MODES[name]},
                "quadrature x2": dict(n_quad=2 * q),
                "quadrature x4": dict(n_quad=4 * q)}
    if name == "mahalanobis":
        over = {"angular 6": dict(N_angular_max=6),
                "angular 10": dict(N_angular_max=10),
                "radial 5": dict(M_radial_max=5),
                "radial 7": dict(M_radial_max=7),
                "radial quadrature x2": dict(N_r_quad=80),
                "angular quadrature x2": dict(N_theta_quad=192)}
    else:
        over = {"angular 6": dict(N_angular_max=6),
                "angular 10": dict(N_angular_max=10),
                "radial 4": dict(M_radial_max=4),
                "radial 6": dict(M_radial_max=6, q_max=400.0, q_scan_points=6000),
                "radial quadrature x2": dict(N_mu_quad=100),
                "angular quadrature x2": dict(N_nu_quad=192),
                "root scan x2": dict(q_scan_points=10500)}
    return {label: dict(overrides={"numerical": o}) for label, o in over.items()}


def _mode_census(basis):
    """Modes per angular family; a family short of M_radial_max roots means the scan
    range did not hold all requested roots."""
    if not hasattr(basis, "M_radial_max"):
        return dict(n_modes=len(basis.modes), short_families=[])
    count = {}
    for md in basis.modes:
        key = (md.meta["n"], md.meta["parity"])
        count[key] = count.get(key, 0) + 1
    short = sorted(f"{k}" for k, v in count.items() if v < basis.M_radial_max)
    return dict(n_modes=len(basis.modes), short_families=short)


def _gammas_at_step(solver, S1, S2, t, h):
    """The three gammas with step h in the central differences of the analytic
    gradient (production h = 1e-4), through the solver's own assembly."""
    tau = solver.market.T - t
    tau_eff = tau - solver.smooth_tau
    gf = solver.market.gauge_factor(S1, S2, tau)
    u, u1, u2 = solver._u_grad_x(S1, S2, tau_eff)
    hess = solver._u_hessian_x(S1, S2, tau_eff, eps=h)
    _, _, g11, g22, g12 = solver._spot_greeks_from_u(S1, S2, gf, u, u1, u2, *hess)
    return dict(gamma1=float(g11), gamma2=float(g22), gamma12=float(g12))


def _conv_points(solver, pts):
    m = solver.market
    for tau in CONV_TAUS:
        for k in CONV_SPOTS:
            x1, x2 = pts[k][:2]
            yield tau, k, m.S1_0 * np.exp(x1), m.S2_0 * np.exp(x2), m.T - tau


def _solver_at(solver, *, reuse_basis=False, **market_kw):
    """The solver at a bumped market, built as in SpectralSolver._market_greeks, which
    solves it anew for every spot; here it is solved once and priced at all of them."""
    m = _bump_market(solver.market, **market_kw)
    b = solver.basis
    if reuse_basis:
        basis = copy.deepcopy(b)
        basis.market = m
    else:
        fitted = ("modes", "market", "domain", "c_focal", "mu_0", "A_semi", "B_semi")
        basis = type(b)(**{f.name: getattr(b, f.name) for f in fields(b)
                           if f.name not in fitted and not f.name.startswith("_")
                           and isinstance(getattr(b, f.name), (int, float, str, bool))})
    s = SpectralSolver(market=m, payoff=solver.payoff, domain=solver.domain, basis=basis,
                       smooth=solver.smooth, smooth_tau_frac=solver.smooth_tau_frac,
                       projection_method=solver.projection_method)
    s._fit_done = reuse_basis
    s.project()
    return s


def _vega_rows(solver, pts, bump):
    m = solver.market
    solvers = [_solver_at(solver, **{key: getattr(m, key) + d})
               for key in ("sigma1", "sigma2") for d in (bump, -bump)]
    return [dict(tau=tau, spot=k,
                 vega1=(solvers[0].price(S1, S2, t) - solvers[1].price(S1, S2, t)) / (2 * bump),
                 vega2=(solvers[2].price(S1, S2, t) - solvers[3].price(S1, S2, t)) / (2 * bump))
            for tau, k, S1, S2, t in _conv_points(solver, pts)]


def _rho_rows(solver, pts, bump):
    m = solver.market
    up, dn = (_solver_at(solver, reuse_basis=True, r=m.r + d) for d in (bump, -bump))
    return [dict(tau=tau, spot=k,
                 rho=(up.price(S1, S2, t) - dn.price(S1, S2, t)) / (2 * bump))
            for tau, k, S1, S2, t in _conv_points(solver, pts)]


def _conv_rows(solver, pts, *, market=True):
    rows = []
    for tau, k, S1, S2, t in _conv_points(solver, pts):
        g = solver.greeks(S1, S2, t)
        rows.append(dict(tau=tau, spot=k, **{q: float(g[q]) for q in SPOT_KEYS}))
    if market:
        for r, v, p in zip(rows, _vega_rows(solver, pts, 0.01), _rho_rows(solver, pts, 1e-4)):
            r.update(vega1=float(v["vega1"]), vega2=float(v["vega2"]), rho=float(p["rho"]))
    return rows


def section_conv(names=CURVED + RECTANGLES):
    """One-at-a-time refinement of the spectral prices and Greeks; one artefact per
    contract. The eigenbasis and the projection do not depend on T, so both
    maturities come from one solve, priced at t = T - tau."""
    for name in names:
        t_start = time.perf_counter()
        base = build(name)
        pts = _spots(base)
        spots = [dict(index=k, x1=pts[k][0], x2=pts[k][1], fraction=pts[k][2],
                      angle_deg=pts[k][3]) for k in CONV_SPOTS]

        variants = {}
        for label, kw in {"production": {}, **_conv_variants(name)}.items():
            t0 = time.perf_counter()
            s = base if not kw else build(name, **kw)
            s.fit()
            census = _mode_census(s.basis)
            variants[label] = dict(settings=kw, **census, rows=_conv_rows(s, pts),
                                   wall_s=time.perf_counter() - t0)
            print(f"[{name}] {label}: {census} "
                  f"{variants[label]['wall_s']:.1f}s", flush=True)

        steps = {}
        for h in (1e-3, 1e-5, 1e-6):
            steps[f"{h:g}"] = [dict(tau=tau, spot=k, **_gammas_at_step(base, S1, S2, t, h))
                               for tau, k, S1, S2, t in _conv_points(base, pts)]
        # The step of production, through the same route, against greeks().
        check = [_gammas_at_step(base, S1, S2, t, 1e-4)
                 for _, _, S1, S2, t in _conv_points(base, pts)]
        prod = variants["production"]["rows"]
        step_check = max(abs(c[q] - r[q]) / abs(r[q]) for c, r in zip(check, prod)
                         for q in ("gamma1", "gamma2", "gamma12"))
        assert step_check < 1e-12, f"gamma route differs from greeks(): {step_check}"
        # The market Greeks of this route against the solver's own, at the centre.
        own = base._market_greeks(base.market.S1_0, base.market.S2_0, 0.0)
        route_check = max(abs(prod[0][q] - own[q]) / abs(own[q]) for q in MARKET_KEYS)
        assert route_check < 1e-10, f"market route differs: {route_check}"

        bumps = {}
        for label, rows in (("vol 0.005", lambda: _vega_rows(base, pts, 0.005)),
                            ("vol 0.02", lambda: _vega_rows(base, pts, 0.02)),
                            ("rate 1e-3", lambda: _rho_rows(base, pts, 1e-3)),
                            ("rate 1e-5", lambda: _rho_rows(base, pts, 1e-5))):
            bumps[label] = rows()
            print(f"[{name}] bumps {label}", flush=True)

        _write(f"conv_{name}", dict(
            config=name, taus=CONV_TAUS, spots=spots,
            production=dict(OPTIMAL[name], gamma_step=1e-4, vol_bump=0.01,
                            rate_bump=1e-4),
            variants=variants, gamma_steps=steps, bumps=bumps,
            wall_s=time.perf_counter() - t_start))


def _rect_spline(r, domain):
    """Cubic spline of a rectangle solution, with its zero boundary values. Bilinear
    interpolation has an error of the order of the second-order scheme, which changes
    with the place of the spot in its cell and would spoil the extrapolation."""
    return RectBivariateSpline(np.r_[domain.x1_lo, r["x1v"], domain.x1_hi],
                               np.r_[domain.x2_lo, r["x2v"], domain.x2_hi],
                               np.pad(r["u_grid"], 1), kx=3, ky=3, s=0)


def _rect_fields(sp, S1_0, S2_0, x1, x2):
    """Price, deltas and gammas of a rectangle solution at one spot, from the
    derivatives of its spline in log-price."""
    S1, S2 = S1_0 * np.exp(x1), S2_0 * np.exp(x2)
    v = lambda dx, dy: float(sp.ev(x1, x2, dx=dx, dy=dy))
    return dict(price=v(0, 0), delta1=v(1, 0) / S1, delta2=v(0, 1) / S2,
                gamma1=(v(2, 0) - v(1, 0)) / S1 ** 2,
                gamma2=(v(0, 2) - v(0, 1)) / S2 ** 2,
                gamma12=v(1, 1) / (S1 * S2))


def section_conv_fd(names=CURVED, tau=0.25):
    """Finite-difference references for all spots of `offcentre`, from full-grid solves
    of the contract with T = tau; one artefact per contract and maturity. The curved
    barriers use the interpolation of `offcentre`. The rectangles need the references
    at both maturities, the second as conv_fd_<contract>_T, and also give deltas and
    gammas. Also records that the spectral price of that contract equals the T = 1
    solution priced at t = 1 - tau."""
    for name in names:
        for tau_now in ((1.0, tau) if name in RECTANGLES else (tau,)):
            short = build(name, overrides={"financial": dict(T=tau_now)})
            long_ = build(name)
            m = short.market
            pts = _spots(short)
            assert pts == _spots(long_), "spots depend on T"
            grids = []
            for N, Nt in GRIDS:
                t0 = time.perf_counter()
                r = fdm_price(m, short.domain, short.payoff, N=N, N_time=Nt,
                              return_full=True)
                print(f"[{name}] FD tau={tau_now} N={N} "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)
                if name in RECTANGLES:
                    sp = _rect_spline(r, short.domain)
                    grids.append([_rect_fields(sp, m.S1_0, m.S2_0, x1, x2)
                                  for x1, x2, _, _ in pts])
                else:
                    vals = [_bilinear(r["u_grid"], r["x1v"], r["x2v"], x1, x2)
                            for x1, x2, _, _ in pts]
                    assert abs(vals[0] - r["price"]) < 1e-12, "interpolation mismatch"
                    grids.append([dict(price=v) for v in vals])
                del r
            rows = []
            for k, (x1, x2, f, th) in enumerate(pts):
                S1, S2 = m.S1_0 * np.exp(x1), m.S2_0 * np.exp(x2)
                spec = float(short.price(S1, S2))
                g = [grids[i][k] for i in range(3)]
                ref = richardson(g[0]["price"], g[1]["price"], g[2]["price"])
                row = dict(x1=x1, x2=x2, fraction=f, angle_deg=th, spectral=spec,
                           spectral_from_T1=float(long_.price(S1, S2, 1.0 - tau_now)),
                           V160=g[0]["price"], V320=g[1]["price"], V640=g[2]["price"],
                           **ref, diff=spec - ref["V_R"],
                           diff_pct=100 * (spec - ref["V_R"]) / ref["V_R"])
                if name in RECTANGLES:
                    row["greeks"] = {q: dict(V160=g[0][q], V320=g[1][q], V640=g[2][q],
                                             **richardson(g[0][q], g[1][q], g[2][q]))
                                     for q in SPOT_KEYS if q in g[0] and q != "price"}
                rows.append(row)
                print({a: b for a, b in row.items() if a != "greeks"}, flush=True)
            tag = "_T" if tau_now == 1.0 else ""
            _write(f"conv_fd_{name}{tag}", dict(config=name, tau=tau_now, grids=GRIDS,
                                                rows=rows))


def section_conv_angular(name="default", tau=0.25):
    """Larger angular ranges for the Euclidean basket, whose production truncation
    n <= 8 leaves undamped modes at tau = T/4. Each rung widens the scan at 15 points
    per unit q and the quadrature with the order. From n = 11 the scan also returns
    roots at q < 0.3, below the smallest root of the production basis; their modes
    have norms of order 1e-12 and are left out of the projection (norm <= 1e-6), but
    they take radial slots of their order, so the rungs above 10 also add radial
    roots. Prices at the spots of `offcentre` at tau and at T, spot Greeks at the
    spots of `conv`."""
    fd = json.loads((Path(ARTIFACTS_DIR) /
                     f"paper1_accuracy_conv_fd_{name}.json").read_text())["rows"]
    off = json.loads((Path(ARTIFACTS_DIR) /
                      f"paper1_accuracy_offcentre_{name}.json").read_text())["configs"][name]
    rungs = []
    for n_ang, m_rad, q_max in ((8, 5, 350.0), (10, 5, 350.0), (12, 7, 700.0),
                                (14, 9, 1000.0)):
        over = dict(N_angular_max=n_ang, M_radial_max=m_rad, q_max=q_max,
                    q_scan_points=int(15 * q_max), N_mu_quad=max(50, 10 * m_rad),
                    N_nu_quad=max(96, 16 * n_ang))
        t0 = time.perf_counter()
        s = build(name, overrides={"numerical": over})
        s.fit()
        b = s.basis
        q_prod = min(md.meta["q"] for md in b.modes      # first root of ce_0, the
                     if md.meta["n"] == 0 and md.meta["parity"] == "even")  # smallest
        low = [dict(n=md.meta["n"], m=md.meta["m"], parity=md.meta["parity"],
                    q=md.meta["q"], norm=md.norm)
               for md in b.modes if md.meta["q"] < q_prod]
        active = sum(1 for md in b.modes if md.norm > 1e-6)
        m = s.market
        pts = _spots(s)
        prices = []
        for k, (x1, x2, f, th) in enumerate(pts):
            S1, S2 = m.S1_0 * np.exp(x1), m.S2_0 * np.exp(x2)
            p_s, p_l = float(s.price(S1, S2, m.T - tau)), float(s.price(S1, S2))
            prices.append(dict(fraction=f, angle_deg=th, short=p_s, long=p_l,
                               diff_pct_short=100 * (p_s - fd[k]["V_R"]) / fd[k]["V_R"],
                               diff_u_short=(p_s - fd[k]["V_R"]) / fd[k]["u_FD"],
                               diff_pct_long=100 * (p_l - off[k]["V_R"]) / off[k]["V_R"]))
        greeks = _conv_rows(s, pts, market=False)
        rungs.append(dict(N_angular_max=n_ang, M_radial_max=m_rad, q_max=q_max,
                          n_modes=len(b.modes), n_active=active, roots_below_production=low,
                          prices=prices, greeks=greeks, wall_s=time.perf_counter() - t0))
        print(f"[{name}] n<={n_ang} m<={m_rad}: {len(b.modes)} modes, {active} active, "
              f"{len(low)} roots below q={q_prod:.2f}; max |diff| tau={tau}: "
              f"{max(abs(p['diff_pct_short']) for p in prices):.4f}%, tau=T: "
              f"{max(abs(p['diff_pct_long']) for p in prices):.4f}%", flush=True)
    _write("conv_angular", dict(config=name, tau=tau, rungs=rungs))


def section_conv_modes(names=RECTANGLES):
    """Spot Greeks on the rectangles at growing modes per side, against the
    finite-difference Greeks of `conv_fd`, with the quadrature at its production rule
    of four nodes per mode (at least 120) so that it does not limit the larger bases."""
    ladders = dict(dnt=(20, 30, 40, 50, 60), worst_of=(32, 40, 48, 56, 64),
                   spread=(40, 50, 60, 70))
    out = {}
    for name in names:
        fd = {1.0: f"conv_fd_{name}_T", 0.25: f"conv_fd_{name}"}
        fd = {tau: json.loads((Path(ARTIFACTS_DIR) / f"paper1_accuracy_{tag}.json")
                              .read_text())["rows"] for tau, tag in fd.items()}
        rungs = []
        for n in ladders[name]:
            t0 = time.perf_counter()
            s = build(name, n_modes=n)
            pts = _spots(s)
            rows = _conv_rows(s, pts, market=False)
            for r in rows:
                f = fd[r["tau"]][r["spot"]]
                r["diff_pct"] = {q: 100 * (r[q] - g["V_R"]) / g["V_R"]
                                 for q, g in f["greeks"].items()}
                r["diff_pct"]["price"] = 100 * (r["price"] - f["V_R"]) / f["V_R"]
            rungs.append(dict(N_modes=n, N_quad=s.basis.N_quad, rows=rows,
                              wall_s=time.perf_counter() - t0))
            print(f"[{name}] N={n} {rungs[-1]['wall_s']:.0f}s", {
                q: round(rows[0]["diff_pct"][q], 2) for q in ("price", "gamma1", "gamma2")},
                flush=True)
        out[name] = rungs
    _write("conv_modes", dict(configs=out))


SECTIONS = dict(quad=section_quad, fquad=section_fquad, rho0=section_rho0,
                margrabe=section_margrabe, offcentre=section_offcentre,
                holdouts=section_holdouts, timestep=section_timestep,
                vega_fixed=section_vega_fixed, conv=section_conv,
                conv_fd=section_conv_fd, conv_angular=section_conv_angular,
                conv_modes=section_conv_modes)

if __name__ == "__main__":
    args = sys.argv[1:] or list(SECTIONS)
    if args[0] in ("offcentre", "conv", "conv_fd") and len(args) > 1:
        SECTIONS[args[0]](tuple(args[1:]))
    else:
        for a in args:
            SECTIONS[a]()
