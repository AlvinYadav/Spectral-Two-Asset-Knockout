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
             contracts with 500, 1000 and 2000 time steps.

The offcentre, holdouts, rho0 and margrabe sections solve finite differences
at N = 640, which needs several GB of memory per solve; run them in sequence
rather than in parallel processes.

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
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twoasset.cli import assemble_solver                      # noqa: E402
from twoasset.core.domain import (EuclideanCircle,            # noqa: E402
                                  MahalanobisEllipse)
from twoasset.projection_comparison import _build_solver      # noqa: E402
from twoasset.runlog import build_provenance                  # noqa: E402
from twoasset.solvers._optimal import ARTIFACTS_DIR, OPTIMAL  # noqa: E402
from twoasset.solvers.fdm import fdm_price                    # noqa: E402

GRIDS = ((160, 250), (320, 500), (640, 1000))


def _write(section, payload):
    payload["provenance"] = build_provenance(section=section)
    out = Path(ARTIFACTS_DIR) / f"paper1_accuracy_{section}.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out}", flush=True)


def build(name, *, overrides=None, n_modes=None):
    """Assemble a production solver as paper1_evidence.build does, with
    optional per-section overrides of the TOML configuration."""
    cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
    cfg = copy.deepcopy(cfg)
    for section, values in (overrides or {}).items():
        cfg.setdefault(section, {}).update(values)
    opt = OPTIMAL[name]
    N = n_modes if n_modes is not None else opt["N_modes"]
    cfg.setdefault("spectral", {})["N_modes"] = N
    cfg["spectral"]["N_quad"] = max(120, N * 4)
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
    spread call is covered by the doubling in `quad` alone; its 2500-mode
    projection at four times the grid would not fit in memory."""
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


SECTIONS = dict(quad=section_quad, fquad=section_fquad, rho0=section_rho0,
                margrabe=section_margrabe, offcentre=section_offcentre,
                holdouts=section_holdouts, timestep=section_timestep)

if __name__ == "__main__":
    args = sys.argv[1:] or list(SECTIONS)
    if args[0] == "offcentre" and len(args) > 1:
        section_offcentre(tuple(args[1:]))
    else:
        for a in args:
            SECTIONS[a]()
