"""One numerical parameter at a time (Paper 1).

Sections (each writes validation/artifacts/paper1_onefactor_{section}.json):

  onefactor   Euclidean basket (Mathieu), Mahalanobis basket (Bessel) and double
              no-touch (Fourier). The price and spot Greeks are evaluated at three
              spots (centre; 60% and 90% of the way to the barrier along 20°) and at
              two times to maturity (tau = 1 and 0.25). One numerical parameter is
              changed at a time: angular order, radial count or Fourier modes,
              quadrature, the gamma step, and the vol bump. References:
                - curved barriers: the z-space finite differences on
                  m = 40..320 (Richardson);
                - rectangle: the x-space solver solved once on N = 160..640.
              Every difference is reported absolute and relative.
  truncation  Artificial outer boundaries of the spread and worst-of calls grown
              from 1.5 to 2.0 and 2.5 (x-space FD at fixed spacing), and the K = 0
              spread against Margrabe's formula on the same domains.

Run (repo root):  PYTHONPATH=. .venv/bin/python3 validation/paper1_onefactor.py SECTION [...]
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
sys.path.insert(0, str(ROOT / "validation"))

from paper1_evidence import build                                    # noqa: E402
from paper1_zfdm import extrapolate                                  # noqa: E402
from twoasset.cli import assemble                                    # noqa: E402
from twoasset.core.domain import EuclideanCircle, MahalanobisEllipse, Rectangle  # noqa: E402
from twoasset.runlog import build_provenance                         # noqa: E402
from twoasset.solvers._optimal import ARTIFACTS_DIR                  # noqa: E402
from twoasset.solvers.fdm_grid_greeks import fdm_x_onesolve          # noqa: E402
from twoasset.solvers.fdm_z import fdm_z_solve                       # noqa: E402

ART = Path(ARTIFACTS_DIR)
KEYS = ("price", "delta1", "delta2", "gamma1", "gamma2", "gamma12", "theta")
TAUS = (1.0, 0.25)
SPOT_FRACTIONS = (("centre", 0.0), ("mid", 0.6), ("near", 0.9))
ANGLE_DEG = 20.0
Z_LEVELS = ((40, 125), (80, 250), (160, 500), (320, 1000))
X_LEVELS = ((160, 250), (320, 500), (640, 1000))
BIG_Q = dict(q_max=800.0, q_scan_points=12000)        # 15 scan points per unit of q

FACTORS = {
    "default": [
        ("production", {}),
        ("angular order 10", dict(numerical=dict(N_angular_max=10, **BIG_Q))),
        ("angular order 12", dict(numerical=dict(N_angular_max=12, **BIG_Q))),
        ("radial roots 6", dict(numerical=dict(M_radial_max=6, **BIG_Q))),
        ("radial roots 7", dict(numerical=dict(M_radial_max=7, **BIG_Q))),
        ("quadrature x2", dict(numerical=dict(N_mu_quad=100, N_nu_quad=192))),
        ("all refined", dict(numerical=dict(N_angular_max=12, M_radial_max=7, N_mu_quad=100,
                                            N_nu_quad=192, **BIG_Q))),
    ],
    "mahalanobis": [
        ("production", {}),
        ("angular order 10", dict(numerical=dict(N_angular_max=10))),
        ("angular order 12", dict(numerical=dict(N_angular_max=12))),
        ("radial zeros 8", dict(numerical=dict(M_radial_max=8))),
        ("radial zeros 10", dict(numerical=dict(M_radial_max=10))),
        ("quadrature x2", dict(numerical=dict(N_r_quad=80, N_theta_quad=192))),
        ("all refined", dict(numerical=dict(N_angular_max=12, M_radial_max=10, N_r_quad=80,
                                            N_theta_quad=192))),
    ],
    "dnt": [
        ("production", {}),
        ("modes 24 per side", dict(n_modes=24)),
        ("modes 28 per side", dict(n_modes=28)),
        ("quadrature 240", dict(N_quad=240)),
        ("quadrature 480", dict(N_quad=480)),
        ("all refined", dict(n_modes=28, N_quad=480)),
    ],
}
EXPECTED_MODES = {   # (2N+1)·M for the curved bases: a short count means missed roots
    ("default", "production"): 85, ("default", "angular order 10"): 105,
    ("default", "angular order 12"): 125, ("default", "radial roots 6"): 102,
    ("default", "radial roots 7"): 119, ("default", "quadrature x2"): 85,
    ("default", "all refined"): 175,
    ("mahalanobis", "production"): 102, ("mahalanobis", "angular order 10"): 126,
    ("mahalanobis", "angular order 12"): 150, ("mahalanobis", "radial zeros 8"): 136,
    ("mahalanobis", "radial zeros 10"): 170, ("mahalanobis", "quadrature x2"): 102,
    ("mahalanobis", "all refined"): 250,
}
PRODUCTION_WIDTH = {"spread": 3.0, "worst_of": 1.5}     # x1-width of the truncated box


def _write(section, payload):
    payload["provenance"] = build_provenance(section=section)
    out = ART / f"paper1_onefactor_{section}.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out}", flush=True)


def _cfg(name, **financial):
    cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
    cfg["financial"].update(financial)
    return cfg


def _distance_to_barrier(market, domain, u):
    if isinstance(domain, EuclideanCircle):
        return domain.R
    if isinstance(domain, MahalanobisEllipse):
        return domain.R / np.sqrt(u @ np.linalg.solve(market.Sigma, u))
    t = []
    for k, (lo, hi) in enumerate(((domain.x1_lo, domain.x1_hi), (domain.x2_lo, domain.x2_hi))):
        if u[k] > 0:
            t.append(hi / u[k])
        elif u[k] < 0:
            t.append(lo / u[k])
    return min(t)


def _spots(market, domain):
    th = np.deg2rad(ANGLE_DEG)
    u = np.array([np.cos(th), np.sin(th)])
    d = _distance_to_barrier(market, domain, u)
    out = []
    for label, f in SPOT_FRACTIONS:
        x = f * d * u
        out.append(dict(label=label, fraction=f, x1=float(x[0]), x2=float(x[1]),
                        S1=float(market.S1_0 * np.exp(x[0])),
                        S2=float(market.S2_0 * np.exp(x[1]))))
    return out


def _spectral_greeks(solver, S1, S2, t, eps=1e-4):
    """Price and spot Greeks of the spectral solver, with gamma step `eps`."""
    if eps == 1e-4:
        g = solver.greeks(S1, S2, t)
    else:
        m = solver.market
        tau = m.T - t
        tau_eff = tau - solver.smooth_tau
        gf = m.gauge_factor(S1, S2, tau)
        g = solver.greeks(S1, S2, t)
        u, ux1, ux2 = solver._u_grad_x(S1, S2, tau_eff)
        h = solver._u_hessian_x(S1, S2, tau_eff, eps=eps)
        d1, d2, g11, g22, g12 = solver._spot_greeks_from_u(S1, S2, gf, u, ux1, ux2, *h)
        g.update(delta1=d1, delta2=d2, gamma1=g11, gamma2=g22, gamma12=g12)
    return {k: float(g[k]) for k in KEYS}


def _build_variant(name, spec):
    spec = dict(spec)
    nq = spec.pop("N_quad", None)
    _c, s = build(name, **spec)
    if nq is not None:
        s.basis.N_quad = nq
    t0 = time.perf_counter()
    s.fit()
    s.project()
    return s, time.perf_counter() - t0


def _references(name, market_T1, spots):
    """FD references per maturity and spot: dict[tau][label][key] = extrapolate(...)."""
    ref = {}
    for tau in TAUS:
        cfg = _cfg(name, T=tau)
        market, payoff, domain, _b, _ = assemble(cfg)
        per_level = []
        if name == "dnt":
            for N, Nt in X_LEVELS:
                one = fdm_x_onesolve(market, domain, payoff, N=N, N_time=Nt)
                per_level.append([one.grid_greeks(sp["S1"], sp["S2"]) for sp in spots])
                print(f"[{name} tau={tau}] x-space N={N} {one.wall['total_s']:.1f}s", flush=True)
                del one
        else:
            for m, Nt in Z_LEVELS:
                Nt_tau = max(int(round(Nt * tau)), 20)
                s = fdm_z_solve(market, domain, payoff, m=m, N_time=Nt_tau)
                per_level.append([s.greeks(sp["S1"], sp["S2"]) for sp in spots])
                print(f"[{name} tau={tau}] z-space m={m} {s.wall['total_s']:.1f}s", flush=True)
                del s
        ref[str(tau)] = {sp["label"]: {k: extrapolate([lv[i][k] for lv in per_level])
                                       for k in KEYS}
                         for i, sp in enumerate(spots)}
    return ref


def _diff(v, ref):
    a = v - ref["V_R"]
    return dict(value=v, abs_diff=a, rel_diff=a / ref["V_R"] if ref["V_R"] else float("nan"),
                diff_over_e=a / ref["e"] if ref["e"] > 0 else float("nan"))


def section_onefactor():
    out = {}
    for name, factors in FACTORS.items():
        base_cfg = _cfg(name)
        market, _payoff, domain, _b, _ = assemble(base_cfg)
        spots = _spots(market, domain)
        ref = _references(name, market, spots)
        rows = []
        base_solver = None
        for label, spec in factors:
            s, t_fit = _build_variant(name, spec)
            if label == "production":
                base_solver = s
            gram = getattr(s.basis, "gram_matrix", None)
            row = dict(factor=label, n_modes=int(s.n_modes()), cold_s=t_fit,
                       expected_modes=EXPECTED_MODES.get((name, label)),
                       gram_max_offdiag=(float(np.max(np.abs(gram - np.diag(np.diag(gram)))))
                                         if gram is not None and gram.size else None),
                       results={})
            for tau in TAUS:
                t = market.T - tau
                for sp in spots:
                    g = _spectral_greeks(s, sp["S1"], sp["S2"], t)
                    row["results"][f"{tau}/{sp['label']}"] = {
                        k: _diff(g[k], ref[str(tau)][sp["label"]][k]) for k in KEYS}
            rows.append(row)
            print(f"[{name}] {label}: {row['n_modes']} modes, {t_fit:.1f}s, centre price "
                  f"{row['results']['1.0/centre']['price']['rel_diff']:+.2e}", flush=True)
            if s is not base_solver:
                del s
        steps = {}
        for eps in (1e-3, 1e-4, 1e-5):
            steps[f"{eps:g}"] = {}
            for tau in TAUS:
                for sp in spots:
                    g = _spectral_greeks(base_solver, sp["S1"], sp["S2"], market.T - tau, eps=eps)
                    steps[f"{eps:g}"][f"{tau}/{sp['label']}"] = {
                        k: _diff(g[k], ref[str(tau)][sp["label"]][k])
                        for k in ("gamma1", "gamma2", "gamma12")}
        vb = {}
        for bump in (0.01, 0.005):
            t0 = time.perf_counter()
            mg = base_solver._market_greeks(market.S1_0, market.S2_0, 0.0, vol_bump=bump)
            vb[f"{bump:g}"] = dict(vega1=mg["vega1"], vega2=mg["vega2"], rho=mg["rho"],
                                   wall_s=time.perf_counter() - t0)
        out[name] = dict(spots=spots, references=ref, factors=rows, gamma_step=steps,
                         vol_bump=vb)
        print(f"[{name}] vol bump", vb, flush=True)
    _write("onefactor", out)


# ═══════════════════════════════════════════════════════════════════════
#  truncation
# ═══════════════════════════════════════════════════════════════════════

def _margrabe(market):
    s2 = market.sigma1**2 + market.sigma2**2 - 2 * market.rho * market.sigma1 * market.sigma2
    st = np.sqrt(s2 * market.T)
    d1 = (np.log(market.S1_0 / market.S2_0) + 0.5 * s2 * market.T) / st
    return float(market.S1_0 * norm.cdf(d1) - market.S2_0 * norm.cdf(d1 - st))


def section_truncation():
    out = {}
    cases = [("spread", w, {"spread": dict(x1_lo=-w, x1_hi=w, x2_lo=-w, x2_hi=w)})
             for w in (1.5, 2.0, 2.5)]
    cases += [("spread_K0", w, {"spread": dict(x1_lo=-w, x1_hi=w, x2_lo=-w, x2_hi=w,
                                               K_spread=0.0)}) for w in (1.5, 2.0, 2.5)]
    cases += [("worst_of", w, {"worstof": dict(R_max=w)}) for w in (1.5, 2.0, 2.5)]
    for label, w, over in cases:
        name = "spread" if label.startswith("spread") else label
        cfg = copy.deepcopy(_cfg(name))
        for sec, kv in over.items():
            cfg[sec].update(kv)
        market, payoff, domain, _b, _ = assemble(cfg)
        width = domain.W1
        res = {}
        for N_prod in (160, 320):           # the production spacing and half of it
            N = int(round(N_prod * width / PRODUCTION_WIDTH[name]))
            one = fdm_x_onesolve(market, domain, payoff, N=N, N_time=500)
            g = one.grid_greeks()
            res[f"N{N}"] = dict(N=N, dx=domain.W1 / (N + 1), price_bilinear=one.price(),
                                **{k: g[k] for k in KEYS}, wall_s=one.wall["total_s"])
            print(f"[{label} {w}] N={N} price {one.price():.8f} "
                  f"({one.wall['total_s']:.1f}s)", flush=True)
            del one
        row = dict(case=label, outer=w, width_x1=domain.W1, width_x2=domain.W2, grids=res)
        if label == "spread_K0":
            row["margrabe"] = _margrabe(market)
        out[f"{label}/{w}"] = row
    _write("truncation", out)


SECTIONS = dict(onefactor=section_onefactor, truncation=section_truncation)

if __name__ == "__main__":
    if len(sys.argv) < 2 or any(a not in SECTIONS for a in sys.argv[1:]):
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        SECTIONS[a]()
