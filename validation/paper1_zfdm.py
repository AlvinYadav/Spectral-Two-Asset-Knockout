"""Finite differences in whitened coordinates: the gates and references for Paper 1.

Sections (run one or several; each writes validation/artifacts/paper1_zfdm_{section}.json):

  z1       closed-form disc, m = 20..160                                (gate Z1)
  z2       M-matrix sign pattern, spectrum, ‖Rⁿ‖ on coarse grids       (gate Z2)
  levels   the three curved contracts on m = 40, 80, 160, 320, Cholesky and eigen
           whitening: price, one-solve spot Greeks, the 36 off-centre prices
                                                                 (gates Z3, Z5)
  rho0     the Euclidean basket at rho = 0, z-space against x-space   (gate Z4)
  sep      separate space and time refinement on the Euclidean basket (gate Z6)
  market   vegas (fixed contract; Mahalanobis also covariance-scaled) and rho on
           m = 80, 160, 320, plus a halved vol bump at m = 160
  xgreeks  x-space solver solved once: the ten bump-and-revalue spot Greeks of
           Table 6.2 read from one solution (gate G1), and one-solve grid Greeks on
           the rectangles at N = 160, 320, 640
  short    the three curved contracts at tau = T/4 at the centre and the twelve
           off-centre spots, against z-space references on the four grids; the
           Euclidean basket also with wider angular ranges (n <= 10, 12, 14)
  xmarket  vegas and rho of the rectangles from re-solved markets, each read from one
           x-space solve, on N = 160, 320, 640

Run (repo root):  PYTHONPATH=. .venv/bin/python3 validation/paper1_zfdm.py SECTION [...]
"""
from __future__ import annotations

import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
from scipy.special import j0, jn_zeros

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twoasset.cli import assemble                                       # noqa: E402
from twoasset.core.market import GBMMarket                              # noqa: E402
from twoasset.runlog import build_provenance                            # noqa: E402
from twoasset.solvers._optimal import ARTIFACTS_DIR, FOURIER_CONFIGS    # noqa: E402
from twoasset.solvers.fdm import fdm_price                              # noqa: E402
from twoasset.solvers.fdm_grid_greeks import fdm_x_onesolve             # noqa: E402
from twoasset.solvers.fdm_z import (assemble_operator, build_grid,      # noqa: E402
                                    fdm_z_greeks, fdm_z_solve, time_step)

ART = Path(ARTIFACTS_DIR)
CURVED = ("default", "tight_barrier", "mahalanobis")
LEVELS = ((40, 125), (80, 250), (160, 500), (320, 1000))
SPOT_KEYS = ("price", "delta1", "delta2", "gamma1", "gamma2", "gamma12", "theta",
             "theta_bdf2")
ORDER_BAND = (1.7, 2.3)


def _write(section, payload):
    payload["provenance"] = build_provenance(section=section)
    out = ART / f"paper1_zfdm_{section}.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out}", flush=True)


def _contract(name, **financial):
    cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
    cfg["financial"].update(financial)
    market, payoff, domain, _b, _ = assemble(cfg)
    return market, payoff, domain


def extrapolate(values):
    """Pairwise observed orders and the Richardson value from the last three levels
    (refinement ratio 2). The extrapolation step e = |V_R - V_finest| is a
    deterministic estimate of the remaining grid correction, not a standard error.
    A non-monotone last triple gets no order and keeps the finest value."""
    v = np.asarray(values, dtype=float)
    d = np.diff(v)
    orders = [float(np.log2(d[k] / d[k + 1])) if d[k] * d[k + 1] > 0 else float("nan")
              for k in range(len(d) - 1)]
    p = orders[-1]
    vr = float(v[-1] + d[-1] / (2.0 ** p - 1.0)) if np.isfinite(p) and p > 0 else float(v[-1])
    vr2 = float(v[-1] + d[-1] / 3.0)                     # assumed second order
    return dict(values=v.tolist(), orders=orders, p=p, V_R=vr, e=abs(vr - v[-1]),
                V_R_p2=vr2, e_p2=abs(vr2 - v[-1]))


def _in_band(p):
    return bool(np.isfinite(p) and ORDER_BAND[0] <= p <= ORDER_BAND[1])


# ═══════════════════════════════════════════════════════════════════════
#  z1, z2
# ═══════════════════════════════════════════════════════════════════════

def section_z1():
    j01 = jn_zeros(0, 1)[0]
    out = {}
    for label, b, r in (("heat", (0.0, 0.0), 0.0), ("drift", (0.3, -0.2), 0.05)):
        rows = []
        for m, Nt in ((20, 25), (40, 50), (80, 100), (160, 200)):
            g = build_grid(np.eye(2), 1.0, m)
            A = assemble_operator(g, b, r)
            ii, jj = np.nonzero(g.inside)
            z1, z2 = (ii - g.K1) * g.h, (jj - g.K2) * g.h
            bz, rr = b[0] * z1 + b[1] * z2, np.hypot(z1, z2)
            V, _, _ = time_step(A, np.exp(-bz) * j0(j01 * rr), 1.0, Nt)
            ex = (np.exp(-bz - (0.5 * (b[0]**2 + b[1]**2) + r) - 0.5 * j01**2)
                  * j0(j01 * rr))
            c = g.idx[g.K1, g.K2]
            rows.append(dict(m=m, N_time=Nt, n=g.n_unknowns, err_centre=abs(V[c] - ex[c]),
                             err_max=float(np.max(np.abs(V - ex)))))
        oc = [float(np.log2(rows[k]["err_centre"] / rows[k + 1]["err_centre"])) for k in range(3)]
        om = [float(np.log2(rows[k]["err_max"] / rows[k + 1]["err_max"])) for k in range(3)]
        passed = all(1.8 <= o <= 2.2 for o in oc[1:] + om[1:])
        out[label] = dict(rows=rows, orders_centre=oc, orders_max=om, gate_Z1=passed)
        print(label, oc, om, passed, flush=True)
    _write("z1", out)


def section_z2():
    out = {}
    for name in CURVED:
        market, payoff, domain = _contract(name)
        for m, Nt in ((15, 47), (30, 94)):
            s = fdm_z_solve(market, domain, payoff, m=m, N_time=Nt)
            A = s.A.toarray()
            n = A.shape[0]
            off = A - np.diag(np.diag(A))
            ev = np.linalg.eigvals(A)
            dt = market.T / Nt
            sup2, supinf = float("nan"), float("nan")
            if m == 15:                                     # dense powers: coarse grid only
                I = np.eye(n)
                M = np.linalg.inv(I - 0.5 * dt * A)
                R = M @ (I + 0.5 * dt * A)
                P = np.linalg.matrix_power(M, 8)           # Rannacher start-up
                sup2, supinf = 0.0, 0.0
                for _ in range(Nt - 4):
                    sup2 = max(sup2, np.linalg.norm(P, 2))
                    supinf = max(supinf, np.linalg.norm(P, np.inf))
                    P = R @ P
            mu = np.array([market.r - 0.5 * market.sigma1**2, market.r - 0.5 * market.sigma2**2])
            bh = float(np.max(np.abs(s.W @ mu)) * s.grid.h)
            row = dict(m=m, N_time=Nt, n=n, min_offdiag=float(off.min()), max_bh=bh,
                       max_row_sum=float(A.sum(axis=1).max()), max_re_eig=float(ev.real.max()),
                       max_abs_im_eig=float(np.abs(ev.imag).max()),
                       sup_norm2_step_powers=float(sup2), sup_norminf_step_powers=float(supinf))
            row["gate_Z2"] = bool(row["min_offdiag"] >= 0 and bh < 1 and row["max_re_eig"] < 0
                                  and (m != 15 or sup2 <= 1 + 1e-8))
            out[f"{name}/m{m}"] = row
            print(name, row, flush=True)
    _write("z2", out)


# ═══════════════════════════════════════════════════════════════════════
#  levels: the three curved contracts
# ═══════════════════════════════════════════════════════════════════════

def _offcentre_spots(name):
    d = json.loads((ART / f"paper1_accuracy_offcentre_{name}.json").read_text())
    return d["configs"][name]


def section_levels():
    out = {}
    for name in CURVED:
        market, payoff, domain = _contract(name)
        spots = _offcentre_spots(name)
        for whitening in ("cholesky", "eigen"):
            rows = []
            for m, Nt in LEVELS:
                s = fdm_z_solve(market, domain, payoff, m=m, N_time=Nt, whitening=whitening)
                t0 = time.perf_counter()
                g = s.greeks()
                t_g = time.perf_counter() - t0
                off = []
                for sp in spots[1:]:
                    try:
                        off.append(s.price(sp["S1"], sp["S2"]))
                    except ValueError:
                        off.append(float("nan"))
                rows.append(dict(m=m, N_time=Nt, n=s.grid.n_unknowns, h=s.grid.h,
                                 wall=dict(s.wall, greeks_s=t_g),
                                 **{k: g[k] for k in SPOT_KEYS}, offcentre=off))
                print(f"[{name}/{whitening}] m={m} n={s.grid.n_unknowns} "
                      f"{s.wall['total_s']:.1f}s price={g['price']:.8f}", flush=True)
                del s
            ex = {k: extrapolate([r[k] for r in rows]) for k in SPOT_KEYS}
            off_ex = [extrapolate([r["offcentre"][i] for r in rows])
                      for i in range(len(spots) - 1)]
            out[f"{name}/{whitening}"] = dict(levels=rows, extrapolated=ex,
                                              offcentre_spots=spots[1:],
                                              offcentre_extrapolated=off_ex,
                                              gate_Z3=_in_band(ex["price"]["orders"][-1])
                                              and _in_band(ex["price"]["orders"][-2]))
        c, e = out[f"{name}/cholesky"], out[f"{name}/eigen"]
        ec = c["extrapolated"]["price"]
        ee = e["extrapolated"]["price"]
        c["gate_Z5"] = bool(abs(ec["V_R"] - ee["V_R"]) <= 3 * max(ec["e"], ee["e"]))
        print(name, "Z3", c["gate_Z3"], ec["orders"], "Z5", c["gate_Z5"],
              ec["V_R"], ee["V_R"], flush=True)
    _write("levels", out)


# ═══════════════════════════════════════════════════════════════════════
#  rho0 and sep
# ═══════════════════════════════════════════════════════════════════════

def section_rho0():
    market, payoff, domain = _contract("default", rho=0.0)
    z = [fdm_z_solve(market, domain, payoff, m=m, N_time=Nt).price() for m, Nt in LEVELS]
    x = []
    for N, Nt in ((80, 125), (160, 250), (320, 500), (640, 1000)):
        x.append(float(fdm_price(market, domain, payoff=payoff, N=N, N_time=Nt)["price"]))
        print("x-space", N, x[-1], flush=True)
    ez, exx = extrapolate(z), extrapolate(x)
    out = dict(z=ez, x=exx,
               gate_Z4_rho0=bool(abs(ez["V_R"] - exx["V_R"]) <= 3 * (ez["e"] + exx["e"])))
    stored = {r["config"]: r["fd"] for r in
              json.loads((ART / "paper1_price_table.json").read_text())["rows"]}
    lv = json.loads((ART / "paper1_zfdm_levels.json").read_text())
    for name in CURVED:
        zr = lv[f"{name}/cholesky"]["extrapolated"]["price"]
        out[f"{name}/production"] = dict(
            z_V_R=zr["V_R"], z_e=zr["e"], x_V_R=stored[name]["V_R"], x_u=stored[name]["u"],
            gate_Z4=bool(abs(zr["V_R"] - stored[name]["V_R"]) <= 3 * stored[name]["u"]))
    print(out, flush=True)
    _write("rho0", out)


def section_sep():
    market, payoff, domain = _contract("default")
    space = [fdm_z_solve(market, domain, payoff, m=m, N_time=2000).price() for m, _ in LEVELS]
    print("space", space, flush=True)
    timeref = [fdm_z_solve(market, domain, payoff, m=320, N_time=Nt).price()
               for Nt in (250, 500, 1000, 2000)]
    print("time", timeref, flush=True)
    es, et = extrapolate(space), extrapolate(timeref)
    _write("sep", dict(space=es, space_N_time=2000, time=et, time_m=320,
                       gate_Z6=_in_band(es["orders"][-1]) and _in_band(et["orders"][-1])))


# ═══════════════════════════════════════════════════════════════════════
#  market Greeks
# ═══════════════════════════════════════════════════════════════════════

def section_market():
    out = {}
    keys = ("vega1", "vega2", "rho", "vega1_fixed", "vega2_fixed")
    for name in CURVED:
        market, payoff, domain = _contract(name)
        rows = []
        for m, Nt in LEVELS[1:]:
            g = fdm_z_greeks(market, domain, payoff, m=m, N_time=Nt)
            rows.append(dict(m=m, N_time=Nt, **{k: g[k] for k in keys if k in g},
                             wall_all_greeks_s=g["wall_all_greeks_s"],
                             wall_one_solve_s=g["wall_one_solve_s"]))
            print(f"[{name}] m={m}", rows[-1], flush=True)
        half = fdm_z_greeks(market, domain, payoff, m=160, N_time=500, vol_bump=0.005)
        out[name] = dict(levels=rows,
                         extrapolated={k: extrapolate([r[k] for r in rows])
                                       for k in keys if k in rows[0]},
                         half_bump_m160={k: half[k] for k in keys if k in half})
    _write("market", out)


# ═══════════════════════════════════════════════════════════════════════
#  xgreeks: the x-space solver solved once
# ═══════════════════════════════════════════════════════════════════════

def section_xgreeks():
    pub = {r["config"]: r["fdm_bump"] for r in
           json.loads((ART / "paper1_greeks.json").read_text())["bump"]}
    out = dict(G1={}, rectangles={})
    for name in pub:
        market, payoff, domain = _contract(name)
        one = fdm_x_onesolve(market, domain, payoff, N=320, N_time=500)
        bg = one.bump_greeks(1.0)
        rel = {k: abs(bg[k] - pub[name][k]) / max(abs(pub[name][k]), 1e-300)
               for k in ("price", "delta1", "delta2", "gamma1", "gamma2", "gamma12")}
        out["G1"][name] = dict(one_solve=bg, published=pub[name], max_rel_diff=max(rel.values()),
                               rel_diff=rel, gate_G1=bool(max(rel.values()) <= 1e-6),
                               one_solve_wall_s=one.wall["total_s"])
        print(f"[G1 {name}] max rel diff {max(rel.values()):.2e} "
              f"({one.wall['total_s']:.1f}s)", flush=True)
        del one
    for name in FOURIER_CONFIGS:
        market, payoff, domain = _contract(name)
        rows = []
        for N, Nt in ((160, 250), (320, 500), (640, 1000)):
            one = fdm_x_onesolve(market, domain, payoff, N=N, N_time=Nt)
            g = one.grid_greeks()
            rows.append(dict(N=N, N_time=Nt, wall=one.wall, **g))
            print(f"[rect {name}] N={N} {one.wall['total_s']:.1f}s", g, flush=True)
            del one
        out["rectangles"][name] = dict(levels=rows, extrapolated={
            k: extrapolate([r[k] for r in rows]) for k in SPOT_KEYS})
    _write("xgreeks", out)


# ═══════════════════════════════════════════════════════════════════════
#  short: the curved contracts at tau = T/4 against z-space references
# ═══════════════════════════════════════════════════════════════════════

SHORT_TAU = 0.25
# Angular rungs of the Euclidean basket, as in paper1_accuracy_checks.py conv_angular:
# (largest order, radial roots per order, top of the root scan).
SHORT_RUNGS = ((8, 5, 350.0), (10, 5, 350.0), (12, 7, 700.0), (14, 9, 1000.0))


def section_short():
    """Prices at tau = T/4 at the centre and the twelve off-centre spots of `levels`.
    The spectral basis is solved for T and evaluated at tau (no new fit); the reference
    solves the contract with maturity tau on the four z-space grids. The Euclidean
    basket also gets the wider angular rungs of SHORT_RUNGS."""
    sys.path.insert(0, str(ROOT / "validation"))
    from paper1_evidence import build
    out = {}
    for name in CURVED:
        market, payoff, domain = _contract(name, T=SHORT_TAU)
        spots = _offcentre_spots(name)
        rows = []
        for m, Nt in LEVELS:
            s = fdm_z_solve(market, domain, payoff, m=m, N_time=Nt)
            vals = []
            for sp in spots:
                try:
                    vals.append(s.price(sp["S1"], sp["S2"]))
                except ValueError:
                    vals.append(float("nan"))
            rows.append(dict(m=m, N_time=Nt, prices=vals, wall_s=s.wall["total_s"]))
            print(f"[short {name}] m={m} {s.wall['total_s']:.1f}s centre {vals[0]:.8f}",
                  flush=True)
            del s
        ref = [extrapolate([r["prices"][i] for r in rows]) for i in range(len(spots))]
        rungs = []
        for n_ang, m_rad, q_max in (SHORT_RUNGS if name != "mahalanobis" else ((None,) * 3,)):
            if n_ang is None or (n_ang, m_rad, q_max) == SHORT_RUNGS[0]:
                _c, sol = build(name)                   # production settings
                label = "production"
            elif name == "default":
                over = dict(N_angular_max=n_ang, M_radial_max=m_rad, q_max=q_max,
                            q_scan_points=int(15 * q_max), N_mu_quad=max(50, 10 * m_rad),
                            N_nu_quad=max(96, 16 * n_ang))
                _c, sol = build(name, numerical=over)
                label = f"n<={n_ang}, m<={m_rad}, q<={q_max:g}"
            else:
                continue
            t0 = time.perf_counter()
            sol.fit()
            sol.project()
            t_fit = time.perf_counter() - t0
            m1 = sol.market
            per = []
            for sp, rf in zip(spots, ref):
                v = float(sol.price(sp["S1"], sp["S2"], m1.T - SHORT_TAU))
                per.append(dict(fraction=sp["fraction"], angle_deg=sp["angle_deg"],
                                spectral=v, V_R=rf["V_R"], e=rf["e"], p=rf["p"],
                                diff_pct=100 * (v - rf["V_R"]) / rf["V_R"],
                                diff_e=(v - rf["V_R"]) / rf["e"] if rf["e"] else float("nan")))
            worst = max(per, key=lambda r: abs(r["diff_pct"]))
            rungs.append(dict(label=label, n_modes=int(sol.n_modes()), cold_s=t_fit,
                              spots=per, centre_diff_pct=per[0]["diff_pct"],
                              max_abs_diff_pct=abs(worst["diff_pct"]),
                              worst_spot=(worst["fraction"], worst["angle_deg"])))
            print(f"[short {name}] {label}: {sol.n_modes()} modes, centre "
                  f"{per[0]['diff_pct']:+.4f}%, worst {worst['diff_pct']:+.4f}% at "
                  f"{worst['fraction']}/{worst['angle_deg']}", flush=True)
            del sol
        out[name] = dict(tau=SHORT_TAU, levels=rows, reference=ref, spectral=rungs)
    _write("short", out)


# ═══════════════════════════════════════════════════════════════════════
#  xmarket: vegas and rho on the rectangles over three grids
# ═══════════════════════════════════════════════════════════════════════

def section_xmarket():
    """Vegas and rho of the rectangle contracts by central differences over re-solved
    markets (bumps 0.01 and 1e-4, as in Table 6.2), each price read from one x-space
    solve as in `xgreeks`, on N = 160, 320 and 640, then extrapolated."""
    from twoasset.solvers.fdm_z import _bumped
    bumps = dict(vega1=("sigma1", 0.01), vega2=("sigma2", 0.01), rho=("r", 1e-4))
    out = {}
    for name in FOURIER_CONFIGS:
        market, payoff, domain = _contract(name)
        rows = []
        for N, Nt in ((160, 250), (320, 500), (640, 1000)):
            t0 = time.perf_counter()
            row = dict(N=N, N_time=Nt)
            for key, (attr, h) in bumps.items():
                up, dn = (fdm_x_onesolve(_bumped(market, **{attr: getattr(market, attr) + s * h}),
                                         domain, payoff, N=N, N_time=Nt).grid_greeks()["price"]
                          for s in (1, -1))
                row[key] = (up - dn) / (2 * h)
            row["wall_s"] = time.perf_counter() - t0
            rows.append(row)
            print(f"[xmarket {name}] N={N} {row['wall_s']:.0f}s",
                  {k: round(row[k], 5) for k in bumps}, flush=True)
        out[name] = dict(levels=rows, extrapolated={k: extrapolate([r[k] for r in rows])
                                                    for k in bumps})
    _write("xmarket", out)


SECTIONS = dict(z1=section_z1, z2=section_z2, levels=section_levels, rho0=section_rho0,
                sep=section_sep, market=section_market, xgreeks=section_xgreeks,
                short=section_short, xmarket=section_xmarket)

if __name__ == "__main__":
    if len(sys.argv) < 2 or any(a not in SECTIONS for a in sys.argv[1:]):
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        SECTIONS[a]()
