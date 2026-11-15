"""Survival probability, exit-time density, hazard rate and rebate at hit from the
fitted Mathieu and Bessel bases (Paper 1, Section 4).

For each curved contract at its production settings (eigenvalue cutoff, Galerkin
projection) this writes validation/artifacts/paper1_survival.json with

  table      at the centre, for horizons 0.25, 0.5, 1, 2 and 5 years: survival S,
             density f = -dS/dtau, hazard f / S, digital knock-out exp(-r tau) S and
             the rebate 1 paid at the hit;
  kappa1     the long-run hazard rate lambda_1^2 / 2 + beta' Sigma beta / 2;
  identity   (I) S(tau) against exp(r tau) times the price of the unit payoff from
             the ordinary pricing path, at three horizons and two spots;
  ddtau      (I) f against a central difference of S in tau (h = 1e-5);
  mc         (S) survival and rebate at T against the bridge-corrected hitting-time
             sampler `mc_hitting_times` (4e5 paths, 1000 steps, antithetic; the
             standard error is taken over antithetic pairs);
  cutoff     (D, Euclidean basket only) survival and rebate at T for the cutoff
             Lambda, 164 and 300 on 100 x 192 nodes: the rebate converges slowly,
             through the sum of A_n / omega_n.

Regression targets from an independent implementation (Euclidean basket, complete basis
to lambda^2 = 164) are stored alongside for comparison.

Run (repo root):  PYTHONPATH=. python3 validation/paper1_survival.py [contract ...]
"""
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

from twoasset.cli import assemble, assemble_solver                    # noqa: E402
from twoasset.core.payoff import UnitPayoff                           # noqa: E402
from twoasset.runlog import build_provenance                          # noqa: E402
from twoasset.solvers._optimal import ARTIFACTS_DIR                   # noqa: E402
from twoasset.solvers.mc import mc_hitting_times                      # noqa: E402

ART = Path(ARTIFACTS_DIR)
CURVED = ("default", "tight_barrier", "mahalanobis")
HORIZONS = (0.25, 0.5, 1.0, 2.0, 5.0)
SPEC_TARGETS = {  # independent implementation, Euclidean basket, centre
    "kappa1": 0.7553622,
    "rows": {0.25: (0.995497, 0.094363, 0.094790, 0.983130, 0.004506),
             0.5: (0.933570, 0.372264, 0.398753, 0.910520, 0.065210),
             1.0: (0.712671, 0.446830, 0.626980, 0.677914, 0.277922),
             2.0: (0.357854, 0.259449, 0.725014, 0.323800, 0.607885),
             5.0: (0.037943, 0.028641, 0.754828, 0.029550, 0.883583)},
    "mc_prototype": {"survival": (0.71272, 0.00071), "rebate": (0.27781, 0.00069)},
}
MC = dict(n_paths=400_000, n_steps=1000, seed=7)


def _cfg(name, **numerical):
    cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
    cfg["numerical"].update(numerical)
    return cfg


def _solver(name, **numerical):
    s, _ = assemble_solver(_cfg(name, **numerical))
    s.project()
    return s


def _spots(s):
    m = s.market
    R = s.domain.R
    out = [("centre", m.S1_0, m.S2_0)]
    x = 0.6 * R * np.array([np.cos(np.radians(50.0)), np.sin(np.radians(50.0))])
    if hasattr(s.basis, "R"):     # Mahalanobis: the fraction is of the whitened radius
        z = 0.6 * R * np.array([np.cos(np.radians(50.0)), np.sin(np.radians(50.0))])
        x = np.array(m.z_to_x(z[0], z[1]))
    out.append(("60% along 50 deg", m.S1_0 * np.exp(x[0]), m.S2_0 * np.exp(x[1])))
    return out


def _pair_mean_sem(v):
    """Mean and standard error over antithetic pairs (first half, second half)."""
    n = len(v) // 2
    pairs = 0.5 * (v[:n] + v[n:2 * n])
    return float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(n))


def run_contract(name):
    t0 = time.perf_counter()
    s = _solver(name)
    m = s.market
    T = m.T
    info = s.basis.truncation_info
    out = dict(n_modes=len(s.basis.modes), lambda2_cut=info["lambda2_cut"],
               lambda2_first=info["lambda2_first"], kappa1=s.long_run_hazard(),
               half_lambda1_sq=0.5 * info["lambda2_first"],
               drift_part=float(-m.gamma - m.r))
    rows = []
    for tau in HORIZONS:
        S = s.survival(tau=tau)
        rows.append(dict(tau=tau, survival=S, density=s.exit_density(tau=tau),
                         hazard=s.hazard(tau=tau), digital=float(np.exp(-m.r * tau) * S),
                         rebate=s.rebate_at_hit(T=tau)))
    out["table"] = rows

    # (I) survival against the ordinary pricing path of the unit payoff
    cfg = _cfg(name)
    dig, _ = assemble_solver(cfg, payoff=UnitPayoff(notional=1.0), smooth=False,
                             projection_method="galerkin")
    dig.project()
    ident = []
    for label, S1, S2 in _spots(s):
        for tau in (0.25 * T, 0.5 * T, T):
            a = s.survival(S1, S2, tau)
            b = float(np.exp(m.r * tau) * dig.price(S1, S2, T - tau))
            ident.append(dict(spot=label, tau=tau, survival=a, exp_rtau_digital=b,
                              rel_diff=abs(a - b) / abs(b)))
    out["identity"] = dict(rows=ident, max_rel_diff=max(r["rel_diff"] for r in ident),
                           threshold=1e-12)

    # (I) density against a central difference of the survival in tau
    h = 1e-5   # central-difference error h^2 S'''/6 stays below 1e-9
    dd = []
    for label, S1, S2 in _spots(s):
        for tau in (0.25 * T, 0.5 * T, T):
            f = s.exit_density(S1, S2, tau)
            fd = -(s.survival(S1, S2, tau + h) - s.survival(S1, S2, tau - h)) / (2 * h)
            dd.append(dict(spot=label, tau=tau, density=f, central_diff=fd,
                           abs_diff=abs(f - fd)))
    out["ddtau"] = dict(rows=dd, h=h, max_abs_diff=max(r["abs_diff"] for r in dd),
                        threshold=1e-8)

    # (S) Monte Carlo at T
    market, payoff, domain, _b, _k = assemble(_cfg(name))
    t1 = time.perf_counter()
    sim = mc_hitting_times(market, payoff, domain, antithetic=True, **MC)
    hit = sim["hit_time"]
    surv_mc, surv_se = _pair_mean_sem(sim["survived"].astype(float))
    reb = np.where(np.isnan(hit), 0.0, np.exp(-m.r * np.nan_to_num(hit)))
    reb_mc, reb_se = _pair_mean_sem(reb)
    S_T, B_T = s.survival(tau=T), s.rebate_at_hit(T=T)
    out["mc"] = dict(**MC, antithetic=True, wall_s=time.perf_counter() - t1,
                     survival=dict(spectral=S_T, mc=surv_mc, sem=surv_se,
                                   n_sem=abs(S_T - surv_mc) / surv_se),
                     rebate=dict(spectral=B_T, mc=reb_mc, sem=reb_se,
                                 n_sem=abs(B_T - reb_mc) / reb_se),
                     threshold_n_sem=3.0)
    print(f"[survival {name}] {len(s.basis.modes)} modes, kappa1 {out['kappa1']:.7f}; "
          f"S(T) {S_T:.6f} vs MC {surv_mc:.6f} +- {surv_se:.6f}; "
          f"rebate {B_T:.6f} vs MC {reb_mc:.6f} +- {reb_se:.6f}; "
          f"identity {out['identity']['max_rel_diff']:.1e}, "
          f"d/dtau {out['ddtau']['max_abs_diff']:.1e}", flush=True)

    if name == "default":
        cut = []
        for lam in (None, 164.0, 300.0):
            kw = dict(N_mu_quad=100, N_nu_quad=192)
            if lam is not None:
                kw["lambda2_max"] = lam
            sl = _solver(name, **kw)
            cut.append(dict(lambda2_cut=sl.basis.truncation_info["lambda2_cut"],
                            n_modes=len(sl.basis.modes), survival=sl.survival(tau=T),
                            rebate=sl.rebate_at_hit(T=T)))
            print(f"[survival cutoff] {cut[-1]}", flush=True)
        out["cutoff"] = cut
        out["spec_targets"] = SPEC_TARGETS
    out["wall_s"] = time.perf_counter() - t0
    return out


def main(names):
    path = ART / "paper1_survival.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        data[name] = run_contract(name)
    data["provenance"] = build_provenance(section="survival")
    path.write_text(json.dumps(data, indent=2, default=float))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(CURVED))
