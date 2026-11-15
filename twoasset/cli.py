"""Command-line entry point: read a TOML config, price it spectrally and optionally
with Monte Carlo and finite differences."""

from __future__ import annotations
import argparse
import os
import sys
import time

try:
    import tomllib
except ImportError:
    import pip._vendor.tomli as tomllib

import numpy as np

from .core.market import GBMMarket
from .core.domain import EuclideanCircle, MahalanobisEllipse, Rectangle
from .core.payoff import BasketCall, Spread, WorstOf, DoubleNoTouch
from .bases.mathieu import MathieuBasis
from .bases.bessel import BesselBasis
from .bases.fourier import FourierBasis
from .solvers.spectral import SpectralSolver
from .solvers.fdm import fdm_price
from .solvers.mc import mc_price


def detect_problem(cfg: dict) -> str:
    if "mahalanobis" in cfg:
        return "mahalanobis"
    if "dnt" in cfg:
        return "dnt"
    if "spread" in cfg:
        return "spread"
    if "worstof" in cfg:
        return "worstof"
    return "elliptical"


def _truncation_kwargs(num: dict, *, mathieu: bool = False) -> dict:
    """Truncation-rule keys of [numerical] for the curved-barrier bases.

    A config without `truncation` keeps the earlier index rule, so frozen and
    sealed configs reproduce their records; production configs select
    `truncation = "eigenvalue"`."""
    kw = dict(truncation=num.get("truncation", "index"),
              tau_min_frac=num.get("tau_min_frac", 0.25),
              eps_trunc=num.get("eps_trunc", 1e-6),
              lambda2_max=num.get("lambda2_max", None))
    if mathieu:
        kw["q_scan_density"] = num.get("q_scan_density", 15.0)
    return kw


def assemble(cfg: dict):
    """Return (market, payoff, domain, basis, numerical_cfg)."""
    fin = cfg["financial"]
    market = GBMMarket.from_config(cfg)

    kind = detect_problem(cfg)

    if kind == "elliptical":
        bar = cfg["barrier"]
        domain = EuclideanCircle(R=bar["R"])
        payoff = BasketCall(K=fin["K"],
                            w1=fin.get("w1", 1.0),
                            w2=fin.get("w2", 1.0))
        num = cfg.get("numerical", {})
        basis = MathieuBasis(
            N_trunc=num.get("N_trunc", 50),
            N_angular_max=num.get("N_angular_max", 6),
            M_radial_max=num.get("M_radial_max", 4),
            q_min=num.get("q_min", 0.1),
            q_max=num.get("q_max", 100.0),
            q_scan_points=num.get("q_scan_points", 1500),
            N_mu_quad=num.get("N_mu_quad", 40),
            N_nu_quad=num.get("N_nu_quad", 96),
            backend=num.get("mathieu_backend", "scipy"),
            **_truncation_kwargs(num, mathieu=True),
        )
        return market, payoff, domain, basis, kind

    if kind == "mahalanobis":
        mh = cfg["mahalanobis"]
        domain = MahalanobisEllipse(R=mh["R"])
        payoff = BasketCall(K=fin["K"],
                            w1=fin.get("w1", 1.0),
                            w2=fin.get("w2", 1.0))
        num = cfg.get("numerical", {})
        basis = BesselBasis(
            N_angular_max=num.get("N_angular_max", 8),
            M_radial_max=num.get("M_radial_max", 6),
            N_r_quad=num.get("N_r_quad", 40),
            N_theta_quad=num.get("N_theta_quad", 96),
            **_truncation_kwargs(num),
        )
        return market, payoff, domain, basis, kind

    if kind == "dnt":
        dnt = cfg["dnt"]
        domain = Rectangle(
            x1_lo=dnt["x1_lo"], x1_hi=dnt["x1_hi"],
            x2_lo=dnt["x2_lo"], x2_hi=dnt["x2_hi"],
        )
        payoff = DoubleNoTouch(notional=dnt.get("notional", 1.0))
        spec = cfg.get("spectral", {})
        basis = FourierBasis(
            N_modes=spec.get("N_modes", 25),
            N_quad=spec.get("N_quad", 120),
        )
        return market, payoff, domain, basis, kind

    if kind == "spread":
        spr = cfg["spread"]
        domain = Rectangle(
            x1_lo=spr.get("x1_lo", -1.5), x1_hi=spr.get("x1_hi", 1.5),
            x2_lo=spr.get("x2_lo", -1.5), x2_hi=spr.get("x2_hi", 1.5),
        )
        payoff = Spread(K=spr.get("K_spread", 10.0),
                        w1=spr.get("w1", 1.0), w2=spr.get("w2", 1.0))
        spec = cfg.get("spectral", {})
        basis = FourierBasis(
            N_modes=spec.get("N_modes", 25),
            N_quad=spec.get("N_quad", 120),
        )
        return market, payoff, domain, basis, kind

    if kind == "worstof":
        wo = cfg["worstof"]
        xb = np.log(wo.get("B", 0.60))
        Rmax = wo.get("R_max", 1.5)
        domain = Rectangle(x1_lo=xb, x1_hi=xb + Rmax,
                           x2_lo=xb, x2_hi=xb + Rmax)
        payoff = WorstOf(K=wo.get("K", 90),
                         S1_0=market.S1_0, S2_0=market.S2_0)
        spec = cfg.get("spectral", {})
        basis = FourierBasis(
            N_modes=spec.get("N_modes", 25),
            N_quad=spec.get("N_quad", 120),
        )
        return market, payoff, domain, basis, kind

    raise ValueError(f"Unrecognised problem kind: {kind}")


def assemble_solver(cfg: dict, **solver_overrides):
    """Assemble a SpectralSolver whose projection method and smoothing come from the
    config's [numerical] section."""
    market, payoff, domain, basis, kind = assemble(cfg)
    num = cfg.get("numerical", {})
    kw = dict(
        market=market, payoff=payoff, domain=domain, basis=basis,
        projection_method=num.get("projection_method", "auto"),
    )
    if "smooth_tau_frac" in num:
        frac = float(num["smooth_tau_frac"])
        kw.update(smooth=frac > 0, smooth_tau_frac=max(frac, 1e-12))
    kw.update(solver_overrides)
    return SpectralSolver(**kw), kind


# ═══════════════════════════════════════════════════════════════════════
#  Output directory management
# ═══════════════════════════════════════════════════════════════════════

def _next_run_dir(config_path: str) -> str:
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results')
    os.makedirs(root, exist_ok=True)
    existing = [d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d)) and d[:4].isdigit()]
    mx = max((int(d[:4]) for d in existing), default=0)
    name = f"{mx+1:04d}_{os.path.splitext(os.path.basename(config_path))[0]}"
    d = os.path.join(root, name)
    os.makedirs(d)
    return d


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description='twoasset spectral pricer (unified)')
    ap.add_argument('config', nargs='?', default='configs/default.toml')
    ap.add_argument('--no-mc', action='store_true')
    ap.add_argument('--no-fdm', action='store_true')
    ap.add_argument('--no-smooth', action='store_true')
    ap.add_argument('--smooth-tau', type=float, default=None,
                    help='override smoothing fraction (default: the '
                         'config\'s smooth_tau_frac, else 0.005)')
    ap.add_argument('--full-greeks', action='store_true',
                    help='Also compute vega1, vega2, rho '
                         '(adds ~6 spectral solves)')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args(argv)

    with open(args.config, 'rb') as f:
        cfg = tomllib.load(f)

    solver_kw = {}
    if args.smooth_tau is not None:
        solver_kw["smooth_tau_frac"] = args.smooth_tau
        solver_kw["smooth"] = args.smooth_tau > 0
    if args.no_smooth:
        solver_kw["smooth"] = False
    solver, kind = assemble_solver(cfg, **solver_kw)
    market, payoff, domain, basis = (solver.market, solver.payoff,
                                     solver.domain, solver.basis)
    run_dir = _next_run_dir(args.config)

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  twoasset — unified spectral pricer                       ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Config:   {args.config}")
    print(f"  Problem:  {kind}")
    print(f"  Domain:   {type(domain).__name__}")
    print(f"  Basis:    {type(basis).__name__}")
    print(f"  Output:   {run_dir}")
    print()
    print("  ── Market ──")
    print("  " + market.summary().replace("\n", "\n  "))
    print()

    # ── Spectral ──
    print("  ── Spectral ──")
    print(f"  Projection: {solver.projection_method}   "
          f"smoothing: {solver.smooth_tau_frac if solver.smooth else 0.0}")
    t0 = time.time()
    solver.fit()
    t_fit = time.time() - t0
    print(f"  Modes found: {solver.n_modes()} (fit {t_fit:.1f}s)")
    active = basis.active_modes()
    print(f"  Active modes (‖ψ‖>1e-6): {len(active)} "
          f"(dropped {solver.n_modes()-len(active)})")

    t0 = time.time()
    info = solver.project()
    t_proj = time.time() - t0
    print(f"  Projection: cond={info['cond']:.2e}, rank={info['rank']}, "
          f"δτ={info['smooth_tau']:.5f}  ({t_proj:.1f}s)")

    t0 = time.time()
    g = solver.greeks(full=args.full_greeks)
    t_eval = (time.time() - t0) * 1000
    print(f"  Price = {g['price']:.6f}")
    print(f"  Δ₁={g['delta1']:+.6f}  Δ₂={g['delta2']:+.6f}")
    print(f"  Θ ={g['theta']:+.6f}  Γ₁={g['gamma1']:+.6f}  "
          f"Γ₂={g['gamma2']:+.6f}  Γ₁₂={g['gamma12']:+.6f}")
    if args.full_greeks:
        print(f"  ν₁={g['vega1']:+.6f}  ν₂={g['vega2']:+.6f}  ρ ={g['rho']:+.6f}")
    print(f"  Eval: {t_eval:.1f} ms")
    print()

    mc_result = None
    if (not args.no_mc) and cfg.get("monte_carlo", {}).get("enabled", True):
        print("  ── Monte Carlo ──")
        mc_cfg = cfg["monte_carlo"]
        t0 = time.time()
        mc_result = mc_price(
            market, payoff, domain,
            n_paths=mc_cfg.get("n_paths", 500_000),
            n_steps=mc_cfg.get("n_steps", 500),
            seed=mc_cfg.get("seed", 42),
            use_brownian_bridge=mc_cfg.get("brownian_bridge", True),
            antithetic=mc_cfg.get("antithetic", True),
            control_variate=mc_cfg.get("control_variate", None),
        )
        flags = []
        if mc_result["brownian_bridge"]: flags.append("BB")
        if mc_result["antithetic"]:      flags.append("AV")
        if mc_result.get("control_variate"): flags.append(f"CV={mc_result['control_variate']}")
        flag_str = f"[{'+'.join(flags)}]" if flags else "[raw]"
        print(f"  Price = {mc_result['price']:.6f} ± {mc_result['std_err']:.6f}"
              f"  surv={mc_result['survival_rate']:.4f} {flag_str}"
              f"  ({time.time()-t0:.1f}s)")
        if "variance_reduction" in mc_result:
            print(f"  CV β={mc_result['cv_beta']:+.4f}  "
                  f"var-reduction={100*mc_result['variance_reduction']:.1f}%")
        print()

    fdm_result = None
    if (not args.no_fdm) and cfg.get("fdm", {}).get("enabled", True):
        print("  ── Finite difference ──")
        fdm_cfg = cfg["fdm"]
        N = fdm_cfg.get("N", fdm_cfg.get("N_mu", 160))
        Nt = fdm_cfg.get("N_time", 500)
        t0 = time.time()
        fdm_result = fdm_price(market, domain, payoff=payoff,
                               N=N, N_time=Nt)
        print(f"  Price = {fdm_result['price']:.6f}  "
              f"(N={N}, Nt={Nt}, {time.time()-t0:.1f}s)")
        print()

    # ── Summary table ──
    print("  ══════════════════════════════════════════════════════════")
    print("   Method          Price      Δ₁        Δ₂        Θ")
    print("  ──────────────────────────────────────────────────────────")
    print(f"   Spectral    {g['price']:10.4f}  {g['delta1']:+9.5f}  "
          f"{g['delta2']:+9.5f}  {g['theta']:+9.4f}")
    if mc_result:
        print(f"   Monte Carlo {mc_result['price']:10.4f}  "
              f"{'':>9s}  {'':>9s}  {'':>9s}  ±{mc_result['std_err']:.4f}")
    if fdm_result:
        print(f"   FDM (C-N)   {fdm_result['price']:10.4f}")
    print("  ══════════════════════════════════════════════════════════")
    if fdm_result and fdm_result['price'] > 0:
        rel = 100 * abs(g['price'] - fdm_result['price']) / fdm_result['price']
        print(f"  Spectral vs FDM: {rel:.4f}%")
    if mc_result and mc_result['price'] > 0:
        rel = 100 * abs(g['price'] - mc_result['price']) / mc_result['price']
        print(f"  Spectral vs MC:  {rel:.4f}%")

    # Save stats
    sp = os.path.join(run_dir, 'stats.toml')
    with open(sp, 'w') as f:
        f.write(f"# {kind}\n[spectral]\nprice = {g['price']:.8f}\n")
        for k in ['delta1', 'delta2', 'theta', 'gamma1', 'gamma2']:
            f.write(f"{k} = {g[k]:.8f}\n")
        f.write(f"n_modes = {solver.n_modes()}\n\n")
        if mc_result:
            f.write(f"[monte_carlo]\nprice = {mc_result['price']:.8f}\n"
                    f"std_err = {mc_result['std_err']:.8f}\n\n")
        if fdm_result:
            f.write(f"[fdm]\nprice = {fdm_result['price']:.8f}\n")
    print(f"\n  Stats: {sp}")
    print(f"  ✓ Run complete → {run_dir}")


if __name__ == "__main__":
    main()
