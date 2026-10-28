"""Validation helpers, including the residual of the pricing equation."""

from __future__ import annotations

import argparse
import os
import time
import tomllib
from pathlib import Path

import numpy as np

from ..cli import assemble
from ..core.market_base import Market
from ..core.market import GBMMarket
from .fdm import fdm_price
from .mc import mc_price
from .spectral import SpectralSolver
from .fdm_greeks import fdm_greeks


# ═══════════════════════════════════════════════════════════════════════
#  PDE residual
# ═══════════════════════════════════════════════════════════════════════

def pde_residual(market: Market, S1: float, S2: float,
                 V: float, delta1: float, delta2: float,
                 gamma11: float, gamma22: float, gamma12: float,
                 theta: float) -> tuple[float, float]:
    """Return (residual, scale) where:"""
    s1, s2 = market.sigma1, market.sigma2
    rho, r = market.rho, market.r

    diff_term = (0.5 * s1*s1 * S1*S1 * gamma11
                 + 0.5 * s2*s2 * S2*S2 * gamma22
                 + rho * s1 * s2 * S1 * S2 * gamma12)
    drift_term = r * S1 * delta1 + r * S2 * delta2
    discount   = -r * V

    residual = theta + diff_term + drift_term + discount
    scale = max(abs(theta),
                abs(0.5 * s1*s1 * S1*S1 * gamma11),
                abs(0.5 * s2*s2 * S2*S2 * gamma22),
                abs(rho * s1 * s2 * S1 * S2 * gamma12),
                abs(r * S1 * delta1),
                abs(r * S2 * delta2),
                abs(r * V))
    return residual, scale


def local_vol_pde_residual(lvmarket, S1: float, S2: float, tau: float,
                           V: float, delta1: float, delta2: float,
                           gamma11: float, gamma22: float, gamma12: float,
                           theta: float) -> tuple[float, float]:
    """Residual of the pricing equation with state-dependent volatility."""
    x1 = np.log(S1 / lvmarket.S1_0)
    x2 = np.log(S2 / lvmarket.S2_0)
    Sig = lvmarket.Sigma_full(np.array([x1, x2]), tau)
    r = lvmarket.r
    S00, S01, S11 = Sig[0, 0], Sig[0, 1], Sig[1, 1]

    diff_term = (0.5 * S00 * S1*S1 * gamma11
                 + 0.5 * S11 * S2*S2 * gamma22
                 + S01 * S1 * S2 * gamma12)
    drift_term = r * S1 * delta1 + r * S2 * delta2
    discount = -r * V

    residual = theta + diff_term + drift_term + discount
    scale = max(abs(theta),
                abs(0.5 * S00 * S1*S1 * gamma11),
                abs(0.5 * S11 * S2*S2 * gamma22),
                abs(S01 * S1 * S2 * gamma12),
                abs(r * S1 * delta1),
                abs(r * S2 * delta2),
                abs(r * V))
    return residual, scale


# ═══════════════════════════════════════════════════════════════════════
#  Validation pipeline
# ═══════════════════════════════════════════════════════════════════════

def validate_config(
    config_path: str,
    *,
    seeds: list[int] = (1, 2, 3, 42, 100),
    n_paths: int = 2_000_000,
    fdm_coarse: tuple[int, int] = (320, 500),
    fdm_fine: tuple[int, int] = (640, 1000),
    do_fdm_greeks: bool = True,
    no_smooth: bool = False,
    do_weak_residual: bool = False,
    out_dir: str = "validation",
) -> dict:
    print(f"\n══ Validating {config_path} ══")

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    market, payoff, domain, basis, _ = assemble(cfg)

    out: dict = {
        "config": str(config_path),
        "config_name": Path(config_path).stem,
        "market": dict(
            S1_0=market.S1_0, S2_0=market.S2_0,
            sigma1=market.sigma1, sigma2=market.sigma2,
            rho=market.rho, r=market.r, T=market.T,
        ),
        "domain": type(domain).__name__,
        "basis": type(basis).__name__,
    }

    # ── 1) MC reference, multi-seed ────────────────────────────────
    print(f"  [MC] {len(seeds)} seeds × {n_paths:,} paths "
          "(BB+AV)…", flush=True)
    t0 = time.time()
    prices = []
    for s in seeds:
        r = mc_price(market, payoff, domain,
                     n_paths=n_paths, n_steps=500, seed=s,
                     use_brownian_bridge=True, antithetic=True)
        prices.append(r["price"])
        print(f"    seed={s:>3}  V = {r['price']:.6f} ± {r['std_err']:.6f}",
              flush=True)
    mc_mean = float(np.mean(prices))
    mc_sem  = float(np.std(prices, ddof=1) / np.sqrt(len(prices)))
    mc_dt = time.time() - t0
    print(f"  [MC] mean = {mc_mean:.6f} ± {mc_sem:.6f} "
          f"(SEM, {mc_dt:.0f}s)")
    out["mc"] = dict(seeds=list(seeds), n_paths=n_paths,
                     prices=prices, mean=mc_mean, sem=mc_sem,
                     wall_s=mc_dt)

    # ── 2) FDM price at two resolutions ────────────────────────────
    print(f"  [FDM] price convergence "
          f"({fdm_coarse} → {fdm_fine})…", flush=True)
    t0 = time.time()
    fdm_c = fdm_price(market, domain, payoff=payoff,
                       N=fdm_coarse[0], N_time=fdm_coarse[1])["price"]
    fdm_f = fdm_price(market, domain, payoff=payoff,
                       N=fdm_fine[0],  N_time=fdm_fine[1])["price"]
    fdm_dt = time.time() - t0
    print(f"    coarse {fdm_coarse}:  V = {fdm_c:.6f}")
    print(f"    fine   {fdm_fine}:    V = {fdm_f:.6f}")
    print(f"    |diff|: {abs(fdm_f - fdm_c):.2e}  ({fdm_dt:.0f}s)")
    out["fdm"] = dict(coarse_N=fdm_coarse[0], coarse_Nt=fdm_coarse[1],
                       coarse_price=fdm_c,
                       fine_N=fdm_fine[0], fine_Nt=fdm_fine[1],
                       fine_price=fdm_f,
                       grid_uncertainty=abs(fdm_f - fdm_c),
                       wall_s=fdm_dt)

    # ── 3) Spectral price + Greeks ─────────────────────────────────
    print(f"  [Spectral] fit + project + greeks…", flush=True)
    t0 = time.time()
    smooth_tau_frac = cfg.get("spectral", {}).get("smooth_tau_frac", 0.005)
    use_smooth = not no_smooth
    solver = SpectralSolver(market=market, payoff=payoff,
                            domain=domain, basis=basis,
                            smooth=use_smooth, smooth_tau_frac=smooth_tau_frac)
    solver.fit(); solver.project()
    g_spec = solver.greeks()
    sp_dt = time.time() - t0
    print(f"    V = {g_spec['price']:.6f}")
    print(f"    Δ₁={g_spec['delta1']:+.6f}  Δ₂={g_spec['delta2']:+.6f}")
    print(f"    Θ ={g_spec['theta']:+.6f}  Γ₁={g_spec['gamma1']:+.6f}  "
          f"Γ₂={g_spec['gamma2']:+.6f}  Γ₁₂={g_spec['gamma12']:+.6f}  "
          f"({sp_dt:.0f}s)")
    out["spectral"] = dict(g_spec, smooth=use_smooth,
                           smooth_tau_frac=smooth_tau_frac if use_smooth else 0.0,
                           wall_s=sp_dt)

    # ── 4) FDM Greeks (optional, slow) ────────────────────────────
    g_fdm = None
    if do_fdm_greeks:
        print(f"  [FDM Greeks] (this is the slow part)…", flush=True)
        t0 = time.time()
        g_fdm_obj = fdm_greeks(market, domain, payoff,
                                N=fdm_coarse[0], N_time=fdm_coarse[1],
                                verbose=False)
        fg_dt = time.time() - t0
        g_fdm = dict(price=g_fdm_obj.price,
                     delta1=g_fdm_obj.delta1, delta2=g_fdm_obj.delta2,
                     gamma1=g_fdm_obj.gamma11, gamma2=g_fdm_obj.gamma22,
                     gamma12=g_fdm_obj.gamma12,
                     theta=g_fdm_obj.theta,
                     vega1=g_fdm_obj.vega1, vega2=g_fdm_obj.vega2,
                     rho=g_fdm_obj.rho,
                     N=fdm_coarse[0], N_time=fdm_coarse[1],
                     wall_s=fg_dt)
        out["fdm_greeks"] = g_fdm
        print(f"    Δ₁={g_fdm['delta1']:+.6f}  Δ₂={g_fdm['delta2']:+.6f}")
        print(f"    Θ ={g_fdm['theta']:+.6f}  Γ₁={g_fdm['gamma1']:+.6f}  "
              f"Γ₂={g_fdm['gamma2']:+.6f}  Γ₁₂={g_fdm['gamma12']:+.6f}  "
              f"({fg_dt:.0f}s)")

    # ── 5) PDE residuals ──────────────────────────────────────────
    print(f"  [PDE residual]")
    S1, S2 = market.S1_0, market.S2_0
    res_sp, scale_sp = pde_residual(
        market, S1, S2,
        V=g_spec["price"],
        delta1=g_spec["delta1"], delta2=g_spec["delta2"],
        gamma11=g_spec["gamma1"], gamma22=g_spec["gamma2"],
        gamma12=g_spec["gamma12"], theta=g_spec["theta"],
    )
    rel_sp = abs(res_sp) / scale_sp
    print(f"    spectral:  |r| = {abs(res_sp):.3e}   "
          f"|r|/scale = {rel_sp:.3e}")
    out["pde_residual"] = {
        "spectral": dict(residual=float(res_sp), scale=float(scale_sp),
                          relative=float(rel_sp)),
    }
    if g_fdm is not None:
        res_fd, scale_fd = pde_residual(
            market, S1, S2,
            V=g_fdm["price"],
            delta1=g_fdm["delta1"], delta2=g_fdm["delta2"],
            gamma11=g_fdm["gamma1"], gamma22=g_fdm["gamma2"],
            gamma12=g_fdm["gamma12"], theta=g_fdm["theta"],
        )
        rel_fd = abs(res_fd) / scale_fd
        print(f"    FDM:       |r| = {abs(res_fd):.3e}   "
              f"|r|/scale = {rel_fd:.3e}")
        out["pde_residual"]["fdm"] = dict(
            residual=float(res_fd), scale=float(scale_fd),
            relative=float(rel_fd))

    if do_weak_residual:
        from .weak_residual import (
            pointwise_residual_grid, weak_residual_field,
            l2_weak_residual, modal_projections,
        )
        print(f"  [Weak residual]")
        wr_grid = pointwise_residual_grid(solver, n_grid=15)
        wr_field = weak_residual_field(solver, n_grid=15)
        l2 = l2_weak_residual(wr_field)
        proj = modal_projections(basis, wr_field)
        max_proj = max(abs(v) for v in proj.values()) if proj else float("nan")
        print(f"    L2/domain = {l2:.3e}   max|modal proj| = {max_proj:.3e}   "
              f"grid-max pointwise = {float(np.nanmax(wr_grid['relative'])):.3e}")
        out["weak_residual"] = dict(
            l2_over_domain=float(l2),
            max_modal_projection=float(max_proj),
            pointwise_at_spot=float(rel_sp),
            grid_max_pointwise_relative=float(np.nanmax(wr_grid["relative"])),
        )

    # ── 6) Cross-method price agreement table ─────────────────────
    print(f"  ── Price agreement ──")
    refs = [
        ("MC mean (5 seeds, 2M)", mc_mean, mc_sem),
        ("FDM fine",              fdm_f,   abs(fdm_f - fdm_c)),
        ("Spectral",              g_spec["price"], 0.0),
    ]
    for name, V, unc in refs:
        delta_mc = (V - mc_mean) / mc_mean * 100
        in_band = "*" if abs(V - mc_mean) <= 2.0 * mc_sem else " "
        print(f"   {name:<24} {V:11.6f} ± {unc:.4f}    "
              f"Δ_MC = {delta_mc:+.4f}%  {in_band}")
    out["agreement"] = dict(
        mc_mean=mc_mean, mc_sem=mc_sem,
        fdm_fine=fdm_f, fdm_uncertainty=abs(fdm_f - fdm_c),
        spectral=g_spec["price"],
        spec_minus_mc_pct=(g_spec["price"] - mc_mean) / mc_mean * 100,
        fdm_minus_mc_pct=(fdm_f - mc_mean) / mc_mean * 100,
    )

    # ── 7) Persist ─────────────────────────────────────────────────
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{out['config_name']}.toml"
    _write_toml(out, out_path)
    print(f"  → wrote {out_path}")
    return out


def _write_toml(d: dict, path: Path) -> None:
    """Write a flat TOML dump."""
    def fmt(v):
        if isinstance(v, bool):  return "true" if v else "false"
        if isinstance(v, float): return f"{v:.10g}"
        if isinstance(v, int):   return str(v)
        if isinstance(v, str):   return f'"{v}"'
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        raise TypeError(f"Unsupported value: {type(v)}: {v!r}")

    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append("")
            lines.append(f"[{k}]")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    lines.append("")
                    lines.append(f"[{k}.{kk}]")
                    for kkk, vvv in vv.items():
                        lines.append(f"{kkk} = {fmt(vvv)}")
                else:
                    lines.append(f"{kk} = {fmt(vv)}")
        else:
            lines.append(f"{k} = {fmt(v)}")
    path.write_text("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/default.toml",
                    help="Config file (default: configs/default.toml)")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[1, 2, 3, 42, 100],
                    help="MC seeds (default: 1 2 3 42 100)")
    ap.add_argument("--n-paths", type=int, default=2_000_000,
                    help="MC paths per seed (default: 2,000,000)")
    ap.add_argument("--fdm-coarse-N",  type=int, default=320)
    ap.add_argument("--fdm-coarse-Nt", type=int, default=500)
    ap.add_argument("--fdm-fine-N",    type=int, default=640)
    ap.add_argument("--fdm-fine-Nt",   type=int, default=1000)
    ap.add_argument("--no-fdm-greeks", action="store_true",
                    help="Skip the slow FDM Greeks computation "
                         "(13 FDM solves at the coarse resolution)")
    ap.add_argument("--no-smooth", action="store_true",
                    help="Disable FDM pre-smoothing in the spectral solver")
    ap.add_argument("--weak-residual", action="store_true",
                    help="Also compute the weak/gauged residual "
                         "(L2-over-domain + modal projections)")
    ap.add_argument("--out-dir", default="validation",
                    help="Output directory (default: validation/)")
    return ap.parse_args()


def main():
    args = _parse_args()
    validate_config(
        args.config,
        seeds=args.seeds,
        n_paths=args.n_paths,
        fdm_coarse=(args.fdm_coarse_N, args.fdm_coarse_Nt),
        fdm_fine=(args.fdm_fine_N, args.fdm_fine_Nt),
        do_fdm_greeks=not args.no_fdm_greeks,
        do_weak_residual=args.weak_residual,
        no_smooth=args.no_smooth,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
