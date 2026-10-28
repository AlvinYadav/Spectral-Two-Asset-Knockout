"""FDM-based Greeks for the two-asset knockout basket call."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace

try:
    from ..core.market import GBMMarket
    from ..core.payoff import BasketCall
    from ..core.domain import EuclideanCircle
    from .fdm import fdm_price
except ImportError:
    import sys
    import pathlib
    # Add the repo root (two levels up from this file) to sys.path.
    _here = pathlib.Path(__file__).resolve()
    sys.path.insert(0, str(_here.parents[2]))
    from twoasset.core.market import GBMMarket
    from twoasset.core.payoff import BasketCall
    from twoasset.core.domain import EuclideanCircle
    from twoasset.solvers.fdm import fdm_price


# ═══════════════════════════════════════════════════════════════════════
#  Default config — matches configs/default.toml
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_MARKET = dict(
    S1_0=100.0, S2_0=100.0,
    sigma1=0.20, sigma2=0.30,
    rho=0.50, r=0.05, T=1.0,
)
DEFAULT_R = 0.5            # barrier radius in log-return space
DEFAULT_K = 180.0          # basket-call strike


# ═══════════════════════════════════════════════════════════════════════
#  Single-shot pricing wrapper with target-point flexibility
# ═══════════════════════════════════════════════════════════════════════

def _price(market: GBMMarket, domain: EuclideanCircle, payoff: BasketCall,
           *, N: int, N_time: int,
           S1: float | None = None, S2: float | None = None) -> float:
    """Thin wrapper around fdm_price returning just the scalar price."""
    return fdm_price(
        market, domain, payoff=payoff,
        N=N, N_time=N_time,
        S1_target=S1, S2_target=S2,
    )["price"]


# ═══════════════════════════════════════════════════════════════════════
#  Greeks
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Greeks:
    price:     float
    delta1:    float    # ∂V/∂S₁
    delta2:    float    # ∂V/∂S₂
    gamma11:   float    # ∂²V/∂S₁²
    gamma22:   float    # ∂²V/∂S₂²
    gamma12:   float    # ∂²V/∂S₁∂S₂
    theta:     float    # ∂V/∂t
    vega1:     float    # ∂V/∂σ₁  (per unit vol; divide by 100 for per-%)
    vega2:     float    # ∂V/∂σ₂
    rho:       float    # ∂V/∂r
    wall_s:    float    # total wall-clock time for all bumps

    def print(self) -> None:
        print(f"  V  = {self.price:10.6f}")
        print(f"  Δ₁ = {self.delta1:10.6f}    Δ₂ = {self.delta2:10.6f}")
        print(f"  Γ₁₁= {self.gamma11:10.6f}    Γ₂₂= {self.gamma22:10.6f}"
              f"    Γ₁₂= {self.gamma12:10.6f}")
        print(f"  Θ  = {self.theta:10.6f}   (per year)")
        print(f"  ν₁ = {self.vega1:10.6f}    ν₂ = {self.vega2:10.6f}"
              f"   (per unit vol)")
        print(f"  ρ  = {self.rho:10.6f}     (per unit rate)")
        print(f"  wall time: {self.wall_s:.1f} s over all FDM solves")


def _bumped_market(base: GBMMarket, **kw) -> GBMMarket:
    """Rebuild a GBMMarket with `kw` overriding the base fields."""
    raw = dict(
        S1_0=base.S1_0, S2_0=base.S2_0,
        sigma1=base.sigma1, sigma2=base.sigma2,
        rho=base.rho, r=base.r, T=base.T,
    )
    raw.update(kw)
    return GBMMarket(**raw)


def fdm_greeks(
    market: GBMMarket,
    domain: EuclideanCircle,
    payoff: BasketCall,
    *,
    N: int = 320,
    N_time: int = 500,
    bump_pct: float = 1.0,       # spot bump, percent of S
    vol_bump: float = 0.01,       # absolute vol bump
    rate_bump: float = 1e-4,      # absolute rate bump
    time_bump_frac: float = 0.01, # theta bump, fraction of T
    verbose: bool = True,
) -> Greeks:
    """Return finite-difference Greeks against the FDM solver."""
    t0 = time.time()

    h1 = market.S1_0 * bump_pct / 100.0
    h2 = market.S2_0 * bump_pct / 100.0

    def P(**kw) -> float:
        m = kw.pop("market", market)
        S1 = kw.pop("S1", None)
        S2 = kw.pop("S2", None)
        assert not kw, f"stray kw: {kw}"
        return _price(m, domain, payoff, N=N, N_time=N_time, S1=S1, S2=S2)

    V0 = P()

    # ── Delta₁, Gamma₁₁ (bump S₁ at fixed S₂) ────────────────────
    Vp1 = P(S1=market.S1_0 + h1)
    Vm1 = P(S1=market.S1_0 - h1)
    delta1  = (Vp1 - Vm1) / (2*h1)
    gamma11 = (Vp1 - 2*V0 + Vm1) / h1**2

    # ── Delta₂, Gamma₂₂ (bump S₂) ────────────────────────────────
    Vp2 = P(S2=market.S2_0 + h2)
    Vm2 = P(S2=market.S2_0 - h2)
    delta2  = (Vp2 - Vm2) / (2*h2)
    gamma22 = (Vp2 - 2*V0 + Vm2) / h2**2

    # ── Cross-gamma Γ₁₂ (4-point stencil) ────────────────────────
    Vpp = P(S1=market.S1_0 + h1, S2=market.S2_0 + h2)
    Vpm = P(S1=market.S1_0 + h1, S2=market.S2_0 - h2)
    Vmp = P(S1=market.S1_0 - h1, S2=market.S2_0 + h2)
    Vmm = P(S1=market.S1_0 - h1, S2=market.S2_0 - h2)
    gamma12 = (Vpp - Vpm - Vmp + Vmm) / (4*h1*h2)

    dt = time_bump_frac * market.T
    mT_minus = _bumped_market(market, T=market.T - dt)
    V_Tm = P(market=mT_minus)
    theta = -(V0 - V_Tm) / dt      # ∂V/∂t

    # ── Vegas (central difference in each σ) ─────────────────────
    m_sp1 = _bumped_market(market, sigma1=market.sigma1 + vol_bump)
    m_sm1 = _bumped_market(market, sigma1=market.sigma1 - vol_bump)
    vega1 = (P(market=m_sp1) - P(market=m_sm1)) / (2*vol_bump)

    m_sp2 = _bumped_market(market, sigma2=market.sigma2 + vol_bump)
    m_sm2 = _bumped_market(market, sigma2=market.sigma2 - vol_bump)
    vega2 = (P(market=m_sp2) - P(market=m_sm2)) / (2*vol_bump)

    # ── Rho ──────────────────────────────────────────────────────
    m_rp = _bumped_market(market, r=market.r + rate_bump)
    m_rm = _bumped_market(market, r=market.r - rate_bump)
    rho  = (P(market=m_rp) - P(market=m_rm)) / (2*rate_bump)

    wall = time.time() - t0

    g = Greeks(
        price=V0,
        delta1=delta1, delta2=delta2,
        gamma11=gamma11, gamma22=gamma22, gamma12=gamma12,
        theta=theta,
        vega1=vega1, vega2=vega2,
        rho=rho,
        wall_s=wall,
    )
    if verbose:
        print(f"FDM Greeks  (N={N}, N_time={N_time}, "
              f"bump_pct={bump_pct}%, vol_bump={vol_bump}, "
              f"rate_bump={rate_bump}, time_bump_frac={time_bump_frac})")
        print(f"  market: σ₁={market.sigma1}, σ₂={market.sigma2}, "
              f"ρ={market.rho}, r={market.r}, T={market.T}")
        print(f"  domain: {domain!r}, payoff: {payoff!r}")
        g.print()
    return g


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def fdm_greeks_converged(
    market: GBMMarket,
    domain: EuclideanCircle,
    payoff: BasketCall,
    *,
    N_coarse: int = 240,
    N_fine: int = 480,
    Nt_coarse: int = 400,
    Nt_fine: int = 800,
    **kw,
) -> tuple[Greeks, Greeks, dict]:
    """Run the Greeks twice — at (N_coarse, Nt_coarse) and (N_fine, Nt_fine) — and
    return both results plus a dict of Richardson error estimates."""
    kw.pop("verbose", None)
    print(f"── Convergence check: coarse (N={N_coarse}, Nt={Nt_coarse}) "
          f"→ fine (N={N_fine}, Nt={Nt_fine}) ──")

    print("  [coarse]")
    g_c = fdm_greeks(market, domain, payoff,
                     N=N_coarse, N_time=Nt_coarse, verbose=False, **kw)
    print(f"    V={g_c.price:.6f}  Δ₁={g_c.delta1:+.5f}  Δ₂={g_c.delta2:+.5f}  "
          f"Θ={g_c.theta:+.4f}  Γ₁₁={g_c.gamma11:+.5f}  Γ₁₂={g_c.gamma12:+.5f}")

    print("  [fine]")
    g_f = fdm_greeks(market, domain, payoff,
                     N=N_fine, N_time=Nt_fine, verbose=False, **kw)
    print(f"    V={g_f.price:.6f}  Δ₁={g_f.delta1:+.5f}  Δ₂={g_f.delta2:+.5f}  "
          f"Θ={g_f.theta:+.4f}  Γ₁₁={g_f.gamma11:+.5f}  Γ₁₂={g_f.gamma12:+.5f}")

    fields = ["price", "delta1", "delta2", "gamma11", "gamma22", "gamma12",
              "theta", "vega1", "vega2", "rho"]
    err = {f: abs(getattr(g_f, f) - getattr(g_c, f)) for f in fields}
    relerr = {f: (err[f] / abs(getattr(g_f, f)) if abs(getattr(g_f, f)) > 1e-12
                  else float("nan")) for f in fields}

    print("  ── Discretisation error (|fine − coarse|) ──")
    print(f"  {'field':<10}{'coarse':>12}{'fine':>12}{'|diff|':>12}{'rel':>10}")
    for f in fields:
        cv, fv = getattr(g_c, f), getattr(g_f, f)
        r = relerr[f]
        r_str = f"{100*r:>9.3f}%" if r == r else "     — "
        print(f"  {f:<10}{cv:>12.6f}{fv:>12.6f}{err[f]:>12.2e}{r_str}")

    return g_c, g_f, {"err": err, "relerr": relerr}


def _parse_args():
    ap = argparse.ArgumentParser(
        description="FDM finite-difference Greeks for the default "
                    "two-asset knockout basket call (default.toml).")
    ap.add_argument("--N",      type=int,   default=320,
                    help="FDM grid size (default 320, matches default.toml).")
    ap.add_argument("--Nt",     type=int,   default=500,
                    help="FDM time steps (default 500, matches default.toml).")
    ap.add_argument("--bump-pct",  type=float, default=1.0,
                    help="Spot bump as percent of spot (default 1.0).")
    ap.add_argument("--vol-bump",  type=float, default=0.01,
                    help="Absolute vol bump (default 0.01).")
    ap.add_argument("--rate-bump", type=float, default=1e-4,
                    help="Absolute rate bump (default 1e-4).")
    ap.add_argument("--time-bump-frac", type=float, default=0.01,
                    help="Theta bump as fraction of T (default 0.01).")
    ap.add_argument("--conv-check", action="store_true",
                    help="Run at two grid resolutions and report "
                         "Richardson discretisation error per field. "
                         "Doubles the runtime.")
    ap.add_argument("--N-coarse",   type=int, default=240)
    ap.add_argument("--N-fine",     type=int, default=480)
    ap.add_argument("--Nt-coarse",  type=int, default=400)
    ap.add_argument("--Nt-fine",    type=int, default=800)
    return ap.parse_args()


def main():
    args = _parse_args()
    market = GBMMarket(**DEFAULT_MARKET)
    domain = EuclideanCircle(R=DEFAULT_R)
    payoff = BasketCall(K=DEFAULT_K, w1=1.0, w2=1.0)

    if args.conv_check:
        fdm_greeks_converged(
            market, domain, payoff,
            N_coarse=args.N_coarse, N_fine=args.N_fine,
            Nt_coarse=args.Nt_coarse, Nt_fine=args.Nt_fine,
            bump_pct=args.bump_pct,
            vol_bump=args.vol_bump,
            rate_bump=args.rate_bump,
            time_bump_frac=args.time_bump_frac,
        )
        return

    fdm_greeks(
        market, domain, payoff,
        N=args.N, N_time=args.Nt,
        bump_pct=args.bump_pct,
        vol_bump=args.vol_bump,
        rate_bump=args.rate_bump,
        time_bump_frac=args.time_bump_frac,
    )


if __name__ == "__main__":
    main()
