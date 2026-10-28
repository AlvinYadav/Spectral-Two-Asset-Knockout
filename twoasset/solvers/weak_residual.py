"""Pointwise and Galerkin residuals of the spectral expansions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..bases.mathieu import MathieuBasis
from ..bases.bessel import BesselBasis
from ..bases.fourier import FourierBasis
from ._optimal import ALL_CONFIGS, PAPER_FIGURES_DIR, build_solver, load_mc_reference


# ═══════════════════════════════════════════════════════════════════════
#  Shared grid + quadrature-weight helpers
# ═══════════════════════════════════════════════════════════════════════

def _natural_grid(basis, n_grid: int, margin: float = 0.05):
    """Build a FRESH, independent n_grid x n_grid tensor grid in the basis's own
    natural coordinates, at the requested resolution — NOT a subsample of the
    fitting-time quadrature grid."""
    if isinstance(basis, MathieuBasis):
        mu = np.linspace(margin * basis.mu_0, (1 - margin) * basis.mu_0, n_grid)
        nu = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
        dmu = mu[1] - mu[0] if n_grid > 1 else basis.mu_0
        dnu = 2 * np.pi / n_grid
        mu_flat = np.repeat(mu, n_grid)
        nu_flat = np.tile(nu, n_grid)
        S1 = np.empty(n_grid * n_grid)
        S2 = np.empty(n_grid * n_grid)
        for i in range(len(mu_flat)):
            S1[i], S2[i] = basis._elliptic_to_spot(mu_flat[i], nu_flat[i])
        h2 = basis.c_focal ** 2 * (np.sinh(mu_flat) ** 2 + np.sin(nu_flat) ** 2)
        weights = dmu * dnu * h2
        return S1, S2, weights

    if isinstance(basis, BesselBasis):
        r = np.linspace(margin * basis.R, (1 - margin) * basis.R, n_grid)
        theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
        dr = r[1] - r[0] if n_grid > 1 else basis.R
        dtheta = 2 * np.pi / n_grid
        r_flat = np.repeat(r, n_grid)
        th_flat = np.tile(theta, n_grid)
        S1 = np.empty(n_grid * n_grid)
        S2 = np.empty(n_grid * n_grid)
        for i in range(len(r_flat)):
            S1[i], S2[i] = basis._polar_to_spot(r_flat[i], th_flat[i])
        weights = dr * dtheta * r_flat
        return S1, S2, weights

    if isinstance(basis, FourierBasis):
        x1_lo = basis.x1_lo + margin * basis.W1
        x1_hi = basis.x1_lo + basis.W1 - margin * basis.W1
        x2_lo = basis.x2_lo + margin * basis.W2
        x2_hi = basis.x2_lo + basis.W2 - margin * basis.W2
        x1 = np.linspace(x1_lo, x1_hi, n_grid)
        x2 = np.linspace(x2_lo, x2_hi, n_grid)
        dx1 = x1[1] - x1[0] if n_grid > 1 else basis.W1
        dx2 = x2[1] - x2[0] if n_grid > 1 else basis.W2
        x1_flat = np.repeat(x1, n_grid)
        x2_flat = np.tile(x2, n_grid)
        S1 = basis.market.S1_0 * np.exp(x1_flat)
        S2 = basis.market.S2_0 * np.exp(x2_flat)
        weights = np.full(n_grid * n_grid, dx1 * dx2)
        return S1, S2, weights

    raise ValueError(f"unrecognised basis type for natural grid: {type(basis)}")


# ═══════════════════════════════════════════════════════════════════════
#  Pointwise residual grid (reuses validate.pde_residual verbatim)
# ═══════════════════════════════════════════════════════════════════════

def pointwise_residual_grid(solver, *, n_grid: int | None = 15) -> dict:
    """Evaluate the pointwise relative BS-PDE residual over a grid inside the
    survival domain."""
    from .validate import pde_residual

    basis = solver.basis
    S1_sub, S2_sub, w_sub = _natural_grid(basis, n_grid or 15)
    n = len(S1_sub)

    residual = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    relative = np.full(n, np.nan)
    inside = np.zeros(n, dtype=bool)

    for j in range(n):
        S1, S2 = float(S1_sub[j]), float(S2_sub[j])
        if not solver._inside(S1, S2):
            continue
        inside[j] = True
        g = solver.greeks(S1, S2, t=0.0)
        res, sc = pde_residual(
            solver.market, S1, S2, V=g["price"],
            delta1=g["delta1"], delta2=g["delta2"],
            gamma11=g["gamma1"], gamma22=g["gamma2"],
            gamma12=g["gamma12"], theta=g["theta"],
        )
        residual[j] = res
        scale[j] = sc
        relative[j] = abs(res) / sc if sc > 0 else np.nan

    return dict(S1=S1_sub, S2=S2_sub, weights=w_sub, inside=inside,
                residual=residual, scale=scale, relative=relative)


# ═══════════════════════════════════════════════════════════════════════
#  Independent FD Laplacian in z-space (the "weak residual" second term)
# ═══════════════════════════════════════════════════════════════════════

def _stencil_valid(u0: float, neighbours: list[float]) -> bool:
    """Detects the failure mode where a central-difference neighbour lands outside
    the survival domain: `eval_expansion` (via `psi_at_spot`) returns an EXACT 0.0
    for any out-of-domain point (Dirichlet clip), which corrupts a
    second-derivative stencil with a spurious kink at domain- adjacent collocation
    nodes — not a real PDE defect."""
    if abs(u0) < 1e-12:
        return True
    return not any(nv == 0.0 for nv in neighbours)


def _fd_laplacian_z(basis, market, S1: float, S2: float, tau_eff: float,
                     eps: float) -> tuple[float, bool]:
    """½Δ_z u at (S1,S2), via a 5-point central-difference stencil in WHITENED
    z-space."""
    x1, x2 = market.spot_to_x(S1, S2)
    z1, z2 = market.x_to_z(x1, x2)

    def u_at_z(zz1, zz2):
        xx1, xx2 = market.z_to_x(zz1, zz2)
        SS1, SS2 = market.x_to_spot(xx1, xx2)
        return basis.eval_expansion(float(SS1), float(SS2), tau_eff)

    u0 = u_at_z(z1, z2)
    up1, um1 = u_at_z(z1 + eps, z2), u_at_z(z1 - eps, z2)
    up2, um2 = u_at_z(z1, z2 + eps), u_at_z(z1, z2 - eps)
    valid = _stencil_valid(u0, [up1, um1, up2, um2])
    d2u_dz1 = (up1 - 2 * u0 + um1) / eps ** 2
    d2u_dz2 = (up2 - 2 * u0 + um2) / eps ** 2
    return 0.5 * (d2u_dz1 + d2u_dz2), valid


def _fd_generator_x(basis, market, S1: float, S2: float, tau_eff: float,
                     eps: float) -> tuple[float, bool]:
    """a11·u_x1x1 + a22·u_x2x2 + a12·u_x1x2 at (S1,S2), via FD directly in x-space."""
    x1, x2 = market.spot_to_x(S1, S2)

    def u_at_x(xx1, xx2):
        SS1 = market.S1_0 * np.exp(xx1)
        SS2 = market.S2_0 * np.exp(xx2)
        return basis.eval_expansion(float(SS1), float(SS2), tau_eff)

    u0 = u_at_x(x1, x2)
    u_x1x1_p, u_x1x1_m = u_at_x(x1 + eps, x2), u_at_x(x1 - eps, x2)
    u_x2x2_p, u_x2x2_m = u_at_x(x1, x2 + eps), u_at_x(x1, x2 - eps)
    u_pp = u_at_x(x1 + eps, x2 + eps)
    u_pm = u_at_x(x1 + eps, x2 - eps)
    u_mp = u_at_x(x1 - eps, x2 + eps)
    u_mm = u_at_x(x1 - eps, x2 - eps)
    valid = _stencil_valid(
        u0, [u_x1x1_p, u_x1x1_m, u_x2x2_p, u_x2x2_m, u_pp, u_pm, u_mp, u_mm])
    u_x1x1 = (u_x1x1_p - 2 * u0 + u_x1x1_m) / eps ** 2
    u_x2x2 = (u_x2x2_p - 2 * u0 + u_x2x2_m) / eps ** 2
    u_x1x2 = (u_pp - u_pm - u_mp + u_mm) / (4 * eps ** 2)
    return (market.a11 * u_x1x1 + market.a22 * u_x2x2 + market.a12 * u_x1x2,
            valid)


def fd_generator_term(basis, market, S1: float, S2: float, tau_eff: float,
                       eps: float = 1e-3) -> tuple[float, bool]:
    """The FD-independent second term of the weak/gauged residual, using whichever
    second-order operator the basis's OWN eigenproblem actually diagonalises
    (isotropic ½Δ_z for Mathieu/Bessel; anisotropic a11∂x1x1+a22∂x2x2+a12∂x1x2 for
    Fourier) — see `_fd_laplacian_z` / `_fd_generator_x` docstrings."""
    if isinstance(basis, (MathieuBasis, BesselBasis)):
        return _fd_laplacian_z(basis, market, S1, S2, tau_eff, eps)
    if isinstance(basis, FourierBasis):
        return _fd_generator_x(basis, market, S1, S2, tau_eff, eps)
    raise ValueError(f"unrecognised basis type: {type(basis)}")


def _fourier_theta_and_generator_batch(basis, market, S1_arr, S2_arr, tau_eff):
    """Vectorised, EXACT (non-FD) (theta_analytic, generator_term, valid) for a whole
    batch of points at once, for FourierBasis only."""
    S1_arr = np.asarray(S1_arr, dtype=float)
    S2_arr = np.asarray(S2_arr, dtype=float)
    x1_arr, x2_arr = market.spot_to_x(S1_arr, S2_arr)
    inside = np.asarray(basis.domain.contains(x1_arr, x2_arr), dtype=bool)

    N = basis.N_modes
    n_arr = np.arange(1, N + 1)
    p1 = n_arr * np.pi / basis.W1
    p2 = n_arr * np.pi / basis.W2

    ang1 = n_arr[None, :] * np.pi * (x1_arr[:, None] - basis.x1_lo) / basis.W1
    ang2 = n_arr[None, :] * np.pi * (x2_arr[:, None] - basis.x2_lo) / basis.W2
    sin1, cos1 = np.sin(ang1), np.cos(ang1)
    sin2, cos2 = np.sin(ang2), np.cos(ang2)

    M = len(S1_arr)
    ss = (sin1[:, :, None] * sin2[:, None, :]).reshape(M, N * N)
    cc = (cos1[:, :, None] * cos2[:, None, :]).reshape(M, N * N)
    pref_x1 = np.repeat(p1, N)
    pref_x2 = np.tile(p2, N)

    L_raw = (-market.a11 * pref_x1 ** 2 * ss
             - market.a22 * pref_x2 ** 2 * ss
             + market.a12 * pref_x1 * pref_x2 * cc)

    coeffs_psi = np.array([md.coeff for md in basis.modes])
    ksq = np.array([md.k_sq for md in basis.modes])
    decays = np.exp(-0.5 * ksq * tau_eff)
    norm_common = basis.modes[0].norm

    c_raw = (basis._evecs @ (coeffs_psi * decays)) / norm_common
    c_raw_tau = (basis._evecs @ (coeffs_psi * decays * (-0.5 * ksq))) / norm_common

    theta_analytic = ss @ c_raw_tau
    generator = L_raw @ c_raw

    theta_analytic = np.where(inside, theta_analytic, 0.0)
    generator = np.where(inside, generator, 0.0)
    return theta_analytic, generator, inside


def weak_residual_field(solver, *, n_grid: int | None = 15, eps: float = 1e-3,
                         batch_size: int = 4000) -> dict:
    """r(z) = theta_sum(analytic) − generator_term(...) over a grid, plus the raw
    analytic theta_sum (kept only as an internal self-consistency reference — NOT
    the gated quantity)."""
    basis = solver.basis
    market = solver.market
    tau_eff = market.T - solver.smooth_tau

    S1_sub, S2_sub, w_sub = _natural_grid(basis, n_grid or 15)
    n = len(S1_sub)

    if isinstance(basis, FourierBasis):
        theta_analytic = np.empty(n)
        lap_fd = np.empty(n)
        valid = np.empty(n, dtype=bool)
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            theta_analytic[sl], lap_fd[sl], valid[sl] = \
                _fourier_theta_and_generator_batch(
                    basis, market, S1_sub[sl], S2_sub[sl], tau_eff)
    else:
        theta_analytic = np.empty(n)
        lap_fd = np.empty(n)
        valid = np.empty(n, dtype=bool)
        for j in range(n):
            S1, S2 = float(S1_sub[j]), float(S2_sub[j])
            _, _, _, theta_sum = basis.eval_with_derivatives(S1, S2, tau_eff)
            theta_analytic[j] = theta_sum
            lap_fd[j], valid[j] = fd_generator_term(basis, market, S1, S2, tau_eff, eps=eps)

    residual = theta_analytic - lap_fd
    residual[~valid] = np.nan
    w_sub = np.where(valid, w_sub, 0.0)
    n_dropped = int(np.sum(~valid))
    if n_dropped:
        print(f"  weak_residual_field: dropped {n_dropped}/{n} boundary-"
              f"adjacent points (outside domain / FD stencil clipped)")

    return dict(S1=S1_sub, S2=S2_sub, weights=w_sub,
                theta_analytic=theta_analytic, lap_fd=lap_fd,
                residual=residual, valid=valid, tau_eff=tau_eff, eps=eps)


def fourier_galerkin_self_consistency(basis, market) -> dict:
    """Exact, non-tautological substitute for a grid-quadrature weak- residual gate
    on FourierBasis."""
    from ..bases.fourier import _sc_integral

    N = basis.N_modes
    raw_ml = basis._raw_ml
    M = len(raw_ml)
    diag = np.array([market.a11 * (n * np.pi / basis.W1) ** 2
                     + market.a22 * (k * np.pi / basis.W2) ** 2
                     for n, k in raw_ml])
    H = np.diag(diag)
    if abs(market.a12) > 1e-15:
        nf = 4.0 / (basis.W1 * basis.W2)
        sc1 = np.array([_sc_integral(a[0], b[0]) for a in raw_ml for b in raw_ml]).reshape(M, M)
        sc2 = np.array([_sc_integral(a[1], b[1]) for a in raw_ml for b in raw_ml]).reshape(M, M)
        off = -market.a12 * nf * sc1 * sc2
        np.fill_diagonal(off, 0.0)
        H = H + off

    evals, evecs = basis._evals, basis._evecs
    active_idx = [k for k, md in enumerate(basis.modes) if md.norm > 1e-6]
    resid = np.empty(len(active_idx))
    for j, k in enumerate(active_idx):
        v = evecs[:, k]
        r = H @ v - evals[k] * v
        vnorm = np.linalg.norm(v)
        resid[j] = np.linalg.norm(r) / vnorm if vnorm > 0 else np.linalg.norm(r)

    return dict(max_residual=float(np.max(resid)), mean_residual=float(np.mean(resid)),
                n_checked=len(active_idx))


# ═══════════════════════════════════════════════════════════════════════
#  Aggregate scalars
# ═══════════════════════════════════════════════════════════════════════

def l2_weak_residual(field: dict) -> float:
    """||r||_L2(Omega) / scale, domain-averaged scale convention: scale = Sum(w_i *
    max(|theta_analytic_i|, |lap_fd_i|)) / Sum(w_i)."""
    valid = field.get("valid", np.isfinite(field["residual"]))
    w, r = field["weights"][valid], field["residual"][valid]
    scale_i = np.maximum(np.abs(field["theta_analytic"][valid]),
                         np.abs(field["lap_fd"][valid]))
    scale_i = np.maximum(scale_i, 1e-300)
    W = np.sum(w)
    scale = np.sum(w * scale_i) / W
    l2 = np.sqrt(np.sum(w * r ** 2) / W)
    return float(l2 / scale) if scale > 0 else float("nan")


def modal_projections(basis, field: dict) -> dict:
    """<Psi_m, r> / ||Psi_m||^2 for every retained (active) mode, via the
    already-normalised design matrix Psi_m/||Psi_m|| dotted against the weighted
    residual field."""
    valid = field.get("valid", np.isfinite(field["residual"]))
    S1, S2 = field["S1"][valid], field["S2"][valid]
    w, r = field["weights"][valid], field["residual"][valid]
    A = basis.design_matrix_at_spot(S1, S2)
    active_idx = [k for k, md in enumerate(basis.modes) if md.norm > 1e-6]
    proj = {}
    for k in active_idx:
        norm_m = basis.modes[k].norm
        proj[k] = float(np.sum(w * A[:, k] * r) / norm_m)
    return proj


def weak_residual_table_row(config_name: str, pointwise_at_spot: float,
                             l2_over_domain: float, max_modal_projection: float,
                             galerkin_self_consistency: float | None = None) -> dict:
    return dict(config=config_name, pointwise_at_spot=pointwise_at_spot,
                l2_over_domain=l2_over_domain,
                max_modal_projection=max_modal_projection,
                galerkin_self_consistency=galerkin_self_consistency)


# ═══════════════════════════════════════════════════════════════════════
#  Spatial residual map (spread, worst_of only)
# ═══════════════════════════════════════════════════════════════════════

def plot_residual_heatmap(config_name: str, S1_grid, S2_grid, residual_abs,
                           payoff, out_dir: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = PAPER_FIGURES_DIR
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.tricontourf(S1_grid, S2_grid, np.log10(np.maximum(residual_abs, 1e-16)),
                        levels=20, cmap="Blues")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10 |r~(S1,S2)|")

    s1_lo, s1_hi = float(np.min(S1_grid)), float(np.max(S1_grid))
    s2_lo, s2_hi = float(np.min(S2_grid)), float(np.max(S2_grid))
    xs = np.linspace(s1_lo, s1_hi, 200)
    if hasattr(payoff, "K") and hasattr(payoff, "w1") and hasattr(payoff, "w2") \
            and not hasattr(payoff, "S1_0"):
        # Spread: w1*S1 - w2*S2 = K  =>  S2 = (w1*S1 - K)/w2
        ys = (payoff.w1 * xs - payoff.K) / payoff.w2
    elif hasattr(payoff, "S1_0") and hasattr(payoff, "S2_0"):
        # WorstOf: S1/S1_0 = S2/S2_0  =>  S2 = (S2_0/S1_0)*S1
        ys = (payoff.S2_0 / payoff.S1_0) * xs
    else:
        ys = None
    if ys is not None:
        mask = (ys >= s2_lo) & (ys <= s2_hi)
        ax.plot(xs[mask], ys[mask], color="#E24B4A", lw=1.6, label="payoff kink locus")
        ax.legend(fontsize=8)

    ax.set_xlabel("S1")
    ax.set_ylabel("S2")
    ax.set_title(f"{config_name}: spectral PDE residual (log10 |r~|)")
    plt.tight_layout()

    pdf_path = Path(out_dir) / f"{config_name}_residual_heatmap.pdf"
    png_path = Path(out_dir) / f"{config_name}_residual_heatmap.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return pdf_path, png_path


# ═══════════════════════════════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════════════════════════════

def run_weak_residual_study(
    config_names: tuple[str, ...] = ALL_CONFIGS,
    n_grid: int = 15,
    heatmap_configs: tuple[str, ...] = ("spread", "worst_of"),
    heatmap_n_grid: int = 30,
    out_dir: str = None,
) -> dict:
    if out_dir is None:
        out_dir = PAPER_FIGURES_DIR
    print("== Weak-residual metric and spatial residual maps ==")
    results = {}
    for name in config_names:
        solver, market, payoff, domain, basis = build_solver(name)
        S1_0, S2_0 = market.S1_0, market.S2_0

        pw_grid = pointwise_residual_grid(solver, n_grid=n_grid)
        wr_field = weak_residual_field(solver, n_grid=n_grid)
        l2 = l2_weak_residual(wr_field)
        proj = modal_projections(basis, wr_field)
        max_proj = max(abs(v) for v in proj.values()) if proj else float("nan")

        galerkin_sc = None
        if isinstance(basis, FourierBasis):
            galerkin_sc = fourier_galerkin_self_consistency(basis, market)["max_residual"]

        g_spot = solver.greeks(S1_0, S2_0, t=0.0)
        from .validate import pde_residual
        res_spot, scale_spot = pde_residual(
            market, S1_0, S2_0, V=g_spot["price"],
            delta1=g_spot["delta1"], delta2=g_spot["delta2"],
            gamma11=g_spot["gamma1"], gamma22=g_spot["gamma2"],
            gamma12=g_spot["gamma12"], theta=g_spot["theta"],
        )
        pointwise_at_spot = abs(res_spot) / scale_spot if scale_spot > 0 else float("nan")

        row = weak_residual_table_row(name, pointwise_at_spot, l2, max_proj, galerkin_sc)
        results[name] = dict(row, n_active_modes=len(basis.active_modes()),
                             grid_max_pointwise_relative=float(
                                 np.nanmax(pw_grid["relative"])))
        gsc_str = f"  galerkin_self_consistency={galerkin_sc:.3e}" if galerkin_sc is not None else ""
        print(f"  {name:>16}: pointwise@spot={pointwise_at_spot:.3e}  "
              f"L2/domain(grid,documented)={l2:.3e}  "
              f"max|modal proj|(grid,documented)={max_proj:.3e}{gsc_str}  "
              f"grid-max pointwise={results[name]['grid_max_pointwise_relative']:.3e}")

        if name in heatmap_configs:
            hm_grid = pointwise_residual_grid(solver, n_grid=heatmap_n_grid)
            mask = hm_grid["inside"] & np.isfinite(hm_grid["relative"])
            plot_residual_heatmap(
                name, hm_grid["S1"][mask], hm_grid["S2"][mask],
                np.abs(hm_grid["relative"][mask]), payoff, out_dir=out_dir,
            )

    print("\n-- Weak-residual table (Fourier configs) --")
    print("(L2/domain and max|modal proj| are grid-quadrature diagnostics —")
    print(" documented, not gated: a Riemann-sum grid can't resolve high-")
    print(" frequency sine/cosine products to 1e-8. galerkin_self_consistency")
    print(" is the gated quantity — see fourier_galerkin_self_consistency docstring.)")
    print(f"{'config':>12} {'pointwise@spot':>16} {'L2/domain':>12} "
          f"{'max|modal proj|':>16} {'galerkin_self_consist':>22}")
    for name in ("spread", "worst_of"):
        if name in results:
            r = results[name]
            gsc = r["galerkin_self_consistency"]
            gsc_str = f"{gsc:.3e}" if gsc is not None else "n/a"
            print(f"{name:>12} {r['pointwise_at_spot']:>16.3e} "
                  f"{r['l2_over_domain']:>12.3e} {r['max_modal_projection']:>16.3e} "
                  f"{gsc_str:>22}")

    Path("validation").mkdir(parents=True, exist_ok=True)
    with open("validation/weak_residual_study.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\n  -> wrote validation/weak_residual_study.json")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Strong-vs-weak evidence: per-mode operator eigenrelation residual
# ═══════════════════════════════════════════════════════════════════════

def per_mode_operator_residual(solver, *, n_modes: int = 8, n_points: int = 24,
                               eps: float = 2e-3, seed: int = 0) -> dict:
    """How well does each SINGLE basis function satisfy its own operator
    eigenrelation, pointwise, at interior points?"""
    from ..bases.mathieu import MathieuBasis
    from ..bases.bessel import BesselBasis

    basis, market = solver.basis, solver.market
    isotropic = isinstance(basis, (MathieuBasis, BesselBasis))
    op = _fd_laplacian_z if isotropic else _fd_generator_x

    S1_flat, S2_flat, _w = _natural_grid(
        basis, int(np.ceil(np.sqrt(n_points))) + 2, margin=0.22)
    pts = [(float(a), float(b)) for a, b in zip(S1_flat, S2_flat)]
    rng = np.random.default_rng(seed)
    if len(pts) > n_points:
        pts = [pts[i] for i in rng.choice(len(pts), n_points, replace=False)]

    active = [md for md in basis.modes if md.norm > 1e-6]
    active = sorted(active, key=lambda m: m.k_sq)[:n_modes]
    saved = [md.coeff for md in basis.modes]

    rows = []
    try:
        for md in active:
            for other in basis.modes:
                other.coeff = 0.0
            md.coeff = 1.0
            num_sq = den_sq = 0.0
            n_valid = 0
            half_k = 0.5 * md.k_sq
            for (S1, S2) in pts:
                psi = basis.eval_expansion(S1, S2, 0.0)
                val, ok = op(basis, market, S1, S2, 0.0, eps)
                if not ok or not np.isfinite(val) or not np.isfinite(psi):
                    continue
                num_sq += (val + half_k * psi) ** 2
                den_sq += (half_k * psi) ** 2
                n_valid += 1
            if n_valid >= 3 and den_sq > 0:
                rows.append(dict(idx=str(md.idx), k_sq=float(md.k_sq),
                                 residual=float(np.sqrt(num_sq / den_sq)),
                                 n_points=n_valid))
    finally:
        for md, c in zip(basis.modes, saved):
            md.coeff = c

    res = [r["residual"] for r in rows]
    return dict(rows=rows, n_modes=len(rows),
                operator="isotropic_laplacian_z" if isotropic
                         else "anisotropic_generator_x",
                median=float(np.median(res)) if res else float("nan"),
                worst=float(np.max(res)) if res else float("nan"))
