"""Monte Carlo pricer for two-asset knockout options with Brownian-bridge barrier
correction."""

from __future__ import annotations
import numpy as np
from scipy.stats import norm

from ..core.domain import EuclideanCircle, MahalanobisEllipse, Rectangle


def _bridge_survive_halfplane(d1, d2, var_dt):
    """Survival probability of a 1-D Brownian bridge on a half-plane."""
    alive = (d1 > 0) & (d2 > 0)
    p_hit = np.zeros_like(d1, dtype=float)
    # var_dt may be scalar or per-path; index if it's an array.
    if np.isscalar(var_dt) or np.ndim(var_dt) == 0:
        p_hit[alive] = np.exp(-2.0 * d1[alive] * d2[alive] / var_dt)
    else:
        p_hit[alive] = np.exp(-2.0 * d1[alive] * d2[alive] / var_dt[alive])
    surv = 1.0 - p_hit
    surv[~alive] = 0.0
    return np.clip(surv, 0.0, 1.0)


def _bridge_step_euclidean_circle(x1_prev, x2_prev, x1, x2, R, Sigma, dt):
    """Bridge survival factor for the barrier {x₁² + x₂² ≤ R²}."""
    r_prev = np.hypot(x1_prev, x2_prev)
    r_curr = np.hypot(x1, x2)
    # Radial direction at the midpoint (approximation)
    xm1 = 0.5 * (x1_prev + x1); xm2 = 0.5 * (x2_prev + x2)
    rm = np.hypot(xm1, xm2)
    # Avoid div-by-zero near origin where the radial direction is ill-defined —
    # when r ≈ 0 we're nowhere near the barrier so hit prob is zero anyway.
    safe = rm > 1e-12
    u1 = np.where(safe, xm1 / np.where(rm > 0, rm, 1.0), 1.0)
    u2 = np.where(safe, xm2 / np.where(rm > 0, rm, 1.0), 0.0)
    # Effective variance in the radial direction: uᵀ Σ u
    sigma_r2 = (u1 * u1 * Sigma[0, 0] + 2 * u1 * u2 * Sigma[0, 1]
                + u2 * u2 * Sigma[1, 1])
    d1 = R - r_prev
    d2 = R - r_curr
    return _bridge_survive_halfplane(d1, d2, sigma_r2 * dt)


def _bridge_step_mahalanobis(x1_prev, x2_prev, x1, x2, R, market, dt):
    """Bridge survival for {z₁² + z₂² ≤ R²} in whitened z-coordinates, where the
    Brownian motion is isotropic with unit variance per axis."""
    z1p, z2p = market.x_to_z(x1_prev, x2_prev)
    z1c, z2c = market.x_to_z(x1, x2)
    rp = np.hypot(z1p, z2p)
    rc = np.hypot(z1c, z2c)
    d1 = R - rp
    d2 = R - rc
    # In z-space the BM covariance is the identity: uᵀ I u = 1.
    return _bridge_survive_halfplane(d1, d2, dt)


def _bridge_step_rectangle(x1_prev, x2_prev, x1, x2, dom, Sigma, dt):
    """Bridge survival for a rectangle in x-space."""
    s = np.ones_like(x1)
    # Each edge has a constant outward normal e_i with Σ-variance Σ_ii
    for lo, hi, idx in [(dom.x1_lo, dom.x1_hi, 0), (dom.x2_lo, dom.x2_hi, 1)]:
        coord_prev = x1_prev if idx == 0 else x2_prev
        coord_curr = x1     if idx == 0 else x2
        var_dt = Sigma[idx, idx] * dt
        # lower edge: d = coord - lo
        s *= _bridge_survive_halfplane(coord_prev - lo, coord_curr - lo, var_dt)
        # upper edge: d = hi - coord
        s *= _bridge_survive_halfplane(hi - coord_prev, hi - coord_curr, var_dt)
    return s


# ═══════════════════════════════════════════════════════════════════════
#  Geometric-basket control variate (closed form, vanilla — no knockout)
# ═══════════════════════════════════════════════════════════════════════

def _geometric_basket_cv_price(market, K_geo):
    """European vanilla geometric-average basket call in Black–Scholes."""
    m = market
    G0 = np.sqrt(m.S1_0 * m.S2_0)
    sig2 = 0.25 * (m.sigma1**2 + m.sigma2**2 + 2*m.rho*m.sigma1*m.sigma2)
    # Drift of G under risk-neutral measure:  μ_G = r − ½(σ₁² + σ₂²)/2
    # (arithmetic drift of log mean = average of individual log-drifts)
    mu_logG = m.r - 0.25*(m.sigma1**2 + m.sigma2**2)
    # Forward value of G:  F_G = G_0 · exp(μ_G · T)
    F = G0 * np.exp(mu_logG * m.T)
    sigT = np.sqrt(sig2 * m.T)
    if sigT <= 0 or K_geo <= 0:
        intrinsic = max(F - K_geo, 0.0) * np.exp(-m.r * m.T)
        return float(intrinsic)
    d1 = (np.log(F / K_geo) + 0.5*sig2*m.T) / sigT
    d2 = d1 - sigT
    price = np.exp(-m.r * m.T) * (F * norm.cdf(d1) - K_geo * norm.cdf(d2))
    return float(price)


def _geometric_basket_payoff(S1, S2, K_geo):
    return np.maximum(np.sqrt(S1 * S2) - K_geo, 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Shared path simulation (log-returns + barrier survival weight)
# ═══════════════════════════════════════════════════════════════════════

def _simulate_paths(
    market, domain, *,
    n_paths: int,
    n_steps: int,
    seed: int,
    use_brownian_bridge: bool,
    antithetic: bool,
    x1_init: float = 0.0,
    x2_init: float = 0.0,
) -> dict:
    """Simulate correlated log-return paths and the barrier survival weight."""
    if antithetic:
        n_half = n_paths // 2
        n_paths = 2 * n_half
    else:
        n_half = n_paths

    rng = np.random.default_rng(seed)
    dt = market.T / n_steps
    s1, s2 = market.sigma1, market.sigma2
    rho = market.rho
    Sigma = market.Sigma
    Lchol = np.array([[s1, 0.0],
                      [rho * s2, np.sqrt(max(1 - rho*rho, 0.0)) * s2]])

    x1 = np.full(n_paths, float(x1_init))   # log returns: x = log(S/S₀)
    x2 = np.full(n_paths, float(x2_init))
    drift = np.array([market.r - 0.5*s1*s1, market.r - 0.5*s2*s2])

    # Soft survival weight in [0,1] under Brownian-bridge correction;
    # hard 0/1 under uncorrected discrete monitoring.
    survive = np.ones(n_paths)

    for _ in range(n_steps):
        if antithetic:
            Z_half = rng.standard_normal((2, n_half))
            Z = np.concatenate([Z_half, -Z_half], axis=1)
        else:
            Z = rng.standard_normal((2, n_paths))

        dW = Lchol @ Z * np.sqrt(dt)   # scaled correlated Brownian increments
        x1_prev = x1.copy(); x2_prev = x2.copy()
        x1 = x1 + drift[0]*dt + dW[0]
        x2 = x2 + drift[1]*dt + dW[1]

        if use_brownian_bridge:
            if isinstance(domain, EuclideanCircle):
                factor = _bridge_step_euclidean_circle(
                    x1_prev, x2_prev, x1, x2, domain.R, Sigma, dt)
            elif isinstance(domain, MahalanobisEllipse):
                factor = _bridge_step_mahalanobis(
                    x1_prev, x2_prev, x1, x2, domain.R, market, dt)
            elif isinstance(domain, Rectangle):
                factor = _bridge_step_rectangle(
                    x1_prev, x2_prev, x1, x2, domain, Sigma, dt)
            else:
                raise ValueError(f"Unsupported domain for MC: "
                                 f"{type(domain).__name__}")
            survive *= factor
        else:
            # Discrete monitoring: hard kill on endpoint violation
            if isinstance(domain, EuclideanCircle):
                alive = (x1*x1 + x2*x2) <= domain.R**2
            elif isinstance(domain, MahalanobisEllipse):
                z1, z2 = market.x_to_z(x1, x2)
                alive = (z1*z1 + z2*z2) <= domain.R**2
            elif isinstance(domain, Rectangle):
                alive = ((x1 >= domain.x1_lo) & (x1 <= domain.x1_hi)
                         & (x2 >= domain.x2_lo) & (x2 <= domain.x2_hi))
            else:
                raise ValueError(f"Unsupported domain for MC: "
                                 f"{type(domain).__name__}")
            survive *= alive.astype(float)

    return dict(x1=x1, x2=x2, survive=survive, n_paths=n_paths, dt=dt)


def _nearest_correlation_matrix(C: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix with unit diagonal onto the nearest valid
    correlation matrix by clipping negative eigenvalues (standard technique --
    e.g."""
    vals, vecs = np.linalg.eigh(C)
    if vals.min() >= min_eig:
        return C
    vals_clipped = np.maximum(vals, min_eig)
    C_psd = vecs @ np.diag(vals_clipped) @ vecs.T
    d = np.sqrt(np.diag(C_psd))
    C_corr = C_psd / np.outer(d, d)
    frob_dist = float(np.linalg.norm(C_corr - C, ord="fro"))
    print(f"  [heston_exotic_mc] WARNING: naive per-pair correlation matrix "
          f"was not PSD (min eigenvalue {vals.min():.4f}); repaired via "
          f"nearest-correlation-matrix projection (Frobenius distance from "
          f"the nominal matrix: {frob_dist:.4f}).")
    return C_corr


def _heston_correlation_cholesky(rho_sv1: float, rho_sv2: float,
                                 rho_assets: float) -> np.ndarray:
    """Cholesky factor of the 4x4 correlation matrix over (Z_S1, Z_v1, Z_S2, Z_v2)
    with corr=rho_sv1, corr=rho_sv2, corr(S1,S2)=rho_assets, and every other pair
    uncorrelated (see module docstring above this section)."""
    C = np.array([
        [1.0,        rho_sv1, rho_assets, 0.0],
        [rho_sv1,    1.0,     0.0,        0.0],
        [rho_assets, 0.0,     1.0,        rho_sv2],
        [0.0,        0.0,     rho_sv2,    1.0],
    ])
    C = _nearest_correlation_matrix(C)
    return np.linalg.cholesky(C)


def _simulate_heston_paths(
    market, domain, params1, params2, rho_assets, *,
    n_paths: int,
    n_steps: int,
    seed: int,
    use_brownian_bridge: bool = True,
    antithetic: bool = True,
) -> dict:
    """Simulate correlated two-asset Heston log-return paths plus the barrier
    survival weight, reusing the GBM bridge-step helpers with a per-step
    instantaneous covariance."""
    if antithetic:
        n_half = n_paths // 2
        n_paths = 2 * n_half
    else:
        n_half = n_paths

    rng = np.random.default_rng(seed)
    dt = market.T / n_steps
    L4 = _heston_correlation_cholesky(params1.rho_sv, params2.rho_sv, rho_assets)

    x1 = np.zeros(n_paths)
    x2 = np.zeros(n_paths)
    v1 = np.full(n_paths, params1.v0)
    v2 = np.full(n_paths, params2.v0)
    survive = np.ones(n_paths)

    for _ in range(n_steps):
        if antithetic:
            Z4_half = rng.standard_normal((4, n_half))
            Z4 = np.concatenate([Z4_half, -Z4_half], axis=1)
        else:
            Z4 = rng.standard_normal((4, n_paths))
        W = L4 @ Z4   # rows: W_S1, W_v1, W_S2, W_v2, each correlated N(0,1)

        v1_plus = np.maximum(v1, 0.0)
        v2_plus = np.maximum(v2, 0.0)
        sqrt_dt = np.sqrt(dt)

        x1_prev = x1.copy(); x2_prev = x2.copy()
        x1 = x1 + (market.r - 0.5*v1_plus)*dt + np.sqrt(v1_plus)*sqrt_dt*W[0]
        x2 = x2 + (market.r - 0.5*v2_plus)*dt + np.sqrt(v2_plus)*sqrt_dt*W[2]
        v1 = v1 + params1.kappa*(params1.theta - v1_plus)*dt \
            + params1.xi*np.sqrt(v1_plus)*sqrt_dt*W[1]
        v2 = v2 + params2.kappa*(params2.theta - v2_plus)*dt \
            + params2.xi*np.sqrt(v2_plus)*sqrt_dt*W[3]

        if use_brownian_bridge:
            cross = rho_assets * np.sqrt(v1_plus * v2_plus)
            if isinstance(domain, EuclideanCircle) or isinstance(domain, Rectangle):
                Sigma_00, Sigma_01, Sigma_11 = v1_plus, cross, v2_plus
                if isinstance(domain, EuclideanCircle):
                    factor = _bridge_step_euclidean_circle_pervar(
                        x1_prev, x2_prev, x1, x2, domain.R,
                        Sigma_00, Sigma_01, Sigma_11, dt)
                else:
                    factor = _bridge_step_rectangle_pervar(
                        x1_prev, x2_prev, x1, x2, domain, Sigma_00, Sigma_11, dt)
            elif isinstance(domain, MahalanobisEllipse):
                factor = _bridge_step_mahalanobis(
                    x1_prev, x2_prev, x1, x2, domain.R, market, dt)
            else:
                raise ValueError(f"Unsupported domain for Heston MC: "
                                 f"{type(domain).__name__}")
            survive *= factor
        else:
            if isinstance(domain, EuclideanCircle):
                alive = (x1*x1 + x2*x2) <= domain.R**2
            elif isinstance(domain, MahalanobisEllipse):
                z1, z2 = market.x_to_z(x1, x2)
                alive = (z1*z1 + z2*z2) <= domain.R**2
            elif isinstance(domain, Rectangle):
                alive = ((x1 >= domain.x1_lo) & (x1 <= domain.x1_hi)
                         & (x2 >= domain.x2_lo) & (x2 <= domain.x2_hi))
            else:
                raise ValueError(f"Unsupported domain for Heston MC: "
                                 f"{type(domain).__name__}")
            survive *= alive.astype(float)

    return dict(x1=x1, x2=x2, survive=survive, n_paths=n_paths, dt=dt)


def _bridge_step_euclidean_circle_pervar(x1_prev, x2_prev, x1, x2, R,
                                         Sigma_00, Sigma_01, Sigma_11, dt):
    """Same formula as `_bridge_step_euclidean_circle`, but with per-path (not just
    per-step-scalar) covariance entries -- needed because Heston's instantaneous
    variance differs path-by-path, unlike GBM's fixed Sigma."""
    r_prev = np.hypot(x1_prev, x2_prev)
    r_curr = np.hypot(x1, x2)
    xm1 = 0.5 * (x1_prev + x1); xm2 = 0.5 * (x2_prev + x2)
    rm = np.hypot(xm1, xm2)
    safe = rm > 1e-12
    u1 = np.where(safe, xm1 / np.where(rm > 0, rm, 1.0), 1.0)
    u2 = np.where(safe, xm2 / np.where(rm > 0, rm, 1.0), 0.0)
    sigma_r2 = (u1*u1*Sigma_00 + 2*u1*u2*Sigma_01 + u2*u2*Sigma_11)
    d1 = R - r_prev
    d2 = R - r_curr
    return _bridge_survive_halfplane(d1, d2, np.maximum(sigma_r2 * dt, 1e-300))


def _bridge_step_rectangle_pervar(x1_prev, x2_prev, x1, x2, dom,
                                  Sigma_00, Sigma_11, dt):
    """Same formula as `_bridge_step_rectangle`, but with per-path variance (edges
    are axis-aligned, so only the diagonal Sigma entries matter -- no cross term
    needed, same as the GBM version)."""
    s = np.ones_like(x1)
    for lo, hi, idx in [(dom.x1_lo, dom.x1_hi, 0), (dom.x2_lo, dom.x2_hi, 1)]:
        coord_prev = x1_prev if idx == 0 else x2_prev
        coord_curr = x1 if idx == 0 else x2
        var_dt = np.maximum((Sigma_00 if idx == 0 else Sigma_11) * dt, 1e-300)
        s *= _bridge_survive_halfplane(coord_prev - lo, coord_curr - lo, var_dt)
        s *= _bridge_survive_halfplane(hi - coord_prev, hi - coord_curr, var_dt)
    return s


def heston_exotic_mc(
    market, payoff, domain, params1, params2, rho_assets, *,
    n_paths: int = 200_000,
    n_steps: int = 250,
    seed: int = 42,
    use_brownian_bridge: bool = True,
    antithetic: bool = True,
) -> dict:
    """Monte Carlo price under per-asset Heston dynamics."""
    sim = _simulate_heston_paths(
        market, domain, params1, params2, rho_assets,
        n_paths=n_paths, n_steps=n_steps, seed=seed,
        use_brownian_bridge=use_brownian_bridge, antithetic=antithetic,
    )
    S1 = market.S1_0 * np.exp(sim["x1"])
    S2 = market.S2_0 * np.exp(sim["x2"])
    raw = np.asarray(payoff(S1, S2), dtype=float)
    disc = np.exp(-market.r * market.T)
    payoff_samples = disc * sim["survive"] * raw

    price = float(np.mean(payoff_samples))
    se = float(np.std(payoff_samples, ddof=1) / np.sqrt(sim["n_paths"]))
    return dict(price=price, std_err=se, n_paths=sim["n_paths"], n_steps=n_steps,
               survival_rate=float(np.mean(sim["survive"])))


def localvol_exotic_mc(
    lvmarket, payoff, domain, *,
    n_paths: int = 400_000,
    n_steps: int = 400,
    seed: int = 42,
    use_brownian_bridge: bool = True,
    antithetic: bool = True,
) -> dict:
    """Monte Carlo price under state-dependent local volatility."""
    if antithetic:
        n_half = n_paths // 2
        n_paths = 2 * n_half
    else:
        n_half = n_paths

    rng = np.random.default_rng(seed)
    T = lvmarket.T
    r = lvmarket.r
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    x1 = np.zeros(n_paths)
    x2 = np.zeros(n_paths)
    survive = np.ones(n_paths)

    for step in range(n_steps):
        tau_mid = T - (step + 0.5) * dt   # time-to-maturity at the step midpoint
        Sig = lvmarket.Sigma_full(np.array([x1, x2]), tau_mid)
        S00 = np.asarray(Sig[0, 0], dtype=float) + np.zeros(n_paths)
        S01 = np.asarray(Sig[0, 1], dtype=float) + np.zeros(n_paths)
        S11 = np.asarray(Sig[1, 1], dtype=float) + np.zeros(n_paths)

        # Per-path Cholesky of the instantaneous covariance.
        L00 = np.sqrt(np.maximum(S00, 1e-300))
        L10 = S01 / L00
        L11 = np.sqrt(np.maximum(S11 - L10 * L10, 1e-300))

        if antithetic:
            Z_half = rng.standard_normal((2, n_half))
            Z = np.concatenate([Z_half, -Z_half], axis=1)
        else:
            Z = rng.standard_normal((2, n_paths))

        dW1 = L00 * Z[0] * sqrt_dt
        dW2 = (L10 * Z[0] + L11 * Z[1]) * sqrt_dt

        x1_prev = x1.copy(); x2_prev = x2.copy()
        x1 = x1 + (r - 0.5 * S00) * dt + dW1
        x2 = x2 + (r - 0.5 * S11) * dt + dW2

        if use_brownian_bridge:
            if isinstance(domain, EuclideanCircle):
                factor = _bridge_step_euclidean_circle_pervar(
                    x1_prev, x2_prev, x1, x2, domain.R, S00, S01, S11, dt)
            elif isinstance(domain, Rectangle):
                factor = _bridge_step_rectangle_pervar(
                    x1_prev, x2_prev, x1, x2, domain, S00, S11, dt)
            else:
                raise ValueError(
                    f"local-vol MC bridge not wired for domain "
                    f"{type(domain).__name__} (EuclideanCircle/Rectangle only)")
            survive *= factor
        else:
            if isinstance(domain, EuclideanCircle):
                alive = (x1 * x1 + x2 * x2) <= domain.R**2
            elif isinstance(domain, Rectangle):
                alive = ((x1 >= domain.x1_lo) & (x1 <= domain.x1_hi)
                         & (x2 >= domain.x2_lo) & (x2 <= domain.x2_hi))
            else:
                raise ValueError(
                    f"local-vol MC not wired for domain "
                    f"{type(domain).__name__}")
            survive *= alive.astype(float)

    S1 = lvmarket.S1_0 * np.exp(x1)
    S2 = lvmarket.S2_0 * np.exp(x2)
    raw = np.asarray(payoff(S1, S2), dtype=float)
    disc = np.exp(-r * T)
    payoff_samples = disc * survive * raw

    price = float(np.mean(payoff_samples))
    se = float(np.std(payoff_samples, ddof=1) / np.sqrt(n_paths))
    return dict(price=price, std_err=se, n_paths=n_paths, n_steps=n_steps,
               survival_rate=float(np.mean(survive)))


# ═══════════════════════════════════════════════════════════════════════
#  Main MC routine
# ═══════════════════════════════════════════════════════════════════════

def mc_price(
    market, payoff, domain, *,
    n_paths: int = 500_000,
    n_steps: int = 500,
    seed: int = 42,
    use_brownian_bridge: bool = True,
    antithetic: bool = True,
    control_variate: str | None = None,
) -> dict:
    """Monte Carlo knockout price with optional variance-reduction upgrades."""
    sim = _simulate_paths(
        market, domain, n_paths=n_paths, n_steps=n_steps, seed=seed,
        use_brownian_bridge=use_brownian_bridge, antithetic=antithetic,
    )
    x1, x2, survive, n_paths = sim["x1"], sim["x2"], sim["survive"], sim["n_paths"]

    # Endpoint spot prices and payoffs
    S1 = market.S1_0 * np.exp(x1)
    S2 = market.S2_0 * np.exp(x2)
    raw = np.asarray(payoff(S1, S2), dtype=float)
    disc = np.exp(-market.r * market.T)
    payoff_samples = disc * survive * raw

    # ── Control variate ──
    cv_info = {}
    if control_variate == "geometric_basket":
        # Strike: match payoff.K if available, else use S1_0 (ATM-like)
        K_cv = getattr(payoff, "K", market.S1_0)
        analytic = _geometric_basket_cv_price(market, K_cv)
        cv_pathwise = disc * _geometric_basket_payoff(S1, S2, K_cv)
        # β-optimal control variate coefficient
        cov = np.cov(payoff_samples, cv_pathwise, ddof=1)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        adjusted = payoff_samples - beta * (cv_pathwise - analytic)
        cv_info = dict(cv_beta=float(beta),
                       cv_analytic=float(analytic),
                       cv_mc_mean=float(np.mean(cv_pathwise)),
                       variance_reduction=float(
                           1 - np.var(adjusted, ddof=1) /
                           max(np.var(payoff_samples, ddof=1), 1e-30)))
        payoff_samples = adjusted

    price = float(np.mean(payoff_samples))
    se = float(np.std(payoff_samples, ddof=1) / np.sqrt(n_paths))

    return dict(
        price=price,
        std_err=se,
        ci_lo=price - 1.96*se,
        ci_hi=price + 1.96*se,
        survival_rate=float(np.mean(survive)),
        n_paths=n_paths,
        n_steps=n_steps,
        brownian_bridge=bool(use_brownian_bridge),
        antithetic=bool(antithetic),
        control_variate=control_variate,
        **cv_info,
    )


def mc_hitting_times(
    market, payoff, domain, *,
    n_paths: int = 200_000,
    n_steps: int = 500,
    seed: int = 42,
    antithetic: bool = True,
) -> dict:
    """Simulate paths and sample a bridge-corrected first-passage time per path (NaN
    = survived to T)."""
    if antithetic:
        n_half = n_paths // 2
        n_paths = 2 * n_half
    else:
        n_half = n_paths

    rng = np.random.default_rng(seed)
    rng_hit = np.random.default_rng(seed + 1)
    dt = market.T / n_steps
    s1, s2 = market.sigma1, market.sigma2
    rho = market.rho
    Sigma = market.Sigma
    Lchol = np.array([[s1, 0.0],
                      [rho * s2, np.sqrt(max(1 - rho * rho, 0.0)) * s2]])

    x1 = np.zeros(n_paths)
    x2 = np.zeros(n_paths)
    drift = np.array([market.r - 0.5 * s1 * s1, market.r - 0.5 * s2 * s2])

    hit_time = np.full(n_paths, np.nan)
    alive = np.ones(n_paths, dtype=bool)

    for step in range(n_steps):
        if antithetic:
            Z_half = rng.standard_normal((2, n_half))
            Z = np.concatenate([Z_half, -Z_half], axis=1)
        else:
            Z = rng.standard_normal((2, n_paths))
        dW = Lchol @ Z * np.sqrt(dt)
        x1_prev = x1.copy(); x2_prev = x2.copy()
        x1 = x1 + drift[0] * dt + dW[0]
        x2 = x2 + drift[1] * dt + dW[1]

        if isinstance(domain, EuclideanCircle):
            factor = _bridge_step_euclidean_circle(
                x1_prev, x2_prev, x1, x2, domain.R, Sigma, dt)
        elif isinstance(domain, MahalanobisEllipse):
            factor = _bridge_step_mahalanobis(
                x1_prev, x2_prev, x1, x2, domain.R, market, dt)
        elif isinstance(domain, Rectangle):
            factor = _bridge_step_rectangle(
                x1_prev, x2_prev, x1, x2, domain, Sigma, dt)
        else:
            raise ValueError(f"Unsupported domain for hitting-time MC: "
                             f"{type(domain).__name__}")

        p_hit_step = np.clip(1.0 - factor, 0.0, 1.0)
        u = rng_hit.random(n_paths)
        newly_hit = alive & (u < p_hit_step)
        t_within = rng_hit.random(n_paths) * dt
        hit_time[newly_hit] = step * dt + t_within[newly_hit]
        alive = alive & ~newly_hit

    return dict(hit_time=hit_time, survived=alive, n_paths=n_paths)


def mc_delta_crn(
    market, payoff, domain, *,
    n_paths: int = 500_000,
    n_steps: int = 500,
    seed: int = 42,
    bump_pct: float = 1.0,
    use_brownian_bridge: bool = True,
    antithetic: bool = True,
) -> dict:
    """Central-difference MC delta (Δ₁, Δ₂) with common random numbers (CRN)."""
    disc = np.exp(-market.r * market.T)
    h1 = market.S1_0 * bump_pct / 100.0
    h2 = market.S2_0 * bump_pct / 100.0

    def _run(x1_init: float, x2_init: float) -> dict:
        return _simulate_paths(
            market, domain, n_paths=n_paths, n_steps=n_steps, seed=seed,
            use_brownian_bridge=use_brownian_bridge, antithetic=antithetic,
            x1_init=x1_init, x2_init=x2_init,
        )

    base = _run(0.0, 0.0)
    n_paths_actual = base["n_paths"]
    S1_base = market.S1_0 * np.exp(base["x1"])
    S2_base = market.S2_0 * np.exp(base["x2"])
    price_samples = disc * base["survive"] * np.asarray(
        payoff(S1_base, S2_base), dtype=float)
    price = float(np.mean(price_samples))
    price_sem = float(np.std(price_samples, ddof=1) / np.sqrt(n_paths_actual))

    # ── Delta₁: bump asset 1's current spot, asset 2 stays at base ──
    p1 = _run(np.log((market.S1_0 + h1) / market.S1_0), 0.0)
    m1 = _run(np.log((market.S1_0 - h1) / market.S1_0), 0.0)
    S1p = market.S1_0 * np.exp(p1["x1"]); S2p1 = market.S2_0 * np.exp(p1["x2"])
    S1m = market.S1_0 * np.exp(m1["x1"]); S2m1 = market.S2_0 * np.exp(m1["x2"])
    d1_samples = disc * (
        p1["survive"] * np.asarray(payoff(S1p, S2p1), dtype=float)
        - m1["survive"] * np.asarray(payoff(S1m, S2m1), dtype=float)
    ) / (2 * h1)
    delta1 = float(np.mean(d1_samples))
    delta1_sem = float(np.std(d1_samples, ddof=1) / np.sqrt(n_paths_actual))

    # ── Delta₂: bump asset 2's current spot, asset 1 stays at base ──
    p2 = _run(0.0, np.log((market.S2_0 + h2) / market.S2_0))
    m2 = _run(0.0, np.log((market.S2_0 - h2) / market.S2_0))
    S1p2 = market.S1_0 * np.exp(p2["x1"]); S2p = market.S2_0 * np.exp(p2["x2"])
    S1m2 = market.S1_0 * np.exp(m2["x1"]); S2m = market.S2_0 * np.exp(m2["x2"])
    d2_samples = disc * (
        p2["survive"] * np.asarray(payoff(S1p2, S2p), dtype=float)
        - m2["survive"] * np.asarray(payoff(S1m2, S2m), dtype=float)
    ) / (2 * h2)
    delta2 = float(np.mean(d2_samples))
    delta2_sem = float(np.std(d2_samples, ddof=1) / np.sqrt(n_paths_actual))

    return dict(
        price=price, price_sem=price_sem,
        delta1=delta1, delta1_sem=delta1_sem,
        delta2=delta2, delta2_sem=delta2_sem,
        survival_rate=float(np.mean(base["survive"])),
        n_paths=n_paths_actual, n_steps=n_steps, seed=seed,
        bump_pct=bump_pct,
        brownian_bridge=bool(use_brownian_bridge),
        antithetic=bool(antithetic),
    )
