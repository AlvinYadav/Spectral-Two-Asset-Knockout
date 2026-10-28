"""Generic spectral solver."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..core.market_base import Market
from ..core.market import GBMMarket
from ..core.domain import EuclideanCircle, MahalanobisEllipse, Rectangle
from ..bases.base import EigenBasis
from ..bases.mathieu import MathieuBasis
from ..bases.bessel import BesselBasis
from ..bases.fourier import FourierBasis


def _bump_market(base: GBMMarket, **kw) -> GBMMarket:
    """Rebuild a GBMMarket with `kw` overriding the seven raw inputs."""
    raw = dict(
        S1_0=base.S1_0, S2_0=base.S2_0,
        sigma1=base.sigma1, sigma2=base.sigma2,
        rho=base.rho, r=base.r, T=base.T,
    )
    raw.update(kw)
    return GBMMarket(**raw)


@dataclass
class SpectralSolver:
    market: Market
    payoff: object
    domain: object
    basis: EigenBasis

    smooth: bool = True
    smooth_tau_frac: float = 0.005
    smooth_N_grid: int = 100
    smooth_N_steps: int = 50

    projection_method: str = "auto"

    smooth_tau: float = 0.0
    _fit_done: bool = False
    _proj_info: dict = None
    _projection_geometry_cache: tuple = None

    # ── Setup ────────────────────────────────────────────────────
    def fit(self):
        if not self._fit_done:
            self.basis.fit(self.market, self.domain)
            self._fit_done = True
            self._projection_geometry_cache = None
        return self

    def project(self, *, reuse_design: bool = False):
        """Build target `b`, solve SVD, fill mode.coeff."""
        self.fit()

        # Zero the coefficients of all modes up front so inactive ones stay
        # at zero and don't leak into the expansion.
        for md in self.basis.modes:
            md.coeff = 0.0

        if hasattr(self.basis, "project_galerkin") and isinstance(self.domain, Rectangle):
            info = self.basis.project_galerkin(
                self.payoff, smooth=self.smooth,
                smooth_tau_frac=self.smooth_tau_frac,
            )
            self.smooth_tau = info["smooth_tau"]
            self._active_modes = self.basis.active_modes()
            self._proj_info = dict(info, smoothing=bool(self.smooth))
            return self._proj_info

        active = self.basis.active_modes()
        self._active_modes = active

        if reuse_design and self._projection_geometry_cache is not None:
            S1_arr, S2_arr, A_full, active_idx = self._projection_geometry_cache
        else:
            S1_arr, S2_arr = self.basis.collocation_grid_spot()
            A_full = self.basis.design_matrix_at_spot(S1_arr, S2_arr)
            active_idx = [k for k, md in enumerate(self.basis.modes)
                          if md.norm > 1e-6]
            if reuse_design:
                self._projection_geometry_cache = (
                    S1_arr, S2_arr, A_full, active_idx,
                )
        n_pts = len(S1_arr)

        smooth_tau = 0.0
        if self.smooth and not isinstance(self.domain, Rectangle):
            # FDM pre-smooth on the gauged equation
            from .fdm import fdm_smooth
            if hasattr(self.basis, "_colloc_mu"):
                nat1 = self.basis._colloc_mu
                nat2 = self.basis._colloc_nu
            elif hasattr(self.basis, "_colloc_r"):
                nat1 = self.basis._colloc_r
                nat2 = self.basis._colloc_theta
            else:
                raise ValueError("basis does not expose natural-coord grid")
            delta_tau = self.smooth_tau_frac * self.market.T
            b, smooth_tau = fdm_smooth(
                self.market, self.domain, nat1, nat2, self.basis,
                payoff=self.payoff, delta_tau=delta_tau,
                N_grid=self.smooth_N_grid, N_steps=self.smooth_N_steps,
            )
        else:
            # No smoothing — use raw gauged payoff on the grid
            b = np.zeros(n_pts)
            for i in range(n_pts):
                raw = float(self.payoff(S1_arr[i], S2_arr[i]))
                gauge = self.market.gauge_factor(S1_arr[i], S2_arr[i], 0.0)
                b[i] = raw / gauge if abs(gauge) > 1e-100 else 0.0

        A = A_full[:, active_idx]

        method = self.projection_method
        if method == "auto":
            tridiag_mathieu = (isinstance(self.basis, MathieuBasis)
                               and self.basis.backend == "tridiagonal")
            method = ("galerkin"
                      if (tridiag_mathieu or isinstance(self.basis, BesselBasis))
                      else "collocation")
            if isinstance(self.basis, BesselBasis) and not self.payoff.boundary_compatible:
                method = "collocation"
        if method not in ("collocation", "weighted", "galerkin"):
            raise ValueError(
                f"unknown projection_method {method!r}; expected 'auto', "
                "'collocation', 'weighted' or 'galerkin'"
            )
        if method == "galerkin":
            w = self._projection_row_weights(n_pts)
            coeffs = A.T @ (w * b)
            rank = A.shape[1]
            cond = float("nan")   # no least-squares system is solved
        else:
            if method == "weighted":
                sqrt_w = np.sqrt(self._projection_row_weights(n_pts))
                A_fit = A * sqrt_w[:, None]
                b_fit = b * sqrt_w
            else:
                A_fit, b_fit = A, b

            # Column equilibration for numerical stability
            scale = np.max(np.abs(A_fit), axis=0)
            scale[scale == 0] = 1.0
            A_scaled = A_fit / scale
            coeffs_scaled, _, rank, sv = np.linalg.lstsq(A_scaled, b_fit,
                                                         rcond=1e-10)
            coeffs = coeffs_scaled / scale
            cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf

        for j, k in enumerate(active_idx):
            self.basis.modes[k].coeff = float(coeffs[j])

        self.smooth_tau = smooth_tau
        self._proj_info = dict(
            n_pts=n_pts, n_modes=len(self.basis.modes),
            n_active=len(active_idx),
            rank=int(rank), cond=float(cond),
            max_coeff=float(np.max(np.abs(coeffs))),
            smooth_tau=float(smooth_tau),
            smoothing=bool(self.smooth),
            method=method,
        )
        return self._proj_info

    def _projection_row_weights(self, n_pts: int) -> np.ndarray:
        """Row weights that discretise the L2(Omega, dz) measure."""
        b = self.basis
        if isinstance(b, MathieuBasis):
            if b.backend != "tridiagonal":
                raise ValueError(
                    f"projection_method={self.projection_method!r} requires "
                    "MathieuBasis with backend='tridiagonal'"
                )
            if not hasattr(b, "_colloc_mu") or len(b._colloc_mu) != n_pts:
                raise ValueError(
                    "weighted/galerkin projection requires the collocation "
                    "quadrature grid; custom spot grids are not supported"
                )
            J = b.c_focal ** 2 * (np.sinh(b._colloc_mu) ** 2
                                  + np.sin(b._colloc_nu) ** 2)
            w = b._colloc_mu_weights * b._dnu * J
        elif isinstance(b, BesselBasis):
            if not hasattr(b, "_colloc_r") or len(b._colloc_r) != n_pts:
                raise ValueError(
                    "weighted/galerkin projection requires the collocation "
                    "quadrature grid; custom spot grids are not supported"
                )
            n_theta = int(round(len(b._colloc_r) / b.N_r_quad))
            dtheta = 2.0 * np.pi / n_theta
            J = b._colloc_r                      # polar area element
            w = b._colloc_r_weights * dtheta * J
        else:
            raise ValueError(
                f"projection_method={self.projection_method!r} is not "
                f"supported for {type(b).__name__}; the weighted measure is "
                "implemented for MathieuBasis (elliptic) and BesselBasis "
                "(polar). Rectangular domains project through "
                "FourierBasis.project_galerkin, where the Jacobian is 1."
            )
        if not np.all(w > 0):
            raise AssertionError(
                "non-positive projection row weight (coordinate degeneracy "
                "on the collocation grid)"
            )
        return w

    # ── Price ────────────────────────────────────────────────────
    def price(self, S1=None, S2=None, t: float = 0.0) -> float:
        if self._proj_info is None:
            self.project()
        S1 = self.market.S1_0 if S1 is None else S1
        S2 = self.market.S2_0 if S2 is None else S2
        tau = self.market.T - t
        if not self._inside(S1, S2):
            return 0.0
        tau_eff = tau - self.smooth_tau
        u = self.basis.eval_expansion(S1, S2, tau_eff)
        gauge = self.market.gauge_factor(S1, S2, tau)
        return float(gauge * u)

    def _inside(self, S1, S2) -> bool:
        x1, x2 = self.market.spot_to_x(S1, S2)
        if isinstance(self.domain, Rectangle):
            return bool(self.domain.contains(x1, x2))
        if isinstance(self.domain, EuclideanCircle):
            return x1*x1 + x2*x2 < self.domain.R**2
        if isinstance(self.domain, MahalanobisEllipse):
            return bool(self.domain.contains_spot(S1, S2, self.market))
        return True

    # ── Greeks ───────────────────────────────────────────────────
    def _u_grad_x(self, S1, S2, tau_eff):
        """Analytic (u, ∂u/∂x₁, ∂u/∂x₂) at a single spot point."""
        m = self.market
        b = self.basis

        if isinstance(b, MathieuBasis):
            u_val, du_dmu, du_dnu, _ = b.eval_with_derivatives(S1, S2, tau_eff)
            mu, nu = b.mu_nu_at_spot(S1, S2)
            sh = b.c_focal * np.sinh(mu + 1j*nu)
            if abs(sh) < 1e-14:
                return u_val, 0.0, 0.0
            inv_sh = 1.0 / sh
            dmu_dz2, dnu_dz2 = np.real(inv_sh), np.imag(inv_sh)
            dmu_dz1, dnu_dz1 = np.real(1j*inv_sh), np.imag(1j*inv_sh)
            Mx = m.M_xz
            dmu_dx1 = dmu_dz1 * Mx[0, 0] + dmu_dz2 * Mx[1, 0]
            dnu_dx1 = dnu_dz1 * Mx[0, 0] + dnu_dz2 * Mx[1, 0]
            dmu_dx2 = dmu_dz1 * Mx[0, 1] + dmu_dz2 * Mx[1, 1]
            dnu_dx2 = dnu_dz1 * Mx[0, 1] + dnu_dz2 * Mx[1, 1]
            du_dx1 = du_dmu * dmu_dx1 + du_dnu * dnu_dx1
            du_dx2 = du_dmu * dmu_dx2 + du_dnu * dnu_dx2
            return u_val, du_dx1, du_dx2

        if isinstance(b, BesselBasis):
            u_val, du_dz1, du_dz2, _ = b.eval_with_derivatives(S1, S2, tau_eff)
            Mx = m.M_xz
            du_dx1 = du_dz1 * Mx[0, 0] + du_dz2 * Mx[1, 0]
            du_dx2 = du_dz1 * Mx[0, 1] + du_dz2 * Mx[1, 1]
            return u_val, du_dx1, du_dx2

        if isinstance(b, FourierBasis):
            # Fourier's natural coordinates ARE x — no chain rule.
            u_val, du_dx1, du_dx2, _ = b.eval_with_derivatives(S1, S2, tau_eff)
            return u_val, du_dx1, du_dx2

        # Unknown basis: caller must fall back to FD on the price.
        return None, None, None

    def _u_hessian_x(self, S1, S2, tau_eff, eps: float = 1e-4):
        """Second derivatives of the *gauged function u* w.r.t. log-spot x_i."""
        def g(dS1_frac, dS2_frac):
            # Bump in log space: S → S·exp(ε)
            s1p = S1 * np.exp(dS1_frac)
            s2p = S2 * np.exp(dS2_frac)
            return self._u_grad_x(s1p, s2p, tau_eff)

        # u_x1x1: bump x1, read u_x1
        _, uxp, _ = g(+eps, 0.0)
        _, uxm, _ = g(-eps, 0.0)
        u_x1x1 = (uxp - uxm) / (2 * eps)

        # u_x2x2: bump x2, read u_x2
        _, _, uyp = g(0.0, +eps)
        _, _, uym = g(0.0, -eps)
        u_x2x2 = (uyp - uym) / (2 * eps)

        # u_x1x2: two ways, symmetrise for stability
        #   (a) bump x2, read u_x1
        _, uxp2, _ = g(0.0, +eps)
        _, uxm2, _ = g(0.0, -eps)
        u_x1x2_a = (uxp2 - uxm2) / (2 * eps)
        #   (b) bump x1, read u_x2
        _, _, uyp1 = g(+eps, 0.0)
        _, _, uym1 = g(-eps, 0.0)
        u_x1x2_b = (uyp1 - uym1) / (2 * eps)
        u_x1x2 = 0.5 * (u_x1x2_a + u_x1x2_b)

        return u_x1x1, u_x2x2, u_x1x2

    def _spot_greeks_from_u(self, S1, S2, gf, u_val, u_x1, u_x2,
                            u_x1x1, u_x2x2, u_x1x2):
        """Assemble spot-space Greeks from u and its log-space derivatives."""
        m = self.market
        b1, b2 = m.beta1, m.beta2
        delta1 = gf * (b1 * u_val + u_x1) / S1
        delta2 = gf * (b2 * u_val + u_x2) / S2
        gamma11 = gf * (b1*(b1 - 1.0) * u_val + (2*b1 - 1.0) * u_x1 + u_x1x1) / (S1*S1)
        gamma22 = gf * (b2*(b2 - 1.0) * u_val + (2*b2 - 1.0) * u_x2 + u_x2x2) / (S2*S2)
        gamma12 = gf * (b1*b2 * u_val + b2*u_x1 + b1*u_x2 + u_x1x2) / (S1*S2)
        return delta1, delta2, gamma11, gamma22, gamma12

    def greeks(self, S1=None, S2=None, t: float = 0.0,
               full: bool = False,
               vol_bump: float = 0.01,
               rate_bump: float = 1e-4) -> dict:
        """Return a dict of Greeks for the solver's current config."""
        if self._proj_info is None:
            self.project()
        S1 = self.market.S1_0 if S1 is None else S1
        S2 = self.market.S2_0 if S2 is None else S2
        tau = self.market.T - t
        if not self._inside(S1, S2):
            out = dict(price=0.0, delta1=0, delta2=0,
                       theta=0, gamma1=0, gamma2=0, gamma12=0)
            if full:
                out.update(vega1=0.0, vega2=0.0, rho=0.0)
            return out
        tau_eff = tau - self.smooth_tau
        gf = self.market.gauge_factor(S1, S2, tau)

        if isinstance(self.basis, MathieuBasis):
            out = self._greeks_mathieu(S1, S2, tau, tau_eff, gf)
        elif isinstance(self.basis, BesselBasis):
            out = self._greeks_bessel(S1, S2, tau, tau_eff, gf)
        elif isinstance(self.basis, FourierBasis):
            out = self._greeks_fourier(S1, S2, tau, tau_eff, gf)
        else:
            out = self._greeks_fd(S1, S2, t)

        if full:
            out.update(self._market_greeks(
                S1, S2, t, vol_bump=vol_bump, rate_bump=rate_bump,
            ))
        return out

    def _market_greeks(self, S1, S2, t, *,
                       vol_bump: float = 0.01,
                       rate_bump: float = 1e-4) -> dict:
        """Central-difference vega1, vega2, rho by rebuilding the solver at bumped
        markets."""
        from dataclasses import replace, fields

        def fresh_basis():
            FITTED_STATE = {
                "modes", "market", "domain",
                "c_focal", "mu_0", "A_semi", "B_semi",
            }
            kw = {}
            for f in fields(self.basis):
                if f.name in FITTED_STATE or f.name.startswith("_"):
                    continue
                v = getattr(self.basis, f.name)
                if isinstance(v, (int, float, str, bool)):
                    kw[f.name] = v
            return type(self.basis)(**kw)

        def price_at(**market_kw):
            m_new = _bump_market(self.market, **market_kw)
            solver = SpectralSolver(
                market=m_new, payoff=self.payoff,
                domain=self.domain, basis=fresh_basis(),
                smooth=self.smooth, smooth_tau_frac=self.smooth_tau_frac,
                smooth_N_grid=self.smooth_N_grid,
                smooth_N_steps=self.smooth_N_steps,
                projection_method=self.projection_method,
            )
            solver.fit(); solver.project()
            return solver.price(S1, S2, t)

        vp1 = price_at(sigma1=self.market.sigma1 + vol_bump)
        vm1 = price_at(sigma1=self.market.sigma1 - vol_bump)
        vega1 = (vp1 - vm1) / (2 * vol_bump)

        vp2 = price_at(sigma2=self.market.sigma2 + vol_bump)
        vm2 = price_at(sigma2=self.market.sigma2 - vol_bump)
        vega2 = (vp2 - vm2) / (2 * vol_bump)

        rp = price_at(r=self.market.r + rate_bump)
        rm = price_at(r=self.market.r - rate_bump)
        rho = (rp - rm) / (2 * rate_bump)

        return dict(vega1=vega1, vega2=vega2, rho=rho)

    # ---- Mathieu-specific analytic greeks --------------------------
    def _greeks_mathieu(self, S1, S2, tau, tau_eff, gf):
        m = self.market
        b = self.basis
        u_val, du_dmu, du_dnu, theta_sum = b.eval_with_derivatives(S1, S2, tau_eff)
        V = gf * u_val
        mu, nu = b.mu_nu_at_spot(S1, S2)
        sh = b.c_focal * np.sinh(mu + 1j*nu)
        if abs(sh) < 1e-14:
            t_val = self.market.T - tau
            return self._greeks_fd(S1, S2, t_val)

        inv_sh = 1.0 / sh
        dmu_dz2, dnu_dz2 = np.real(inv_sh), np.imag(inv_sh)
        dmu_dz1, dnu_dz1 = np.real(1j*inv_sh), np.imag(1j*inv_sh)
        Mx = m.M_xz
        dmu_dx1 = dmu_dz1 * Mx[0, 0] + dmu_dz2 * Mx[1, 0]
        dnu_dx1 = dnu_dz1 * Mx[0, 0] + dnu_dz2 * Mx[1, 0]
        dmu_dx2 = dmu_dz1 * Mx[0, 1] + dmu_dz2 * Mx[1, 1]
        dnu_dx2 = dnu_dz1 * Mx[0, 1] + dnu_dz2 * Mx[1, 1]
        u_x1 = du_dmu * dmu_dx1 + du_dnu * dnu_dx1
        u_x2 = du_dmu * dmu_dx2 + du_dnu * dnu_dx2

        # Second derivatives by FD of the analytic gradient
        u_x1x1, u_x2x2, u_x1x2 = self._u_hessian_x(S1, S2, tau_eff)

        # Assemble spot-space Greeks with exact gauge chain rule
        d1, d2, g11, g22, g12 = self._spot_greeks_from_u(
            S1, S2, gf, u_val, u_x1, u_x2, u_x1x1, u_x2x2, u_x1x2,
        )
        theta_val = -gf * (m.gamma * u_val + theta_sum)
        return dict(price=V, delta1=d1, delta2=d2,
                    theta=theta_val,
                    gamma1=g11, gamma2=g22, gamma12=g12)

    # ---- Bessel-specific analytic greeks ---------------------------
    def _greeks_bessel(self, S1, S2, tau, tau_eff, gf):
        m = self.market
        b = self.basis
        u_val, du_dz1, du_dz2, theta_sum = b.eval_with_derivatives(S1, S2, tau_eff)
        V = gf * u_val
        Mx = m.M_xz
        u_x1 = du_dz1 * Mx[0, 0] + du_dz2 * Mx[1, 0]
        u_x2 = du_dz1 * Mx[0, 1] + du_dz2 * Mx[1, 1]

        u_x1x1, u_x2x2, u_x1x2 = self._u_hessian_x(S1, S2, tau_eff)
        d1, d2, g11, g22, g12 = self._spot_greeks_from_u(
            S1, S2, gf, u_val, u_x1, u_x2, u_x1x1, u_x2x2, u_x1x2,
        )
        theta_val = -gf * (m.gamma * u_val + theta_sum)
        return dict(price=V, delta1=d1, delta2=d2,
                    theta=theta_val,
                    gamma1=g11, gamma2=g22, gamma12=g12)

    # ---- Fourier-specific analytic greeks --------------------------
    def _greeks_fourier(self, S1, S2, tau, tau_eff, gf):
        """Same machinery as Mathieu/Bessel but skipping the chain rule from natural
        coords to x — Fourier's natural coords ARE x."""
        m = self.market
        u_val, u_x1, u_x2, theta_sum = self.basis.eval_with_derivatives(
            S1, S2, tau_eff)
        V = gf * u_val
        u_x1x1, u_x2x2, u_x1x2 = self._u_hessian_x(S1, S2, tau_eff)
        d1, d2, g11, g22, g12 = self._spot_greeks_from_u(
            S1, S2, gf, u_val, u_x1, u_x2, u_x1x1, u_x2x2, u_x1x2,
        )
        theta_val = -gf * (m.gamma * u_val + theta_sum)
        return dict(price=V, delta1=d1, delta2=d2,
                    theta=theta_val,
                    gamma1=g11, gamma2=g22, gamma12=g12)

    # ---- Finite-diff fallback (unrecognised bases) -----------------
    def _greeks_fd(self, S1, S2, t):
        V = self.price(S1, S2, t)
        d1, d2 = self._fd_deltas(S1, S2, t)
        g1, g2 = self._fd_gammas(S1, S2, t)
        # 4-point cross-gamma stencil in spot
        h1 = 0.01 * S1
        h2 = 0.01 * S2
        Vpp = self.price(S1+h1, S2+h2, t)
        Vpm = self.price(S1+h1, S2-h2, t)
        Vmp = self.price(S1-h1, S2+h2, t)
        Vmm = self.price(S1-h1, S2-h2, t)
        g12 = (Vpp - Vpm - Vmp + Vmm) / (4 * h1 * h2)
        et = 1e-3 * self.market.T
        theta = (self.price(S1, S2, t + et) - self.price(S1, S2, t - et)) / (2*et)
        return dict(price=V, delta1=d1, delta2=d2, theta=theta,
                    gamma1=g1, gamma2=g2, gamma12=g12)

    def _fd_deltas(self, S1, S2, t):
        eps = 1e-4
        d1 = (self.price(S1*np.exp(eps), S2, t) - self.price(S1*np.exp(-eps), S2, t)) / (2*eps*S1)
        d2 = (self.price(S1, S2*np.exp(eps), t) - self.price(S1, S2*np.exp(-eps), t)) / (2*eps*S2)
        return d1, d2

    def _fd_gammas(self, S1, S2, t):
        V = self.price(S1, S2, t)
        e1 = 0.01 * S1
        g1 = (self.price(S1+e1, S2, t) - 2*V + self.price(S1-e1, S2, t)) / e1**2
        e2 = 0.01 * S2
        g2 = (self.price(S1, S2+e2, t) - 2*V + self.price(S1, S2-e2, t)) / e2**2
        return g1, g2

    # ── Diagnostics ──────────────────────────────────────────────
    def n_modes(self) -> int:
        return self.basis.n_modes()

    @property
    def projection_info(self) -> dict:
        return self._proj_info
