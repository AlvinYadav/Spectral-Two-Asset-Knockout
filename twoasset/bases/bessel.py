"""Fourier–Bessel eigenbasis for the Mahalanobis barrier."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.special import jn, jn_zeros, jvp

from .base import EigenBasis, EigenMode, gram_dedupe


@dataclass
class BesselBasis(EigenBasis):
    """Fourier–Bessel basis on a disk of radius R (in whitened z-coords)."""
    N_angular_max: int = 8
    M_radial_max: int = 6
    N_r_quad: int = 40
    N_theta_quad: int = 96

    # Filled by .fit():
    modes: list = field(default_factory=list)
    market: object = None
    domain: object = None
    R: float = 0.0
    gram_matrix: np.ndarray = field(default=None, repr=False)
    mode_ledger: dict = field(default=None, repr=False)
    _last_dedup_dropped: list = field(default_factory=list, repr=False)

    def fit(self, market, domain):
        self.market = market
        self.domain = domain
        g = domain.geometry(market)
        self.R = g["radius"]
        self._build_modes()
        constructed = len(self.modes)
        self._compute_norms()
        self._dedupe_near_degenerate_modes()
        low_norm = [md.meta for md in self.modes if md.norm <= 1e-6]
        n_candidates = sum(self.M_radial_max for _ in range(self.N_angular_max + 1))
        self.mode_ledger = dict(
            basis="BesselBasis", candidates=n_candidates, constructed=constructed,
            dedup_dropped=self._last_dedup_dropped,
            n_dedup_dropped=len(self._last_dedup_dropped),
            post_dedup=len(self.modes), downstream_low_norm=low_norm,
            n_downstream_low_norm=len(low_norm),
            active=int(sum(1 for md in self.modes if md.norm > 1e-6)),
        )
        return self

    def _dedupe_near_degenerate_modes(self):
        """Check that the modes are orthonormal on the quadrature grid and drop
        near-duplicates."""
        gl_x, gl_w = np.polynomial.legendre.leggauss(self.N_r_quad)
        r_nodes = 0.5 * self.R * (gl_x + 1)
        r_weights = 0.5 * self.R * gl_w
        theta_nodes = np.linspace(0, 2 * np.pi, self.N_theta_quad, endpoint=False)
        dtheta = 2 * np.pi / self.N_theta_quad
        quad_weight_flat = (r_weights[:, None] * dtheta * r_nodes[:, None]
                           * np.ones_like(theta_nodes)[None, :]).ravel()

        psi_flat = np.empty((len(self.modes), quad_weight_flat.shape[0]))
        for k, md in enumerate(self.modes):
            n, j_nm, parity = md.meta["n"], md.meta["j_nm"], md.meta["parity"]
            radial = jn(n, j_nm * r_nodes / self.R)
            angular = (np.cos(n * theta_nodes) if parity == "even"
                      else np.sin(n * theta_nodes))
            norm = md.norm if md.norm > 1e-15 else 1.0
            psi_flat[k] = (radial[:, None] * angular[None, :]).ravel() / norm

        self.modes, self.gram_matrix, self._last_dedup_dropped = gram_dedupe(
            self.modes, psi_flat, quad_weight_flat, label="BesselBasis",
        )

    # ── Eigenvalues = zeros of J_n ───────────────────────────────
    def _build_modes(self):
        modes = []
        for n in range(self.N_angular_max + 1):
            zeros = jn_zeros(n, self.M_radial_max)
            for m, j_nm in enumerate(zeros, start=1):
                k_sq = (j_nm / self.R) ** 2
                # Even cosine mode always exists
                modes.append(EigenMode(
                    idx=(n, m, "even"),
                    k_sq=float(k_sq),
                    meta=dict(n=n, m=m, parity="even", j_nm=float(j_nm)),
                ))
                # Sine mode exists only for n ≥ 1
                if n >= 1:
                    modes.append(EigenMode(
                        idx=(n, m, "odd"),
                        k_sq=float(k_sq),
                        meta=dict(n=n, m=m, parity="odd", j_nm=float(j_nm)),
                    ))
        modes.sort(key=lambda md: md.k_sq)
        self.modes = modes

    def _compute_norms(self):
        """Closed-form L² norms on the disk of radius R."""
        for md in self.modes:
            n, m = md.meta["n"], md.meta["m"]
            j_nm = md.meta["j_nm"]
            parity = md.meta["parity"]
            ang = 2 * np.pi if (n == 0 and parity == "even") else np.pi
            radial = 0.5 * self.R**2 * jn(n + 1, j_nm) ** 2
            md.norm = float(np.sqrt(ang * radial))

    # ── Coord helpers ───────────────────────────────────────────
    def _spot_to_polar(self, S1, S2):
        """(S1, S2) → (r, θ) in whitened z-coords."""
        x1, x2 = self.market.spot_to_x(S1, S2)
        z1, z2 = self.market.x_to_z(x1, x2)
        r = np.hypot(z1, z2)
        theta = np.arctan2(z2, z1)   # θ∈(-π,π]; sign convention fixed here
        return r, theta

    def _polar_to_spot(self, r, theta):
        z1 = r * np.cos(theta)
        z2 = r * np.sin(theta)
        x1, x2 = self.market.z_to_x(z1, z2)
        return self.market.S1_0 * np.exp(x1), self.market.S2_0 * np.exp(x2)

    # ── ψ evaluation ────────────────────────────────────────────
    def _psi_polar(self, mode, r, theta):
        n = mode.meta["n"]
        j_nm = mode.meta["j_nm"]
        radial = jn(n, j_nm * r / self.R)
        if mode.meta["parity"] == "even":
            angular = np.cos(n * theta)
        else:
            angular = np.sin(n * theta)
        return radial * angular

    def psi_at_spot(self, mode, S1, S2):
        r, theta = self._spot_to_polar(S1, S2)
        if r > self.R:
            return 0.0
        return self._psi_polar(mode, r, theta)

    # ── Gauss–Legendre × uniform collocation grid on the disk ───
    def collocation_grid_spot(self):
        gl_x, gl_w = np.polynomial.legendre.leggauss(self.N_r_quad)
        # Map Gauss–Legendre nodes to (0, R) — open interval so r > 0 always
        r_nodes = 0.5 * self.R * (gl_x + 1)
        r_weights = 0.5 * self.R * gl_w
        # Drop any node exactly on the boundary (unlikely but defensive)
        keep = r_nodes < self.R - 1e-12
        r_nodes = r_nodes[keep]
        r_weights = r_weights[keep]
        theta_nodes = np.linspace(0, 2 * np.pi, self.N_theta_quad,
                                  endpoint=False)
        r_flat = np.repeat(r_nodes, len(theta_nodes))
        th_flat = np.tile(theta_nodes, len(r_nodes))
        S1_flat = np.empty_like(r_flat)
        S2_flat = np.empty_like(r_flat)
        for i in range(len(r_flat)):
            S1_flat[i], S2_flat[i] = self._polar_to_spot(r_flat[i], th_flat[i])
        self._colloc_r = r_flat
        self._colloc_theta = th_flat
        self._colloc_r_weights = np.repeat(r_weights, len(theta_nodes))
        return S1_flat, S2_flat

    def design_matrix_at_spot(self, S1_arr, S2_arr):
        n_pts = len(S1_arr)
        n_modes = len(self.modes)
        if hasattr(self, "_colloc_r") and len(self._colloc_r) == n_pts:
            r_flat = self._colloc_r
            th_flat = self._colloc_theta
        else:
            r_flat = np.empty(n_pts)
            th_flat = np.empty(n_pts)
            for i in range(n_pts):
                r_flat[i], th_flat[i] = self._spot_to_polar(
                    S1_arr[i], S2_arr[i])

        # Precompute J_n(j_nm r/R) and cos/sin(nθ) for speed
        A = np.zeros((n_pts, n_modes))
        for k, md in enumerate(self.modes):
            n = md.meta["n"]; j_nm = md.meta["j_nm"]
            parity = md.meta["parity"]
            radial = jn(n, j_nm * r_flat / self.R)
            angular = np.cos(n * th_flat) if parity == "even" else np.sin(n * th_flat)
            norm = md.norm if md.norm > 1e-15 else 1.0
            A[:, k] = radial * angular / norm
        return A

    # ── Analytic derivatives for Greeks ─────────────────────────
    def eval_with_derivatives(self, S1, S2, tau_eff):
        """Return u, ∂u/∂z₁, ∂u/∂z₂, Σc·ψ·(−k²/2) (for θ)."""
        x1, x2 = self.market.spot_to_x(S1, S2)
        z1, z2 = self.market.x_to_z(x1, x2)
        r = np.hypot(z1, z2)
        if r > self.R:
            return 0.0, 0.0, 0.0, 0.0
        theta = np.arctan2(z2, z1)

        u = du_dz1 = du_dz2 = theta_sum = 0.0
        for md in self.modes:
            c = md.coeff
            if abs(c) < 1e-15:
                continue
            n = md.meta["n"]; j_nm = md.meta["j_nm"]
            parity = md.meta["parity"]
            norm = md.norm if md.norm > 1e-15 else 1.0
            tf = md.decay(tau_eff)

            kr = j_nm / self.R
            rad = jn(n, kr * r)
            radp = jvp(n, kr * r, 1) * kr        # d/dr  J_n(kr·r)
            if parity == "even":
                ang = np.cos(n * theta)
                angp = -n * np.sin(n * theta)
            else:
                ang = np.sin(n * theta)
                angp = n * np.cos(n * theta)

            psi_n = rad * ang / norm
            dpsi_dr = radp * ang / norm
            dpsi_dth = rad * angp / norm

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            if r > 1e-12:
                dpsi_dz1 = dpsi_dr * cos_t - dpsi_dth * sin_t / r
                dpsi_dz2 = dpsi_dr * sin_t + dpsi_dth * cos_t / r
            elif n == 1:
                half_kr = kr / (2.0 * norm)
                dpsi_dz1 = half_kr if parity == "even" else 0.0
                dpsi_dz2 = half_kr if parity == "odd" else 0.0
            else:
                dpsi_dz1 = 0.0
                dpsi_dz2 = 0.0

            u         += c * psi_n * tf
            du_dz1    += c * dpsi_dz1 * tf
            du_dz2    += c * dpsi_dz2 * tf
            theta_sum += c * psi_n * (-md.k_sq / 2) * tf
        return u, du_dz1, du_dz2, theta_sum
