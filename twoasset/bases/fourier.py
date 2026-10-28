"""Tensor-product Fourier-sine basis on an axis-aligned rectangle in x-space."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .base import EigenBasis, EigenMode, gram_dedupe


def _sc_integral(n, m):
    """∫₀^W sin(nπx/W) · (mπ/W)cos(mπx/W) dx / W, integer indices."""
    if n == m or (n + m) % 2 == 0:
        return 0.0
    return n * m / (n * n - m * m) * (-2.0)


@dataclass
class FourierBasis(EigenBasis):
    """Fourier-sine basis (n,m = 1 … N) on a rectangle in x-space."""
    N_modes: int = 25          # per dimension — total modes = N_modes²
    N_quad: int = 120

    modes: list = field(default_factory=list)
    market: object = None
    domain: object = None
    gram_matrix: np.ndarray = field(default=None, repr=False)
    mode_ledger: dict = field(default=None, repr=False)
    _last_dedup_dropped: list = field(default_factory=list, repr=False)

    # Filled by _build_modes:
    _evals: np.ndarray = field(default=None, repr=False)
    _evecs: np.ndarray = field(default=None, repr=False)
    _raw_ml: list = field(default_factory=list, repr=False)
    _use_gal: bool = False

    def fit(self, market, domain):
        self.market = market
        self.domain = domain
        g = domain.geometry(market)
        self.W1 = g["W1"]
        self.W2 = g["W2"]
        self.x1_lo = g["x1_lo"]
        self.x2_lo = g["x2_lo"]
        self._build_modes()
        constructed = len(self.modes)
        self._dedupe_near_degenerate_modes()
        low_norm = [md.meta for md in self.modes if md.norm <= 1e-6]
        self.mode_ledger = dict(
            basis="FourierBasis", candidates=self.N_modes ** 2,
            constructed=constructed, dedup_dropped=self._last_dedup_dropped,
            n_dedup_dropped=len(self._last_dedup_dropped),
            post_dedup=len(self.modes), downstream_low_norm=low_norm,
            n_downstream_low_norm=len(low_norm),
            active=int(sum(1 for md in self.modes if md.norm > 1e-6)),
        )
        return self

    def _dedupe_near_degenerate_modes(self):
        """Check that the modes are orthonormal on a grid fine enough to avoid
        aliasing, and drop near-duplicates."""
        n_grid = max(40, self.N_modes + 1)
        x1 = np.linspace(self.x1_lo, self.x1_lo + self.W1, n_grid + 2)[1:-1]
        x2 = np.linspace(self.x2_lo, self.x2_lo + self.W2, n_grid + 2)[1:-1]
        dx1 = x1[1] - x1[0]
        dx2 = x2[1] - x2[0]
        quad_weight_flat = np.full(n_grid * n_grid, dx1 * dx2)

        raw_vecs = np.empty((len(x1) * len(x2), len(self._raw_ml)))
        idx = 0
        for xx1 in x1:
            for xx2 in x2:
                raw_vecs[idx] = self._sine_vec(xx1, xx2)
                idx += 1

        psi_flat = np.empty((len(self.modes), n_grid * n_grid))
        for k, md in enumerate(self.modes):
            ev_idx = md.meta["eval_idx"]
            norm = md.norm if md.norm > 1e-15 else 1.0
            psi_flat[k] = (raw_vecs @ self._evecs[:, ev_idx]) / norm

        self.modes, self.gram_matrix, self._last_dedup_dropped = gram_dedupe(
            self.modes, psi_flat, quad_weight_flat, label="FourierBasis",
        )

    def _build_modes(self):
        m = self.market
        N = self.N_modes
        raw_ml = [(n, k) for n in range(1, N + 1) for k in range(1, N + 1)]
        M = len(raw_ml)
        diag = np.array([m.a11 * (n * np.pi / self.W1)**2
                         + m.a22 * (k * np.pi / self.W2)**2
                         for n, k in raw_ml])
        if abs(m.a12) > 1e-15:
            H = np.diag(diag)
            nf = 4.0 / (self.W1 * self.W2)
            for i, (n1, k1) in enumerate(raw_ml):
                for j, (n2, k2) in enumerate(raw_ml):
                    if i == j:
                        continue
                    s1 = _sc_integral(n1, n2)
                    s2 = _sc_integral(k1, k2)
                    if s1 != 0.0 and s2 != 0.0:
                        H[i, j] -= m.a12 * nf * s1 * s2
            evals, evecs = np.linalg.eigh(H)
            idx = np.argsort(evals)
            self._evals = evals[idx]
            self._evecs = evecs[:, idx]
            self._use_gal = True
        else:
            idx = np.argsort(diag)
            self._evals = diag[idx]
            self._evecs = np.eye(M)[:, idx]
            self._use_gal = False
        self._raw_ml = raw_ml

        norm_common = np.sqrt(self.W1 * self.W2 / 4.0)

        self.modes = []
        for k in range(M):
            self.modes.append(EigenMode(
                idx=(k,),
                k_sq=float(2.0 * self._evals[k]),
                norm=float(norm_common),
                meta=dict(eval_idx=k),
            ))

    # ── ψ evaluation ────────────────────────────────────────────
    def _sine_vec(self, x1, x2):
        """Values of every raw sine product at (x1, x2)."""
        N = self.N_modes
        sv1 = np.sin(np.arange(1, N + 1) * np.pi * (x1 - self.x1_lo) / self.W1)
        sv2 = np.sin(np.arange(1, N + 1) * np.pi * (x2 - self.x2_lo) / self.W2)
        return np.outer(sv1, sv2).ravel()

    def psi_at_spot(self, mode, S1, S2):
        x1, x2 = self.market.spot_to_x(S1, S2)
        if not self.domain.contains(x1, x2):
            return 0.0
        raw = self._sine_vec(x1, x2)
        k = mode.meta["eval_idx"]
        return float(self._evecs[:, k] @ raw)

    # ── Derivatives in the natural (x₁, x₂) coordinates ───────────
    def _sine_cos_vecs(self, x1, x2):
        """Return (sin·sin, cos·sin, sin·cos) raw-mode vectors."""
        N = self.N_modes
        n_arr = np.arange(1, N + 1)
        s1 = np.sin(n_arr * np.pi * (x1 - self.x1_lo) / self.W1)
        c1 = np.cos(n_arr * np.pi * (x1 - self.x1_lo) / self.W1)
        s2 = np.sin(n_arr * np.pi * (x2 - self.x2_lo) / self.W2)
        c2 = np.cos(n_arr * np.pi * (x2 - self.x2_lo) / self.W2)
        ss = np.outer(s1, s2).ravel()    # value
        cs = np.outer(c1, s2).ravel()    # ∂/∂x₁ before prefactor
        sc = np.outer(s1, c2).ravel()    # ∂/∂x₂ before prefactor
        return ss, cs, sc

    def eval_with_derivatives(self, S1, S2, tau_eff):
        """Evaluate u, ∂u/∂x₁, ∂u/∂x₂, and ∂u/∂τ at a single spot point."""
        x1, x2 = self.market.spot_to_x(S1, S2)
        if not self.domain.contains(x1, x2):
            return 0.0, 0.0, 0.0, 0.0

        ss, cs, sc = self._sine_cos_vecs(x1, x2)

        # Prefactors n·π/W₁ and m·π/W₂ live on raw-mode index axes.
        N = self.N_modes
        n_arr = np.arange(1, N + 1) * np.pi / self.W1
        m_arr = np.arange(1, N + 1) * np.pi / self.W2
        pref_x1 = np.repeat(n_arr, N)        # (N²,) — n-index outer
        pref_x2 = np.tile(m_arr, N)          # (N²,) — m-index inner

        norm_common = self.modes[0].norm
        coeffs_psi = np.array([md.coeff for md in self.modes])
        decays = np.array([np.exp(-0.5 * md.k_sq * tau_eff) for md in self.modes])
        # Coefficient on raw-sine basis at this τ:
        c_raw = (self._evecs @ (coeffs_psi * decays)) / norm_common

        u_val   = float(c_raw @ ss)
        du_dx1  = float((c_raw * pref_x1) @ cs)
        du_dx2  = float((c_raw * pref_x2) @ sc)

        ksq = np.array([md.k_sq for md in self.modes])
        c_raw_tau = (self._evecs @ (coeffs_psi * decays * (-0.5 * ksq))) / norm_common
        theta_sum = float(c_raw_tau @ ss)

        return u_val, du_dx1, du_dx2, theta_sum

    # ── Collocation ─────────────────────────────────────────────
    def collocation_grid_spot(self):
        Nq = self.N_quad
        x1q = np.linspace(self.x1_lo, self.x1_lo + self.W1, Nq + 2)[1:-1]
        x2q = np.linspace(self.x2_lo, self.x2_lo + self.W2, Nq + 2)[1:-1]
        x1_flat = np.repeat(x1q, len(x2q))
        x2_flat = np.tile(x2q, len(x1q))
        S1_flat = self.market.S1_0 * np.exp(x1_flat)
        S2_flat = self.market.S2_0 * np.exp(x2_flat)
        self._colloc_x1 = x1_flat
        self._colloc_x2 = x2_flat
        return S1_flat, S2_flat

    def design_matrix_at_spot(self, S1_arr, S2_arr):
        n_pts = len(S1_arr)
        n_modes = len(self.modes)
        if hasattr(self, "_colloc_x1") and len(self._colloc_x1) == n_pts:
            x1_arr = self._colloc_x1
            x2_arr = self._colloc_x2
        else:
            x1_arr, x2_arr = self.market.spot_to_x(S1_arr, S2_arr)
        # Precompute raw sine values: shape (n_pts, M)
        N = self.N_modes
        n_arr = np.arange(1, N + 1)
        sin1 = np.sin(n_arr[None, :] * np.pi * (x1_arr[:, None] - self.x1_lo) / self.W1)
        sin2 = np.sin(n_arr[None, :] * np.pi * (x2_arr[:, None] - self.x2_lo) / self.W2)
        raw = (sin1[:, :, None] * sin2[:, None, :]).reshape(n_pts, N * N)
        # A = raw @ evecs / ‖ψ‖
        norm_common = self.modes[0].norm
        A = (raw @ self._evecs) / norm_common
        return A

    def project_galerkin(self, payoff, *, smooth=True, smooth_tau_frac=0.005):
        m = self.market
        Nq = self.N_quad
        x1q = np.linspace(self.x1_lo, self.x1_lo + self.W1, Nq + 2)[1:-1]
        x2q = np.linspace(self.x2_lo, self.x2_lo + self.W2, Nq + 2)[1:-1]
        dx1 = x1q[1] - x1q[0]
        dx2 = x2q[1] - x2q[0]

        u0 = np.empty((Nq, Nq))
        for i in range(Nq):
            S1 = m.S1_0 * np.exp(x1q[i])
            for j in range(Nq):
                S2 = m.S2_0 * np.exp(x2q[j])
                g = m.gauge_factor(S1, S2, 0.0)
                u0[i, j] = float(payoff(S1, S2)) / g if abs(g) > 1e-100 else 0.0

        dtau = 0.0
        if smooth:
            dtau = smooth_tau_frac * m.T
            ns = max(int(dtau / (0.3 * min(dx1, dx2) ** 2)), 10)
            dt_s = dtau / ns
            for _ in range(ns):
                un = u0.copy()
                uxx = (u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[:-2, 1:-1]) / dx1**2
                uyy = (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2]) / dx2**2
                uxy = (u0[2:, 2:] - u0[2:, :-2] - u0[:-2, 2:] + u0[:-2, :-2]) \
                      / (4 * dx1 * dx2)
                un[1:-1, 1:-1] += dt_s * (m.a11 * uxx + m.a12 * uxy + m.a22 * uyy)
                u0 = un

        # Raw-sine Galerkin projection onto the untransformed (sin_n sin_k) basis
        N = self.N_modes
        sin1 = np.array([np.sin(n * np.pi * (x1q - self.x1_lo) / self.W1)
                         for n in range(1, N + 1)])
        sin2 = np.array([np.sin(k * np.pi * (x2q - self.x2_lo) / self.W2)
                         for k in range(1, N + 1)])
        nsq = self.W1 * self.W2 / 4.0
        rc = np.empty(N * N)
        for p_idx, (n, k) in enumerate(self._raw_ml):
            rc[p_idx] = (np.sum(u0 * sin1[n - 1][:, None] * sin2[k - 1][None, :])
                         * dx1 * dx2 / nsq)

        # Rotate into the true eigenbasis and fold in the stored ‖ψ‖
        if self._use_gal:
            c_raw = self._evecs.T @ rc
        else:
            # argsort permutation was applied to evecs as identity columns;
            # use the same permutation on rc
            c_raw = self._evecs.T @ rc
        # Our stored modes use ψ/‖ψ‖, so coeff_on_normalised = c_raw · ‖ψ‖
        norm_common = self.modes[0].norm
        coeffs = c_raw * norm_common

        for k_idx, md in enumerate(self.modes):
            md.coeff = float(coeffs[k_idx])

        return dict(
            n_modes=len(self.modes),
            n_active=len(self.modes),
            rank=len(self.modes),
            cond=1.0,
            max_coeff=float(np.max(np.abs(coeffs))),
            smooth_tau=float(dtau),
            smoothing=bool(smooth),
        )
