"""
Finite differences in whitened coordinates for the curved knockout barriers.

The x-space solver (`fdm.py`) converges only at first order on curved barriers: its
mixed-derivative stencil sets diagonal neighbours outside the region to zero, and at
ρ = 0, where that term vanishes, the same solver is second order. After a whitening z = W x with W Σ Wᵀ = I the
price satisfies

    ∂τ V = ½ Δ_z V + b·∇_z V − r V,      b = W μ,  μ = (r − ½σ₁², r − ½σ₂²),

which has no mixed derivative, so Shortley–Weller treats every term next to the
barrier. The solver works on the undiscounted price V, with no exponential
transformation, and the default whitening is the Cholesky factor W = L⁻¹
(Σ = L Lᵀ), whose axes are not those of the spectral basis. It therefore shares
neither the gauge nor the whitening map with the spectral pipeline.

  • Barrier xᵀ A x < R² in x  →  zᵀ B z < R² with B = W⁻ᵀ A W⁻¹:
      EuclideanCircle      A = I
      MahalanobisEllipse   A = Σ_b⁻¹, with Σ_b the market's Σ by default (the
                           covariance-scaled barrier, as the other solvers do) or a
                           fixed contract matrix (`barrier_Sigma`) for fixed-contract
                           vegas, whose image under a bumped Σ is a rotated ellipse.
  • Grid: uniform spacing h in both z-axes, h = max_k(half_k)/m, nodes at (k₁h, k₂h);
    the spot z = 0 is a node on every grid.
  • Space: three-point central differences, Shortley–Weller non-uniform stencils
    next to the barrier with the arm from the exact root of the quadratic form.
    Off-diagonals are non-negative while |b_k| h ≤ 1 (asserted), so −A_h is a
    diagonally dominant M-matrix.
  • Time: Crank–Nicolson with Rannacher start-up (2·rannacher_steps implicit-Euler
    half-steps); the half-step matrix equals the Crank–Nicolson one, so a single LU.

Rectangles are not handled here: their grid is aligned with the barrier in x.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..core.domain import EuclideanCircle, MahalanobisEllipse
from .fdm_grid_greeks import bdf2_theta, local_derivatives, spot_greeks_from_x

_ARM_MIN = 1e-6        # smallest Shortley–Weller arm, in units of h (as in fdm.py)
# Fill-reducing LU ordering. The runs of 28 Oct 2026 used this one: with it the stored
# levels of paper1_zfdm_levels.json are reproduced bit for bit, and with SciPy's
# default (COLAMD) only to about 1e-13 relative.
_PERMC = "MMD_AT_PLUS_A"


# ═══════════════════════════════════════════════════════════════════════
#  Coordinates and barrier
# ═══════════════════════════════════════════════════════════════════════

def covariance(sigma1: float, sigma2: float, rho: float) -> np.ndarray:
    return np.array([[sigma1**2, rho * sigma1 * sigma2],
                     [rho * sigma1 * sigma2, sigma2**2]])


def whitening_matrix(Sigma: np.ndarray, kind: str = "cholesky") -> np.ndarray:
    """W with W Σ Wᵀ = I: inverse Cholesky factor, or D^(-1/2) Pᵀ ("eigen")."""
    if kind == "cholesky":
        return np.linalg.inv(np.linalg.cholesky(Sigma))
    if kind == "eigen":
        d, P = np.linalg.eigh(Sigma)
        return np.diag(1.0 / np.sqrt(d)) @ P.T
    raise ValueError(f"unknown whitening {kind!r}")


def barrier_form_x(domain, Sigma_barrier: np.ndarray):
    """(A, R²) of the barrier xᵀ A x < R² in log-return coordinates."""
    if isinstance(domain, EuclideanCircle):
        return np.eye(2), float(domain.R**2)
    if isinstance(domain, MahalanobisEllipse):
        return np.linalg.inv(Sigma_barrier), float(domain.R**2)
    raise ValueError(f"z-space FD treats the curved barriers only, not "
                     f"{type(domain).__name__}")


# ═══════════════════════════════════════════════════════════════════════
#  Grid and operator
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ZGrid:
    """Nodes (k₁h, k₂h), |k_i| ≤ K_i, of the region zᵀ B z < R²."""
    h: float
    R2: float
    K1: int
    K2: int
    B: np.ndarray = field(repr=False)
    inside: np.ndarray = field(repr=False)      # (2K1+1, 2K2+1) bool
    idx: np.ndarray = field(repr=False)         # flat index of inside nodes, else -1
    n_unknowns: int = 0

    @property
    def ii(self):
        return np.nonzero(self.inside)[0]

    @property
    def jj(self):
        return np.nonzero(self.inside)[1]


def build_grid(B: np.ndarray, R2: float, m: int) -> ZGrid:
    half = np.sqrt(R2 * np.diag(np.linalg.inv(B)))     # bounding-box half-widths
    h = float(half.max() / m)
    K1, K2 = (int(np.ceil(hk / h)) + 1 for hk in half)
    Z1, Z2 = np.meshgrid(np.arange(-K1, K1 + 1) * h, np.arange(-K2, K2 + 1) * h,
                         indexing="ij")
    inside = (B[0, 0] * Z1**2 + 2 * B[0, 1] * Z1 * Z2 + B[1, 1] * Z2**2) < R2
    idx = np.full(inside.shape, -1, dtype=np.int64)
    n = int(inside.sum())
    idx[inside] = np.arange(n)
    return ZGrid(h=h, R2=float(R2), K1=K1, K2=K2, B=np.asarray(B, float),
                 inside=inside, idx=idx, n_unknowns=n)


def _arm_fraction(grid: ZGrid, Q0, z1, z2, axis: int, sign: int):
    """t ∈ (0, 1] with the barrier at z + t·sign·h·e_axis (positive root)."""
    B, h = grid.B, grid.h
    a = h * h * B[axis, axis]
    bq = 2.0 * sign * h * (B[axis, 0] * z1 + B[axis, 1] * z2)
    c = Q0 - grid.R2                                        # < 0 inside
    t = (-bq + np.sqrt(bq * bq - 4.0 * a * c)) / (2.0 * a)
    return np.clip(t, _ARM_MIN, 1.0)


def assemble_operator(grid: ZGrid, b, r: float):
    """Sparse A_h of ∂τV = ½ΔV + b·∇V − rV with V = 0 on the barrier."""
    b = np.asarray(b, dtype=float)
    if np.max(np.abs(b)) * grid.h > 1.0:
        raise ValueError(f"|b| h = {np.max(np.abs(b)) * grid.h:.3f} > 1: the "
                         "central drift stencil would lose its M-matrix sign pattern")
    ii, jj = np.nonzero(grid.inside)
    rows = grid.idx[ii, jj]
    z1 = (ii - grid.K1) * grid.h
    z2 = (jj - grid.K2) * grid.h
    B = grid.B
    Q0 = B[0, 0] * z1**2 + 2 * B[0, 1] * z1 * z2 + B[1, 1] * z2**2
    diag = np.full(len(rows), -float(r))
    R_, C_, V_ = [], [], []
    for axis in (0, 1):
        nbr, arm = {}, {}
        for sign in (1, -1):
            di, dj = (sign, 0) if axis == 0 else (0, sign)
            nidx = grid.idx[ii + di, jj + dj]
            t = np.ones(len(rows))
            out = nidx < 0
            t[out] = _arm_fraction(grid, Q0[out], z1[out], z2[out], axis, sign)
            nbr[sign], arm[sign] = nidx, t * grid.h
        hp, hm, bk = arm[1], arm[-1], b[axis]
        c_plus = (1.0 + bk * hm) / (hp * (hp + hm))
        c_minus = (1.0 - bk * hp) / (hm * (hp + hm))
        diag += -1.0 / (hp * hm) + bk * (hp - hm) / (hp * hm)
        for sign, coef in ((1, c_plus), (-1, c_minus)):
            ok = nbr[sign] >= 0
            R_.append(rows[ok]); C_.append(nbr[sign][ok]); V_.append(coef[ok])
    R_.append(rows); C_.append(rows); V_.append(diag)
    n = grid.n_unknowns
    return sparse.csr_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))),
                             shape=(n, n))


def time_step(A, V0: np.ndarray, T: float, N_time: int, *, rannacher_steps: int = 4,
              permc_spec: str = _PERMC, timings: dict | None = None):
    """Crank–Nicolson with Rannacher start-up; returns (V(T), last three levels, dt).
    If `timings` is given, the factorisation and stepping times are stored in it."""
    t0 = time.perf_counter()
    n = A.shape[0]
    I = sparse.identity(n, format="csc")
    A = A.tocsc()
    dt = T / N_time
    lu = splu((I - 0.5 * dt * A).tocsc(), permc_spec=permc_spec)
    t1 = time.perf_counter()
    V = V0.copy()
    n_cn = N_time
    if rannacher_steps > 0:
        for _ in range(2 * rannacher_steps):     # implicit Euler, step dt/2
            V = lu.solve(V)
        n_cn -= rannacher_steps
    Bm = (I + 0.5 * dt * A).tocsr()
    levels = deque([V], maxlen=3)
    for _ in range(n_cn):
        V = lu.solve(Bm @ V)
        levels.append(V)
    if timings is not None:
        timings.update(lu_s=t1 - t0, steps_s=time.perf_counter() - t1)
    return V, list(levels), dt


# ═══════════════════════════════════════════════════════════════════════
#  Market-level solve
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ZSolution:
    S1_0: float
    S2_0: float
    m: int
    N_time: int
    dt: float
    grid: ZGrid = field(repr=False)
    W: np.ndarray = field(repr=False)
    A: object = field(repr=False)
    V: np.ndarray = field(repr=False)
    levels: list = field(repr=False)
    wall: dict = field(default_factory=dict)
    _fields: dict = field(default_factory=dict, repr=False)

    def _field(self, vec) -> np.ndarray:
        U = np.zeros(self.grid.inside.shape)
        U[self.grid.inside] = vec
        return U

    def fields(self):
        """Grid arrays of V, of the generator A_h V and of the BDF2 theta,
        built once per solution."""
        if not self._fields:
            self._fields.update(V=self._field(self.V), AV=self._field(self.A @ self.V),
                                bdf2=self._field(bdf2_theta(self.levels, self.dt)))
        return self._fields

    def _fractional_index(self, S1, S2):
        x = np.array([np.log(S1 / self.S1_0), np.log(S2 / self.S2_0)])
        z = self.W @ x
        return z[0] / self.grid.h + self.grid.K1, z[1] / self.grid.h + self.grid.K2

    def price(self, S1=None, S2=None) -> float:
        S1 = self.S1_0 if S1 is None else S1
        S2 = self.S2_0 if S2 is None else S2
        fi, fj = self._fractional_index(S1, S2)
        v, _, _ = local_derivatives(self.fields()["V"], self.grid.inside, fi, fj,
                                    self.grid.h, self.grid.h)
        return v

    def greeks(self, S1=None, S2=None) -> dict:
        """Price and spot Greeks from this solution: derivatives in z by central
        differences at a node (4×4 Lagrange otherwise), mapped to x by
        ∇ₓ = Wᵀ∇_z and Hₓ = Wᵀ H_z W; theta from the semidiscrete generator
        (Θ = −A_h V) and, as a check, from BDF2 over the last three levels."""
        S1 = self.S1_0 if S1 is None else S1
        S2 = self.S2_0 if S2 is None else S2
        fi, fj = self._fractional_index(S1, S2)
        h, ins, F = self.grid.h, self.grid.inside, self.fields()
        v, gz, Hz = local_derivatives(F["V"], ins, fi, fj, h, h)
        av, _, _ = local_derivatives(F["AV"], ins, fi, fj, h, h)
        tb, _, _ = local_derivatives(F["bdf2"], ins, fi, fj, h, h)
        gx = self.W.T @ gz
        Hx = self.W.T @ Hz @ self.W
        out = dict(price=v, **spot_greeks_from_x(S1, S2, gx, Hx))
        out.update(theta=-av, theta_bdf2=tb)
        return out


def fdm_z_solve(market, domain, payoff, *, m: int = 160, N_time: int = 500,
                whitening: str = "cholesky", barrier_Sigma=None,
                rannacher_steps: int = 4, permc_spec: str = _PERMC) -> ZSolution:
    """Solve once in whitened coordinates.

    Only the raw inputs of `market` are read (S1_0, S2_0, sigma1, sigma2, rho, r, T).
    `barrier_Sigma` fixes the covariance in a Mahalanobis barrier; by default it is
    the market's own Σ, as in the spectral, x-space and Monte Carlo solvers."""
    t0 = time.perf_counter()
    Sigma = covariance(market.sigma1, market.sigma2, market.rho)
    W = whitening_matrix(Sigma, whitening)
    Winv = np.linalg.inv(W)
    A_x, R2 = barrier_form_x(domain, Sigma if barrier_Sigma is None else barrier_Sigma)
    B = Winv.T @ A_x @ Winv
    mu = np.array([market.r - 0.5 * market.sigma1**2, market.r - 0.5 * market.sigma2**2])
    grid = build_grid(B, R2, m)
    A = assemble_operator(grid, W @ mu, market.r)
    ii, jj = np.nonzero(grid.inside)
    z = np.vstack([(ii - grid.K1) * grid.h, (jj - grid.K2) * grid.h])
    x = Winv @ z
    V0 = np.asarray(payoff(market.S1_0 * np.exp(x[0]), market.S2_0 * np.exp(x[1])),
                    dtype=float)
    t1 = time.perf_counter()
    split = {}
    V, levels, dt = time_step(A, V0, market.T, N_time, rannacher_steps=rannacher_steps,
                              permc_spec=permc_spec, timings=split)
    t2 = time.perf_counter()
    return ZSolution(S1_0=market.S1_0, S2_0=market.S2_0, m=m, N_time=N_time, dt=dt,
                     grid=grid, W=W, A=A, V=V, levels=levels,
                     wall=dict(assemble_s=t1 - t0, solve_s=t2 - t1, total_s=t2 - t0,
                               n_unknowns=grid.n_unknowns, **split))


def _bumped(market, **kw):
    from ..core.market import GBMMarket
    raw = dict(S1_0=market.S1_0, S2_0=market.S2_0, sigma1=market.sigma1,
               sigma2=market.sigma2, rho=market.rho, r=market.r, T=market.T)
    raw.update(kw)
    return GBMMarket(**raw)


def fdm_z_greeks(market, domain, payoff, *, m: int = 160, N_time: int = 500,
                 vol_bump: float = 0.01, rate_bump: float = 1e-4,
                 market_greeks: bool = True, **solve_kw) -> dict:
    """Price, spot Greeks from one solve, and the market Greeks from re-solves.

    The vegas hold the contract fixed (barrier matrix A fixed in x). For a
    Mahalanobis barrier the covariance-scaled vegas (A = Σ(σ)⁻¹, what the other
    solvers report) are `vega1`/`vega2`, and the fixed-contract ones
    (A = Σ₀⁻¹) are `vega1_fixed`/`vega2_fixed`."""
    t0 = time.perf_counter()
    base = fdm_z_solve(market, domain, payoff, m=m, N_time=N_time, **solve_kw)
    out = base.greeks()
    out["wall_one_solve_s"] = base.wall["total_s"]
    out["wall_spot_greeks_s"] = time.perf_counter() - t0
    out["n_unknowns"] = base.grid.n_unknowns
    if not market_greeks:
        return out

    def P(mk, **kw):
        return fdm_z_solve(mk, domain, payoff, m=m, N_time=N_time, **solve_kw, **kw).price()

    Sigma0 = covariance(market.sigma1, market.sigma2, market.rho)
    for i, s in ((1, "sigma1"), (2, "sigma2")):
        up = _bumped(market, **{s: getattr(market, s) + vol_bump})
        dn = _bumped(market, **{s: getattr(market, s) - vol_bump})
        out[f"vega{i}"] = (P(up) - P(dn)) / (2 * vol_bump)
        if isinstance(domain, MahalanobisEllipse):
            out[f"vega{i}_fixed"] = (P(up, barrier_Sigma=Sigma0)
                                     - P(dn, barrier_Sigma=Sigma0)) / (2 * vol_bump)
    rp = _bumped(market, r=market.r + rate_bump)
    rm = _bumped(market, r=market.r - rate_bump)
    out["rho"] = (P(rp) - P(rm)) / (2 * rate_bump)
    out["wall_all_greeks_s"] = time.perf_counter() - t0
    return out
