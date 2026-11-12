"""
Spot Greeks from ONE finite-difference solve.

One solve gives the price at every grid node, so the deltas, gammas and theta follow
from that solution without further solves. This module holds the pieces shared by the
two grid solvers:

  • lagrange_weights / choose_stencil — derivatives of a local 4×4 tensor Lagrange
    interpolant (values O(h⁴), first derivatives O(h³), second O(h²)), with the
    stencil shifted inward when it would leave the survival region;
  • spot_greeks_from_x — spot Greeks from V and its log-price derivatives;
  • fdm_x_onesolve — the x-space solver of `fdm.py`, solved once, with the price read
    exactly as `fdm_price` reads it (bit-identical) and the spot Greeks taken either by
    re-reading the solution at the bump-and-revalue points of `fdm_greeks` (which
    shows those ten solves are one solve) or by differentiating the grid.

`fdm.py` is not modified: the system builders and the interpolation are imported
from it, and the time stepping below repeats `fdm_price` operation for operation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..core.domain import Rectangle
from .fdm import (_build_curved_system, _build_rect_system,
                  _initial_condition_curved, _interp2d)


# ═══════════════════════════════════════════════════════════════════════
#  Local interpolation and the spot chain rule
# ═══════════════════════════════════════════════════════════════════════

def lagrange_weights(nodes, x) -> np.ndarray:
    """(3, n) weights of the value, first and second derivative at `x` of the
    Lagrange interpolant through `nodes` (all in the same units)."""
    nodes = np.asarray(nodes, dtype=float)
    w = np.empty((3, len(nodes)))
    for j in range(len(nodes)):
        others = np.delete(nodes, j)
        c = np.poly(others) / np.prod(nodes[j] - others)
        w[0, j] = np.polyval(c, x)
        w[1, j] = np.polyval(np.polyder(c), x)
        w[2, j] = np.polyval(np.polyder(c, 2), x)
    return w


def choose_stencil(inside: np.ndarray, fi: float, fj: float, size: int = 4):
    """Lower-left array index (i0, j0) of a size×size block of inside nodes that
    contains the fractional position (fi, fj) as centrally as possible, or None."""
    ci = int(np.floor(fi)) - (size // 2 - 1)
    cj = int(np.floor(fj)) - (size // 2 - 1)
    shifts = sorted(((di, dj) for di in range(-size + 1, size) for dj in range(-size + 1, size)),
                    key=lambda s: (abs(s[0]) + abs(s[1]), abs(s[0]), s))
    n1, n2 = inside.shape
    for di, dj in shifts:
        i0, j0 = ci + di, cj + dj
        if not (i0 <= fi <= i0 + size - 1 and j0 <= fj <= j0 + size - 1):
            continue
        if i0 < 0 or j0 < 0 or i0 + size > n1 or j0 + size > n2:
            continue
        if inside[i0:i0 + size, j0:j0 + size].all():
            return i0, j0
    return None


def local_derivatives(U: np.ndarray, inside: np.ndarray, fi: float, fj: float,
                      h1: float, h2: float):
    """Value, gradient and Hessian at fractional array index (fi, fj) of the grid
    field U (NaN or anything outside `inside` is never read).

    At a node (fi, fj integral) with all eight neighbours inside, second-order
    central differences are used; elsewhere a 4×4 tensor Lagrange interpolant, or a
    3×3 one where no 4×4 block of inside nodes contains the point. Within about a
    cell of the barrier neither exists and ValueError is raised."""
    ri, rj = round(fi), round(fj)
    if abs(fi - ri) < 1e-9 and abs(fj - rj) < 1e-9:
        i, j = int(ri), int(rj)
        block = inside[i - 1:i + 2, j - 1:j + 2]
        if block.shape == (3, 3) and block.all():
            u = U[i - 1:i + 2, j - 1:j + 2]
            g = np.array([(u[2, 1] - u[0, 1]) / (2 * h1), (u[1, 2] - u[1, 0]) / (2 * h2)])
            H = np.empty((2, 2))
            H[0, 0] = (u[2, 1] - 2 * u[1, 1] + u[0, 1]) / h1**2
            H[1, 1] = (u[1, 2] - 2 * u[1, 1] + u[1, 0]) / h2**2
            H[0, 1] = H[1, 0] = (u[2, 2] - u[2, 0] - u[0, 2] + u[0, 0]) / (4 * h1 * h2)
            return float(u[1, 1]), g, H
    for size in (4, 3):                 # cubic, then quadratic next to the barrier
        st = choose_stencil(inside, fi, fj, size)
        if st is not None:
            break
    else:
        raise ValueError(f"no interior stencil around index ({fi:.3f}, {fj:.3f})")
    i0, j0 = st
    wi = lagrange_weights(np.arange(i0, i0 + size), fi)
    wj = lagrange_weights(np.arange(j0, j0 + size), fj)
    u = U[i0:i0 + size, j0:j0 + size]
    v = float(wi[0] @ u @ wj[0])
    g = np.array([wi[1] @ u @ wj[0] / h1, wi[0] @ u @ wj[1] / h2])
    H = np.empty((2, 2))
    H[0, 0] = wi[2] @ u @ wj[0] / h1**2
    H[1, 1] = wi[0] @ u @ wj[2] / h2**2
    H[0, 1] = H[1, 0] = wi[1] @ u @ wj[1] / (h1 * h2)
    return v, g, H


def spot_greeks_from_x(S1: float, S2: float, Vx: np.ndarray, Hx: np.ndarray) -> dict:
    """Deltas and gammas from the log-price gradient and Hessian of V,
    using ∂/∂S = S⁻¹∂/∂x and ∂²/∂S² = S⁻²(∂²/∂x² − ∂/∂x)."""
    return dict(
        delta1=float(Vx[0] / S1),
        delta2=float(Vx[1] / S2),
        gamma1=float((Hx[0, 0] - Vx[0]) / S1**2),
        gamma2=float((Hx[1, 1] - Vx[1]) / S2**2),
        gamma12=float(Hx[0, 1] / (S1 * S2)),
    )


def bdf2_theta(levels, dt: float) -> np.ndarray:
    """Θ = ∂V/∂t = −∂V/∂τ from the last three time levels (second order)."""
    v2, v1, v0 = levels[-3], levels[-2], levels[-1]
    return -(3.0 * v0 - 4.0 * v1 + v2) / (2.0 * dt)


# ═══════════════════════════════════════════════════════════════════════
#  x-space solver, solved once
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class XSolution:
    """One solve of the x-space solver of `fdm.py`, with what the Greeks need."""
    market: object
    domain: object
    N: int
    N_time: int
    dt: float
    x1: np.ndarray = field(repr=False)
    x2: np.ndarray = field(repr=False)
    inside: np.ndarray = field(repr=False)
    flat_idx: np.ndarray = field(repr=False)
    L: object = field(repr=False)
    u: np.ndarray = field(repr=False)
    levels: list = field(repr=False)
    wall: dict = field(default_factory=dict)
    _fields: dict = field(default_factory=dict, repr=False)

    def fields(self):
        """Grid arrays of u, of the generator L u and of the BDF2 theta, built once."""
        if not self._fields:
            self._fields.update(u=self._grid(self.u), Lu=self._grid(self.L @ self.u),
                                bdf2=self._grid(bdf2_theta(self.levels, self.dt)))
        return self._fields

    # ── price exactly as fdm_price reads it ──
    def price(self, S1=None, S2=None) -> float:
        m = self.market
        S1 = m.S1_0 if S1 is None else S1
        S2 = m.S2_0 if S2 is None else S2
        x1t = float(np.log(S1 / m.S1_0))
        x2t = float(np.log(S2 / m.S2_0))
        return _interp2d(self.u, x1t, x2t, self.x1, self.x2, self.inside,
                         self.flat_idx, len(self.x1))

    def _grid(self, vec) -> np.ndarray:
        U = np.full(self.inside.shape, np.nan)
        U[self.inside] = vec[self.flat_idx[self.inside]]
        return U

    def bump_greeks(self, bump_pct: float = 1.0) -> dict:
        """The spot Greeks of `fdm_greeks.fdm_greeks`, read from this one solution:
        the same central differences in S with bumps of `bump_pct` % of spot, each
        bumped price interpolated from the same grid (theta excluded: `fdm_greeks`
        takes it from a re-solve at a shorter maturity)."""
        m = self.market
        h1 = m.S1_0 * bump_pct / 100.0
        h2 = m.S2_0 * bump_pct / 100.0
        P = self.price
        V0 = P()
        Vp1, Vm1 = P(S1=m.S1_0 + h1), P(S1=m.S1_0 - h1)
        Vp2, Vm2 = P(S2=m.S2_0 + h2), P(S2=m.S2_0 - h2)
        Vpp = P(m.S1_0 + h1, m.S2_0 + h2)
        Vpm = P(m.S1_0 + h1, m.S2_0 - h2)
        Vmp = P(m.S1_0 - h1, m.S2_0 + h2)
        Vmm = P(m.S1_0 - h1, m.S2_0 - h2)
        return dict(
            price=V0,
            delta1=(Vp1 - Vm1) / (2 * h1), delta2=(Vp2 - Vm2) / (2 * h2),
            gamma1=(Vp1 - 2 * V0 + Vm1) / h1**2, gamma2=(Vp2 - 2 * V0 + Vm2) / h2**2,
            gamma12=(Vpp - Vpm - Vmp + Vmm) / (4 * h1 * h2),
        )

    def grid_greeks(self, S1=None, S2=None) -> dict:
        """Spot Greeks by differentiating the grid solution (local 4×4 Lagrange
        interpolant), theta from the semidiscrete generator and from BDF2."""
        m = self.market
        S1 = m.S1_0 if S1 is None else S1
        S2 = m.S2_0 if S2 is None else S2
        x1t, x2t = np.log(S1 / m.S1_0), np.log(S2 / m.S2_0)
        h1 = self.x1[1] - self.x1[0]
        h2 = self.x2[1] - self.x2[0]
        fi = (x1t - self.x1[0]) / h1
        fj = (x2t - self.x2[0]) / h2
        F = self.fields()
        V, g, H = local_derivatives(F["u"], self.inside, fi, fj, h1, h2)
        LV, _, _ = local_derivatives(F["Lu"], self.inside, fi, fj, h1, h2)
        th_b, _, _ = local_derivatives(F["bdf2"], self.inside, fi, fj, h1, h2)
        out = dict(price=V, **spot_greeks_from_x(S1, S2, g, H))
        out.update(theta=-LV, theta_bdf2=th_b)
        return out


def fdm_x_onesolve(market, domain, payoff, *, N: int = 320, N_time: int = 500,
                   rannacher: bool = True, rannacher_steps: int = 4,
                   permc_spec: str = "COLAMD") -> XSolution:
    """Solve the x-space problem once, exactly as `fdm_price` does, and keep the
    operator and the last three time levels for the Greeks.

    With the default `permc_spec` (SuperLU's own default, as in `fdm_price`) the
    solution is bit-identical to `fdm_price`. "MMD_AT_PLUS_A" orders the structurally
    symmetric matrix with less fill and faster solves; it changes the result only
    at rounding level and is what the cost comparison uses."""
    import time
    t0 = time.perf_counter()
    if isinstance(domain, Rectangle):
        x1, x2, ins, fidx, n_unk, L, _dx = _build_rect_system(market, domain, N)
    else:
        x1, x2, ins, fidx, n_unk, L, _dx = _build_curved_system(market, domain, N)
    t1 = time.perf_counter()
    dt = market.T / N_time
    I_sp = sparse.eye(n_unk, format='csr')
    u = _initial_condition_curved(market, payoff, x1, x2, ins, fidx, n_unk)
    t_ic = time.perf_counter()
    n_cn = N_time
    t_lu = 0.0
    if rannacher and rannacher_steps > 0:
        dt_half = dt / 2.0
        ta = time.perf_counter()
        ie_solver = splu((I_sp - dt_half * L).tocsc(), permc_spec=permc_spec)
        t_lu += time.perf_counter() - ta
        for _ in range(2 * rannacher_steps):
            u = ie_solver.solve(u)
        n_cn -= rannacher_steps
    ta = time.perf_counter()
    solver = splu((I_sp - 0.5 * dt * L).tocsc(), permc_spec=permc_spec)
    t_lu += time.perf_counter() - ta
    B = I_sp + 0.5 * dt * L
    levels = deque([u], maxlen=3)
    for _ in range(n_cn):
        u = solver.solve(B @ u)
        levels.append(u)
    t2 = time.perf_counter()
    return XSolution(market=market, domain=domain, N=N, N_time=N_time, dt=dt,
                     x1=x1, x2=x2, inside=ins, flat_idx=fidx, L=L, u=u,
                     levels=list(levels),
                     wall=dict(assemble_s=t1 - t0, solve_s=t2 - t1, total_s=t2 - t0,
                               payoff_s=t_ic - t1, lu_s=t_lu, steps_s=t2 - t_ic - t_lu))
