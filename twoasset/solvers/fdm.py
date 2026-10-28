"""Finite-difference solver for the two-asset gauged heat equation."""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..core.domain import EuclideanCircle, MahalanobisEllipse, Rectangle
from ..core.payoff import BasketCall


# ═══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _quad_theta(x1, x2, step_x1, step_x2, a, b, c):
    """Find θ ∈ (0, 1] s.t. (x1+θ·dx1, x2+θ·dx2) satisfies a·X₁² + b·X₂² + c·X₁X₂ =
    R² i.e. the earliest forward intersection with the barrier curve."""
    # Substitute X_i = x_i + θ·dx_i and collect in θ:
    # A θ² + B θ + C = 0  where
    A = a*step_x1**2 + b*step_x2**2 + c*step_x1*step_x2
    B = 2*a*x1*step_x1 + 2*b*x2*step_x2 + c*(x1*step_x2 + x2*step_x1)
    C = a*x1**2 + b*x2**2 + c*x1*x2
    # C is the "how far inside" term; we want smallest positive root ≤ 1.
    if abs(A) < 1e-30:
        return 1.0
    disc = B*B - 4*A*(C - 0.0)  # RHS is 0 after shifting by R² — caller
    if disc < 0:
        return 1.0
    sd = np.sqrt(disc)
    roots = [(-B + sd)/(2*A), (-B - sd)/(2*A)]
    valid = [t for t in roots if 1e-10 < t <= 1.0 + 1e-8]
    if not valid:
        return 1.0
    return max(min(min(valid), 1.0), 1e-6)


def _barrier_form(market, domain):
    """Return (a, b, c, R²) coefficients of the barrier xᵀM x = R² in x-space."""
    if isinstance(domain, EuclideanCircle):
        return 1.0, 1.0, 0.0, domain.R**2
    if isinstance(domain, MahalanobisEllipse):
        Sinv = np.linalg.inv(market.Sigma)
        return float(Sinv[0, 0]), float(Sinv[1, 1]), float(2*Sinv[0, 1]), domain.R**2
    raise ValueError(f"Unsupported curved domain: {type(domain).__name__}")


# ═══════════════════════════════════════════════════════════════════════
#  Curved-barrier (circle / ellipse in x-space) FDM
# ═══════════════════════════════════════════════════════════════════════

def _build_curved_system(market, domain, N, offset=None):
    """Build the Shortley–Weller FDM operator for a quadratic barrier in x-space."""
    a, b, c, R2 = _barrier_form(market, domain)
    R = np.sqrt(R2)
    s1, s2, rho, r = market.sigma1, market.sigma2, market.rho, market.r
    if isinstance(domain, EuclideanCircle):
        half = R
    else:
        M = np.linalg.inv(market.Sigma)
        evs = np.linalg.eigvalsh(M)
        half = R / np.sqrt(max(evs.min(), 1e-12))
    dx = 2.0 * half / (N + 1)

    if offset is None:
        x1_pts = np.array([-half + (i+1)*dx for i in range(N)])
        x2_pts = np.array([-half + (j+1)*dx for j in range(N)])
    else:
        o1, o2 = (float(o) for o in offset)
        if not (-0.5 <= o1 < 0.5 and -0.5 <= o2 < 0.5):
            raise ValueError(f"grid offset must lie in [-0.5, 0.5), got {offset}")
        x1_pts = np.array([-half + (i + o1)*dx for i in range(N + 2)])
        x2_pts = np.array([-half + (j + o2)*dx for j in range(N + 2)])
        N = N + 2

    inside = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            inside[i, j] = (a*x1_pts[i]**2 + b*x2_pts[j]**2
                            + c*x1_pts[i]*x2_pts[j]) < R2

    flat_idx = -np.ones((N, N), dtype=int)
    count = 0
    for i in range(N):
        for j in range(N):
            if inside[i, j]:
                flat_idx[i, j] = count
                count += 1
    n_unk = count

    a11 = 0.5 * s1**2
    a22 = 0.5 * s2**2
    a12 = rho * s1 * s2
    b1 = r - 0.5 * s1**2
    b2 = r - 0.5 * s2**2

    rows, cols, vals = [], [], []

    def is_in(ii, jj):
        return 0 <= ii < N and 0 <= jj < N and inside[ii, jj]

    def add_entry(row, col, val):
        rows.append(row); cols.append(col); vals.append(val)

    def theta_to_barrier(x1, x2, dx1, dx2):
        # Solve  a(x1+θdx1)² + b(x2+θdx2)² + c(x1+θdx1)(x2+θdx2) = R²
        A = a*dx1**2 + b*dx2**2 + c*dx1*dx2
        B = 2*a*x1*dx1 + 2*b*x2*dx2 + c*(x1*dx2 + x2*dx1)
        C = a*x1**2 + b*x2**2 + c*x1*x2 - R2
        if abs(A) < 1e-30:
            return 1.0
        disc = B*B - 4*A*C
        if disc < 0:
            return 1.0
        sd = np.sqrt(disc)
        roots = [(-B + sd)/(2*A), (-B - sd)/(2*A)]
        valid = [t for t in roots if 1e-10 < t <= 1.0 + 1e-8]
        if not valid:
            return 1.0
        return max(min(min(valid), 1.0), 1e-6)

    for i in range(N):
        for j in range(N):
            if not inside[i, j]:
                continue
            k = flat_idx[i, j]
            x1c, x2c = x1_pts[i], x2_pts[j]
            diag = -r

            # ── x₁ direction ──
            ip_in = is_in(i+1, j)
            im_in = is_in(i-1, j)
            hp = dx if ip_in else theta_to_barrier(x1c, x2c, dx, 0) * dx
            hm = dx if im_in else theta_to_barrier(x1c, x2c, -dx, 0) * dx
            coeff_plus_2  = a11 / (hp * (hp + hm) / 2)
            coeff_minus_2 = a11 / (hm * (hp + hm) / 2)
            coeff_cent_2  = -(coeff_plus_2 + coeff_minus_2)
            if ip_in: add_entry(k, flat_idx[i+1, j], coeff_plus_2)
            if im_in: add_entry(k, flat_idx[i-1, j], coeff_minus_2)
            diag += coeff_cent_2

            denom_1 = hp * hm * (hp + hm)
            coeff_plus_1  = b1 * hm**2 / denom_1
            coeff_minus_1 = -b1 * hp**2 / denom_1
            coeff_cent_1  = b1 * (hp**2 - hm**2) / denom_1
            if ip_in: add_entry(k, flat_idx[i+1, j], coeff_plus_1)
            if im_in: add_entry(k, flat_idx[i-1, j], coeff_minus_1)
            diag += coeff_cent_1

            # ── x₂ direction ──
            jp_in = is_in(i, j+1)
            jm_in = is_in(i, j-1)
            hp2 = dx if jp_in else theta_to_barrier(x1c, x2c, 0, dx) * dx
            hm2 = dx if jm_in else theta_to_barrier(x1c, x2c, 0, -dx) * dx
            coeff_plus_2  = a22 / (hp2 * (hp2 + hm2) / 2)
            coeff_minus_2 = a22 / (hm2 * (hp2 + hm2) / 2)
            coeff_cent_2  = -(coeff_plus_2 + coeff_minus_2)
            if jp_in: add_entry(k, flat_idx[i, j+1], coeff_plus_2)
            if jm_in: add_entry(k, flat_idx[i, j-1], coeff_minus_2)
            diag += coeff_cent_2

            denom_1 = hp2 * hm2 * (hp2 + hm2)
            coeff_plus_1  = b2 * hm2**2 / denom_1
            coeff_minus_1 = -b2 * hp2**2 / denom_1
            coeff_cent_1  = b2 * (hp2**2 - hm2**2) / denom_1
            if jp_in: add_entry(k, flat_idx[i, j+1], coeff_plus_1)
            if jm_in: add_entry(k, flat_idx[i, j-1], coeff_minus_1)
            diag += coeff_cent_1

            # ── Cross derivative ──
            c12 = a12 / (4.0 * dx**2)
            if is_in(i+1, j+1): add_entry(k, flat_idx[i+1, j+1],  c12)
            if is_in(i+1, j-1): add_entry(k, flat_idx[i+1, j-1], -c12)
            if is_in(i-1, j+1): add_entry(k, flat_idx[i-1, j+1], -c12)
            if is_in(i-1, j-1): add_entry(k, flat_idx[i-1, j-1],  c12)

            add_entry(k, k, diag)

    L = sparse.csr_matrix((vals, (rows, cols)), shape=(n_unk, n_unk))
    return x1_pts, x2_pts, inside, flat_idx, n_unk, L, dx


def _initial_condition_curved(market, payoff, x1_pts, x2_pts, inside,
                              flat_idx, n_unk):
    """Raw payoff on the grid — no gauge applied yet."""
    u = np.zeros(n_unk)
    N = len(x1_pts)
    for i in range(N):
        for j in range(N):
            if inside[i, j]:
                S1 = market.S1_0 * np.exp(x1_pts[i])
                S2 = market.S2_0 * np.exp(x2_pts[j])
                u[flat_idx[i, j]] = float(payoff(S1, S2))
    return u


def _interp2d(u, x1_pt, x2_pt, x1_grid, x2_grid, inside, flat_idx, N):
    dx1 = x1_grid[1] - x1_grid[0]
    dx2 = x2_grid[1] - x2_grid[0]
    i_f = (x1_pt - x1_grid[0]) / dx1
    j_f = (x2_pt - x2_grid[0]) / dx2
    i_lo = max(0, min(int(np.floor(i_f)), N - 2))
    j_lo = max(0, min(int(np.floor(j_f)), N - 2))
    fx = max(0.0, min(1.0, i_f - i_lo))
    fy = max(0.0, min(1.0, j_f - j_lo))

    def val(i, j):
        if 0 <= i < N and 0 <= j < N and inside[i, j]:
            return u[flat_idx[i, j]]
        return 0.0

    return ((1-fx)*(1-fy)*val(i_lo, j_lo) + fx*(1-fy)*val(i_lo+1, j_lo)
            + (1-fx)*fy*val(i_lo, j_lo+1) + fx*fy*val(i_lo+1, j_lo+1))


# ═══════════════════════════════════════════════════════════════════════
#  Rectangular FDM
# ═══════════════════════════════════════════════════════════════════════

def _build_rect_system(market, domain, N):
    """Uniform grid on the rectangle with Dirichlet on all four edges."""
    dx1 = domain.W1 / (N + 1)
    dx2 = domain.W2 / (N + 1)
    x1_pts = np.array([domain.x1_lo + (i+1)*dx1 for i in range(N)])
    x2_pts = np.array([domain.x2_lo + (j+1)*dx2 for j in range(N)])

    a11 = market.a11
    a22 = market.a22
    a12 = market.a12
    b1 = market.r - 0.5 * market.sigma1**2
    b2 = market.r - 0.5 * market.sigma2**2
    r = market.r

    n_unk = N * N
    rows, cols, vals = [], [], []
    fl = lambda i, j: i * N + j

    def add(rr, cc, vv):
        rows.append(rr); cols.append(cc); vals.append(vv)

    for i in range(N):
        for j in range(N):
            k = fl(i, j)
            diag = -r

            # a11 u_x1x1 + b1 u_x1
            if i + 1 < N: add(k, fl(i+1, j), a11/dx1**2 + b1/(2*dx1))
            if i - 1 >= 0: add(k, fl(i-1, j), a11/dx1**2 - b1/(2*dx1))
            diag += -2*a11/dx1**2

            # a22 u_x2x2 + b2 u_x2
            if j + 1 < N: add(k, fl(i, j+1), a22/dx2**2 + b2/(2*dx2))
            if j - 1 >= 0: add(k, fl(i, j-1), a22/dx2**2 - b2/(2*dx2))
            diag += -2*a22/dx2**2

            # a12 u_x1x2 cross
            c12 = a12 / (4.0 * dx1 * dx2)
            if i+1 < N and j+1 < N: add(k, fl(i+1, j+1),  c12)
            if i+1 < N and j-1 >= 0: add(k, fl(i+1, j-1), -c12)
            if i-1 >= 0 and j+1 < N: add(k, fl(i-1, j+1), -c12)
            if i-1 >= 0 and j-1 >= 0: add(k, fl(i-1, j-1), c12)

            add(k, k, diag)

    L = sparse.csr_matrix((vals, (rows, cols)), shape=(n_unk, n_unk))
    # Represent "inside" mask and flat index as dense for interface uniformity
    inside = np.ones((N, N), dtype=bool)
    flat_idx = np.arange(n_unk).reshape(N, N)
    return x1_pts, x2_pts, inside, flat_idx, n_unk, L, dx1


# ═══════════════════════════════════════════════════════════════════════
#  Public entry points
# ═══════════════════════════════════════════════════════════════════════

def fdm_price(market, domain, payoff=None, *,
              N: int = 160, N_time: int = 500,
              S1_target: float = None, S2_target: float = None,
              rannacher: bool = True, rannacher_steps: int = 4,
              return_full: bool = False, offset=None) -> dict:
    """Solve the raw (non-gauged) Black–Scholes PDE to t = 0 and return V at the spot
    (S₁⁰, S₂⁰) by default, or at (S1_target, S2_target) if given."""
    if payoff is None:
        raise ValueError("fdm_price requires an explicit payoff callable.")
    if isinstance(domain, Rectangle):
        if offset is not None:
            raise ValueError("grid offset applies to curved barriers only")
        x1, x2, ins, fidx, n_unk, L, dx = _build_rect_system(market, domain, N)
    else:
        x1, x2, ins, fidx, n_unk, L, dx = _build_curved_system(
            market, domain, N, offset=offset)
    n_grid = len(x1)    # N, or N + 2 on a translated grid
    if n_unk == 0:
        out = dict(price=0.0, N=N, N_time=N_time)
        if return_full:
            out.update(u_grid=np.full((n_grid, n_grid), np.nan),
                        x1v=x1, x2v=x2)
        return out

    dt = market.T / N_time
    I_sp = sparse.eye(n_unk, format='csr')
    u = _initial_condition_curved(market, payoff, x1, x2, ins, fidx, n_unk)

    # ── Rannacher warm-up: 2·rannacher_steps half-steps of implicit Euler ──
    n_cn = N_time
    if rannacher and rannacher_steps > 0:
        dt_half = dt / 2.0
        ie_solver = splu((I_sp - dt_half * L).tocsc())
        for _ in range(2 * rannacher_steps):
            u = ie_solver.solve(u)   # implicit Euler: (I − dt·L) u_{n+1} = u_n
        n_cn -= rannacher_steps      # the warm-up advanced by exactly
                                     # rannacher_steps full CN equivalents

    # ── Crank–Nicolson sweep for the remainder ──
    solver = splu((I_sp - 0.5*dt*L).tocsc())
    B = I_sp + 0.5*dt*L
    for _ in range(n_cn):
        u = solver.solve(B @ u)

    # Evaluate at the requested log-price point
    if S1_target is None: S1_target = market.S1_0
    if S2_target is None: S2_target = market.S2_0
    x1_tgt = float(np.log(S1_target / market.S1_0))
    x2_tgt = float(np.log(S2_target / market.S2_0))
    price = _interp2d(u, x1_tgt, x2_tgt, x1, x2, ins, fidx, n_grid)

    out = dict(price=price, N=N, N_time=N_time)
    if return_full:
        # Reconstitute the full N×N grid; NaN outside the domain.
        U = np.full((n_grid, n_grid), np.nan, dtype=float)
        for i in range(n_grid):
            for j in range(n_grid):
                if ins[i, j]:
                    U[i, j] = u[fidx[i, j]]
        out.update(u_grid=U, x1v=x1, x2v=x2)
    return out


def fdm_smooth(market, domain, mu_flat, nu_flat, basis, *,
               payoff=None, delta_tau=None,
               N_grid: int = 100, N_steps: int = 50):
    """Advance the gauged payoff by a small δτ, then sample the result in the basis's
    natural coordinates."""
    if delta_tau is None:
        delta_tau = 0.02 * market.T
    if payoff is None:
        raise ValueError("fdm_smooth requires an explicit payoff callable.")

    if isinstance(domain, Rectangle):
        x1, x2, ins, fidx, n_unk, L, dx = _build_rect_system(market, domain, N_grid)
    else:
        x1, x2, ins, fidx, n_unk, L, dx = _build_curved_system(market, domain, N_grid)
    if n_unk == 0:
        return np.zeros(len(mu_flat)), delta_tau

    dt = delta_tau / N_steps
    I_sp = sparse.eye(n_unk, format='csr')
    u = _initial_condition_curved(market, payoff, x1, x2, ins, fidx, n_unk)
    solver = splu((I_sp - 0.5*dt*L).tocsc())
    B = I_sp + 0.5*dt*L
    for _ in range(N_steps):
        u = solver.solve(B @ u)

    # Sample at the basis's natural-coord nodes
    b_smooth = np.zeros(len(mu_flat))
    for i in range(len(mu_flat)):
        if hasattr(basis, "_elliptic_to_spot"):
            S1, S2 = basis._elliptic_to_spot(mu_flat[i], nu_flat[i])
        elif hasattr(basis, "_polar_to_spot"):
            S1, S2 = basis._polar_to_spot(mu_flat[i], nu_flat[i])
        else:
            raise ValueError("basis must expose _elliptic_to_spot or "
                             "_polar_to_spot")
        x1v, x2v = market.spot_to_x(S1, S2)
        # Skip any sample outside the domain
        if isinstance(domain, Rectangle):
            if not domain.contains(x1v, x2v):
                continue
        elif isinstance(domain, EuclideanCircle):
            if x1v**2 + x2v**2 >= domain.R**2:
                continue
        else:  # MahalanobisEllipse
            a, b, c, R2 = _barrier_form(market, domain)
            if a*x1v**2 + b*x2v**2 + c*x1v*x2v >= R2:
                continue
        U = _interp2d(u, x1v, x2v, x1, x2, ins, fidx, N_grid)
        lnS1 = x1v + np.log(market.S1_0)
        lnS2 = x2v + np.log(market.S2_0)
        gauge = np.exp(market.beta1*lnS1 + market.beta2*lnS2 + market.gamma*delta_tau)
        b_smooth[i] = U / gauge if abs(gauge) > 1e-100 else 0.0

    return b_smooth, delta_tau
